"""Hierarchical (H-matrix) operator assembled from coefficient queries.

Panels are clustered by the spatial bisection of ``spatial_order``; a block of
two clusters is admissible when ``min(diam) <= ETA * dist`` of their bounding
boxes, inflated by each cluster's largest panel. Inadmissible leaf blocks are
evaluated exactly with the ordinary coefficient oracle (and so with its near,
junction, thin-gap and acute-corner quadrature). The admissible blocks of one
row cluster at one tree level (its interaction list) are compressed together
by a batched adaptive cross approximation that only evaluates selected rows
and columns, then recompressed by QR/SVD. No dense or full-tile matrix is
formed: storage and coefficient work are proportional to the retained ranks.

The approximation is not certified by a deterministic bound. Its error is
measured: every leaf cluster contributes exactly evaluated sample rows, whose
complete row error ``sum_j |A_ij - H_ij|`` is compared with the operator norm
after assembly, and the same exact rows check every published solve's
residual (``hmatrix_factor``).
"""
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.linalg as la
from ghost_backend.linalg.hierarchical import spatial_order, HierarchicalRejected
from ghost_backend.execution.metrics import timed_stage

# Unknowns from which the compressed backend assembles this operator instead
# of evaluating every coefficient tile (see NUMERICAL_METHODS.md, "Hierarchical
# 2-D operator"); GHOST_HMATRIX_MIN_UNKNOWNS overrides it, 0 disables it.
HMATRIX_MIN_UNKNOWNS = 8192
# Leaf clusters hold about this many distinct points: coincident densities
# (coatings, region interfaces) multiply the unknowns per leaf (at most 512,
# the inverse tree's leaf limit).
LEAF = 128
ETA = 1.
# Mean row-sum error target of one compressed interaction list, relative to
# the sampled operator norm ||A||_inf. Rows meet about one list per tree level.
TOLERANCE = 1e-12
# The measured relative row error after assembly must stay below this.
OPERATOR_ERROR_LIMIT = 1e-11
# Exact sample rows: at least this many, one per leaf, at most this many bytes.
SAMPLE_MIN_ROWS = 64
SAMPLE_BYTES = 256 * 1024**2
# The coefficient oracles cap one query at 16 MiB of 24-byte entries.
QUERY_ENTRIES = (16 * 1024**2) // 24
PROBES = 6
# Interaction lists whose expected rank stores more than this share of
# their coefficients are evaluated exactly instead.
DENSE_LIST_FRACTION = .5
DENSE_LIST_ENTRIES = 1 << 20
# Lists of one level compressed first to calibrate the level's rank guess.
PILOT_LISTS = 8
MAX_PASSES = 64


class HMatrixRejected(HierarchicalRejected):
    pass


def minimum_unknowns():
    from ghost_backend.execution.options import environment_value
    raw = environment_value('GHOST_HMATRIX_MIN_UNKNOWNS', '').strip()
    if not raw:
        return HMATRIX_MIN_UNKNOWNS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError('GHOST_HMATRIX_MIN_UNKNOWNS must be a nonnegative integer.')
    if value < 0:
        raise ValueError('GHOST_HMATRIX_MIN_UNKNOWNS must be a nonnegative integer.')
    return value


def selected(n):
    """Whether a compressed system of ``n`` unknowns uses the hierarchical operator."""
    threshold = minimum_unknowns()
    return threshold > 0 and int(n) >= threshold


def multiplicity(coordinates):
    """Unknowns per distinct coordinate (coincident densities share points)."""
    coordinates = np.asarray(coordinates, float)
    if not len(coordinates):
        return 1.
    span = max(float(np.ptp(coordinates)), 1e-300)
    distinct = len(np.unique(np.round(coordinates/span, 12), axis=0))
    return max(1., len(coordinates)/max(1, distinct))


def leaf_size(coordinates):
    return int(min(512, max(LEAF, round(LEAF*multiplicity(coordinates)))))


def unsupported(oracle):
    """Why a coefficient oracle keeps the tile operator, or None."""
    from ghost_backend.execution.options import option
    if option('compressed_far_method', 'full') != 'full':
        return 'experimental compressed_far_method requested'
    if hasattr(oracle, 'oracles'):
        reasons = [unsupported(source) for source in oracle.oracles]
        return next((reason for reason in reasons if reason is not None), None)
    if not hasattr(oracle, 'mesh') or not hasattr(oracle, 'get_with_error'):
        return 'oracle without mesh geometry'
    if getattr(oracle, 'hmatrix_rejected', None):
        return oracle.hmatrix_rejected
    if getattr(oracle, 'kind', None) == 'thin' and getattr(oracle, 'B', 0) != 0:
        # Every coefficient column is a solve with the inverse mass matrix.
        return 'thin dielectric layer with a magnetic contrast couples rows through the inverse mass matrix'
    return None


def support_radius(mesh, coordinates):
    """Largest adjacent panel length at each DOF coordinate (basis support)."""
    from scipy.spatial import cKDTree
    nodes = np.asarray([node.xy for node in mesh.nodes], float)
    radius = np.zeros(len(nodes))
    for element in mesh.elements:
        for node in element.node_ids:
            radius[node] = max(radius[node], float(element.length))
    distance, nearest = cKDTree(nodes).query(np.asarray(coordinates, float))
    longest = float(radius.max()) if len(radius) else 0.
    # A DOF not located on a mesh node keeps the conservative global value.
    return np.where(distance <= 1e-9 * max(longest, 1e-300) + 1e-15, radius[nearest], longest)


class ClusterTree:
    """Bisection tree in ``spatial_order``'s permutation: clusters are ranges."""
    def __init__(self, coordinates, radius, leaf):
        n = len(coordinates)
        self.order = spatial_order(coordinates, np.arange(n), leaf)
        self.position = np.empty(n, np.int64)
        self.position[self.order] = np.arange(n)
        xy = coordinates[self.order]
        pad = radius[self.order]
        start, stop, left, right, level, lo, hi = [], [], [], [], [], [], []
        pending = [(0, n, 0, -1, 0)]
        while pending:
            a, b, depth, parent, side = pending.pop()
            index = len(start)
            start.append(a); stop.append(b); level.append(depth)
            left.append(-1); right.append(-1)
            inflate = float(pad[a:b].max())
            lo.append(xy[a:b].min(axis=0) - inflate); hi.append(xy[a:b].max(axis=0) + inflate)
            if parent >= 0:
                (left if side == 0 else right)[parent] = index
            # spatial_order splits every cluster above the leaf size at len//2.
            if b - a > leaf:
                mid = a + (b - a)//2
                pending.append((mid, b, depth+1, index, 1))
                pending.append((a, mid, depth+1, index, 0))
        self.start, self.stop = np.asarray(start), np.asarray(stop)
        self.left, self.right, self.level = np.asarray(left), np.asarray(right), np.asarray(level)
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.diameter = np.linalg.norm(self.hi - self.lo, axis=1)

    def size(self, k):
        return int(self.stop[k] - self.start[k])

    def leaf(self, k):
        return self.left[k] < 0

    def distance(self, a, b):
        gap = np.maximum(0., np.maximum(self.lo[a] - self.hi[b], self.lo[b] - self.hi[a]))
        return float(np.linalg.norm(gap))

    def partition(self, eta):
        """Interaction lists: far[k] (admissible) and near[k] (dense) partners."""
        far, near = {}, {}
        pending = [(0, 0)]
        while pending:
            a, b = pending.pop()
            if min(self.diameter[a], self.diameter[b]) <= eta*self.distance(a, b):
                far.setdefault(a, []).append(b)
            elif self.leaf(a) or self.leaf(b):
                near.setdefault(a, []).append(b)
            else:
                for x in (self.right[a], self.left[a]):
                    for y in (self.right[b], self.left[b]):
                        pending.append((x, y))
        for lists in (far, near):
            for k in lists:
                lists[k].sort(key=lambda c: self.start[c])
        return far, near


def _ranges(tree, partners):
    return [(int(tree.start[c]), int(tree.stop[c])) for c in partners]


def _range_ids(ranges):
    return np.concatenate([np.arange(a, b) for a, b in ranges]) if ranges else np.empty(0, np.int64)


class BlockRow:
    """One row cluster's partners at one level: dense ``u`` (v None) or ``u @ v``."""
    __slots__ = ('row', 'start', 'stop', 'ranges', 'columns', 'u', 'v', 'offsets', 'kind')

    def __init__(self, row, start, stop, ranges, u, v, kind=0):
        self.row, self.start, self.stop, self.ranges, self.kind = row, start, stop, ranges, kind
        self.columns = _range_ids(ranges)
        self.u, self.v = u, v
        self.offsets = np.r_[0, np.cumsum([b - a for a, b in ranges])].astype(np.int64)

    @property
    def nbytes(self):
        return self.u.nbytes + (0 if self.v is None else self.v.nbytes) + self.columns.nbytes

    @property
    def rank(self):
        return self.u.shape[1] if self.v is not None else None

    def values(self, rows=None, columns=None):
        """Entries for local row/column positions (all when None)."""
        u = self.u if rows is None else self.u[rows]
        if self.v is None:
            return u if columns is None else u[:, columns]
        return u @ (self.v if columns is None else self.v[:, columns])

    def local_columns(self, positions):
        """Local column indices and the selected entries of sorted permuted ``positions``."""
        local, selected = [], []
        for (a, b), offset in zip(self.ranges, self.offsets):
            i, j = np.searchsorted(positions, (a, b))
            if j > i:
                local.append(positions[i:j] - a + offset)
                selected.append(np.arange(i, j))
        if not local:
            return None, None
        return np.concatenate(local), np.concatenate(selected)


class _Fetcher:
    """Exact coefficients of permuted positions, within the oracle's query cap.

    A paired oracle (``oracle.oracles``) answers one query for every
    polarization through one kernel traversal; results are lists by target.
    """
    def __init__(self, oracle, order, checkpoint, tails=False):
        from ghost_backend.compressed.coefficients import PairedNativeOracle
        if type(oracle) is PairedNativeOracle:
            # Its shared primitive cache lives on the source oracles for one
            # call; concurrent queries each need their own (shallow) sources.
            import copy
            oracle = PairedNativeOracle(*(copy.copy(source) for source in oracle.oracles))
        self.oracle, self.order, self.checkpoint = oracle, order, checkpoint
        self.count = len(getattr(oracle, 'oracles', [oracle]))
        self.calls = self.entries = 0
        self.tails = [np.zeros(len(order)) for _ in range(self.count)] if tails else None

    def __call__(self, rows, cols):
        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        results = [np.empty((len(rows), len(cols)), complex) for _ in range(self.count)]
        if not len(rows) or not len(cols):
            return results
        width = max(1, QUERY_ENTRIES // max(1, len(cols)))
        step = max(1, QUERY_ENTRIES // max(1, min(width, len(rows))))
        for r0 in range(0, len(rows), width):
            for c0 in range(0, len(cols), step):
                self.checkpoint()
                rr, cc = rows[r0:r0+width], cols[c0:c0+step]
                answers = self.oracle.get_with_error(self.order[rr], self.order[cc])
                if self.count == 1:
                    answers = [answers]
                for index, (value, error) in enumerate(answers):
                    if value.shape != (len(rr), len(cc)) or not np.all(np.isfinite(value)):
                        raise ValueError('Oracle returned invalid coefficients.')
                    if error.shape != value.shape or not np.all(np.isfinite(error)) or np.any(error < 0):
                        raise ValueError('Oracle returned invalid coefficient error bounds.')
                    results[index][r0:r0+width, c0:c0+step] = value
                    if self.tails is not None and np.any(error):
                        np.add.at(self.tails[index], rr, np.sum(error, axis=1))
                self.calls += 1
                self.entries += len(rr)*len(cc)
        return results


def _spread(count, size):
    if size <= 0 or count <= 0:
        return np.empty(0, np.int64)
    return np.unique(np.linspace(0, size - 1, min(count, size)).round().astype(np.int64))


def _fresh(used, count, rng):
    free = np.flatnonzero(~used)
    if not len(free):
        return np.empty(0, np.int64)
    return np.sort(rng.choice(free, min(count, len(free)), replace=False))


def _leading_columns(rows, threshold, keep=None):
    """Pivoted-QR column skeleton of evaluated rows: indices with |R_kk| > threshold.

    Columns in ``keep`` (an earlier skeleton) are retained; only the part of
    the rows they do not span selects further columns.
    """
    keep = np.empty(0, np.int64) if keep is None else np.asarray(keep, np.int64)
    residual = rows
    if len(keep):
        q, _ = la.qr(rows[:, keep], mode='economic', check_finite=False)
        residual = rows - q @ (q.conj().T @ rows)
        residual[:, keep] = 0
    _, r, permutation = la.qr(residual, mode='economic', pivoting=True, check_finite=False)
    diagonal = abs(np.diag(r))
    scale = max(float(diagonal[0]) if len(diagonal) else 0., float(np.max(np.linalg.norm(rows, axis=0))) if rows.size else 0.)
    if not scale > 0:
        return keep
    rank = int(np.count_nonzero(diagonal > max(threshold, 1e-14*scale)))
    return np.union1d(keep, permutation[:rank]).astype(np.int64)


def _solve(matrix, rhs):
    """Minimum-norm least-squares solution (rank-revealing QR, cutoff 1e-14)."""
    if not matrix.size or not rhs.size:
        return np.zeros((matrix.shape[1], rhs.shape[1]), complex)
    return la.lstsq(matrix, rhs, cond=1e-14, lapack_driver='gelsy', check_finite=False)[0]


def cross_approximation(rows, columns, shape, tolerance, start_rank, rng, max_rank=None):
    """Sampled column skeleton of implicit blocks: ``rows(I)``/``columns(J)`` evaluate them.

    ``rows`` and ``columns`` return one array per target block (polarizations
    of one paired query); ``tolerance`` is a matching sequence. A pass
    evaluates sample rows (spread, then random and the rows the skeleton
    weights most), takes one pivoted-QR column skeleton ``J`` of all targets'
    sampled rows (each scaled by its tolerance; a Gaussian sketch of the
    samples once they far exceed the skeleton) and fits every
    ``A ~ A[:, J] X`` to its sampled rows by least squares. Averaging over
    more rows than the rank keeps rounding-level noise in the coefficients
    from being interpolated into the approximation. Fresh probe rows and then
    fresh probe columns must estimate a Frobenius residual within each
    tolerance, with no probe above four times its mean share; otherwise the
    probes join the sample and the sample grows (doubling while far from the
    target). Each result is recompressed by QR/SVD (``tolerance/2`` tail).
    Returns a list of ``(u, v)`` and the largest residual estimate relative to
    its tolerance.
    """
    m, n = shape
    tolerance = np.atleast_1d(np.asarray(tolerance, float))
    limit = min(m, n) if max_rank is None else int(max_rank)
    unit = 1. / math.sqrt(m*n)
    sampled = np.zeros(m, bool)
    first = _spread(min(m, max(8, int(1.25*start_rank) + 16)), m)
    sampled[first] = True
    index = first
    values = rows(first)
    count = len(values)
    have = np.zeros(n, bool)
    known = [dict() for _ in range(count)]
    estimate = math.inf
    skeleton = None
    bases = inverses = None
    # Skeleton threshold per column of the sampled rows; it halves whenever a
    # failed pass did not add columns or nearly met the target (a skeleton
    # too coarse for the target).
    factor, previous_skeleton = .5, -1
    for _ in range(MAX_PASSES):
        stacked = np.vstack([value/t for value, t in zip(values, tolerance)])
        width = 0 if skeleton is None else len(skeleton)
        sketch = max(2*width, width + 64, int(1.5*start_rank) + 32)
        if stacked.shape[0] > 2*sketch:
            # Column selection needs the sampled rows' span, not every row.
            omega = (rng.standard_normal((sketch, stacked.shape[0])) +
                     1j*rng.standard_normal((sketch, stacked.shape[0])))/math.sqrt(2*sketch)
            stacked = omega @ stacked
        skeleton = _leading_columns(stacked, factor*unit*math.sqrt(len(index)), skeleton)
        stacked = omega = None
        if len(skeleton) > limit:
            raise HMatrixRejected('Interaction list rank exceeds {}.'.format(limit))
        missing = skeleton[~have[skeleton]]
        if len(missing):
            for store, fetched in zip(known, columns(missing)):
                for j, column in zip(missing, fetched.T):
                    store[int(j)] = column
            have[missing] = True
        bases = [np.stack([store[int(j)] for j in skeleton], axis=1) if len(skeleton) else np.empty((m, 0), complex)
                 for store in known]
        # A ~ B (B_I)^+ V_I: the least-squares fit, applied lazily (only probes
        # need it before convergence); B_P (B_I)^+ is the minimum-norm
        # solution of B_I^H T^H = B_P^H.
        inverses = [basis[index] for basis in bases]
        probes = _fresh(sampled, PROBES, rng)
        if len(skeleton) and np.any(~sampled):
            weight = sum(np.linalg.norm(basis/np.maximum(np.linalg.norm(basis, axis=0), 1e-300), axis=1) for basis in bases)
            weight[sampled] = -1; weight[probes] = -1
            heavy = np.argsort(-weight)[:PROBES]
            probes = np.union1d(probes, heavy[weight[heavy] > 0]).astype(np.int64)
        if not len(probes):
            estimate = 0.
            break
        probe_values = rows(probes)
        passed, ratios = True, []
        for basis, inverse, value, probe, t in zip(bases, inverses, values, probe_values, tolerance):
            transfer = _solve(inverse.conj().T, basis[probes].conj().T).conj().T if basis.shape[1] else None
            residual = probe - transfer @ value if basis.shape[1] else probe
            norms = np.linalg.norm(residual, axis=1)
            current = math.sqrt(m * float(np.mean(norms**2)))
            ratios.append(current/t)
            passed &= current <= t and float(norms.max()) <= 4*t/math.sqrt(m)
        estimate = max(ratios)
        if passed:
            probe_cols = _fresh(have, PROBES, rng)
            if not len(probe_cols):
                break
            checks = columns(probe_cols)
            for store, check in zip(known, checks):
                for j, column in zip(probe_cols, check.T):
                    store[int(j)] = column
            have[probe_cols] = True
            column_ratios = []
            worst = np.zeros(m)
            for basis, inverse, value, check, t in zip(bases, inverses, values, checks, tolerance):
                if basis.shape[1]:
                    check = check - basis @ _solve(inverse, value[:, probe_cols])
                column_norms = np.linalg.norm(check, axis=0)
                current = math.sqrt(n * float(np.mean(column_norms**2)))
                column_ratios.append(current/t)
                passed &= current <= t and float(column_norms.max()) <= 4*t/math.sqrt(n)
                worst += np.linalg.norm(check, axis=1)/t
            if passed:
                estimate = max(estimate, max(column_ratios))
                break
            # Rows where the probe columns disagree join the sample.
            worst[sampled] = -1; worst[probes] = -1
            extra = np.argsort(-worst)[:max(PROBES, len(index)//2)]
            extra = extra[worst[extra] > 0]
        else:
            # Far from the target the sample doubles, near it it grows by half.
            grow = len(index) if estimate > 1e3 else len(index)//2
            extra = _fresh(sampled | np.isin(np.arange(m), probes), max(PROBES, grow), rng)
        if len(skeleton) <= previous_skeleton or 1 < estimate < 100 and 2*len(skeleton) < len(index):
            # A residual within two decades of the target lacks a few weak
            # skeleton columns rather than sample rows.
            factor = max(factor/2, 1/16)
        previous_skeleton = len(skeleton)
        sampled[probes] = True
        add = extra[~sampled[extra]]
        sampled[add] = True
        added = rows(add) if len(add) else [None]*count
        index = np.r_[index, probes, add].astype(np.int64)
        values = [np.vstack([x for x in (value, probe, more) if x is not None])
                  for value, probe, more in zip(values, probe_values, added)]
    else:
        raise HMatrixRejected('Cross approximation did not converge in {} passes.'.format(MAX_PASSES))
    return [recompress(basis, _solve(inverse, value), t/2) for basis, inverse, value, t
            in zip(bases, inverses, values, tolerance)], estimate


def recompress(u, v, tolerance):
    """Truncated SVD of ``u @ v`` through two QR factorizations (Frobenius tail <= tolerance)."""
    if not u.shape[1]:
        return u, v
    qu, ru = la.qr(u, mode='economic', check_finite=False)
    qv, rv = la.qr(v.conj().T, mode='economic', check_finite=False)
    w, s, zh = la.svd(ru @ rv.conj().T, full_matrices=False, check_finite=False)
    tail = np.sqrt(np.cumsum((s**2)[::-1])[::-1])
    keep = int(np.count_nonzero(tail > tolerance))
    return np.ascontiguousarray(qu @ (w[:, :keep] * s[:keep])), np.ascontiguousarray(zh[:keep] @ qv.conj().T)


def _run(tasks):
    """Results of ``tasks`` in order; single-threaded coefficient queries run concurrently.

    Each task owns its random stream and inputs, so results do not depend on
    scheduling or on the number of threads.
    """
    # Coefficient queries are native kernel loops outside the GIL: unlike
    # assembly tiles they gain from SMT siblings (1.7x from 8 to 16 threads on
    # the 8-core host), so they use every allocated logical CPU.
    from ghost_backend.execution.options import allocated_cpu_budget
    budget = allocated_cpu_budget()
    workers = min(budget, len(tasks))
    if workers < 2:
        return [task() for task in tasks]
    import contextvars
    from ghost_backend.execution.options import single_thread_blas, cpu_allocation_scope
    # Fewer tasks than CPUs (the coarsest interaction lists): each task's
    # coefficient queries use the remaining CPUs for their own tiles.
    share = max(1, budget // len(tasks))
    def run(task):
        with cpu_allocation_scope(share):
            return task()
    with single_thread_blas(), ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ghost-hmatrix') as pool:
        futures = [pool.submit(contextvars.copy_context().run, run, task) for task in tasks]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


class _Ledger:
    """Retained bytes of operators built together, against one storage cap."""
    def __init__(self, budget):
        self.budget, self.bytes = budget, 0

    def reserve(self, count):
        self.bytes += int(count)
        if self.bytes > self.budget:
            raise MemoryError('Hierarchical operator exceeded its retained-storage cap.')


@timed_stage('hmatrix_assembly')
def _assemble(oracle, targets):
    """Fill ``targets`` (one per polarization of ``oracle``) from shared queries."""
    first = targets[0]
    tree, order, n, checkpoint = first.tree, first.order, first.n, first.checkpoint
    far, near = tree.partition(first.eta)
    statistics = dict(queries=0, coefficients=0)
    def count(fetch):
        statistics['queries'] += fetch.calls; statistics['coefficients'] += fetch.entries
    for target in targets:
        target.near, target.far, target.dense_far = [], [], []
        target.dense_far_rows = 0
        target._reserve(sum(a.nbytes for a in (target.order, target.position, target.coordinates)) + 64*len(tree.start))
    # Inadmissible blocks: exact coefficients, one query set per row cluster.
    plans = []
    for k in sorted(near, key=lambda c: tree.start[c]):
        a, b = int(tree.start[k]), int(tree.stop[k])
        ranges = _ranges(tree, near[k])
        columns = _range_ids(ranges)
        for target in targets:
            target._reserve(16*(b - a)*len(columns) + 8*len(columns))
        plans.append((k, a, b, ranges, columns))
    def exact(plan):
        k, a, b, ranges, columns = plan
        fetch = _Fetcher(oracle, order, checkpoint, tails=True)
        values = fetch(np.arange(a, b), columns)
        return values, [tail[a:b].copy() for tail in fetch.tails], fetch
    for target in targets:
        target.near_tail = np.zeros(n)
    for (k, a, b, ranges, _), (values, tails, fetch) in zip(plans, _run([lambda p=p: exact(p) for p in plans])):
        for target, value, tail in zip(targets, values, tails):
            target.near.append(BlockRow(k, a, b, ranges, value, None, 0))
            target.near_tail[a:b] += tail
        count(fetch)
    near_entries = statistics['coefficients']
    # Operator scales from exactly evaluated sample rows (one per leaf at least).
    _sample(oracle, targets, count)
    scales = [float(target.sample_norm.max()) for target in targets]
    if not all(scale > 0 for scale in scales):
        raise HMatrixRejected('Sampled operator rows vanish.')
    # Interaction lists, finest level first; a level's rank guess comes from
    # the finer one, so every list's inputs are fixed in advance.
    levels = sorted({int(tree.level[k]) for k in far}, reverse=True)
    previous, finer = int(16*first.density), None
    for level in levels:
        checkpoint()
        lists = sorted((k for k in far if tree.level[k] == level), key=lambda c: tree.start[c])
        def compress(k, guess=previous):
            a, b = int(tree.start[k]), int(tree.stop[k])
            ranges = _ranges(tree, far[k])
            columns = _range_ids(ranges)
            m, w = b - a, len(columns)
            # Mean row-sum error tolerance*scale: Frobenius tolerance*scale*sqrt(m/w).
            targets_tolerance = [target.tolerance*scale*math.sqrt(m / w) for target, scale in zip(targets, scales)]
            fetch = _Fetcher(oracle, order, checkpoint)
            if m*w <= DENSE_LIST_ENTRIES and guess*(m + w) >= DENSE_LIST_FRACTION*m*w:
                # The expected rank leaves little to compress: one exact query.
                # Its rank stays unmeasured and does not feed later guesses.
                dense = fetch(np.arange(a, b), columns)
                return [BlockRow(k, a, b, ranges, value, None, 2) for value in dense], None, fetch
            rng = np.random.default_rng((first.seed, int(k)))
            # Partner lists are wider than their rows: sample (cheap) columns
            # and keep a skeleton of (long) rows, i.e. compress the transpose,
            # so long rows are evaluated about once per rank.
            factors, _ = cross_approximation(lambda c: [value.T for value in fetch(np.arange(a, b), columns[c])],
                                             lambda r: [value.T for value in fetch(a + r, columns)],
                                             (w, m), targets_tolerance, guess, rng)
            result, ranks, dense = [], [], None
            for vt, ut in factors:
                u, v = np.ascontiguousarray(ut.T), np.ascontiguousarray(vt.T)
                ranks.append(u.shape[1])
                if u.shape[1]*(m + w) >= m*w:
                    # No storage gain: keep the exact coefficients instead.
                    if dense is None:
                        dense = fetch(np.arange(a, b), columns)
                    u, v = dense[len(result)], None
                result.append(BlockRow(k, a, b, ranges, u, v, 1 if v is not None else 2))
            return result, max(ranks), fetch
        # A spread pilot wave calibrates the guess for the rest of the level.
        pilot = sorted(set(lists[::max(1, len(lists)//PILOT_LISTS)][:PILOT_LISTS]))
        rest = [k for k in lists if k not in set(pilot)]
        results = dict(zip(pilot, _run([lambda k=k: compress(k) for k in pilot])))
        measured = [rank for _, rank, _ in results.values() if rank is not None]
        guess = int(1.1*max(measured)) + 8 if measured else previous
        results.update(zip(rest, _run([lambda k=k: compress(k, guess) for k in rest])))
        rank_max = None
        for k in lists:
            rows, rank, fetch = results.pop(k)
            count(fetch)
            if rank is not None:
                rank_max = max(rank_max or 0, rank)
            for target, row in zip(targets, rows):
                target._reserve(row.nbytes)
                if row.v is None:
                    target.dense_far_rows += 1
                (target.far if row.v is not None else target.dense_far).append(row)
        # Ranks grow by a level-dependent factor toward the root (about 1.1
        # where clusters are electrically small, 2 to 3 at high frequency).
        if rank_max is not None:
            growth = max(1., min(3., rank_max/finer)) if finer else 1.5
            previous, finer = int(growth*rank_max) + 8, max(rank_max, 1)
    for target in targets:
        target.fetch_calls, target.fetch_entries = statistics['queries'], statistics['coefficients']
        target.shared_query_targets = len(targets)
        target._index()
        target._norms()
        target._validate()
        target._evidence(near_entries)


def _sample(oracle, targets, count):
    first = targets[0]
    tree, n = first.tree, first.n
    leaves = np.flatnonzero(tree.left < 0)
    total = max(SAMPLE_MIN_ROWS, len(leaves))
    total = max(1, min(total, SAMPLE_BYTES // (16*n), n))
    # One row near the middle of every leaf (strata), then random rows.
    middle = ((tree.start[leaves] + tree.stop[leaves])//2).astype(np.int64)
    middle = middle[_spread(total, len(middle))] if len(middle) > total else middle
    used = np.zeros(n, bool); used[middle] = True
    rows = np.sort(np.r_[middle, _fresh(used, total - len(middle), first.rng)]).astype(np.int64)
    for target in targets:
        target._reserve(16*len(rows)*n + 8*len(rows))
        target.sample_rows = rows
        # Exact rows in permuted column order; the tail holds dropped-route bounds.
        target.sample_values = np.empty((len(rows), n), complex)
        target.sample_tail = np.zeros(len(rows))
    step = max(1, QUERY_ENTRIES // n)
    def part(start):
        fetch = _Fetcher(oracle, first.order, first.checkpoint, tails=True)
        chunk = rows[start:start + step]
        return fetch(chunk, np.arange(n)), [tail[chunk].copy() for tail in fetch.tails], fetch
    starts = list(range(0, len(rows), step))
    for start, (values, tails, fetch) in zip(starts, _run([lambda s=s: part(s) for s in starts])):
        for target, value, tail in zip(targets, values, tails):
            target.sample_values[start:start + step] = value
            target.sample_tail[start:start + step] = tail
        count(fetch)
    for target in targets:
        target.sample_norm = np.sum(abs(target.sample_values), axis=1) + target.sample_tail


# Forecast ranks (calibration: PEC, coated and airfoil systems at 10 GHz,
# NUMERICAL_METHODS.md): an interaction list of a cluster of electrical
# diameter kD needs about 12 + 0.21 kD terms per distinct coincident density,
# the inverse's off-diagonal blocks about 0.25 kD at the root (where a closed
# body's halves face each other) and 0.09 kD below it (kD of the
# bounding-box diagonal). The forecast adds margins.
FORECAST_RANK_BASE = 16
FORECAST_RANK_SLOPE = .3
FORECAST_INVERSE_BASE = 32
FORECAST_INVERSE_SLOPE = (.3, .12)
# Coefficients evaluated per retained far coefficient (samples and probes).
FORECAST_FETCH_RATIO = 1.8


def forecast_storage(coordinates, radius, wavenumber, leaf=None, eta=ETA):
    """Retained bytes and coefficient work of the operator and its inverse, from geometry only.

    The cluster tree and interaction lists are those assembly would build;
    ranks follow the calibrated model above (electrical cluster size times
    the number of densities sharing a point), so no coefficient is evaluated.
    """
    coordinates = np.asarray(coordinates, float)
    n = len(coordinates)
    radius = np.zeros(n) if radius is None else np.asarray(radius, float)
    tree = ClusterTree(coordinates, radius, leaf_size(coordinates) if leaf is None else leaf)
    far, near = tree.partition(eta)
    density = multiplicity(coordinates)
    k = abs(complex(wavenumber))
    near_entries = sum(tree.size(a)*sum(tree.size(b) for b in partners) for a, partners in near.items())
    far_entries, top_lists = 0, []
    for a, partners in far.items():
        m = tree.size(a); w = sum(tree.size(b) for b in partners)
        rank = min(m, w, int(math.ceil(density*(FORECAST_RANK_BASE + FORECAST_RANK_SLOPE*k*tree.diameter[a]))))
        far_entries += min(rank*(m + w), m*w)
        top_lists.append((rank*(m + w), m, w, rank))
    samples = max(1, min(max(SAMPLE_MIN_ROWS, int(np.count_nonzero(tree.left < 0))), SAMPLE_BYTES // (16*n), n))
    operator = 16*(near_entries + far_entries + samples*n) + 24*n*int(tree.level.max() + 2) + 64*len(tree.start)
    # Inverse (compressed.inverse tree): per node v12, v21, e1, e2 and a
    # 2r Woodbury LU; leaves keep LU factors of their diagonal blocks.
    inverse = peak = 0
    inverse_rank = 0
    pending = [0]
    while pending:
        c = pending.pop()
        size = tree.size(c)
        if tree.leaf(c):
            inverse += 16*size*size + 8*size
            continue
        slope = FORECAST_INVERSE_SLOPE[0 if c == 0 else 1]
        rank = min(size//2, int(math.ceil(density*(FORECAST_INVERSE_BASE + slope*k*tree.diameter[c]))))
        inverse_rank = max(inverse_rank, rank)
        inverse += 32*size*rank + 16*(2*rank)**2 + 8*2*rank
        # Construction also holds u12/u21 and the range sampler's workspace.
        peak = max(peak, 48*size*rank)
        pending.extend((tree.left[c], tree.right[c]))
    top_lists.sort(reverse=True)
    from ghost_backend.compressed.operator import _thread_budget
    threads = max(1, _thread_budget())
    # Concurrent list compressions hold samples, skeleton rows and fits.
    workspace = sum(16*4*(m + w)*rank for _, m, w, rank in top_lists[:threads])
    return dict(method='hmatrix_model', operator_bytes=int(operator), operator_allowance_bytes=int(operator),
                inverse_bytes=int(inverse), inverse_peak_bytes=int(inverse + peak), assembly_workspace_bytes=int(workspace),
                coefficients=int(near_entries + samples*n + FORECAST_FETCH_RATIO*far_entries),
                retained_entries=int(near_entries + far_entries), inverse_rank=int(inverse_rank),
                density_multiplicity=float(density), wavenumber=float(k),
                sampled=False, samples=0, hmatrix=True)


def build_pair(oracle, coordinates, radius=None, **options):
    """Hierarchical operators of both polarizations of a paired oracle, one traversal."""
    sources = list(oracle.oracles)
    ledger = _Ledger(options.pop('budget', 512*1024**2))
    targets = [HMatrixOperator(source, coordinates, radius, assemble=False, ledger=ledger, **options)
               for source in sources]
    _assemble(oracle, targets)
    return targets


class HMatrixOperator:
    """Hierarchical operator with the compressed backend's operator interface."""
    compressed_operator = True

    def __init__(self, oracle, coordinates, radius=None, leaf=None, eta=ETA, tolerance=TOLERANCE,
                 budget=512*1024**2, checkpoint=None, seed=20261017, assemble=True, ledger=None):
        coordinates = np.asarray(coordinates, float)
        if oracle.n <= 0 or coordinates.ndim != 2 or coordinates.shape[0] != oracle.n or not np.all(np.isfinite(coordinates)):
            raise ValueError('Coordinates must be finite and match the nonempty system.')
        if not np.isfinite(tolerance) or not 0 < tolerance < 1 or not np.isfinite(budget) or budget <= 0:
            raise ValueError('Invalid tolerance or storage budget.')
        self.n = int(oracle.n); self.shape = (self.n, self.n)
        self.coordinates = coordinates.copy()
        self.checkpoint = checkpoint or (lambda: None)
        self.recycling_identity = getattr(oracle, 'recycling_identity', None)
        self.density = multiplicity(coordinates)
        self.leaf = leaf_size(coordinates) if leaf is None else int(leaf)
        self.eta, self.tolerance = float(eta), float(tolerance)
        self.ledger = ledger or _Ledger(budget)
        self.budget = self.ledger.budget
        self.radius = np.zeros(self.n) if radius is None else np.asarray(radius, float)
        self.tree = ClusterTree(self.coordinates, self.radius, self.leaf)
        self.order, self.position = self.tree.order, self.tree.position
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.reserved_partner_bytes = 0
        self.entries = self.calls = self.max_entries = 0
        self.oracle = oracle
        self.bytes = 0
        if assemble:
            _assemble(oracle, [self])

    # ---- assembly ---------------------------------------------------------
    def _reserve(self, count):
        self.bytes += int(count)
        self.ledger.reserve(count)

    def _evidence(self, near_entries):
        far_bytes = sum(row.nbytes for row in self.far); near_bytes = sum(row.nbytes for row in self.near + self.dense_far)
        far_ranks = [row.rank for row in self.far]
        self.evidence = dict(operator='hmatrix', unknowns=self.n, leaf=self.leaf, eta=self.eta,
            tolerance=self.tolerance, retained_bytes=self.bytes, storage_budget=self.budget,
            near_bytes=near_bytes, far_bytes=far_bytes, dense_equivalent_bytes=16*self.n**2,
            near_rows=len(self.near), interaction_lists=len(self.far), dense_interaction_lists=self.dense_far_rows,
            max_rank=max(far_ranks or [0]), mean_rank=float(np.mean(far_ranks)) if far_ranks else 0.,
            geometry_queries=self.fetch_calls, geometry_coefficients=self.fetch_entries,
            shared_query_targets=self.shared_query_targets,
            near_coefficients=near_entries, coefficient_fraction=self.fetch_entries/float(self.n)**2,
            sampled_rows=len(self.sample_rows), sampled_row_error=self.sampled_row_error,
            operator_error_limit=OPERATOR_ERROR_LIMIT,
            error_semantics='measured on exactly evaluated sample rows; not a deterministic bound')

    def _index(self):
        """Block rows grouped by cluster level: ranges of one level are disjoint."""
        self.groups = {}
        for group in (self.near, self.far, self.dense_far):
            for row in group:
                self.groups.setdefault((int(self.tree.level[row.row]), row.kind), []).append(row)
        clusters = {}
        for (level, _), rows in self.groups.items():
            for row in rows:
                clusters.setdefault(level, {}).setdefault(row.start, []).append(row)
        self.group_index = {}
        for key, rows in self.groups.items():
            rows.sort(key=lambda r: r.start)
            self.group_index[key] = (np.asarray([r.start for r in rows]), np.asarray([r.stop for r in rows]), rows)
        self.levels = []
        for level in sorted(clusters):
            starts = np.asarray(sorted(clusters[level]))
            members = [sorted(clusters[level][a], key=lambda r: r.kind) for a in starts]
            stops = np.asarray([rows[0].stop for rows in members])
            self.levels.append((starts, stops, members))

    def _passes(self, function):
        """``function(row)`` for every block row: groups in a fixed order, rows of one group on threads."""
        from ghost_backend.compressed.operator import _thread_budget
        workers = min(8, _thread_budget())
        for key in sorted(self.groups):
            group = self.groups[key]
            self.checkpoint()
            if workers < 2 or len(group) < 2:
                for row in group:
                    function(row)
                continue
            from ghost_backend.execution.options import single_thread_blas
            with single_thread_blas(), ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ghost-hmatrix') as pool:
                for _ in pool.map(function, group):
                    pass

    def _absolute(self, row_scale=None):
        """Row sums/maxima and column sums/maxima of |H| (optionally row-scaled), one pass.

        Column partial sums are kept per fixed chunk of block rows and added
        in chunk order, so the result does not depend on the thread count.
        """
        n = self.n
        rows_sum, rows_max = np.zeros(n), np.zeros(n)
        chunks = {}
        def visit(row):
            a, b = row.start, row.stop
            width = max(1, (4*1024**2)//max(1, 16*(b - a)))
            local_sum, local_max = np.zeros(b - a), np.zeros(b - a)
            col_sum, col_max = np.zeros(len(row.columns)), np.zeros(len(row.columns))
            scale = None if row_scale is None else row_scale[a:b, None]
            for c0 in range(0, len(row.columns), width):
                value = abs(row.values(None, slice(c0, c0 + width)))
                if scale is not None:
                    value /= scale
                local_sum += value.sum(axis=1); np.maximum(local_max, value.max(axis=1), out=local_max)
                col_sum[c0:c0+width] = value.sum(axis=0); col_max[c0:c0+width] = value.max(axis=0)
            chunks[(row.kind, row.row)] = (row, local_sum, local_max, col_sum, col_max)
        self._passes(visit)
        cols_sum, cols_max = np.zeros(n), np.zeros(n)
        for key in sorted(chunks):
            row, local_sum, local_max, col_sum, col_max = chunks[key]
            rows_sum[row.start:row.stop] += local_sum
            np.maximum(rows_max[row.start:row.stop], local_max, out=rows_max[row.start:row.stop])
            np.add.at(cols_sum, row.columns, col_sum)
            np.maximum.at(cols_max, row.columns, col_max)
        return rows_sum, rows_max, cols_sum, cols_max

    @timed_stage('hmatrix_norms')
    def _norms(self):
        rows_sum, rows_max, cols_sum, _ = self._absolute()
        # Original (unpermuted) numbering, as the tile operator exposes it.
        self.row_norm = np.empty(self.n); self.row_norm[self.order] = rows_sum + self.near_tail
        self.row_max = np.empty(self.n); self.row_max[self.order] = rows_max
        self.column_norm = np.empty(self.n); self.column_norm[self.order] = cols_sum
        # Compression error is measured on sample rows, not bounded per row:
        # the residual gate of hmatrix_factor uses the exact sample rows.
        self.row_error = np.zeros(self.n); self.row_error[self.order] = self.near_tail
        self.column_error = np.zeros(self.n)

    @timed_stage('hmatrix_validation')
    def _validate(self):
        approximate = self._get_positions(self.sample_rows, np.arange(self.n))
        difference = np.sum(abs(self.sample_values - approximate), axis=1)
        norm = max(float(self.row_norm.max()), float(self.sample_norm.max()))
        self.operator_norm = norm
        self.sampled_row_errors = (difference + self.sample_tail)/norm
        self.sampled_row_error = float(self.sampled_row_errors.max())
        if not np.isfinite(self.sampled_row_error) or self.sampled_row_error > OPERATOR_ERROR_LIMIT:
            raise HMatrixRejected('Sampled hierarchical operator row error {:.3g} exceeds {:.3g}.'.format(
                self.sampled_row_error, OPERATOR_ERROR_LIMIT))

    # ---- interface ----------------------------------------------------------
    def __len__(self):
        return self.n

    def __matmul__(self, value):
        return self.matmul(value)

    @property
    def nbytes(self):
        return self.bytes

    def load(self):
        pass

    def close(self):
        pass

    def matmul(self, b, trans=0):
        if trans not in (0, 1, 2):
            raise ValueError('Invalid transpose mode')
        b = np.asarray(b); vector = b.ndim == 1
        if vector:
            b = b[:, None]
        if b.ndim != 2 or b.shape[0] != self.n or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid operator RHS.')
        x = np.ascontiguousarray(b[self.order], dtype=complex)
        out = self.matmul_positions(x, trans)
        result = np.empty_like(out); result[self.order] = out
        return result[:, 0] if vector else result

    def matmul_positions(self, x, trans=0):
        """Product in the permuted numbering; deterministic for any thread count."""
        out = np.zeros((self.n, x.shape[1]), complex)
        if trans == 0:
            def visit(row):
                gathered = x[row.columns]
                value = row.u @ gathered if row.v is None else row.u @ (row.v @ gathered)
                out[row.start:row.stop] += value
            self._passes(visit)
            return out
        def adj(a):
            return a.T if trans == 1 else a.conj().T
        parts = {}
        def visit(row):
            y = x[row.start:row.stop]
            parts[(row.kind, row.row)] = (row, adj(row.u) @ y if row.v is None else adj(row.v) @ (adj(row.u) @ y))
        self._passes(visit)
        for key in sorted(parts):
            row, value = parts.pop(key)
            for (a, b), offset in zip(row.ranges, row.offsets):
                out[a:b] += value[offset:offset + b - a]
        return out

    def _blocks(self, positions):
        """Block rows meeting sorted permuted row ``positions``: (row, local rows, selected)."""
        for starts, stops, members in self.levels:
            index = np.searchsorted(stops, positions, side='right')
            inside = index < len(starts)
            inside[inside] &= starts[index[inside]] <= positions[inside]
            for k in np.unique(index[inside]):
                i, j = np.searchsorted(positions, (starts[k], stops[k]))
                for row in members[k]:
                    yield row, positions[i:j] - row.start, np.arange(i, j)

    def _get_positions(self, rows, cols):
        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        row_order, col_order = np.argsort(rows, kind='stable'), np.argsort(cols, kind='stable')
        sorted_rows, sorted_cols = rows[row_order], cols[col_order]
        result = np.zeros((len(rows), len(cols)), complex)
        for row, local_rows, selected_rows in self._blocks(sorted_rows):
            local_cols, selected_cols = row.local_columns(sorted_cols)
            if local_cols is None:
                continue
            result[np.ix_(row_order[selected_rows], col_order[selected_cols])] += row.values(local_rows, local_cols)
        return result

    def get(self, rows, cols):
        from ghost_backend.twod.assembly.compact import CompactOperator
        rows, cols = CompactOperator._ids(rows, self.n), CompactOperator._ids(cols, self.n)
        return self._get(rows, cols)

    def _get(self, rows, cols, row_plan=None, col_plan=None):
        if len(rows)*len(cols)*16 > 16*1024**2:
            raise MemoryError('Coefficient query exceeds the 16 MiB workspace limit.')
        self.entries += len(rows)*len(cols); self.calls += 1
        self.max_entries = max(self.max_entries, len(rows)*len(cols))
        return self._get_positions(self.position[np.asarray(rows, np.int64)], self.position[np.asarray(cols, np.int64)])

    def plan(self, ids):
        return None

    def block_matmul(self, rows, cols, x, row_plan=None, col_plan=None, trans=0):
        """``A[rows, cols] @ x`` (or its transpose/adjoint) from retained factors."""
        if trans not in (0, 1, 2):
            raise ValueError('Invalid transpose mode')
        rows, cols = self.position[np.asarray(rows, np.int64)], self.position[np.asarray(cols, np.int64)]
        row_order, col_order = np.argsort(rows, kind='stable'), np.argsort(cols, kind='stable')
        sorted_rows, sorted_cols = rows[row_order], cols[col_order]
        x = np.asarray(x)
        def adj(a):
            return a.T if trans == 1 else a.conj().T
        if trans == 0:
            result = np.zeros((len(rows), x.shape[1]), complex)
        else:
            result = np.zeros((len(cols), x.shape[1]), complex)
        for row, local_rows, selected_rows in self._blocks(sorted_rows):
            local_cols, selected_cols = row.local_columns(sorted_cols)
            if local_cols is None:
                continue
            u = row.u[local_rows]
            if trans == 0:
                rhs = x[col_order[selected_cols]]
                result[row_order[selected_rows]] += u[:, local_cols] @ rhs if row.v is None else u @ (row.v[:, local_cols] @ rhs)
            else:
                rhs = x[row_order[selected_rows]]
                result[col_order[selected_cols]] += (adj(u[:, local_cols]) @ rhs if row.v is None
                                                     else adj(row.v[:, local_cols]) @ (adj(u) @ rhs))
        return result

    def _within(self, r0, r1, c0, c1):
        """Block rows inside permuted rows [r0, r1), with their partner pieces inside [c0, c1).

        Hierarchical inverse blocks are unions of clusters, so every block
        row and partner range is either inside such a block or disjoint.
        """
        selected = {}
        for key in sorted(self.groups):
            starts, stops, rows = self.group_index[key]
            first, last = np.searchsorted(starts, r0), np.searchsorted(stops, r1, side='right')
            for row in rows[first:last]:
                pieces = [(int(offset), a - c0, b - c0) for (a, b), offset in zip(row.ranges, row.offsets)
                          if a >= c0 and b <= c1]
                if pieces:
                    selected.setdefault(key, []).append((row, pieces))
        return selected

    def range_matmul(self, r0, r1, c0, c1, x, adjoint=False):
        """``H[r0:r1, c0:c1] @ x`` (or its adjoint applied to ``x``) in permuted numbering."""
        selected = self._within(r0, r1, c0, c1)
        if not adjoint:
            out = np.zeros((r1 - r0, x.shape[1]), complex)
            def visit(item):
                row, pieces = item
                if row.v is None:
                    value = sum(row.u[:, o:o + b - a] @ x[a:b] for o, a, b in pieces)
                else:
                    value = row.u @ sum(row.v[:, o:o + b - a] @ x[a:b] for o, a, b in pieces)
                out[row.start - r0:row.stop - r0] += value
            for key in sorted(selected):
                self._map(visit, selected[key])
            return out
        out = np.zeros((c1 - c0, x.shape[1]), complex)
        def visit(item):
            row, pieces = item
            y = x[row.start - r0:row.stop - r0]
            if row.v is None:
                return [(a, b, row.u[:, o:o + b - a].conj().T @ y) for o, a, b in pieces]
            t = row.u.conj().T @ y
            return [(a, b, row.v[:, o:o + b - a].conj().T @ t) for o, a, b in pieces]
        for key in sorted(selected):
            for parts in self._map(visit, selected[key]):
                for a, b, value in parts:
                    out[a:b] += value
        return out

    def _map(self, function, items):
        """Ordered results of ``function`` over items with disjoint outputs, on threads when large."""
        from ghost_backend.compressed.operator import _thread_budget
        workers = min(8, _thread_budget(), len(items))
        self.checkpoint()
        if workers < 2:
            return [function(item) for item in items]
        from ghost_backend.execution.options import single_thread_blas
        with single_thread_blas(), ThreadPoolExecutor(max_workers=workers, thread_name_prefix='ghost-hmatrix') as pool:
            return list(pool.map(function, items))

    def equilibrate(self):
        row = np.where(self.row_max > 0, self.row_max, 1.)
        _, _, sums, column = self._absolute(row[self.order])
        scale = np.empty(self.n); scale[self.order] = column
        total = np.empty(self.n); total[self.order] = sums
        scale = np.where(scale > 0, scale, 1.)
        return row, scale, float(np.max(total/scale))

    def rebuild(self, tolerance):
        """Assemble again at another tolerance, in place (factors keep this object)."""
        self.tolerance = float(tolerance)
        self.ledger.bytes -= self.bytes
        self.bytes = 0
        self.near = self.far = self.dense_far = self.sample_values = None
        _assemble(self.oracle, [self])

    def balanced_scaling(self, row_scale, col_scale, starts):
        """``combined_regions.balanced_condition_scaling`` for this operator (one pass)."""
        from ghost_backend.twod.formulations.combined_regions import _balance
        starts = np.asarray(starts, int)
        labels = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, self.n)))[self.order]
        inverse_rows = (1./np.asarray(row_scale, float))[self.order]
        parts = {}
        def visit(row):
            total = np.zeros((len(starts), len(row.columns)))
            weights = np.zeros((len(starts), row.stop - row.start))
            weights[labels[row.start:row.stop], np.arange(row.stop - row.start)] = inverse_rows[row.start:row.stop]
            width = max(1, (4*1024**2)//max(1, 16*(row.stop - row.start)))
            for c0 in range(0, len(row.columns), width):
                total[:, c0:c0 + width] = weights @ abs(row.values(None, slice(c0, c0 + width)))
            parts[(row.kind, row.row)] = (row, total)
        self._passes(visit)
        partial = np.zeros((len(starts), self.n))
        for key in sorted(parts):
            row, total = parts.pop(key)
            for (a, b), offset in zip(row.ranges, row.offsets):
                partial[:, a:b] += total[:, offset:offset + b - a]
        original = np.empty_like(partial); original[:, self.order] = partial
        return _balance(original/np.asarray(col_scale, float)[None, :], starts, row_scale, col_scale)

    @property
    def factor_class(self):
        from ghost_backend.compressed.hmatrix_factor import HMatrixFactor
        return HMatrixFactor

    def sample_residual(self, x, b):
        """Exact-row residuals of the sample rows (original numbering of ``x``, ``b``)."""
        rows = self.order[self.sample_rows]
        return self.sample_values @ x[self.order] - b[rows], rows
