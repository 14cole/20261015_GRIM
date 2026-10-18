"""Semi-analytic S/K moments of separated straight panel pairs across a thin gap.

The tensor recursion of a close pair resolves a gap d with O(L/d) boxes. Here
the inner integral over the longer panel takes the kernels' singular parts in
closed form,

    G = (1/2pi) J0(kr) ln r + smooth,
    G'(r)/r = (1/2pi) [1/r^2 - (k J1(kr)/r) ln r] + smooth,

with J0 and k J1(kr)/r truncated after (kr)^(2 SERIES_TERMS). The remainder,
O((kr)^(2 SERIES_TERMS + 2) ln r), takes a Gauss rule split at the foot point.
The outer integral over the shorter panel bisects with the tensor recursion's
error test (block change against the children's summed norms), so a gap costs
O(log(L/d)) intervals instead of O(L/d) boxes.

Results are monomial moments M[n, m] = int int x^n y^m kernel ds_x ds_y in the
observation and source parameters, for any nodal basis of degree <= DEGREE.
"""
from math import factorial, comb
import numpy as np

SERIES_TERMS = 2
DEGREE = 3
# A pair qualifies when its gap is below THIN_GAP of the longer (inner) panel
# and the outer panel lies within one inner length of the inner panel's line
# and foot range. The closed forms expand in powers of the foot coordinate;
# the extent bound limits their cancellation to a few digits.
THIN_GAP = .05
OUTER_ORDER = 16
INNER_ORDER = 16
# Outer points x inner remainder nodes per evaluation batch.
BATCH_SAMPLES = 1 << 17

_S_SERIES = tuple((-1.)**j / (4.**j * factorial(j)**2) for j in range(SERIES_TERMS + 1))
_V_SERIES = tuple((-1.)**j / (2. * 4.**j * factorial(j) * factorial(j + 1))
                  for j in range(SERIES_TERMS + 1))


def qualifies(p0, p1, q0, q1, gap):
    """Pairs (segment ends, shape (n, 2); gap, shape (n,)) this module integrates."""
    lo = np.linalg.norm(p1 - p0, axis=-1)
    ls = np.linalg.norm(q1 - q0, axis=-1)
    swap = (lo > ls)[:, None]
    inner0, inner1 = np.where(swap, p0, q0), np.where(swap, p1, q1)
    length = np.maximum(lo, ls)
    tangent = (inner1 - inner0) / np.where(length > 0, length, 1.)[:, None]
    ok = (length > 0) & (np.minimum(lo, ls) > 0) & (gap > 0) & (gap < THIN_GAP * length)
    for end in (np.where(swap, q0, p0), np.where(swap, q1, p1)):
        d = end - inner0
        t = np.sum(d * tangent, axis=-1)
        h = d[:, 1] * tangent[:, 0] - d[:, 0] * tangent[:, 1]
        ok &= (np.abs(h) <= length) & (t >= -length) & (t <= 2. * length)
    return ok


def _gauss(order):
    x, w = np.polynomial.legendre.leggauss(order)
    return .5 * (x + 1.), .5 * w


def _polymul(a, b):
    out = [0.] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = out[i + j] + x * y
    return out


def _inner(x, q0, tangent, length, normal, sigma, k, degree, want_s, want_k, part, table=None):
    """One part ('analytic' or 'remainder') of the inner-panel integrals at
    outer points x against (s/L)^m, m <= degree.

    Rows of q0, tangent, length, normal and sigma belong to the points.
    Returns S, K of shape (P, degree + 1): S = int (s/L)^m G ds and
    K = sigma int (s/L)^m V(r) n.(x - y) ds with V = G'(r)/r. Kernel values
    are the near batches' (real Bessel functions, or the validated lossy-medium
    ``table``, else scaled Hankel functions).
    """
    from ghost_backend.twod.polynomial_quadrature import _kernels
    k = complex(k)
    d = x - q0
    t = np.sum(d * tangent, axis=1)
    h = d[:, 1] * tangent[:, 0] - d[:, 0] * tangent[:, 1]
    delta = np.abs(h)
    d2 = delta * delta
    u0, u1 = -t, length - t
    from ghost_backend.twod.assembly.native import near as native
    if native.available():
        return _native_inner(t, h, delta, d2, u0, u1, length, normal, tangent, sigma, k, degree,
                             want_s, want_k, part, table)
    qmax = degree + 1 + 2 * SERIES_TERMS
    # C_q = int u^q / (u^2 + delta^2) du, with delta^2 C_0 = delta times the
    # subtended angle; B_q = int u^q ln(u^2 + delta^2) du.
    angle = np.arctan2(delta * length, d2 + u0 * u1)
    l0, l1 = np.log(u0 * u0 + d2), np.log(u1 * u1 + d2)
    power0, power1 = [np.ones_like(t)], [np.ones_like(t)]
    for _ in range(qmax + 2):
        power0.append(power0[-1] * u0)
        power1.append(power1[-1] * u1)
    c = [None, .5 * (l1 - l0)]
    for q in range(2, qmax + 3):
        c.append((power1[q - 1] - power0[q - 1]) / (q - 1) - (delta * angle if q == 2 else d2 * c[q - 2]))
    b = []
    with np.errstate(invalid='ignore', divide='ignore'):
        for q in range(qmax + 1):
            ends = (np.where(power1[q + 1] == 0., 0., power1[q + 1] * l1)
                    - np.where(power0[q + 1] == 0., 0., power0[q + 1] * l0))
            b.append((ends - 2. * c[q + 2]) / (q + 1))

    def log_series(coefficients, poly, k_power):
        # sum_j a_j k^(2j + k_power) int poly(u) (u^2 + delta^2)^j ln(u^2 + delta^2) du
        total = 0.
        for j, a in enumerate(coefficients):
            moment = 0.
            for l in range(j + 1):
                moment = moment + comb(j, l) * d2**(j - l) * sum(
                    g * b[2 * l + i] for i, g in enumerate(poly))
            total = total + (a * k**(2 * j + k_power)) * moment
        return total

    # (s/L)^m = ((u + t)/L)^m as coefficients of u^i.
    monomials = [[comb(m, i) * t**(m - i) / length**m for i in range(m + 1)] for m in range(degree + 1)]
    m0 = normal[:, 1] * tangent[:, 0] - normal[:, 0] * tangent[:, 1]
    m1 = np.sum(normal * tangent, axis=1)
    s_out = np.zeros((len(t), degree + 1), dtype=np.complex128) if want_s else None
    k_out = np.zeros((len(t), degree + 1), dtype=np.complex128) if want_k else None
    if part == 'analytic':
        for m, poly in enumerate(monomials):
            if want_s:
                s_out[:, m] = (1. / (4. * np.pi)) * log_series(_S_SERIES, poly, 0)
            if want_k:
                # n.(x - y) = h m0 - u m1; int u^0 h / (u^2 + delta^2) du = sign(h) angle.
                singular = m0 * poly[0] * (np.sign(h) * angle) - m1 * poly[0] * c[1]
                for i, coefficient in enumerate(poly[1:], start=1):
                    singular = singular + coefficient * (h * m0 * c[i] - m1 * c[i + 1])
                singular = singular - .5 * log_series(_V_SERIES, _polymul(poly, (h * m0, -m1)), 2)
                k_out[:, m] = sigma * (1. / (2. * np.pi)) * singular
        return s_out, k_out

    qt, qw = _gauss(INNER_ORDER)
    split = np.where((t > 0.) & (t < length), t, .5 * length)
    starts = np.stack([np.zeros_like(t), split], axis=1)
    widths = np.stack([split, length - split], axis=1)
    s = (starts[:, :, None] + widths[:, :, None] * qt).reshape(len(t), -1)
    w = (widths[:, :, None] * qw).reshape(len(t), -1)
    u = s - t[:, None]
    r2 = u * u + d2[:, None]
    r = np.sqrt(r2)
    log_r2 = np.log(r2)
    green, derivative = _kernels(k, r, want_k, table)
    ratio = s / length[:, None]
    weights = [w]
    for _ in range(degree):
        weights.append(weights[-1] * ratio)
    if want_s:
        rem = green - (1. / (4. * np.pi)) * sum(a * (k * k * r2)**j for j, a in enumerate(_S_SERIES)) * log_r2
        for m in range(degree + 1):
            s_out[:, m] = np.sum(weights[m] * rem, axis=1)
    if want_k:
        # V = G'(r)/r = -(j/4) k H1(kr)/r.
        v = -derivative / r - (1. / (2. * np.pi)) * (1. / r2 - .5 * sum(
            a * k**(2 * j + 2) * r2**j for j, a in enumerate(_V_SERIES)) * log_r2)
        v = v * (h[:, None] * m0[:, None] - u * m1[:, None])
        for m in range(degree + 1):
            k_out[:, m] = sigma * np.sum(weights[m] * v, axis=1)
    return s_out, k_out


def _native_inner(t, h, delta, d2, u0, u1, length, normal, tangent, sigma, k, degree, want_s, want_k,
                  part, table):
    """``_inner`` with its arithmetic in assembly.native.near, bitwise the NumPy route.

    The logarithms, arctangent, powers and kernel values are NumPy's own; the
    closed forms, series and remainder sums take NumPy's operation order there.
    """
    from ghost_backend.twod.polynomial_quadrature import _kernels
    from ghost_backend.twod.assembly.native import near as native
    m0 = normal[:, 1] * tangent[:, 0] - normal[:, 0] * tangent[:, 1]
    m1 = np.sum(normal * tangent, axis=1)
    v_series = [a * k**(2 * j + 2) for j, a in enumerate(_V_SERIES)]
    if part == 'analytic':
        angle = np.arctan2(delta * length, d2 + u0 * u1)
        l0, l1 = np.log(u0 * u0 + d2), np.log(u1 * u1 + d2)
        t3, l3 = (t**3, length**3) if degree == 3 else (None, None)
        return native.thin_analytic(degree, t, h, delta, d2, u0, u1, angle, l0, l1, length, t3, l3, m0, m1, sigma,
                                    [a * k**(2 * j) for j, a in enumerate(_S_SERIES)], v_series, want_s, want_k)
    qt, qw = _gauss(INNER_ORDER)
    r2 = native.thin_squares(t, length, d2, qt, qw)
    r = np.sqrt(r2)
    log_r2 = np.log(r2)
    green, derivative = _kernels(k, r, want_k, table)
    return native.thin_remainder(degree, t, length, d2, h, m0, m1, sigma, qt, qw, r, log_r2, green, derivative,
                                 k * k, _S_SERIES, v_series, want_s, want_k)


def _contract(outer, values):
    """Moments sum_q outer[q, a] values[q, b] per pair (native or einsum, the same bits)."""
    from ghost_backend.twod.assembly.native import near as native
    blocks = native.thin_contract(outer, values)
    return np.einsum('nqa,nqb->nab', outer, values) if blocks is None else blocks


def thin_gap_moments(obs, src, k, obs_normal_deriv, want_s=True, want_k=True, degree=DEGREE,
                     rtol=1.0e-9, max_depth=24, map_jobs=None, checkpoint=None):
    """Monomial S and K moments (n, 2, degree + 1, degree + 1) of separated pairs.

    ``obs`` and ``src`` are (p0, p1, normal) arrays of the pairs' panels.
    ``map_jobs(function, jobs)`` may run evaluation batches in parallel; the
    batching does not change any result (a lossy medium's kernel table covers
    the pairs of one call).
    """
    (op0, op1, on), (sp0, sp1, sn) = obs, src
    npairs = len(op0)
    lo, ls = np.linalg.norm(op1 - op0, axis=1), np.linalg.norm(sp1 - sp0, axis=1)
    swap = lo > ls
    sw = swap[:, None]
    outer0, outer1 = np.where(sw, sp0, op0), np.where(sw, sp1, op1)
    inner0, inner1 = np.where(sw, op0, sp0), np.where(sw, op1, sp1)
    normal = on if obs_normal_deriv else sn
    # Swapping the panels reverses x - y.
    sigma = (1. if obs_normal_deriv else -1.) * np.where(swap, -1., 1.)
    inner_length, outer_length = np.maximum(lo, ls), np.minimum(lo, ls)
    tangent = (inner1 - inner0) / inner_length[:, None]
    floor = np.finfo(float).eps * np.maximum(1., inner_length * outer_length)
    qt, qw = _gauss(OUTER_ORDER)
    size = degree + 1
    from ghost_backend.twod.polynomial_quadrature import _table_for_ends
    table = _table_for_ends(k, outer0, outer1, inner0, inner1) if npairs else None

    def evaluate_batch(job):
        pair, a, b, parts = job
        xi = a[:, None] + (b - a)[:, None] * qt
        x = (outer0[pair][:, None, :] + xi[:, :, None] * (outer1 - outer0)[pair][:, None, :]).reshape(-1, 2)
        rows = np.repeat(pair, len(qt))
        weight = qw * ((b - a) * outer_length[pair])[:, None]
        outer = [weight]
        for _ in range(degree):
            outer.append(outer[-1] * xi)
        outer = np.stack(outer, axis=-1)
        blocks = []
        for part in parts:
            for values in _inner(x, inner0[rows], tangent[rows], inner_length[rows], normal[rows],
                                 sigma[rows], k, degree, want_s, want_k, part, table):
                blocks.append(np.zeros((len(pair), size, size), dtype=np.complex128) if values is None else
                              _contract(outer, values.reshape(len(pair), len(qt), size)))
        return blocks

    def evaluate(pair, a, b, parts):
        per = max(1, BATCH_SAMPLES // (len(qt) * (2 * INNER_ORDER if 'remainder' in parts else 4)))
        jobs = [(pair[i:i + per], a[i:i + per], b[i:i + per], parts) for i in range(0, len(pair), per)]
        results = list(map(evaluate_batch, jobs) if map_jobs is None else map_jobs(evaluate_batch, jobs))
        if checkpoint is not None:
            checkpoint()
        if not results:
            return [np.zeros((0, size, size), complex)] * (2 * len(parts))
        return [np.concatenate([r[i] for r in results]) for i in range(2 * len(parts))]

    def norm(blocks):
        flat = blocks.reshape(len(blocks), -1)
        return np.sqrt(np.sum(flat.real**2 + flat.imag**2, axis=1))

    def integrate(parts):
        """Bisect outer intervals until the blocks of parts[0] converge.

        The analytic part is tested against its own children; the remainder
        (parts = remainder, analytic) against the children's whole blocks.
        """
        total = np.zeros((npairs, 2, size, size), dtype=np.complex128)
        pair = np.arange(npairs)
        a, b = np.zeros(npairs), np.ones(npairs)
        coarse = evaluate(pair, a, b, parts)[:2]
        depth = 0
        while len(pair):
            m = .5 * (a + b)
            cp, ca, cb = np.repeat(pair, 2), np.column_stack((a, m)).ravel(), np.column_stack((m, b)).ravel()
            children = [values.reshape(-1, 2, size, size) for values in evaluate(cp, ca, cb, parts)]
            whole = children[:2] if len(parts) == 1 else [x + y for x, y in zip(children[:2], children[2:])]
            fine = [values[:, 0] + values[:, 1] for values in children[:2]]
            error = np.zeros(len(pair))
            for want, f, c, ch in zip((want_s, want_k), fine, coarse, whole):
                if want:
                    scale = norm(ch[:, 0]) + norm(ch[:, 1])
                    error = np.maximum(error, norm(f - c) / np.maximum(scale, floor[pair]))
            done = error <= rtol
            for channel in (0, 1):
                np.add.at(total[:, channel], pair[done], fine[channel][done])
            refine = ~done
            if np.any(refine) and depth >= max_depth:
                i = int(np.flatnonzero(refine)[0])
                raise FloatingPointError(
                    "Thin-gap Galerkin quadrature did not converge: estimated relative "
                    f"block error={error[i]:.3e} after depth {depth} for panels "
                    f"{op0[pair[i]]}-{op1[pair[i]]} / {sp0[pair[i]]}-{sp1[pair[i]]}. "
                    "Refine the boundary mesh.")
            keep = refine.repeat(2)
            pair, a, b = cp[keep], ca[keep], cb[keep]
            coarse = [values[refine].reshape(-1, size, size) for values in children[:2]]
            depth += 1
        return total

    # The closed-form part needs the outer bisection toward the gap; the
    # smooth remainder, which holds every kernel evaluation, converges on a
    # few intervals.
    total = integrate(('analytic',)) + integrate(('remainder', 'analytic'))
    # Moments are indexed [outer, inner]; swapped pairs return [obs, src].
    total[swap] = total[swap].transpose(0, 1, 3, 2)
    return total


_HATS = np.array([[1., 0.], [-1., 1.]])


def thin_gap_linear_blocks(obs, src, k, obs_normal_deriv, want_s=True, want_k=True, **options):
    """S and K 2x2 hat-function blocks of linear pairs (see ``thin_gap_moments``)."""
    moments = thin_gap_moments(obs, src, k, obs_normal_deriv, want_s, want_k, degree=1, **options)
    blocks = _HATS.T @ moments @ _HATS
    return blocks[:, 0], blocks[:, 1]
