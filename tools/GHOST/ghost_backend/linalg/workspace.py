"""Linear-system checks with bounded temporary arrays (Python 3.6+)."""
import numpy as np
from ghost_backend.execution.metrics import timed_stage


def first_nonfinite(value, block_bytes=1024*1024):
    array = np.asarray(value)
    if array.ndim == 0:
        return None if np.isfinite(array) else ()
    row_entries = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    rows = max(1, int(block_bytes) // max(1, row_entries))
    for start in range(0, len(array), rows):
        finite = np.isfinite(array[start:start+rows])
        if not np.all(finite):
            index = np.unravel_index(int(np.argmin(finite)), finite.shape)
            return (start + int(index[0]),) + tuple(int(i) for i in index[1:])
    return None


def matrix_inf_norm(matrix, block_bytes=1024*1024):
    a = np.asarray(matrix)
    if not a.size:
        return 0.
    if a.flags.f_contiguous and not a.flags.c_contiguous:
        sums = np.zeros(a.shape[0])
        width = max(1, int(block_bytes)//max(1, a.shape[0]*8))
        for start in range(0, a.shape[1], width):
            sums += np.sum(np.abs(a[:, start:start+width]), axis=1)
        return float(np.max(sums))
    rows = max(1, int(block_bytes) // max(1, a.shape[1]*8))
    largest = 0.0
    for start in range(0, len(a), rows):
        largest = max(largest, float(np.max(np.sum(np.abs(a[start:start+rows]), axis=1))))
    return largest


def checked_matrix_norms(matrix, one_norm=False, checkpoint=None):
    """Finite check, infinity norm and optional 1-norm with shared magnitudes.

    Keep the BoR 1-norm's 64-row reduction order. For Fortran-only matrices,
    retain the original column-blocked infinity-norm reduction as well.
    """
    a = np.asarray(matrix)
    if not a.size:
        return None, 0., 0. if one_norm else None
    checkpoint = checkpoint or (lambda: None)
    rows = 64 if one_norm else max(1, 1024**2 // max(1, a.shape[1]*8))
    columns = np.zeros(a.shape[1]) if one_norm else None
    fortran = a.flags.f_contiguous and not a.flags.c_contiguous
    largest = 0.
    for start in range(0, len(a), rows):
        checkpoint()
        magnitude = np.abs(a[start:start+rows])
        if not np.isfinite(magnitude).all():
            first = first_nonfinite(a)
            if first is not None:
                return first, None, None
        if not fortran:
            largest = max(largest, float(np.max(np.sum(magnitude, axis=1))))
        if one_norm:
            columns += np.sum(magnitude, axis=0)
    if fortran:
        largest = matrix_inf_norm(a)
    return None, largest, float(np.max(columns)) if one_norm else None


# Pre-factor passes over matrices of at least this size run their blocks on
# several threads (NumPy's abs/sum/max release the GIL). The blocks and the
# order in which their partial results are combined are those of the serial
# pass, so the results are bitwise identical for any thread count; below the
# threshold a thread pool costs more than it saves.
PARALLEL_PASS_MIN_BYTES = 32 * 1024**2
# Consecutive blocks per thread task, and tasks in flight per thread (bounds
# the per-block partial results retained ahead of the in-order reduction).
_PASS_GROUP_BYTES = 8 * 1024**2
# Real workspace per concurrent equilibration block (see below).
_PASS_PARALLEL_BLOCK_BYTES = 1024**2
_PASS_TASKS_PER_WORKER = 2


def pass_workers(nbytes, workers=None):
    """Threads for a blocked pass over ``nbytes`` of matrix (1 = serial)."""
    if workers is not None:
        return max(1, int(workers))
    if int(nbytes) < PARALLEL_PASS_MIN_BYTES:
        return 1
    from ghost_backend.execution.options import blas_core_budget
    return max(1, int(blas_core_budget()))


def _ordered_blocks(function, starts, workers, block_bytes):
    """Yield ``function(start)`` for every block start, in order.

    With several workers, groups of consecutive blocks (about
    ``_PASS_GROUP_BYTES`` of matrix each) run as thread tasks, at most
    ``_PASS_TASKS_PER_WORKER`` per worker in flight; the caller reduces the
    yielded per-block results in block order exactly as a serial loop would.
    """
    starts = list(starts)
    if workers <= 1 or len(starts) <= 1:
        for start in starts:
            yield function(start)
        return
    group = max(1, min(int(_PASS_GROUP_BYTES // max(1, int(block_bytes))),
                       -(-len(starts) // workers)))
    groups = [starts[i:i + group] for i in range(0, len(starts), group)]
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def run(members):
        return [function(start) for start in members]

    pool = ThreadPoolExecutor(max_workers=min(workers, len(groups)), thread_name_prefix='ghost-matrix-pass')
    try:
        pending = deque()
        submitted = 0
        while submitted < len(groups) and len(pending) < _PASS_TASKS_PER_WORKER * workers:
            pending.append(pool.submit(run, groups[submitted]))
            submitted += 1
        while pending:
            results = pending.popleft().result()
            if submitted < len(groups):
                pending.append(pool.submit(run, groups[submitted]))
                submitted += 1
            for result in results:
                yield result
            results = None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def checked_row_norms(matrix, block_bytes=1024*1024, workers=None):
    """``(first non-finite index or None, infinity norm, row maxima of |a|)``
    of a 2-D matrix in one blocked pass.

    It replaces a :func:`first_nonfinite` pass followed by a
    :func:`matrix_inf_norm` pass, and the row maxima are the first pass of the
    row/column equilibration.  Blocks and sums follow :func:`matrix_inf_norm`
    exactly, so the norm is bitwise identical to it; on a non-finite entry the
    norm and maxima are None and the index is the one :func:`first_nonfinite`
    reports.  Large matrices evaluate their blocks on threads
    (:func:`pass_workers`) and add the per-block row sums in block order, so
    the results do not depend on the thread count.
    """
    a = np.asarray(matrix)
    n_rows = a.shape[0]
    row_max = np.zeros(n_rows)
    if not a.size:
        return None, 0., row_max
    threads = pass_workers(a.nbytes, workers)

    if a.flags.f_contiguous and not a.flags.c_contiguous:
        sums = np.zeros(n_rows)
        width = max(1, int(block_bytes)//max(1, n_rows*8))

        def column_block(start):
            magnitude = np.abs(a[:, start:start+width])
            # |z| is non-finite for every non-finite entry (and on overflow).
            if not np.isfinite(magnitude).all():
                return None
            return np.sum(magnitude, axis=1), np.max(magnitude, axis=1)

        for part in _ordered_blocks(column_block, range(0, a.shape[1], width), threads, n_rows*width*16):
            if part is None:
                return first_nonfinite(a), None, None
            sums += part[0]
            np.maximum(row_max, part[1], out=row_max)
        return None, float(np.max(sums)), row_max
    rows = max(1, int(block_bytes) // max(1, a.shape[1]*8))
    largest = 0.0

    def row_block(start):
        magnitude = np.abs(a[start:start+rows])
        if not np.isfinite(magnitude).all():
            return None
        return float(np.max(np.sum(magnitude, axis=1))), np.max(magnitude, axis=1)

    for start, part in zip(range(0, n_rows, rows),
                           _ordered_blocks(row_block, range(0, n_rows, rows), threads, rows*a.shape[1]*16)):
        if part is None:
            return first_nonfinite(a), None, None
        largest = max(largest, part[0])
        row_max[start:start+rows] = part[1]
    return None, largest, row_max


@timed_stage('condition_scaling')
def equilibrated_scaling_and_norm_1(a_mat, max_block_bytes=16 * 1024 * 1024, row_max=None, workers=None):
    """Row/column equilibration scales and the equilibrated 1-norm.

    The same column-blocked computation as
    ``twod.solver._equilibrated_scaling_and_norm_1`` (bitwise identical
    results: every column block is independent and the norm is a maximum),
    with the blocks of a large matrix on threads (:func:`pass_workers`).
    """
    a_eval = np.asarray(a_mat, dtype=np.complex128)
    if a_eval.ndim != 2 or a_eval.shape[0] != a_eval.shape[1]:
        raise ValueError("Condition estimation requires a square matrix.")
    n = int(a_eval.shape[0])
    if n < 1:
        raise ValueError("Condition estimation requires a non-empty matrix.")
    workspace_bytes = max(8, int(max_block_bytes))
    threads = pass_workers(a_eval.nbytes, workers)
    if threads > 1 and a_eval.flags.f_contiguous:
        # Concurrent blocks stay cache-sized (eight 16 MiB blocks spill the
        # last-level cache and run memory-bound). Every output depends on its
        # own column alone -- elementwise scaling, a maximum over the rows,
        # and NumPy's pairwise sum of one contiguous column -- so the block
        # width does not change any bit of the result.
        workspace_bytes = min(workspace_bytes, _PASS_PARALLEL_BLOCK_BYTES)
    block_columns = max(1, min(n, workspace_bytes // (8 * n)))
    starts = range(0, n, block_columns)

    if row_max is None:
        def row_maxima(start):
            stop = min(n, start + block_columns)
            return np.max(np.abs(a_eval[:, start:stop]), axis=1)
        row_scale = np.zeros(n, dtype=float)
        for part in _ordered_blocks(row_maxima, starts, threads, 16*n*block_columns):
            np.maximum(row_scale, part, out=row_scale)
    else:
        row_scale = np.asarray(row_max, dtype=float)
        if row_scale.shape != (n,):
            raise ValueError("Row maxima do not match the matrix.")
    row_scale = np.where(row_scale > 0.0, row_scale, 1.0)

    col_scale = np.ones(n, dtype=float)

    def column_pass(start):
        stop = min(n, start + block_columns)
        equilibrated = np.abs(a_eval[:, start:stop])
        equilibrated /= row_scale[:, None]
        local_col_scale = np.max(equilibrated, axis=0)
        local_col_scale = np.where(local_col_scale > 0.0, local_col_scale, 1.0)
        col_scale[start:stop] = local_col_scale
        equilibrated /= local_col_scale[None, :]
        return float(np.max(np.sum(equilibrated, axis=0)))

    norm_a = 0.0
    for value in _ordered_blocks(column_pass, starts, threads, 16*n*block_columns):
        norm_a = max(norm_a, value)
    return row_scale, col_scale, norm_a
