"""Ordered, bounded thread-parallel execution of independent grid tiles.

NumPy releases the GIL inside ufuncs, copies, fancy indexing, sorting and
pocketfft, so disjoint tiles of a dataset operation can be computed at the
same time. Only per-tile bodies that read shared inputs and write their own
disjoint destination tile belong here; any reduction of tile results is done
by the caller in submission order, so results are identical to a serial loop.

Controls (one switch disables all threading):

* ``GRIM_DATASET_THREADS`` environment variable, read at import;
* ``configure_dataset_threads(count)`` for the application (``None`` restores
  the automatic default, ``1`` disables threading);
* ``dataset_threads(count)`` to scope a different count to one worker.

The automatic default is the number of CPUs this process may run on (its
affinity), at most ``_DEFAULT_MAXIMUM``. Callers keep their memory budgets by
dividing a per-tile allowance with :func:`scaled_cells` so that the tiles in
flight together use no more scratch than one serial tile did.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import concurrent.futures
import itertools
import os
import threading

_DEFAULT_MAXIMUM = 8
_SCOPED = ContextVar("grim_dataset_threads", default=None)
_CONFIGURED = None
_POOLS = {}
_POOL_LOCK = threading.Lock()
_WORKER = threading.local()
_END = object()


def _checked_count(count):
    if isinstance(count, bool):
        raise ValueError("dataset thread count must be a positive integer")
    count = int(count)
    if count < 1:
        raise ValueError("dataset thread count must be a positive integer")
    return count


def _environment_count():
    raw = os.environ.get("GRIM_DATASET_THREADS", "").strip()
    if not raw:
        return None
    try:
        return _checked_count(raw)
    except (TypeError, ValueError):
        return None


def available_cpus():
    """CPUs this process may use (its affinity), never less than one."""
    try:
        import psutil
        return max(1, len(psutil.Process().cpu_affinity()))
    except Exception:  # noqa: BLE001 - psutil is optional; fall back to the OS view
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def configure_dataset_threads(count=None):
    """Set the application-wide tile thread count; None selects the default."""
    global _CONFIGURED
    _CONFIGURED = None if count is None else _checked_count(count)


@contextmanager
def dataset_threads(count):
    """Scope a tile thread count (``1`` = serial) to this thread/context."""
    token = _SCOPED.set(_checked_count(count))
    try:
        yield
    finally:
        _SCOPED.reset(token)


def worker_count():
    """Threads the next tile loop may use (1 inside a tile worker)."""
    if getattr(_WORKER, "active", False):
        return 1
    for count in (_SCOPED.get(), _CONFIGURED, _ENVIRONMENT):
        if count is not None:
            return count
    return min(_DEFAULT_MAXIMUM, available_cpus())


def scaled_cells(cells, workers=None, *, minimum=1):
    """Per-tile allowance so ``workers`` tiles in flight fit one serial tile."""
    workers = worker_count() if workers is None else max(1, int(workers))
    return max(int(minimum), int(cells) // workers)


def _pool(workers):
    with _POOL_LOCK:
        pool = _POOLS.get(workers)
        if pool is None:
            pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="grim-tiles",
                initializer=_mark_worker)
            _POOLS[workers] = pool
        return pool


def _mark_worker():
    _WORKER.active = True


def map_tiles(function, tiles, *, workers=None):
    """Yield ``function(tile)`` for each tile, in order, using worker threads.

    At most ``workers`` tiles run at once and at most ``2 * workers`` are
    queued. Each tile runs in a copy of the caller's context (result storage
    policy, work control). Selection generators (which call ``checkpoint``)
    advance on the caller's thread, so cancellation raises there; in-flight
    tiles are cancelled or awaited before the exception propagates, so no
    tile writes into a destination after the caller has failed.
    """
    workers = worker_count() if workers is None else max(1, int(workers))
    if workers <= 1:
        for tile in tiles:
            yield function(tile)
        return
    tiles = iter(tiles)
    first = next(tiles, _END)
    second = next(tiles, _END) if first is not _END else _END
    if second is _END:
        # Zero or one tile: nothing to overlap, so skip the thread hand-off.
        if first is not _END:
            yield function(first)
        return
    tiles = itertools.chain((first, second), tiles)
    pool = _pool(workers)
    pending = deque()
    try:
        for tile in tiles:
            pending.append(pool.submit(copy_context().run, function, tile))
            if len(pending) >= 2 * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        concurrent.futures.wait(list(pending))


def run_tiles(function, tiles, *, workers=None):
    """Run ``function`` over every tile for its side effects."""
    for _ in map_tiles(function, tiles, workers=workers):
        pass


_ENVIRONMENT = _environment_count()
