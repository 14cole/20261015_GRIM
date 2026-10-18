"""Bounded process preparation of independent, contracted BoR near blocks.

Workers receive geometry and quadrature parameters, never solver objects,
checkpoints, dense matrices, or mutable caches. Results are consumed in pair
order by the owning process.

One persistent pool per process serves every preparation (the base and fine
meshes of a certified solve, and consecutive solves of a driver worker) as
long as their admitted pool size is the same: spawning and importing eight
workers cost 1-2 s per preparation (27 % of a 6 GHz certified sphere).  The
pool is started when a preparation scope opens, so its spawn overlaps the
far-block build, and is released by :func:`close_near_process_pool`, at
interpreter exit, at the exit of a multiprocessing worker process (before
``multiprocessing`` joins that process's children), after any failed
preparation, and when a later preparation cannot use it.  Its idle workers'
memory is priced by the mode-phase admission (``PROCESS_IDLE_BYTES``).
"""
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import atexit
import multiprocessing
import multiprocessing.util
import os
import sys
import threading
from pathlib import Path
from functools import lru_cache
from ghost_backend.execution.runtime import (ScopedValue, pin_worker_environment,
                                             release_worker_environment)

_POOL = ScopedValue('ghost_bor_near_process_pool', default=None)
# Resident memory of one worker process beyond its task scratch (interpreter,
# numpy/scipy, the BoR modules and native kernel, and what finished tasks leave
# in its heap). Measured 16 Oct 2026 (Windows 11, Python 3.11, numpy 2.4,
# psutil RSS every 20 ms of an 8-worker pool, private and shared pages):
# 62-64 MB once initialized, 67-69 MB idle after the near preparation of the
# largest test bodies (PEC sphere 400 el certified at 12 GHz and 800 el at
# 30 GHz, coated sphere 4 GHz certified, coated/dielectric spheres 200/240 el
# at 6 GHz, cylinder 6 GHz). Charged at 1.5x the 69 MB measured (formerly
# 256 MiB, 3.7x the measurement).
PROCESS_OVERHEAD_BYTES = 104 * 1024**2
# One idle persistent worker kept through the mode phase (same measurement);
# the mode-phase admission charges it for every worker the pool keeps alive.
PROCESS_IDLE_BYTES = PROCESS_OVERHEAD_BYTES

_PERSISTENT_LOCK = threading.RLock()
_PERSISTENT = dict(executor=None, size=0, pid=None, pinned=False, uses=0, created=0, factory=None)


@lru_cache(maxsize=16)
def _guarded_main(filename):
    # Spawn imports the entry script. An unguarded top-level solve would run
    # again in every child; keep those direct/interactive callers on threads.
    import ast
    try:
        statements = ast.parse(Path(filename).read_text(encoding='utf-8-sig')).body
    except (OSError,UnicodeError,SyntaxError):
        return False
    for statement in statements:
        test = statement.test if isinstance(statement,ast.If) else None
        if isinstance(test,ast.Compare) and len(test.ops)==1 and isinstance(test.ops[0],ast.Eq):
            left,right = test.left,test.comparators[0]
            if isinstance(left,ast.Constant): left,right = right,left
            if (isinstance(left,ast.Name) and left.id=='__name__' and
                isinstance(right,ast.Constant) and right.value=='__main__'):
                return True
    return False


def process_capable():
    main = sys.modules.get('__main__')
    filename = getattr(main, '__file__', '')
    # A daemonic process (a multiprocessing.Pool worker) may not start
    # children; a concurrent.futures process worker is not daemonic and may,
    # so a driver that runs its units in such workers keeps process
    # preparation inside them.
    return (sys.version_info >= (3, 9) and not getattr(sys, 'frozen', False)
            and not multiprocessing.current_process().daemon
            and bool(filename) and _guarded_main(filename))


def _initialize():
    # Each process does quadrature; nested BLAS teams waste the CPU allocation.
    import numpy
    import scipy.linalg
    from ghost_backend.execution.thread_control import threadpool_limits
    global _limits
    _limits = threadpool_limits(limits=1)


def _watch_parent():
    """End this worker when its parent process is gone.

    A persistent pool idles between preparations; a parent that is killed
    (a crashed or terminated driver worker) never sends the shutdown
    sentinel, and the idle worker would otherwise outlive it.
    """
    parent = multiprocessing.parent_process()
    sentinel = getattr(parent, 'sentinel', None)
    if sentinel is None:
        return

    def watch():
        from multiprocessing.connection import wait
        try:
            wait([sentinel])
        except Exception:
            return
        os._exit(0)

    threading.Thread(target=watch, name='ghost-near-parent-watch', daemon=True).start()


def _initialize_persistent():
    _initialize()
    _watch_parent()
    # Import the near kernels while the parent builds its far blocks; the
    # first task then finds them loaded.
    import ghost_backend.bor.solver  # noqa: F401


def _worker_started():
    return os.getpid()


def _acquire_persistent(size):
    """The persistent pool of ``size`` workers and whether it already existed."""
    size = int(size)
    with _PERSISTENT_LOCK:
        pool = _PERSISTENT
        executor = pool['executor']
        if executor is not None and (pool['size'] != size or pool['pid'] != os.getpid()
                                     or pool['factory'] is not ProcessPoolExecutor
                                     or getattr(executor, '_broken', False)):
            _close_locked(wait=True)
            executor = None
        if executor is not None:
            pool['uses'] += 1
            return executor, True
        # Workers must start with one BLAS thread (SINGLE_THREAD_WORKER_ENVIRONMENT).
        # Every worker is spawned here, so the parent's environment is pinned
        # only while they start, not for the pool's lifetime.
        pin_worker_environment()
        pinned = True
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=size,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_persistent)
            # Each submit made before any worker is idle spawns one more
            # process (3.11 spawns on demand), so ``size`` submits start all.
            # A pool that could not start them all keeps the environment
            # pinned and is not kept past its preparation (process_scope).
            submit = getattr(executor, 'submit', None)
            if submit is not None:
                for _ in range(size):
                    submit(_worker_started)
            processes = getattr(executor, '_processes', None)
            if isinstance(processes, dict) and len(processes) >= size:
                release_worker_environment()
                pinned = False
        except BaseException:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if pinned:
                release_worker_environment()
            raise
        pool.update(executor=executor, size=size, pid=os.getpid(), pinned=pinned,
                    uses=1, created=pool['created'] + 1, factory=ProcessPoolExecutor)
        return executor, False


def _close_locked(wait=True):
    pool = _PERSISTENT
    executor, pinned = pool['executor'], pool['pinned']
    pool.update(executor=None, size=0, pid=None, pinned=False, uses=0, factory=None)
    try:
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
    finally:
        if pinned:
            release_worker_environment()


def close_near_process_pool(wait=True):
    """Shut down this process's persistent near-preparation pool, if any."""
    with _PERSISTENT_LOCK:
        if _PERSISTENT['executor'] is not None:
            _close_locked(wait=wait)


def near_process_pool_info():
    """Size, use count and worker process ids of the live persistent pool."""
    with _PERSISTENT_LOCK:
        executor = _PERSISTENT['executor']
        processes = getattr(executor, '_processes', None) or {}
        return dict(alive=executor is not None, size=int(_PERSISTENT['size']),
                    uses=int(_PERSISTENT['uses']), created=int(_PERSISTENT['created']),
                    pids=sorted(int(pid) for pid in list(processes)))


def persistent_pool_idle_bytes(size=None):
    """Resident bytes of ``size`` idle workers (default: the live pool)."""
    if size is None:
        with _PERSISTENT_LOCK:
            size = _PERSISTENT['size'] if _PERSISTENT['executor'] is not None else 0
    return int(size) * PROCESS_IDLE_BYTES


# Interpreter exit; and the exit of a multiprocessing worker process, whose
# ``_exit_function`` runs priority finalizers before it joins every
# non-daemonic child -- an idle pool would otherwise block that join forever
# (a driver worker reaching max_tasks_per_child could never exit).
atexit.register(close_near_process_pool)
multiprocessing.util.Finalize(None, close_near_process_pool, exitpriority=100)


# Spawn/import costs dominate small solves: automatic selection keeps the
# thread path below this many (near pairs) x (prepared modes).
AUTO_PROCESS_WORK_THRESHOLD = 8000


def process_backend_possible(workers):
    """Whether this host and option set can use process workers at all."""
    from ghost_backend.bor.options import current_options
    return (int(workers) > 1 and current_options()['near_backend'] != 'threads'
            and process_capable())


def processes_selected(pair_count=None, mode_tasks=None, workers=None):
    """The one workload policy shared by memory planning and the executor.

    ``mode_tasks`` is the number of prepared modes (cap + 1); ``workers`` is
    accepted for interface compatibility and does not change the policy. An
    unknown workload is not selected in advance: the planner then keeps the
    thread reservation and records the process pool size that was verified
    to fit, which bounds the executor if it meets a large job later.
    """
    from ghost_backend.bor.options import current_options
    backend = current_options()['near_backend']
    if backend == 'processes':
        return True
    if backend != 'auto' or pair_count is None or mode_tasks is None:
        return False
    return int(pair_count) * int(mode_tasks) >= AUTO_PROCESS_WORK_THRESHOLD


@contextmanager
def process_scope(workers, process_workers=None, prewarm=False):
    """``process_workers`` is the pool size admitted by the memory plan.

    ``prewarm`` (the plan selected process workers for this preparation)
    starts the persistent pool on entry, so its spawn overlaps the work the
    preparation does before its first near pair.  A persistent pool this
    preparation cannot use is released on entry; one it did not use, one
    whose workers did not all start up front, and any pool after a failure
    or cancellation, on exit.
    """
    admitted = int(workers) if process_workers is None else int(process_workers)
    state = dict(workers=int(workers), process_workers=admitted, executor=None, jobs=0,
                 environment_pinned=False, persistent=True, reused=False, prewarmed=False)
    usable = admitted > 1 and process_backend_possible(admitted)
    with _PERSISTENT_LOCK:
        if _PERSISTENT['executor'] is not None and (not usable or _PERSISTENT['size'] != admitted):
            _close_locked(wait=True)
    completed = False
    with _POOL.override(state):
        try:
            if prewarm and usable:
                _, state['reused'] = _acquire_persistent(admitted)
                state['prewarmed'] = True
            yield state
            completed = True
        finally:
            with _PERSISTENT_LOCK:
                if not completed or state['executor'] is None or _PERSISTENT['pinned']:
                    close_near_process_pool()


def executor_for(pair_count, mode_cap):
    state = _POOL.get()
    if state is None:
        return None
    if state['executor'] is None:
        if (state['process_workers'] <= 1
                or not process_backend_possible(state['process_workers'])
                or not processes_selected(pair_count, mode_cap+1, state['process_workers'])):
            return None
        state['executor'], reused = _acquire_persistent(state['process_workers'])
        state['reused'] = state['reused'] or reused
    # Once started, the pool serves every later call of this preparation, small
    # ones included. A thread team sized without its workers would overlap
    # their memory, which the plan prices as alternatives (threads OR
    # processes), never as a sum; the idle pool kept for the mode phase is
    # priced there (PROCESS_IDLE_BYTES).
    state['jobs'] += pair_count
    return state['executor']


class NearTask:
    """One near element pair, integrated for the nonnegative modes only.

    Negative modes follow by exact tangential-block parity (near_storage),
    and every consumer retains only ``0..m_max``, so the signed half is never
    contracted here.
    """

    def __init__(self, gp, gq, k, m_max, kinds, depth=4, pair_kind=None,
                 near_order=12, near_rtol=2e-5, near_max_order=192,
                 mode_start=0, return_families=False, check_pairs=frozenset()):
        self.gp, self.gq, self.k = gp, gq, k
        self.m_max, self.kinds, self.depth = m_max, kinds, depth
        self.mode_start, self.return_families = int(mode_start), bool(return_families)
        self.pair_kind = pair_kind
        # Touching pairs whose blocks are also integrated by the self-check
        # rule (their result carries the relative change instead of None).
        self.check_pairs = frozenset(check_pairs)
        from ghost_backend.bor.options import current_options
        self.junction_refinement = current_options()['near_refinement']
        self.near_order, self.near_rtol, self.near_max_order = near_order, near_rtol, near_max_order

    def __call__(self, pair):
        return self.run_batch([pair])[0]

    def _points(self, e, f, cell, check=False):
        from ghost_backend.bor.solver import _same_surface_points, _junction_points
        from ghost_backend.bor.sharp_pairs import pair_points, junction_points
        if self.pair_kind is None:
            return (pair_points(self.gp, e, f, self.depth, self.m_max, check=True) if check else
                    _same_surface_points(self.gp, e, f, self.kinds, self.depth, self.m_max))
        if check:
            return junction_points(self.gp, e, self.gq, f, cell, self.junction_refinement, self.m_max, check=True)
        return _junction_points(self.gp, e, self.gq, f, cell, self.junction_refinement, self.m_max)

    def run_batch(self, pairs):
        """Results of every pair, in order, as separate calls would return them.

        Touching pairs are integrated together, and so are the disjoint pairs
        at each refinement level (``_contract_near_batch`` and
        ``_converged_disjoint_batch``): one graded-rule call per batch instead
        of several per pair, bitwise the same blocks.  Same-surface and
        junction pairs use the same rules as the in-process path
        (``_same_surface_points``, ``_junction_points``).
        """
        from ghost_backend.bor.solver import _contract_near_batch, _converged_disjoint_batch
        from ghost_backend.bor.sharp_pairs import relative_change
        results = [None] * len(pairs)
        touching, disjoint, jobs = [], [], []
        for index, pair in enumerate(pairs):
            e, f = pair
            if self.pair_kind is None:
                cell = ('diag' if e == f else 'corner10' if f == e+1 else 'corner01') if abs(e-f) <= 1 else None
            else:
                cell = self.pair_kind.get(tuple(pair))
            if cell is None:
                disjoint.append((index, pair))
                continue
            checked = tuple(pair) in self.check_pairs
            touching.append((index, len(jobs), checked))
            jobs.append((self.gp, e, self.gq, f, self._points(e, f, cell)))
            if checked:
                jobs.append((self.gp, e, self.gq, f, self._points(e, f, cell, check=True)))
        pick = ((lambda blocks: blocks[self.kinds[0]])
                if self.pair_kind is None and not self.return_families else (lambda blocks: blocks))
        if touching:
            outs = _contract_near_batch(jobs, self.k, self.m_max, self.kinds,
                                        signed=False, mode_start=self.mode_start)
            for index, job, checked in touching:
                change = relative_change(outs[job], outs[job + 1]) if checked else None
                results[index] = (pick(outs[job]), change)
        if disjoint:
            outs = _converged_disjoint_batch(self.gp, self.gq, [pair for _, pair in disjoint],
                self.k, self.m_max, self.kinds, self.near_order, self.near_rtol, self.near_max_order,
                signed=False, mode_start=self.mode_start)
            for (index, _), (blocks, order, error) in zip(disjoint, outs):
                results[index] = (pick(blocks), (order, error))
        return results


def run_near_batch(task, batch):
    """Process-pool entry: one pickled task serves a small batch of pairs."""
    return task.run_batch(batch)
