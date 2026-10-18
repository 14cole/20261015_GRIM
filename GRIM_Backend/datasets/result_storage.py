"""Operation-scoped allocation of ordinary RAM or private disk-backed results.

Only retained destinations use this allocator. Bounded numerical scratch stays
in RAM. Mapped arrays own their private files through NumPy base/view chains,
so a dataset, undo entry, or held view can safely outlive the operation scope.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import operator
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import weakref

import numpy as np

BLOCK_CELLS = 65_536
_POLICY = ContextVar("grim_operation_result_storage", default=None)
_pending_cleanup = set()
_RAM_LOCK = threading.RLock()
_LIVE_RAM_BYTES = 0
_AUTO_RAM_BUDGET = None
_UNSET = object()
_WORKING_FOLDER = None


def configure_working_folder(path=None):
    """Select the folder for private temporary files; None restores the system temp.

    Operations that name no folder of their own (plot preparation, statistics
    workspaces, assembly builds, report scratch) then use the folder the user
    chose for results instead of always filling the system drive.
    """
    global _WORKING_FOLDER
    _WORKING_FOLDER = os.fspath(path) if path else None


def configured_working_folder():
    """The folder chosen with configure_working_folder, or None."""
    return _WORKING_FOLDER


def working_folder():
    """The active operation's folder, else the configured folder, else system temp."""
    policy = _POLICY.get()
    if policy is not None and policy.cache_dir:
        return policy.cache_dir
    return _WORKING_FOLDER or tempfile.gettempdir()


def _checked_application_budget(nbytes):
    if nbytes is not None:
        if isinstance(nbytes, (bool, np.bool_)):
            raise ValueError('application RAM budget must be a nonnegative integer')
        nbytes = operator.index(nbytes)
        if nbytes < 0:
            raise ValueError('application RAM budget must be a nonnegative integer')
    return nbytes


def configure_auto_ram_budget(nbytes=None):
    """Set the application-wide Auto allowance; None selects a system default.

    Existing buffers stay resident. Explicit RAM operations remain available.
    The allowance covers managed buffers, not all Python/library allocations.
    """
    global _AUTO_RAM_BUDGET
    nbytes = _checked_application_budget(nbytes)
    with _RAM_LOCK:
        _AUTO_RAM_BUDGET = nbytes


def _application_budget():
    # A scope may carry its own allowance; it never changes the global one.
    policy = _POLICY.get()
    override = policy.application_ram_budget_bytes if policy is not None else _UNSET
    configured = _AUTO_RAM_BUDGET if override is _UNSET else override
    if configured is not None:
        return configured
    from GRIM_Backend.datasets.memory import _available_import_memory_bytes
    available = _available_import_memory_bytes()
    # Include our existing reservations so retaining a result does not itself
    # reduce the allowance twice. Other applications' pressure still matters.
    return (max(1, (available + _LIVE_RAM_BYTES) // 4)
            if available is not None else 512 * 1024**2)


def application_result_storage_info():
    with _RAM_LOCK:
        return {'reserved_ram_bytes': _LIVE_RAM_BYTES,
                'ram_budget_bytes': _application_budget(),
                'automatic': _AUTO_RAM_BUDGET is None}


def _remove(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        _pending_cleanup.discard(path)
        from GRIM_Backend.datasets.storage_registry import forget_cache
        forget_cache(path)
    except OSError:
        _pending_cleanup.add(path)
    else:
        _pending_cleanup.discard(path)
        from GRIM_Backend.datasets.storage_registry import forget_cache
        forget_cache(path)


def _retry_cleanup():
    for path in tuple(_pending_cleanup):
        _remove(path)


def _close(path, mappings):
    for mapping in mappings:
        mapping.close()
    _remove(path)


atexit.register(_retry_cleanup)


class _Owner:
    def __init__(self, directory):
        _retry_cleanup()
        self.path = tempfile.mkdtemp(prefix="grim-result-", dir=directory)
        self.mappings = []
        from GRIM_Backend.datasets.storage_registry import register_cache
        register_cache(self.path, kind="result", owner=self)
        self.finalizer = weakref.finalize(self, _close, self.path, self.mappings)


def _shares_buffer(array, source):
    try:
        return bool(np.may_share_memory(array, source))
    except Exception:  # noqa: BLE001 - keep the owner when unsure
        return True


class _ResultArray(np.ndarray):
    def __array_finalize__(self, source):
        owner = getattr(source, "_result_owner", None)
        # Views share the owner's buffer and must keep it alive. Independent
        # results (a * 2, a.copy(), deepcopy) must not pin its temporary file.
        if owner is not None and not _shares_buffer(self, source):
            owner = None
        self._result_owner = owner


@dataclass
class _Policy:
    mode: str
    ram_budget_bytes: int
    cache_dir: str | None
    allow_exceed: bool = False
    application_ram_budget_bytes: object = _UNSET
    reserved_ram_bytes: int = 0
    lock: object = field(default_factory=threading.RLock)


def _release_ram(policy, nbytes):
    global _LIVE_RAM_BYTES
    with _RAM_LOCK, policy.lock:
        policy.reserved_ram_bytes = max(0, policy.reserved_ram_bytes - nbytes)
        _LIVE_RAM_BYTES = max(0, _LIVE_RAM_BYTES - nbytes)


class _MemoryOwner:
    def __init__(self, policy, nbytes):
        self.finalizer = weakref.finalize(self, _release_ram, policy, nbytes)


@contextmanager
def operation_result_storage(mode="memory", *, ram_budget_bytes=256 * 1024**2,
                             cache_dir=None, allow_exceed=False,
                             application_ram_budget_bytes=_UNSET):
    """Select output storage for this operation and its synchronous callees.

    Auto keeps live managed allocations within the RAM budget and maps larger
    destinations. Released buffers return their allowance. This is not a total
    process-memory guarantee; numerical scratch and other libraries use RAM.
    Contexts are nested and thread-local; each worker must enter its own scope.
    """
    mode = str(mode).strip().lower()
    if mode not in {"memory", "disk", "auto"}:
        raise ValueError("operation result storage must be memory, disk, or auto")
    if not isinstance(allow_exceed, bool):
        raise ValueError('allow_exceed must be True or False')
    if isinstance(ram_budget_bytes, (bool, np.bool_)):
        raise ValueError("result RAM budget must be a nonnegative integer")
    try:
        budget = operator.index(ram_budget_bytes)
    except TypeError as exc:
        raise ValueError("result RAM budget must be a nonnegative integer") from exc
    if budget < 0:
        raise ValueError("result RAM budget must be a nonnegative integer")
    # Omitted settings are inherited from an enclosing scope, so a nested
    # scratch scope keeps the folder and allowance chosen for the operation.
    outer = _POLICY.get()
    directory = os.fspath(cache_dir) if cache_dir else (outer.cache_dir if outer is not None else None)
    if application_ram_budget_bytes is not _UNSET:
        # Scoped to this operation and thread; the application-wide value is
        # changed only with configure_auto_ram_budget.
        application_ram_budget_bytes = _checked_application_budget(application_ram_budget_bytes)
    elif outer is not None:
        application_ram_budget_bytes = outer.application_ram_budget_bytes
    token = _POLICY.set(_Policy(mode, budget, directory, allow_exceed=allow_exceed,
                                application_ram_budget_bytes=application_ram_budget_bytes))
    try:
        from GRIM_Backend.datasets.load_policy import import_memory_budget
        with import_memory_budget(sys.maxsize if allow_exceed else None):
            yield
    finally:
        _POLICY.reset(token)
        _retry_cleanup()


def result_storage_mode():
    policy = _POLICY.get()
    return policy.mode if policy is not None else "memory"


def result_storage_info():
    policy = _POLICY.get()
    if policy is None:
        return None
    return {"mode": result_storage_mode(),
            "ram_budget_bytes": policy.ram_budget_bytes if policy else None,
            "reserved_ram_bytes": policy.reserved_ram_bytes if policy else 0,
            "allow_exceed": policy.allow_exceed,
            "cache_dir": policy.cache_dir if policy else None}


def typed_samples(values, dtype):
    """Keep lazy components lazy and promote concrete arrays in bounded tiles."""
    if not hasattr(values, "shape") or not hasattr(values, "dtype"):
        return np.asarray(values, dtype=dtype)
    return values if np.dtype(values.dtype) == np.dtype(dtype) else copy_result(values, dtype=dtype)


def resident_result_bytes(nbytes, *, aggregate=False):
    """Project additional result RAM without allocating or reserving it.

    A single array either fits or spills. For multiple planned allocations,
    ``aggregate=True`` bounds the total RAM those arrays could occupy: some
    small arrays may fit even when their combined size exceeds the budget.
    """
    nbytes = operator.index(nbytes)
    if nbytes < 0:
        raise ValueError("result byte count must be nonnegative")
    policy = _POLICY.get()
    if policy is None or policy.mode == "memory":
        return nbytes
    if policy.mode == "disk":
        return 0
    with _RAM_LOCK, policy.lock:
        remaining = max(0, min(policy.ram_budget_bytes - policy.reserved_ram_bytes,
                               _application_budget() - _LIVE_RAM_BYTES))
        return min(nbytes, remaining) if aggregate else nbytes if nbytes <= remaining else 0


def array_storage(value):
    """Report actual memory backing, including plain ndarray views of maps."""
    seen = set()
    current = value
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            return "disk"
        current = getattr(current, "base", None)
    return "memory"


def allocate_result(shape, dtype, *, fill=None, order="C"):
    """Allocate a writable, privately owned operation destination."""
    global _LIVE_RAM_BYTES
    from GRIM_Backend.datasets.work_control import checkpoint
    checkpoint()
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    shape = tuple(operator.index(extent) for extent in shape)
    if any(extent < 0 for extent in shape):
        raise ValueError("result dimensions must be nonnegative")
    dtype = np.dtype(dtype)
    if order not in {"C", "F"}:
        raise ValueError("result order must be C or F")
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes > np.iinfo(np.intp).max:
        raise MemoryError("operation result exceeds this Python/NumPy build's addressable size")
    policy = _POLICY.get()
    with _RAM_LOCK:
        mapped = policy is not None and resident_result_bytes(nbytes) == 0 and nbytes > 0
        reserved = policy is not None and (not mapped or dtype.hasobject)
        if reserved:
            with policy.lock:
                policy.reserved_ram_bytes += nbytes
                _LIVE_RAM_BYTES += nbytes
    # Object arrays contain process pointers and cannot be persisted in a map.
    # Metadata objects remain resident; scientific sample arrays are numeric.
    if mapped and not dtype.hasobject:
        directory = policy.cache_dir or _WORKING_FOLDER or tempfile.gettempdir()
        if not Path(directory).is_dir():
            raise OSError(f"Operation result working folder does not exist: {directory}")
        free = shutil.disk_usage(directory).free
        if nbytes + 4096 > free:
            raise OSError(f"Operation result needs {nbytes / 1024**2:.1f} MiB of temporary disk space; "
                          f"only {free / 1024**2:.1f} MiB is free in {directory}")
        owner = _Owner(directory)
        try:
            source = np.memmap(Path(owner.path) / "samples.bin", mode="w+", dtype=dtype,
                               shape=shape, order=order)
            owner.mappings.append(source._mmap)
            result = source.view(_ResultArray)
            result._result_owner = owner
        except Exception:
            owner.finalizer()
            raise
    else:
        if policy is None:
            result = np.empty(shape, dtype=dtype, order=order)
        else:
            # Own the allocation in the subclass itself: plain ndarray views must
            # keep this owner alive, rather than bypass it through an ndarray base.
            try:
                result = np.ndarray.__new__(_ResultArray, shape, dtype=dtype, order=order)
            except BaseException:
                _release_ram(policy, nbytes)
                raise
            result._result_owner = _MemoryOwner(policy, nbytes)
    if fill is not None:
        for block in result_blocks(shape):
            result[block] = fill
    return result


def result_blocks(shape, maximum_cells=BLOCK_CELLS):
    """Bounded basic selections, also supporting scalars and empty arrays."""
    from GRIM_Backend.datasets.memory import _bounded_grid_selections
    shape = tuple(shape)
    if not shape:
        yield (Ellipsis,)
    elif all(shape):
        yield from _bounded_grid_selections(shape, maximum_cells)


def copy_result(values, *, dtype=None):
    """Copy array-like values without materializing a derived full-grid source."""
    if not hasattr(values, "shape") or not hasattr(values, "dtype"):
        values = np.asarray(values)
    flags = getattr(values, "flags", None)
    order = "F" if flags is not None and flags.f_contiguous and not flags.c_contiguous else "C"
    result = allocate_result(values.shape, values.dtype if dtype is None else dtype, order=order)
    for block in result_blocks(result.shape):
        result[block] = values[block]
    return result


def _progression_slice(index, extent):
    """Return the basic index equal to *index*, or None.

    *index* is a validated integer, a ``range`` or a 1-D in-bounds integer
    array; integers stay integers so the axis is dropped like before.
    """
    if isinstance(index, int):
        return index
    if isinstance(index, range):
        if len(index) == 0:
            return slice(0, 0)
        if len(index) == 1:
            return slice(index.start, index.start + 1)
        return slice(index.start, index.stop, index.step) if index.step > 0 else None
    if index.size == 0 or np.any(index < 0) or np.any(index >= extent):
        return None   # empty, negative or out of range: keep NumPy's own handling
    first = int(index[0])
    if index.size == 1:
        return slice(first, first + 1)
    step = int(index[1]) - first
    if step <= 0 or not np.all(np.diff(index) == step):
        return None
    return slice(first, int(index[-1]) + 1, step)


def select_result(values, selection=Ellipsis, *, dtype=None):
    """Copy basic or orthogonal (``np.ix_``) selections in bounded tiles."""
    shape = tuple(values.shape)
    selection = selection if isinstance(selection, tuple) else (selection,)
    if isinstance(values, np.ndarray) and all(
            item is Ellipsis or isinstance(item, (slice, int, np.integer)) for item in selection):
        return copy_result(values[selection], dtype=dtype)
    expanded = []
    for item in selection:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - len(selection) + 1))
        else:
            expanded.append(item)
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))
    if len(expanded) != len(shape) or any(item is None for item in expanded):
        return copy_result(values[selection], dtype=dtype)
    indices, output_shape = [], []
    for extent, item in zip(shape, expanded):
        if isinstance(item, (int, np.integer)):
            index = operator.index(item)
            if index < 0:
                index += extent
            if index < 0 or index >= extent:
                raise IndexError("result sample index out of bounds")
            indices.append(index)
        elif isinstance(item, slice):
            index = range(*item.indices(extent))
            indices.append(index)
            output_shape.append(len(index))
        else:
            index = np.asarray(item)
            if index.dtype.kind == "b" and index.ndim == 1:
                if index.size != extent:
                    raise IndexError("boolean result sample index has the wrong length")
                index = np.flatnonzero(index)
            elif index.dtype.kind not in "iu" or sum(size != 1 for size in index.shape) > 1:
                return copy_result(values[selection], dtype=dtype)
            index = index.reshape(-1)
            indices.append(index)
            output_shape.append(index.size)
    # Multiple plain 1D advanced indices use paired indexing, not Cartesian.
    # Operations use np.ix_; retain NumPy semantics for arbitrary public input.
    advanced = [item for item in expanded if isinstance(item, np.ndarray)]
    if len(advanced) > 1 and all(item.ndim == 1 for item in advanced):
        return copy_result(values[selection], dtype=dtype)
    if isinstance(values, np.ndarray):
        # Index vectors that are evenly spaced runs (ranges, strides, single
        # coordinates: every crop and most alignments) select exactly the
        # cells of a basic view, which copies in 64k-cell tiles instead of
        # 4,096-cell np.ix_ gathers.
        basic = tuple(_progression_slice(item, extent) for item, extent in zip(indices, shape))
        if all(item is not None for item in basic):
            return copy_result(values[basic], dtype=dtype)
    result = allocate_result(output_shape, values.dtype if dtype is None else dtype)
    for block in result_blocks(output_shape, maximum_cells=4096):
        selected, output_axis = [], 0
        integer_axes = []
        for axis, item in enumerate(indices):
            if isinstance(item, int):
                selected.append(np.asarray([item], dtype=np.intp))
                integer_axes.append(axis)
            else:
                selected.append(np.asarray(item[block[output_axis]], dtype=np.intp))
                output_axis += 1
        gathered = values[np.ix_(*selected)]
        if integer_axes:
            gathered = np.squeeze(gathered, axis=tuple(integer_axes))
        result[block] = gathered
    return result


def take_result(values, indices, axis=0):
    """Bounded equivalent of np.take with one-dimensional indices."""
    axis = operator.index(axis)
    if axis < 0:
        axis += len(values.shape)
    if axis < 0 or axis >= len(values.shape):
        raise ValueError("take axis is out of range")
    selected = [slice(None)] * len(values.shape)
    selected[axis] = np.asarray(indices, dtype=np.intp).reshape(-1)
    return select_result(values, tuple(selected))
