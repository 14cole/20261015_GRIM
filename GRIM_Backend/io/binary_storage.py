"""Bounded binary-reader destinations using the shared managed allocator."""
from contextlib import contextmanager
from dataclasses import dataclass
import sys

import numpy as np

from GRIM_Backend.datasets.load_policy import import_memory_budget
from GRIM_Backend.datasets.memory import _checked_dense_import_allocation
from GRIM_Backend.datasets.result_storage import operation_result_storage
from GRIM_Backend.io.text_storage import source_signature

BLOCK_CELLS = 32_768


@dataclass(frozen=True)
class BinaryDiscoveryPlan:
    """A validated record layout of one binary source, never decoded samples.

    An estimate scans the records once; a later load of the unchanged file
    reuses that layout instead of scanning again. The source signature ties
    the plan to the exact file contents it was built from.
    """
    signature: tuple
    source_format: str
    layout: object


def valid_binary_layout(path, plan, source_format):
    """Return the plan's layout when it describes this exact file, else None."""
    if not isinstance(plan, BinaryDiscoveryPlan) or plan.source_format != source_format:
        return None
    # A replaced file keeps its own framing; a stale plan is simply ignored.
    return plan.layout if plan.signature == source_signature(path) else None


@contextmanager
def binary_import_storage(storage, *, cache_dir=None, max_output_bytes=None):
    if storage not in {"memory", "disk"}:
        raise ValueError("storage must be memory or disk")
    with operation_result_storage(storage, cache_dir=cache_dir):
        with import_memory_budget(max_output_bytes):
            yield


def binary_import_allocation(shape, dtype, *, storage, source,
                             scratch_bytes=0, retained_metadata_bytes=0, max_output_bytes=None):
    """Keep address/shape validation while budgeting mapped outputs as disk."""
    scratch_bytes = int(scratch_bytes)
    if storage == "memory":
        report = _checked_dense_import_allocation(
            shape, (dtype,), source=source, resident_bytes=scratch_bytes,
            max_output_bytes=max_output_bytes)
    else:
        report = _checked_dense_import_allocation(
            shape, (dtype,), source=source, max_output_bytes=sys.maxsize)
        # The mapped address range is not a retained heap allocation. The binary
        # source, decoded temporary samples, and validation masks are block-sized.
        working = _checked_dense_import_allocation(
            (max(1, scratch_bytes),), (np.uint8,), source=source,
            max_output_bytes=max_output_bytes)
        report.update(peak_bytes=working["peak_bytes"], resident_bytes=scratch_bytes,
                      limit_bytes=working["limit_bytes"])
    report["retained_bytes"] = int(retained_metadata_bytes) + (report["dense_bytes"] if storage == "memory" else 0)
    return report


def loaded_binary_grid(grid, storage):
    grid.load_storage = storage
    # Loading has its own recorder path, separate from derived operations.
    grid.__dict__.pop("_operation_storage_options", None)
    return grid


def estimate_binary_import(path, *, storage="memory", discovery_plan=None):
    """Discover PIO/SS/PTM layout and bounded working memory without decoding cells.

    SS reports carry a ``discovery_plan`` whose validated record layout a
    subsequent ``load_dataset`` call reuses, so files with many records are
    scanned once per import rather than once for planning and once for loading.
    """
    from GRIM_Backend.datasets.grid import RcsGrid
    lower = str(path).lower()
    if lower.endswith(".ptm"):
        from GRIM_Backend.io.ptm import read_ptm
        report = read_ptm(path, storage=storage, max_output_bytes=sys.maxsize,
                          _estimate_only=True)
        return dict(report, estimated_peak_bytes=report["peak_bytes"], source_format="PTM")
    if storage not in {"memory", "disk"}:
        raise ValueError("storage must be memory or disk")
    if lower.endswith((".pio", ".cmplx_di")):
        report = RcsGrid._load_pio_bounded(path, storage=storage, max_output_bytes=sys.maxsize,
                                           _estimate_only=True)
        return dict(report, estimated_peak_bytes=report["peak_bytes"], source_format="PIO")
    if lower.endswith(".ss"):
        report = RcsGrid._load_ss_bounded(path, storage=storage, max_output_bytes=sys.maxsize,
                                          _estimate_only=True, discovery_plan=discovery_plan)
        return dict(report, estimated_peak_bytes=report["peak_bytes"], source_format="SS")
    return None
