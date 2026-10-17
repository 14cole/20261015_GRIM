"""Dataset format dispatch and deterministic folder loading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import glob
import os

from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.coordinates import canonical_angular_coordinate_system
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.work_control import checkpoint, WorkCancelled
from GRIM_Backend.io.csv import TextEncodingError, has_flat_csv_signature, load_flat_csv as _load_flat_csv_schema


SUPPORTED_EXTENSIONS = (
    ".grim",
    ".csv",
    ".cst_data",
    ".txt",
    ".dat",
    ".asc",
    ".ascii",
    ".tsv",
    ".out",
    ".pio",
    ".cmplx_di",
    ".ptm",
    ".ss",
)


class UnrecognizedTableError(ValueError):
    """No standard reader recognized a text table; the GUI may offer mapping."""


def is_supported_path(path: str) -> bool:
    return str(path).lower().endswith(SUPPORTED_EXTENSIONS)


def load_flat_csv(path: str, *, declared_units=None) -> RcsGrid:
    """Load a supported flat RCS table into an RcsGrid.

    ``declared_units`` supplies units (currently ``frequency``) for legacy
    tables that state none; a unit the table declares always wins.
    """

    return _load_flat_csv_schema(
        path,
        grid_class=RcsGrid,
        canonical_angular_coordinate_system=canonical_angular_coordinate_system,
        c0=C0,
        declared_units=declared_units,
    )


def read_CST(path: str) -> RcsGrid:
    """Read a supported CST far-field table.

    Accepts wide theta/phi CSV and row-oriented ``.cst_data`` tables.
    """
    return RcsGrid.read_CST(path)


def read_SENTRi(path: str) -> RcsGrid:
    """Read a CREATE-RF SENTRi RCS table with its vendor conventions."""

    return RcsGrid.read_SENTRi(path)


def load_dataset(path: str, *, allow_legacy_pickle=False, max_output_bytes=None,
                 storage="memory", selection=None, cache_dir=None, discovery_plan=None,
                 declared_units=None) -> RcsGrid:
    """Load a supported file with an optional memory budget and native selection.

    Native selections use inclusive axis ranges in the file's declared units.
    Disk storage applies to native files, PIO/SS/PTM, flat CSV, SENTRi, CST
    and OUT tables. Pre-load coordinate selection applies to .grim files.
    A validated discovery_plan from estimate_text_import or
    estimate_binary_import avoids repeating coordinate or record discovery;
    replacing the source safely invalidates that plan. ``declared_units``
    supplies a frequency unit for legacy flat CSV tables that declare none.
    """
    from GRIM_Backend.datasets.load_policy import import_memory_budget
    with import_memory_budget(max_output_bytes):
        if str(path).lower().endswith(".grim"):
            return RcsGrid.load(path, allow_legacy_pickle=allow_legacy_pickle,
                               max_output_bytes=max_output_bytes, storage=storage,
                               selection=selection, cache_dir=cache_dir)
        if selection:
            raise ValueError("Selection before loading is supported for .grim files")
        if storage not in {"memory", "disk"}:
            raise ValueError("storage must be memory or disk")
        if str(path).lower().endswith((".pio", ".cmplx_di")):
            return RcsGrid.load_pio(path, storage=storage, cache_dir=cache_dir,
                                    max_output_bytes=max_output_bytes)
        if str(path).lower().endswith(".ss"):
            return RcsGrid.load_ss(path, storage=storage, cache_dir=cache_dir,
                                   max_output_bytes=max_output_bytes,
                                   discovery_plan=discovery_plan)
        if str(path).lower().endswith(".ptm"):
            return RcsGrid.load_ptm(path, storage=storage, cache_dir=cache_dir,
                                    max_output_bytes=max_output_bytes)
        from GRIM_Backend.datasets.result_storage import operation_result_storage
        from GRIM_Backend.io.text_storage import discovery_session
        from GRIM_Backend.io.binary_storage import loaded_binary_grid
        with operation_result_storage(storage, cache_dir=cache_dir):
            with discovery_session(path, discovery_plan):
                grid = _load_dataset_dispatch(path, allow_legacy_pickle=allow_legacy_pickle,
                                              declared_units=declared_units)
        return loaded_binary_grid(grid, storage)


def _load_dataset_dispatch(path: str, *, allow_legacy_pickle=False, declared_units=None) -> RcsGrid:
    """Load any GRIM-supported dataset without importing Qt."""
    path = str(path)
    lower = path.lower()
    if declared_units:
        def load_flat_csv_declared(table_path):
            return load_flat_csv(table_path, declared_units=declared_units)
        load_flat_csv_declared.__name__ = load_flat_csv.__name__
        flat_loader = load_flat_csv_declared
    else:
        flat_loader = load_flat_csv
    if lower.endswith(".grim"):
        return RcsGrid.load(path, allow_legacy_pickle=allow_legacy_pickle)
    if lower.endswith(".out"):
        return RcsGrid.load_out(path)
    if lower.endswith(".ss"):
        return RcsGrid.load_ss(path)
    if lower.endswith(".ptm"):
        return RcsGrid.load_ptm(path)
    if lower.endswith((".pio", ".cmplx_di")):
        return RcsGrid.load_pio(path)
    if lower.endswith(".cst_data"):
        return read_CST(path)
    if lower.endswith((".csv", ".txt")) and RcsGrid.has_SENTRi_signature(path):


        return read_SENTRi(path)
    if lower.endswith((".csv", ".txt")) and has_flat_csv_signature(path):


        return flat_loader(path)
    loaders = (
        (flat_loader, read_CST)
        if lower.endswith(".csv")
        else (
            RcsGrid.load_theta_phi_txt,
            flat_loader,
            read_CST,
        )
    )
    errors = []
    for loader in loaders:
        try:
            return loader(path)
        except (MemoryError, WorkCancelled, TextEncodingError):
            # Undecodable text is a file problem, not an unrecognized layout
            # for the next reader (or the GUI column mapper) to try.
            raise
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    error_type = (UnrecognizedTableError if lower.endswith(
        (".csv", ".txt", ".dat", ".asc", ".ascii", ".tsv")) else ValueError)
    raise error_type("; ".join(errors))


def _matching_dataset_paths(folder: str, *, pattern="*", recursive=False):
    search = (
        os.path.join(str(folder), "**", pattern)
        if recursive
        else os.path.join(str(folder), pattern)
    )
    paths = [
        path
        for path in sorted(glob.glob(search, recursive=recursive))
        if os.path.isfile(path) and is_supported_path(path)
    ]
    if not paths:
        raise ValueError(f"no supported datasets matched {search!r}")
    return paths


def load_folder(
    folder: str,
    *,
    pattern="*",
    recursive=False,
    operation="join",
    workers=1,
    overlap="error",
    max_output_bytes=None,
    coherent_metadata_attested=False,
    stitch_policy="priority-first",
    tol=1.0e-6,
    storage="memory",
    cache_dir=None,
    ram_budget_bytes=None,
    cancelled=None,
    progress=None,
):
    """Load and combine in pathname order with adjustable memory/disk storage.

    ``storage='disk'`` retains supported inputs and the output on temporary
    disk. ``'auto'`` chooses disk for the batch when the estimated resident
    inputs and workspace exceed the batch or shared application RAM allowance.
    Its combined output uses only the batch allowance left after retained inputs;
    larger results spill to disk. ``workers`` remains an upper bound;
    simultaneous workspaces must fit the allowance.
    ``max_output_bytes`` also reaches individual readers. Optional cancellation
    and progress callbacks run in worker threads, not a GUI event loop.
    """
    paths = _matching_dataset_paths(
        folder, pattern=pattern, recursive=recursive
    )
    from GRIM_Backend.datasets.result_storage import operation_result_storage
    from GRIM_Backend.datasets.work_control import work_control
    from GRIM_Backend.datasets.load_policy import checked_import_budget
    from GRIM_Backend.datasets.memory import _default_dense_import_limit_bytes

    if storage not in {"memory", "disk", "auto"}:
        raise ValueError("storage must be memory, disk, or auto")
    explicit_budget = checked_import_budget(ram_budget_bytes)
    budget = explicit_budget or checked_import_budget(max_output_bytes) or _default_dense_import_limit_bytes()
    workers = max(1, min(int(workers), len(paths)))
    with work_control(cancelled=cancelled, progress=progress):
        plans = [_folder_file_estimate(path, "memory" if storage == "auto" else storage)
                 for path in paths]
        retained = sum(report["retained_bytes"] for report in plans)
        extras = sorted((max(0, report["estimated_peak_bytes"] - report["retained_bytes"])
                         for report in plans), reverse=True)
        selected_storage = storage
        if storage == "auto":
            from GRIM_Backend.datasets.result_storage import application_result_storage_info
            application = application_result_storage_info()
            available = min(budget, max(0, application["ram_budget_bytes"]
                                        - application["reserved_ram_bytes"]))
            selected_storage = "disk" if retained + max(extras, default=0) > available else "memory"
            if selected_storage == "disk":
                # Keep coordinate discovery, update only resident accounting.
                plans = [_folder_file_estimate(path, "disk", report.get("discovery_plan"))
                         for path, report in zip(paths, plans)]
                retained = sum(report["retained_bytes"] for report in plans)
                extras = sorted((max(0, report["estimated_peak_bytes"] - report["retained_bytes"])
                                 for report in plans), reverse=True)
        scheduling_budget = min(budget, available) if storage == 'auto' and selected_storage == 'memory' else budget
        while workers > 1 and retained + sum(extras[:workers]) > scheduling_budget:
            workers -= 1

        def load_one(item):
            index, (path, report) = item
            checkpoint(index, len(paths), "Loading folder datasets")
            return load_dataset(path, storage=selected_storage, cache_dir=cache_dir,
                                max_output_bytes=max_output_bytes,
                                discovery_plan=report.get("discovery_plan"))

        if workers == 1:
            grids = [load_one(item) for item in enumerate(zip(paths, plans))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(copy_context().run, load_one, item)
                           for item in enumerate(zip(paths, plans))]
                try:
                    grids = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        checkpoint(detail="Combining folder datasets")
        # Source selection does not fix the output policy: Auto must account
        # for the inputs still alive during combination and honor the shared
        # allocator, even when every individual input fitted in RAM.
        output_storage = "auto" if storage == "auto" else selected_storage
        output_budget = max(0, budget - retained) if storage == "auto" else budget
        with operation_result_storage(output_storage, ram_budget_bytes=output_budget,
                                      cache_dir=cache_dir):
            return combine_datasets(
                grids, operation, overlap=overlap, max_output_bytes=max_output_bytes,
                coherent_metadata_attested=coherent_metadata_attested,
                stitch_policy=stitch_policy, tol=tol)


def _folder_file_estimate(path, storage, discovery_plan=None):
    """Headless format planning; malformed inputs retain their reader errors."""
    from GRIM_Backend.io.binary_storage import estimate_binary_import
    from GRIM_Backend.io.text_storage import estimate_text_import
    from GRIM_Backend.datasets.memory import estimate_native_archive_allocation
    checkpoint(detail="Planning folder datasets")
    if str(path).lower().endswith(".grim"):
        return estimate_native_archive_allocation(path, storage=storage)
    report = estimate_binary_import(path, storage=storage, discovery_plan=discovery_plan)
    if report is not None:
        return report
    report = estimate_text_import(path, storage=storage, discovery_plan=discovery_plan)
    if report is not None:
        return report
    size = os.path.getsize(path)
    return {"retained_bytes": 16 * size, "estimated_peak_bytes": max(64 * 1024**2, 96 * size)}
