"""Flat RCS CSV schemas, validation, loading, and export."""
from __future__ import annotations

import codecs
import contextlib
import csv
import ctypes
import io
import math
import os
import tempfile
import warnings

import numpy as np

from GRIM_Backend.datasets.result_storage import allocate_result, resident_result_bytes
from GRIM_Backend.datasets.work_control import checkpoint
from GRIM_Backend.io.text_storage import discover_coordinates, text_reader

from GRIM_Backend.datasets.memory import _bounded_grid_selections
from GRIM_Backend.io.samples import construct_parsed_grid, parsed_sample_memory

_CSV_BLOCK_CELLS = 65_536


FLAT_CSV_SCHEMA = "grim.flat-rcs.v1"


_FREQUENCY_FACTORS = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}


_MAGNITUDE_COLUMNS = (
    "magnitude_power_linear",
    "magnitude_linear",
    "magnitude_db",
    "magnitude_dbsm",
    "magnitude_dbke",
)


_QUANTITY_LOG_UNITS = {
    "sigma_3d": "dBsm",
    "sigma_2d": "dBke",
    "power_ratio": "dB",
}


_V1_REQUIRED_COLUMNS = (
    "grim_csv_schema",
    "azimuth",
    "azimuth_unit",
    "elevation",
    "elevation_unit",
    "frequency",
    "frequency_unit",
    "polarization",
    "rcs_linear_quantity",
    "rcs_log_unit",
    "angular_coordinate_system",
)


class TextEncodingError(ValueError):
    """A text dataset is neither UTF-8 nor BOM-marked UTF-16."""


def text_table_encoding(path):
    """Return the decoder for a text table: BOM-marked UTF-16 or UTF-8."""

    with open(path, "rb") as stream:
        head = stream.read(2)
    if head in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"
    return "utf-8-sig"


@contextlib.contextmanager
def open_text_table(path, *, newline=None):
    """Open a text table, naming the file when its bytes cannot be decoded.

    UTF-16 files are accepted only with a byte-order mark; everything else is
    read as UTF-8 (optional BOM).  Decoding failures raise
    :class:`TextEncodingError` instead of a bare ``UnicodeDecodeError``.
    """

    encoding = text_table_encoding(path)
    try:
        with open(path, "r", newline=newline, encoding=encoding) as stream:
            yield stream
    except UnicodeDecodeError as exc:
        label = "UTF-16" if encoding == "utf-16" else "UTF-8"
        raise TextEncodingError(
            f"{os.path.basename(os.fspath(path))}: file is not valid {label} text "
            f"({exc.reason}); save it as UTF-8 and reload"
        ) from exc


def _field_map(fieldnames):
    mapped = {}
    duplicates = []
    for raw_name in fieldnames or ():
        if raw_name is None:
            continue
        key = str(raw_name).strip().lower()
        if not key:
            continue
        if key in mapped:
            duplicates.append(str(raw_name))
        else:
            mapped[key] = raw_name
    if duplicates:
        raise ValueError(
            "duplicate CSV column name(s), ignoring case: "
            + ", ".join(duplicates)
        )
    return mapped


def has_flat_csv_signature(path):
    """Return whether the file header identifies a supported flat RCS schema."""

    try:
        with open_text_table(path, newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = csv.reader(stream, delimiter=delimiter)
            header = next(reader, None)
    except (OSError, UnicodeError, TextEncodingError, csv.Error):
        return False
    try:
        fields = _field_map(header)
    except ValueError:
        return True
    plain_axes = {"azimuth", "elevation", "frequency", "polarization"}
    cem_axes = {
        "azimuth_deg",
        "elevation_deg",
        "frequency_ghz",
        "polarization",
    }
    return bool(
        (plain_axes.issubset(fields) or cem_axes.issubset(fields))
        and any(name in fields for name in _MAGNITUDE_COLUMNS)
    )


def _canonical_frequency_unit(value):
    aliases = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}
    text = str(value or "GHz").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported frequency unit {!r}".format(value))
    return aliases[text]


def _canonical_angle_unit(value):
    aliases = {
        "deg": "deg",
        "degree": "deg",
        "degrees": "deg",
        "rad": "rad",
        "radian": "rad",
        "radians": "rad",
    }
    text = str(value or "deg").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported angular unit {!r}; use deg or rad".format(value))
    return aliases[text]


def _canonical_log_unit(value):
    aliases = {"db": "dB", "dbsm": "dBsm", "dbke": "dBke"}
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError("unsupported RCS logarithmic unit {!r}".format(value))
    return aliases[text]


def _canonical_quantity(value):
    aliases = {
        "sigma_3d": "sigma_3d",
        "sigma3d": "sigma_3d",
        "sigma_2d": "sigma_2d",
        "sigma2d": "sigma_2d",
        "power_ratio": "power_ratio",
        "ratio": "power_ratio",
    }
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError(
            "unsupported rcs_linear_quantity {!r}; use sigma_3d, sigma_2d, "
            "or power_ratio".format(value)
        )
    return aliases[text]


def _canonical_phase_wrap(value):
    aliases = {
        "": "-180_180",
        "-180_180": "-180_180",
        "-180-180": "-180_180",
        "pm180": "-180_180",
        "0_360": "0_360",
        "0-360": "0_360",
    }
    text = str(value or "").strip().lower()
    if text not in aliases:
        raise ValueError(
            "unsupported phase_wrap {!r}; use -180_180 or 0_360".format(value)
        )
    return aliases[text]


def _legacy_frequency_unit(values):
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    typical = float(np.median(np.abs(finite))) if finite.size else 0.0
    if typical >= 1.0e6:
        unit = "Hz"
    elif typical >= 1.0e3:
        unit = "MHz"
    else:
        unit = "GHz"
    rule = (
        "legacy magnitude heuristic selected {} from median |frequency|={:.9g}; "
        "add frequency_unit to remove this inference"
    ).format(unit, typical)
    return unit, rule


def _available_memory_bytes():
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _allocation_budget_bytes():
    from GRIM_Backend.datasets.load_policy import current_import_memory_budget
    captured = current_import_memory_budget()
    if captured is not None:
        return captured, 'the selected load memory budget'
    declared = str(os.environ.get("GRIM_MAX_CSV_GRID_GB", "") or "").strip()
    if declared:
        try:
            gib = float(declared)
        except ValueError as exc:
            raise ValueError("GRIM_MAX_CSV_GRID_GB must be a positive number") from exc
        if not np.isfinite(gib) or gib <= 0.0:
            raise ValueError("GRIM_MAX_CSV_GRID_GB must be a positive number")
        return int(gib * 1024.0 ** 3), "GRIM_MAX_CSV_GRID_GB"
    available = _available_memory_bytes()
    if available is not None and available > 0:
        return int(available * 0.5), "50% of currently available RAM"
    return 2 * 1024 ** 3, "2 GiB fallback because available RAM is unknown"


class CsvMemoryLimitError(MemoryError, ValueError):
    """Keep legacy ValueError compatibility while identifying budget failures."""


def _preflight_dense_grid(axis_lengths, record_count, *, check_budget=True):
    cells = 1
    for length in axis_lengths:
        cells *= int(length)
    dense_payload, estimated_peak = parsed_sample_memory(axis_lengths)
    budget, source = _allocation_budget_bytes() if check_budget else (None, None)
    if check_budget and estimated_peak > budget:
        raise CsvMemoryLimitError(
            "flat CSV axes form a dense grid of {:,} cells ({} x {} x {} x {}), "
            "requiring about {:.2f} GiB peak for power/phase arrays; {} allows "
            "{:.2f} GiB. The file contains only {:,} row(s), so verify that the "
            "axes were intended to form a Cartesian grid. Increase "
            "GRIM_MAX_CSV_GRID_GB only after confirming sufficient RAM."
            .format(
                cells,
                axis_lengths[0],
                axis_lengths[1],
                axis_lengths[2],
                axis_lengths[3],
                estimated_peak / 1024.0 ** 3,
                source,
                budget / 1024.0 ** 3,
                record_count,
            )
        )
    return cells, dense_payload, estimated_peak


def _grid_scalar(value, label):
    if value is None:
        return ""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("flat CSV {} must be scalar".format(label))
    return str(array.reshape(-1)[0].item()).strip()


def _declared_grid_metadata(grid, key):
    """Return one scalar convention declared in either units or extra.

    RcsGrid owns the canonical compatibility rules, including equivalent time
    convention spellings, so use its helper when available. The dependency-
    light fallback still validates scalar shape and refuses two different
    nonblank declarations rather than silently preferring one container.
    """

    declared_helper = getattr(grid, "_declared_scalar_metadata", None)
    if callable(declared_helper):
        return _grid_scalar(declared_helper(key), key)

    declarations = []
    for container_name in ("units", "extra"):
        container = dict(getattr(grid, container_name, {}) or {})
        if key not in container:
            continue
        value = _grid_scalar(container[key], key)
        if value:
            declarations.append(value)
    normalized = {
        " ".join(value.split()).casefold() for value in declarations
    }
    if len(normalized) > 1:
        raise ValueError(
            "dataset contains contradictory {} metadata".format(key)
        )
    return declarations[0] if declarations else ""


def _finite_csv_number(value):
    number = float(value)
    if np.isnan(number):
        return ""
    if np.isneginf(number):
        return "-inf"
    if not np.isfinite(number):
        raise ValueError("flat CSV cannot represent positive infinity")
    return format(number, ".17g")


def _csv_numbers(values):
    """``_finite_csv_number`` for a float64 array, returned as a list of text."""
    text = []
    for value in values.tolist():
        if value != value:
            text.append("")
        elif value == -math.inf:
            text.append("-inf")
        elif value == math.inf:
            raise ValueError("flat CSV cannot represent positive infinity")
        else:
            text.append(format(value, ".17g"))
    return text


def _csv_fields(fields, delimiter):
    """Render fields exactly as ``csv.writer`` does, without a line ending."""
    buffer = io.StringIO()
    # Keep the normal terminator while quoting: csv.writer only treats CR/LF
    # as special characters when they appear in its configured terminator.
    csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n").writerow(fields)
    return buffer.getvalue()[:-2]


def _write_flat_csv_block(stream, delimiter, grid, selection, power_block, phase_block, *,
                          magnitude_columns, phase_wrap, frequency_unit, leading, labels):
    """Write one sample block of the v1 schema, one numpy pass per column.

    Equivalent to formatting row by row: every value is computed with the same
    float64 expressions, formatted with ``.17g`` and joined in the row-major
    (azimuth, elevation, frequency, polarization) order; text fields go
    through ``csv.writer`` once per distinct value so quoting is unchanged.
    """
    azimuth_unit, elevation_unit = leading
    frequencies = grid.frequencies[selection[2]]
    frequency_hz = []
    for frequency in frequencies:
        value = float(frequency) * _FREQUENCY_FACTORS[frequency_unit]
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("flat CSV frequencies must be positive and finite")
        frequency_hz.append(value)
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    angle_heads = [
        _csv_fields([FLAT_CSV_SCHEMA, azimuth, azimuth_unit, elevation, elevation_unit], delimiter)
        for azimuth in (_finite_csv_number(value) for value in grid.azimuths[selection[0]])
        for elevation in [_finite_csv_number(value) for value in grid.elevations[selection[1]]]
    ]
    frequency_heads = [_csv_fields([_finite_csv_number(value), frequency_unit], delimiter)
                       for value in frequencies]
    polarization_heads = [_csv_fields([str(value)] + labels, delimiter)
                          for value in grid.polarizations[selection[3]]]
    power = np.asarray(power_block, dtype=np.float64)
    finite = np.isfinite(power)
    columns = []
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for column in magnitude_columns:
            if column == "magnitude_power_linear":
                values = np.where(finite, power, np.nan)
            else:
                if column == "magnitude_dbke":
                    wavelength = (299792458.0 / frequency_hz).reshape(1, 1, -1, 1)
                    values = 10.0 * np.log10(power * 2.0 * math.pi / wavelength)
                else:
                    values = 10.0 * np.log10(power)
                values = np.where(power == 0.0, -np.inf, values)
                values = np.where(finite, values, np.nan)
            columns.append(_csv_numbers(values.reshape(-1)))
        if phase_block is not None:
            phase_rad = np.asarray(phase_block, dtype=np.float64)
            degrees = np.degrees(phase_rad)
            if phase_wrap == "0_360":
                phase_deg = np.mod(degrees, 360.0)
            else:
                phase_deg = np.mod(degrees + 180.0, 360.0) - 180.0
            columns.append(_csv_numbers(np.where(np.isfinite(phase_rad), phase_deg, np.nan).reshape(-1)))
    tails = [delimiter + delimiter.join(cells) for cells in zip(*columns)] if columns else None
    lines = []
    index = 0
    for angle in angle_heads:
        for frequency in frequency_heads:
            head = angle + delimiter + frequency + delimiter
            for polarization in polarization_heads:
                lines.append(head + polarization + (tails[index] if tails else "") + "\r\n")
                index += 1
    stream.write("".join(lines))


def _write_flat_csv_direct(
    grid, path, scale="linear", delimiter=",", include_phase=False
):
    """Stream *grid* using ``grim.flat-rcs.v1`` and return *path*.

    ``scale`` mirrors the Plotting export choices: ``linear``, ``db``,
    ``dbsm``, ``dbke``, or ``both``.  Linear output is always stored power,
    never field amplitude.  ``both`` writes power plus the only logarithmic
    column compatible with the declared physical quantity.
    """

    scale = str(scale or "linear").strip().lower()
    if scale not in ("linear", "db", "dbsm", "dbke", "both"):
        raise ValueError("CSV scale must be linear, db, dbsm, dbke, or both")
    if delimiter not in (",", "\t"):
        raise ValueError("flat CSV delimiter must be a comma or tab")

    units = dict(getattr(grid, "units", {}) or {})
    azimuth_unit = _canonical_angle_unit(units.get("azimuth", "deg"))
    elevation_unit = _canonical_angle_unit(units.get("elevation", "deg"))
    frequency_unit = _canonical_frequency_unit(units.get("frequency", "GHz"))
    quantity = (
        _canonical_quantity(grid.linear_quantity())
        if hasattr(grid, "linear_quantity")
        else _canonical_quantity(units.get("rcs_linear_quantity", "sigma_3d"))
    )
    log_unit = (
        _canonical_log_unit(grid.default_log_unit())
        if hasattr(grid, "default_log_unit")
        else _canonical_log_unit(units.get("rcs_log_unit", "dBsm"))
    )
    expected_log = _QUANTITY_LOG_UNITS[quantity]
    if log_unit != expected_log:
        raise ValueError(
            "inconsistent RCS metadata: {} requires rcs_log_unit={}, not {}"
            .format(quantity, expected_log, log_unit)
        )
    required_quantity = {"db": "power_ratio", "dbsm": "sigma_3d", "dbke": "sigma_2d"}
    if scale in required_quantity and quantity != required_quantity[scale]:
        if quantity == "sigma_3d" and scale == "dbke":
            raise ValueError(
                "sigma_3d datasets cannot be labeled dBke; convert to sigma_2d first"
            )
        if quantity == "sigma_2d" and scale == "dbsm":
            raise ValueError(
                "sigma_2d datasets cannot be labeled dBsm; convert to sigma_3d first"
            )
        if quantity == "power_ratio":
            raise ValueError(
                "dimensionless power ratios can be exported only as Linear or dB"
            )
        raise ValueError(
            "{} export requires {}, but the dataset contains {}"
            .format(scale, required_quantity[scale], quantity)
        )

    # The v1 schema keeps its angular frame columns, but GRIM angles are always
    # plain azimuth/elevation so they are written with fixed values.
    angular_coordinate_system = "conic"
    gc_convention = ""
    angular_roll_deg = angular_tilt_deg = 0.0
    polarization_basis = _declared_grid_metadata(grid, "polarization_basis")
    time_convention = _declared_grid_metadata(grid, "time_convention")
    phase_reference = _declared_grid_metadata(grid, "phase_reference")
    phase_wrap = _canonical_phase_wrap((getattr(grid, "units", {}) or {}).get(
        "phase_wrap", "-180_180"
    ))

    magnitude_columns = []
    if scale in ("linear", "both"):
        magnitude_columns.append("magnitude_power_linear")
    if scale == "db":
        magnitude_columns.append("magnitude_db")
    elif scale == "dbsm":
        magnitude_columns.append("magnitude_dbsm")
    elif scale == "dbke":
        magnitude_columns.append("magnitude_dbke")
    elif scale == "both":
        magnitude_columns.append({
            "sigma_3d": "magnitude_dbsm",
            "sigma_2d": "magnitude_dbke",
            "power_ratio": "magnitude_db",
        }[quantity])

    header = [
        "grim_csv_schema", "azimuth", "azimuth_unit", "elevation",
        "elevation_unit", "frequency", "frequency_unit", "polarization",
        "rcs_linear_quantity", "rcs_log_unit", "angular_coordinate_system",
        "great_circle_coordinate_convention", "angular_roll_deg",
        "angular_tilt_deg", "polarization_basis", "time_convention",
        "phase_reference", "phase_wrap",
    ] + magnitude_columns
    if include_phase:
        header.append("phase_deg")

    power_array = grid.power
    phase_array = grid.phase if include_phase else None
    expected_shape = (
        len(grid.azimuths), len(grid.elevations), len(grid.frequencies),
        len(grid.polarizations),
    )
    if power_array.shape != expected_shape or (phase_array is not None and phase_array.shape != expected_shape):
        raise ValueError("dataset power/phase arrays do not match its four axes")

    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(header)
        for selection in _bounded_grid_selections(expected_shape, _CSV_BLOCK_CELLS):
            if grid.storage_kind == "complex":
                # Derive the exported power/phase in float64 from the stored
                # field; complex64 derivation added ~9 dB of round-trip error.
                field_block = np.asarray(grid.rcs_slice(selection), dtype=np.complex128)
                power_block = grid._squared_magnitude(field_block)
                phase_block = (grid._phase_from_field(field_block, dtype=np.float64)
                               if include_phase else None)
            else:
                power_block = np.asarray(power_array[selection])
                phase_block = np.asarray(phase_array[selection]) if include_phase else None
            _write_flat_csv_block(
                stream, delimiter, grid, selection, power_block, phase_block,
                magnitude_columns=magnitude_columns, phase_wrap=phase_wrap,
                frequency_unit=frequency_unit,
                leading=(azimuth_unit, elevation_unit),
                labels=[quantity, log_unit, angular_coordinate_system,
                        gc_convention, _finite_csv_number(angular_roll_deg),
                        _finite_csv_number(angular_tilt_deg), polarization_basis,
                        time_convention, phase_reference, phase_wrap],
            )
    return path


def write_flat_csv(grid, path, scale="linear", delimiter=",", include_phase=False):
    """Atomically publish a versioned flat-RCS table.

    Validation, formatting, or I/O failures leave an existing destination
    untouched. The streaming implementation retains one bounded sample block
    and one formatted row beyond the dataset arrays. Phase is derived only
    when the export includes it.
    """

    requested_path = os.fspath(path)
    output_path = os.path.abspath(requested_path)
    directory = os.path.dirname(output_path) or os.curdir
    fd, stage_path = tempfile.mkstemp(
        prefix=".grim-flat-csv-", suffix=".staging", dir=directory
    )
    os.close(fd)
    try:
        _write_flat_csv_direct(
            grid,
            stage_path,
            scale=scale,
            delimiter=delimiter,
            include_phase=include_phase,
        )
        os.replace(stage_path, output_path)
    finally:
        if os.path.lexists(stage_path):
            try:
                os.unlink(stage_path)
            except OSError:
                pass
    return requested_path


def _one_value(values, label, default=None, required=False):
    if len(values) > 1:
        rendered = ", ".join(repr(value) for value in sorted(values, key=str))
        raise ValueError("one grid cannot contain multiple {} values: {}".format(label, rendered))
    value = next(iter(values)) if values else default
    if required and (value is None or str(value).strip() == ""):
        raise ValueError("{} is required and cannot be blank".format(label))
    return value


def _parse_number(text, label, line_no, strict):
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("line {}: invalid {} ({})".format(line_no, label, exc)) from exc
    # Python-float predicates: the same tests as np.isnan/np.isposinf without
    # a NumPy scalar round trip per cell.
    if value != value:
        if strict:
            raise ValueError("line {}: {} cannot be NaN; leave the cell blank".format(line_no, label))
        return None
    if value == math.inf:
        raise ValueError("line {}: {} cannot be +infinity".format(line_no, label))
    return value


def _power_candidates(record, mode, quantity, frequency_unit, c0):
    line_no = record["line_no"]
    values = record["magnitudes"]
    strict = mode == "v1"
    candidates = []
    for column, text in values.items():
        value = _parse_number(text, column, line_no, strict)
        if value is None:
            continue
        if column in ("magnitude_power_linear", "magnitude_linear"):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "line {}: {} must be finite and >= 0".format(line_no, column)
                )
            if column == "magnitude_linear" and mode == "legacy_cem_amplitude":
                power = value * value
                semantic = "legacy CEM field amplitude squared"
            else:
                power = value
                semantic = "linear power"
        else:
            if value == -math.inf:
                power = 0.0
            elif not math.isfinite(value):
                raise ValueError(
                    "line {}: {} must be finite or -inf".format(line_no, column)
                )
            else:
                try:
                    ratio = 10.0 ** (value / 10.0)
                except OverflowError:
                    raise ValueError(
                        "line {}: {} overflows finite linear power".format(line_no, column)
                    ) from None
                if column == "magnitude_dbke":
                    frequency_hz = record["frequency"] * _FREQUENCY_FACTORS[frequency_unit]
                    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
                        raise ValueError(
                            "line {}: positive frequency is required to convert dBke"
                            .format(line_no)
                        )
                    power = (c0 / (2.0 * math.pi * frequency_hz)) * ratio
                else:
                    power = ratio
            semantic = column
        if not math.isfinite(power) or power < 0.0:
            raise ValueError(
                "line {}: {} converts to an invalid linear power".format(line_no, column)
            )
        candidates.append((column, float(power), semantic))

    if not candidates:
        return float("nan")
    reference_column, reference, _semantic = candidates[0]
    for column, candidate, _semantic in candidates[1:]:
        # np.isclose(reference, candidate, rtol=5e-6, atol=0) for the finite
        # values validated above, evaluated in Python floats.
        if not abs(reference - candidate) <= 0.0 + 5.0e-6 * abs(candidate):
            raise ValueError(
                "line {}: redundant magnitude columns conflict: {}={} W-like "
                "power but {}={} W-like power".format(
                    line_no, reference_column, reference, column, candidate
                )
            )
    return reference


class _FlatCsvSource:
    """Two-pass row access to a delimited flat RCS table on disk."""

    discovery_key = "flat-csv"
    empty_message = "CSV contains no data rows"

    def __init__(self, path):
        self.path = str(path)
        with open_text_table(self.path, newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            self.delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            reader = csv.DictReader(stream, delimiter=self.delimiter)
            if not reader.fieldnames:
                raise ValueError("missing CSV header row")
            self.fieldnames = list(reader.fieldnames)
        self.fields = _field_map(self.fieldnames)

    def rows(self, stage):
        detail = "Discovering CSV coordinates" if stage == "discover" else "Reading CSV samples"
        with open_text_table(self.path, newline="") as stream:
            reader = csv.DictReader(stream, delimiter=self.delimiter)
            if stage == "samples" and _field_map(reader.fieldnames) != self.fields:
                raise ValueError("CSV header changed while the file was being loaded")
            for line_no, row in enumerate(reader, start=2):
                if line_no % 1024 == 0:
                    checkpoint(detail=detail)
                if None in row and any(str(value or "").strip() for value in row[None]):
                    raise ValueError("line {}: more values than header columns".format(line_no))
                if not any(str(value or "").strip() for value in row.values()):
                    continue
                # DictReader fills absent trailing fields with None (blank
                # cells are ""), so None marks a truncated row.
                missing = sum(1 for key, value in row.items()
                              if key is not None and value is None)
                if missing:
                    raise ValueError(
                        "line {}: {} fewer value(s) than header columns "
                        "(truncated row)".format(line_no, missing))
                yield line_no, row


class _SuppliedRowSource:
    """Two-pass access to canonical rows a caller supplies; no file round trip.

    ``rows`` is a sequence of ``(line_no, mapping)`` pairs or a zero-argument
    callable returning a fresh iterator. A one-shot iterator is materialized
    because the loader reads every row twice (coordinates, then samples).
    """

    discovery_key = "supplied-rows"

    def __init__(self, path, fieldnames, rows, *, empty_message=None):
        self.path = str(path)
        self.fieldnames = list(fieldnames)
        self.fields = _field_map(self.fieldnames)
        self.empty_message = empty_message or "table contains no data rows"
        if callable(rows):
            self._factory = rows
        else:
            replayable = rows if isinstance(rows, (list, tuple)) else list(rows)
            self._factory = lambda: replayable

    def rows(self, stage):
        detail = "Discovering table coordinates" if stage == "discover" else "Reading table samples"
        for count, (line_no, row) in enumerate(self._factory(), start=1):
            if count % 1024 == 0:
                checkpoint(detail=detail)
            yield line_no, row


def _load_flat_table(source, grid_class, c0, *, _estimate_only=False, declared_units=None):
    """Load a flat RCS table from a two-pass row source into *grid_class*."""

    fields = source.fields
    path = source.path
    declared = {str(key).strip().lower(): value for key, value in (declared_units or {}).items()
                if value is not None and str(value).strip()}

    if "grim_csv_schema" in fields:
        mode = "v1"
        missing = [name for name in _V1_REQUIRED_COLUMNS if name not in fields]
        if missing:
            raise ValueError(
                "{} is missing required column(s): {}".format(
                    FLAT_CSV_SCHEMA, ", ".join(missing)
                )
            )
        if "magnitude_linear" in fields:
            raise ValueError(
                "{} does not permit ambiguous magnitude_linear; use "
                "magnitude_power_linear".format(FLAT_CSV_SCHEMA)
            )
        axis_fields = {
            "azimuth": "azimuth",
            "elevation": "elevation",
            "frequency": "frequency",
        }
    elif all(name in fields for name in (
        "azimuth_deg", "elevation_deg", "frequency_ghz", "polarization"
    )):
        mode = "legacy_cem_amplitude"
        axis_fields = {
            "azimuth": "azimuth_deg",
            "elevation": "elevation_deg",
            "frequency": "frequency_ghz",
        }
    else:
        mode = "legacy_grim_power"
        required = ("azimuth", "elevation", "frequency", "polarization")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError("missing required column(s): {}".format(", ".join(missing)))
        axis_fields = {
            "azimuth": "azimuth",
            "elevation": "elevation",
            "frequency": "frequency",
        }

    magnitude_columns = [name for name in _MAGNITUDE_COLUMNS if name in fields]
    if not magnitude_columns:
        raise ValueError("missing magnitude column")

    def cell(row, key):
        raw = row.get(fields[key], "")
        return str(raw).strip() if raw is not None else ""

    def _discover():
        metadata_sets = {
            "schema": set(),
            "azimuth_unit": set(),
            "elevation_unit": set(),
            "frequency_unit": set(),
            "rcs_linear_quantity": set(),
            "rcs_log_unit": set(),
            "polarization_basis": set(),
            "time_convention": set(),
            "phase_reference": set(),
            "phase_wrap": set(),
        }
        record_count = 0
        azimuth_values = set()
        elevation_values = set()
        frequency_values = set()
        polarization_order = []
        # Column names resolved once; each cell reads exactly as cell() does.
        metadata_columns = [(metadata_sets[name], fields[name])
                            for name in metadata_sets if name in fields]

        for line_no, row in source.rows("discover"):
            try:
                azimuth = float(cell(row, axis_fields["azimuth"]))
                elevation = float(cell(row, axis_fields["elevation"]))
                frequency = float(cell(row, axis_fields["frequency"]))
            except ValueError as exc:
                raise ValueError("line {}: invalid axis value ({})".format(line_no, exc)) from exc
            if not (math.isfinite(azimuth) and math.isfinite(elevation)
                    and math.isfinite(frequency)):
                raise ValueError("line {}: axis values must be finite".format(line_no))
            if frequency <= 0.0:
                raise ValueError(
                    "line {}: frequency must be positive (got {!r})".format(
                        line_no, cell(row, axis_fields["frequency"])))
            polarization = cell(row, "polarization")
            if not polarization:
                raise ValueError("line {}: polarization is blank".format(line_no))

            for values, column in metadata_columns:
                raw = row.get(column, "")
                values.add(str(raw).strip() if raw is not None else "")
            if mode == "v1":
                schema_value = cell(row, "grim_csv_schema")
                metadata_sets["schema"].add(schema_value)
                if schema_value != FLAT_CSV_SCHEMA:
                    raise ValueError(
                        "line {}: unsupported grim_csv_schema {!r}; expected {!r}"
                        .format(line_no, schema_value, FLAT_CSV_SCHEMA)
                    )
                for name in (
                    "azimuth_unit", "elevation_unit", "frequency_unit",
                    "rcs_linear_quantity", "rcs_log_unit",
                ):
                    if not cell(row, name):
                        raise ValueError("line {}: {} is blank".format(line_no, name))

            azimuth_values.add(azimuth)
            elevation_values.add(elevation)
            frequency_values.add(frequency)
            if polarization not in polarization_order:
                polarization_order.append(polarization)
            record_count += 1
        return metadata_sets, record_count, azimuth_values, elevation_values, frequency_values, polarization_order

    metadata_sets, record_count, azimuth_values, elevation_values, frequency_values, polarization_order = discover_coordinates(path, source.discovery_key, _discover)

    if not record_count:
        raise ValueError(source.empty_message)

    inferred_frequency = False
    inference_rule = ""
    declared_frequency = False
    if mode == "legacy_cem_amplitude":
        azimuth_unit = "deg"
        elevation_unit = "deg"
        frequency_unit = "GHz"
        for name, expected, canonicalizer in (
            ("azimuth_unit", "deg", _canonical_angle_unit),
            ("elevation_unit", "deg", _canonical_angle_unit),
            ("frequency_unit", "GHz", _canonical_frequency_unit),
        ):
            declared_text = _one_value(metadata_sets[name], name, default="")
            if str(declared_text or "").strip() and canonicalizer(declared_text) != expected:
                raise ValueError(
                    "legacy CEM column suffix declares {} but {} says {}"
                    .format(expected, name, declared_text)
                )
    else:
        azimuth_text = _one_value(
            metadata_sets["azimuth_unit"], "azimuth units",
            default="deg", required=mode == "v1",
        )
        elevation_text = _one_value(
            metadata_sets["elevation_unit"], "elevation units",
            default="deg", required=mode == "v1",
        )
        azimuth_unit = _canonical_angle_unit(azimuth_text)
        elevation_unit = _canonical_angle_unit(elevation_text)
        frequency_text = _one_value(
            metadata_sets["frequency_unit"], "frequency units",
            default=None, required=mode == "v1",
        )
        if frequency_text is not None and str(frequency_text).strip():
            frequency_unit = _canonical_frequency_unit(frequency_text)
        elif declared.get("frequency"):
            # The table states no unit; the engineer declared one explicitly,
            # so no magnitude heuristic runs and the choice is recorded.
            frequency_unit = _canonical_frequency_unit(declared["frequency"])
            declared_frequency = True
        else:
            frequency_unit, inference_rule = _legacy_frequency_unit(frequency_values)
            inferred_frequency = True

    log_text = _one_value(
        metadata_sets["rcs_log_unit"], "RCS logarithmic units",
        default=None, required=mode == "v1",
    )
    quantity_text = _one_value(
        metadata_sets["rcs_linear_quantity"], "RCS linear quantities",
        default=None, required=mode == "v1",
    )
    if log_text is None or not str(log_text).strip():
        db_columns = [
            name for name in ("magnitude_db", "magnitude_dbsm", "magnitude_dbke")
            if name in magnitude_columns
        ]
        if len(db_columns) > 1:
            raise ValueError(
                "legacy CSV mixes incompatible logarithmic magnitude columns: {}"
                .format(", ".join(db_columns))
            )
        log_unit = {
            "magnitude_db": "dB",
            "magnitude_dbsm": "dBsm",
            "magnitude_dbke": "dBke",
        }.get(db_columns[0] if db_columns else "", "dBsm")
    else:
        log_unit = _canonical_log_unit(log_text)
    quantity = (
        _canonical_quantity(quantity_text)
        if quantity_text is not None and str(quantity_text).strip()
        else {"dB": "power_ratio", "dBke": "sigma_2d", "dBsm": "sigma_3d"}[log_unit]
    )
    expected_log = _QUANTITY_LOG_UNITS[quantity]
    if log_unit != expected_log:
        raise ValueError(
            "inconsistent physical magnitude metadata: {} requires "
            "rcs_log_unit={}, not {}".format(quantity, expected_log, log_unit)
        )
    required_quantity = {
        "magnitude_db": "power_ratio",
        "magnitude_dbsm": "sigma_3d",
        "magnitude_dbke": "sigma_2d",
    }
    for column, column_quantity in required_quantity.items():
        if column in magnitude_columns and quantity != column_quantity:
            raise ValueError(
                "{} is incompatible with rcs_linear_quantity={}; use the "
                "matching logarithmic column or magnitude_power_linear"
                .format(column, quantity)
            )

    # Angular frame columns (coordinate system, great-circle convention,
    # roll/tilt) are ignored: every dataset loads as azimuth/elevation.
    polarization_basis = _one_value(
        metadata_sets["polarization_basis"], "polarization bases", default=""
    )
    time_convention = _one_value(
        metadata_sets["time_convention"], "time conventions", default=""
    )
    phase_reference = _one_value(
        metadata_sets["phase_reference"], "phase references", default=""
    )
    phase_wrap = _canonical_phase_wrap(_one_value(
        metadata_sets["phase_wrap"], "phase wrap declarations", default="-180_180"
    ))

    axis_lengths = (
        len(azimuth_values),
        len(elevation_values),
        len(frequency_values),
        len(polarization_order),
    )
    cells, dense_bytes, peak_bytes = _preflight_dense_grid(
        axis_lengths, record_count, check_budget=not _estimate_only)
    if _estimate_only:
        return {'shape': axis_lengths, 'retained_bytes': resident_result_bytes(dense_bytes, aggregate=True),
                'estimated_peak_bytes': peak_bytes, 'source_format': 'flat CSV'}

    azimuth_axis = np.asarray(sorted(azimuth_values), dtype=float)
    elevation_axis = np.asarray(sorted(elevation_values), dtype=float)
    frequency_axis = np.asarray(sorted(frequency_values), dtype=float)
    polarization_axis = np.asarray(polarization_order)
    shape = tuple(axis_lengths)
    # The sorted arrays now own the coordinate values. Release discovery sets
    # before building index maps and allocating sample buffers.
    del azimuth_values, elevation_values, frequency_values
    power = allocate_result(shape, np.float64, fill=np.nan)
    phase = allocate_result(shape, np.float64, fill=np.nan)
    axis_maps = (
        dict((value, index) for index, value in enumerate(azimuth_axis)),
        dict((value, index) for index, value in enumerate(elevation_axis)),
        dict((value, index) for index, value in enumerate(frequency_axis)),
        dict((value, index) for index, value in enumerate(polarization_axis)),
    )


    first_line = allocate_result(shape, np.int64, fill=0)
    for line_no, row in source.rows("samples"):
        try:
            record = {
                "azimuth": float(cell(row, axis_fields["azimuth"])),
                "elevation": float(cell(row, axis_fields["elevation"])),
                "frequency": float(cell(row, axis_fields["frequency"])),
                "polarization": cell(row, "polarization"),
                "magnitudes": dict(
                    (name, cell(row, name)) for name in magnitude_columns
                ),
                "phase_deg": (
                    cell(row, "phase_deg") if "phase_deg" in fields else ""
                ),
                "line_no": line_no,
            }
        except ValueError as exc:
            raise ValueError(
                "line {}: invalid axis value ({})".format(line_no, exc)
            ) from exc
        if not (math.isfinite(record["azimuth"]) and math.isfinite(record["elevation"])
                and math.isfinite(record["frequency"])):
            raise ValueError("line {}: axis values must be finite".format(line_no))
        if not record["polarization"]:
            raise ValueError("line {}: polarization is blank".format(line_no))

        sample_power = _power_candidates(
            record, mode, quantity, frequency_unit, c0
        )
        phase_value = float("nan")
        if record["phase_deg"]:
            parsed_phase = _parse_number(
                record["phase_deg"], "phase_deg", record["line_no"], mode == "v1"
            )
            if parsed_phase is not None:
                if not math.isfinite(parsed_phase):
                    raise ValueError(
                        "line {}: phase_deg must be finite".format(
                            record["line_no"]
                        )
                    )
                # x * (pi / 180), the same product np.deg2rad evaluates.
                phase_value = parsed_phase * (math.pi / 180.0)
        try:
            index = (
                axis_maps[0][record["azimuth"]],
                axis_maps[1][record["elevation"]],
                axis_maps[2][record["frequency"]],
                axis_maps[3][record["polarization"]],
            )
        except KeyError as exc:
            raise ValueError(
                "CSV axes changed while the file was being loaded"
            ) from exc
        previous_line = int(first_line[index])
        if previous_line:
            previous_power = float(power[index])
            previous_phase = float(phase[index])
            same_power = bool(np.isclose(
                sample_power, previous_power,
                rtol=1.0e-12, atol=0.0, equal_nan=True,
            ))
            zero_power = same_power and sample_power == 0.0
            same_phase = zero_power or (
                (np.isnan(phase_value) and np.isnan(previous_phase))
                or (
                    np.isfinite(phase_value)
                    and np.isfinite(previous_phase)
                    and abs(np.angle(np.exp(
                        1j * (phase_value - previous_phase)
                    ))) <= 1.0e-12
                )
            )
            if not (same_power and same_phase):
                raise ValueError(
                    "line {}: conflicting duplicate CSV sample; first "
                    "defined on line {}".format(line_no, previous_line)
                )
            continue
        first_line[index] = line_no
        power[index] = sample_power
        phase[index] = phase_value

    if not np.isfinite(power).any():
        raise ValueError("CSV contains no finite magnitude values")


    del first_line

    units = {
        "azimuth": azimuth_unit,
        "elevation": elevation_unit,
        "frequency": frequency_unit,
        "rcs_log_unit": log_unit,
        "rcs_linear_quantity": quantity,
        "angular_coordinate_system": "conic",
    }
    if str(polarization_basis or "").strip():
        units["polarization_basis"] = str(polarization_basis).strip()
    if str(time_convention or "").strip():
        units["time_convention"] = str(time_convention).strip()
    units["phase_wrap"] = phase_wrap

    extra = {
        "flat_csv_schema": FLAT_CSV_SCHEMA if mode == "v1" else mode,
        "source_format": "GRIM flat RCS CSV" if mode == "v1" else mode,
    }
    if str(phase_reference or "").strip():
        extra["phase_reference"] = str(phase_reference).strip()
    if inferred_frequency:
        extra["frequency_unit_inferred"] = True
        extra["frequency_unit_inference"] = inference_rule
    if declared_frequency:
        extra["frequency_unit_declared"] = True

    history = "Loaded flat CSV ({}): {}".format(
        FLAT_CSV_SCHEMA if mode == "v1" else mode.replace("_", " "), path
    )
    if inferred_frequency:
        history += "\nWARNING: " + inference_rule
        # Headless and batch callers never see the history or the GUI
        # confirmation, so surface the guess as a Python warning too.
        warnings.warn(
            "{}: {}".format(os.path.basename(str(path)), inference_rule),
            UserWarning,
            stacklevel=2,
        )
    if declared_frequency:
        history += "\nFrequency unit {} supplied by the user; the table declares none".format(frequency_unit)
    constructor = (construct_parsed_grid if hasattr(grid_class, "_complex_from_power_phase")
                   else lambda cls, *args, **kwargs: cls(*args, **kwargs))
    return constructor(
        grid_class,
        azimuth_axis,
        elevation_axis,
        frequency_axis,
        polarization_axis,
        rcs_power=power,
        rcs_phase=phase,
        source_path=str(path),
        history=history,
        units=units,
        extra=extra,
    )


@text_reader
def load_flat_csv(path, grid_class, canonical_angular_coordinate_system, c0=299792458.0, *,
                  _estimate_only=False, declared_units=None):
    """Load supported flat RCS tables into *grid_class*.

    ``canonical_angular_coordinate_system`` is accepted for caller
    compatibility; angular frame columns are no longer interpreted.
    ``declared_units`` may supply a frequency unit for legacy tables that
    state none; it never overrides a unit the table declares.
    """

    return _load_flat_table(_FlatCsvSource(path), grid_class, c0,
                            _estimate_only=_estimate_only, declared_units=declared_units)


def load_flat_rows(path, fieldnames, rows, grid_class, c0=299792458.0, *, empty_message=None):
    """Load caller-supplied canonical flat-table rows without writing a file.

    ``path`` is recorded as the provenance source. ``rows`` yields
    ``(line_no, mapping)`` pairs keyed by the canonical column names in
    ``fieldnames``; pass a zero-argument callable so the two loader passes can
    each read a fresh iterator instead of a buffered copy.
    """

    source = _SuppliedRowSource(path, fieldnames, rows, empty_message=empty_message)
    return _load_flat_table(source, grid_class, c0)
