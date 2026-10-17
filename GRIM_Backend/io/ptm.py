"""PTM binary parsing, writing, and RCS grid conversion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
import re
import struct
import sys
import tempfile

import numpy as np

from GRIM_Backend.datasets.constants import _PTM_GRIM_GC_MARKER


_PTM_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _ptm_configuration_has_grim_gc_marker(value):
    return bool(
        re.search(
            r"(?<![A-Z0-9_])GRIM_GC_V1(?![A-Z0-9_])",
            str(value or "").upper(),
        )
    )


def _ptm_configuration_with_grim_gc_marker(value):
    """Add the GRIM coordinate marker to the configuration text."""

    text = str(value or "").strip()
    if _ptm_configuration_has_grim_gc_marker(text):
        return text
    if not text:
        return _PTM_GRIM_GC_MARKER
    available = 50 - len(_PTM_GRIM_GC_MARKER) - 1
    return f"{_PTM_GRIM_GC_MARKER};{text[:available]}"


def _ptm_configuration_without_grim_gc_marker(value):
    """Remove only GRIM's semicolon-delimited convention marker."""

    parts = [part.strip() for part in str(value or "").split(";")]
    return ";".join(
        part for part in parts
        if part and part.upper() != _PTM_GRIM_GC_MARKER
    )


PTM_HEADER_SIZE = 291


PTM_WRITABLE_POLARIZATIONS = frozenset({"VV", "HH", "VH", "HV"})


_INT_FIELDS = (
    "num_aspects",
    "corecell",
    "calcell",
    "range_corr",
    "gatemincell",
    "gatemaxcell",
    "cal_type",
    "crosscell",
    "phase_flag",
)


_FLOAT_FIELDS = (
    "start_aspect",
    "aspect_increment",
    "center_frequency",
    "bandwidth",
    "roll",
    "pitch",
    "tilt",
    "cal_dbsm",
    "attenuation",
)


@dataclass(frozen=True)
class PtmHeader:
    """Decoded PTM header plus the frequency count implied by file framing."""

    num_aspects: int = 0
    corecell: int = 0
    calcell: int = 0
    range_corr: int = 0
    gatemincell: int = 0
    gatemaxcell: int = 0
    cal_type: int = 0
    crosscell: int = 0
    phase_flag: int = 0

    start_aspect: float = 0.0
    aspect_increment: float = 0.0
    center_frequency: float = 0.0
    bandwidth: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    tilt: float = 0.0
    cal_dbsm: float = 0.0
    attenuation: float = 0.0

    polarity: str = "NA"
    subject: str = ""
    configuration: str = ""
    operator: str = ""
    filename: str = ""
    date: str = ""
    time: str = ""

    num_frequencies: int = 0
    embedded_num_frequencies: bool = True
    byte_order: str = "little"


@dataclass(frozen=True)
class PtmData:
    """A parsed PTM cut in acquisition order."""

    header: PtmHeader
    aspects_deg: np.ndarray
    frequencies_ghz: np.ndarray
    iq: np.ndarray


def _decode_ascii(raw: bytes, field_name: str) -> str:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"PTM {field_name} is not ASCII") from exc
    return text.rstrip("\x00 *")


def _encode_ascii(value: object, width: int, field_name: str) -> bytes:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"PTM {field_name} must contain ASCII characters only") from exc
    if len(raw) > width:
        raw = raw[:width]
    return raw.ljust(width, b"*")


def _encode_filename(value: object) -> bytes:
    """Truncate a filename to its first 45 and last 5 characters."""

    text = str(value or "").replace("\r", " ").replace("\n", " ")
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("PTM filename must contain ASCII characters only") from exc
    if len(raw) > 50:
        raw = raw[:45] + raw[-5:]
    return raw.ljust(50, b"*")


def _validate_polarity(value: object) -> str:
    polarity = str(value or "").strip().upper()
    if len(polarity) != 2 or not polarity.isascii() or not polarity.isalnum():
        raise ValueError(
            f"PTM polarity must be exactly two ASCII letters/digits, got {value!r}"
        )
    return polarity


def _framing_for_size(file_size: int, num_aspects: int) -> tuple[int, int]:
    """Derive block framing from the actual file size.

    Every count is implied by the byte length (``num_aspects + 1`` blocks of
    8 bytes per frequency), so the counts are bounded by the file itself and
    no separate sample cap is needed. Memory admission for the decoded samples
    belongs to the RAM/disk policy applied by ``read_ptm``.
    """
    if int(num_aspects) < 1:
        raise ValueError(f"num_aspects={num_aspects} must be at least 1")
    block_size, remainder = divmod(int(file_size), int(num_aspects) + 1)
    if remainder:
        raise ValueError(
            f"file size {file_size} is not divisible by num_aspects+1 "
            f"({int(num_aspects) + 1})"
        )
    if block_size < PTM_HEADER_SIZE:
        raise ValueError(
            f"block size {block_size} is smaller than the {PTM_HEADER_SIZE}-byte header"
        )
    if block_size % 8:
        raise ValueError(f"block size {block_size} is not a whole complex-float32 block")
    num_frequencies = block_size // 8
    if num_frequencies < 1:
        raise ValueError(f"num_frequencies={num_frequencies} must be at least 1")
    return block_size, num_frequencies


def _parse_header_candidate(
    raw_header: bytes,
    file_size: int,
    byte_order: str,
) -> tuple[PtmHeader, int]:
    prefix = "<" if byte_order == "little" else ">"
    numeric = struct.unpack_from(f"{prefix}9i9f", raw_header, 0)
    int_values = numeric[:9]
    float_values = numeric[9:]
    num_aspects = int(int_values[0])
    block_size, num_frequencies = _framing_for_size(file_size, num_aspects)

    if not np.all(np.isfinite(np.asarray(float_values, dtype=float))):
        raise ValueError("numeric header contains a non-finite float")
    center_frequency = float(float_values[2])
    bandwidth = float(float_values[3])
    if center_frequency <= 0.0:
        raise ValueError(f"center_frequency={center_frequency!r} must be positive")
    if bandwidth < 0.0:
        raise ValueError(f"bandwidth={bandwidth!r} must be non-negative")
    if num_frequencies > 1 and bandwidth == 0.0:
        raise ValueError(
            "bandwidth must be positive when PTM contains more than one "
            "frequency sample"
        )
    lowest_frequency = center_frequency - bandwidth / 2.0
    if lowest_frequency <= 0.0:
        raise ValueError(
            "center_frequency - bandwidth/2 must be positive; reconstructed "
            f"lowest frequency is {lowest_frequency!r} GHz"
        )

    polarity = _validate_polarity(_decode_ascii(raw_header[72:74], "polarity"))
    subject = _decode_ascii(raw_header[74:124], "subject")
    configuration = _decode_ascii(raw_header[124:174], "configuration")

    operator_raw = raw_header[174:224]
    embedded_count = struct.unpack(f"{prefix}i", operator_raw[46:50])[0]
    has_embedded_count = embedded_count == num_frequencies
    operator_width = 46 if has_embedded_count else 50
    operator = _decode_ascii(operator_raw[:operator_width], "operator")

    values = dict(zip(_INT_FIELDS, (int(v) for v in int_values)))
    values.update(zip(_FLOAT_FIELDS, (float(v) for v in float_values)))
    header = PtmHeader(
        **values,
        polarity=polarity,
        subject=subject,
        configuration=configuration,
        operator=operator,
        filename=_decode_ascii(raw_header[224:274], "filename"),
        date=_decode_ascii(raw_header[274:283], "date"),
        time=_decode_ascii(raw_header[283:291], "time"),
        num_frequencies=num_frequencies,
        embedded_num_frequencies=has_embedded_count,
        byte_order=byte_order,
    )
    return header, block_size


def _ptm_layout(path):
    """Read one PTM cut, accepting little- or big-endian numeric storage."""

    path = os.fspath(path)
    file_size = os.path.getsize(path)
    if file_size < PTM_HEADER_SIZE:
        raise ValueError(
            f"{path}: too small to be PTM ({file_size} < {PTM_HEADER_SIZE} bytes)"
        )
    with open(path, "rb") as stream:
        raw_header = stream.read(PTM_HEADER_SIZE)
    if len(raw_header) != PTM_HEADER_SIZE:
        raise ValueError(f"{path}: truncated PTM header")

    candidates: list[tuple[PtmHeader, int]] = []
    errors: list[str] = []
    for byte_order in ("little", "big"):
        try:
            candidates.append(_parse_header_candidate(raw_header, file_size, byte_order))
        except (ValueError, struct.error) as exc:
            errors.append(f"{byte_order}: {exc}")
    if not candidates:
        raise ValueError(f"{path}: invalid PTM framing/header ({'; '.join(errors)})")


    header, block_size = candidates[0]
    raw_aspects = (
        float(header.start_aspect)
        + np.arange(header.num_aspects, dtype=np.float64)
        * float(header.aspect_increment)
    )


    aspects = np.mod(raw_aspects + 180.0, 360.0) - 180.0
    aspect_order = np.argsort(aspects, kind="stable")
    aspects = aspects[aspect_order]
    duplicate_tol = max(
        1.0e-7,
        8.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(aspects)))),
    )
    if aspects.size > 1 and np.any(np.diff(aspects) <= duplicate_tol):
        raise ValueError(
            f"{path}: PTM aspect axis contains duplicate/seam-alias coordinates "
            "after wrapping to [-180, 180)"
        )
    start_frequency = header.center_frequency - header.bandwidth / 2.0
    stop_frequency = header.center_frequency + header.bandwidth / 2.0
    # The frequency axis is materialized by the caller once the memory policy
    # has admitted the file; a single-aspect file can imply a huge axis.
    return header, block_size, aspects, (start_frequency, stop_frequency), aspect_order


def read_ptm(path: os.PathLike[str] | str, *, storage="memory", cache_dir=None,
             max_output_bytes=None, _estimate_only=False) -> PtmData:
    """Decode PTM in bounded blocks into an independent RAM or disk destination."""
    from GRIM_Backend.datasets.result_storage import allocate_result
    from GRIM_Backend.datasets.work_control import checkpoint
    from GRIM_Backend.io.binary_storage import (
        BLOCK_CELLS, binary_import_allocation, binary_import_storage,
    )
    header, block_size, aspects, frequency_span, order = _ptm_layout(path)
    shape = (header.num_aspects, header.num_frequencies)
    frequency_bytes = 8 * int(header.num_frequencies)
    axis_bytes = (aspects.nbytes + frequency_bytes + order.nbytes)
    report = binary_import_allocation(
        shape, np.complex64, storage=storage, source=f"PTM import {path}",
        scratch_bytes=min(BLOCK_CELLS, shape[1]) * 12 + axis_bytes,
        retained_metadata_bytes=aspects.nbytes + frequency_bytes,
        max_output_bytes=max_output_bytes)
    if _estimate_only:
        return report
    frequencies = np.linspace(
        frequency_span[0],
        frequency_span[1],
        header.num_frequencies,
        dtype=np.float64,
    )
    prefix = "<" if header.byte_order == "little" else ">"
    source = np.memmap(path, mode="r", dtype=np.dtype(f"{prefix}c8"),
                       offset=block_size, shape=shape)
    try:
        with binary_import_storage(storage, cache_dir=cache_dir,
                                   max_output_bytes=max_output_bytes):
            iq = allocate_result(shape, np.complex64)
            for output_row, source_row in enumerate(order):
                for first in range(0, shape[1], BLOCK_CELLS):
                    checkpoint(output_row * shape[1] + first,
                               shape[0] * shape[1], "Reading PTM samples")
                    last = min(shape[1], first + BLOCK_CELLS)
                    part = source[source_row, first:last]
                    if not np.all(np.isfinite(part.real) & np.isfinite(part.imag)):
                        raise ValueError(
                            f"{path}: PTM IQ contains a non-finite sample; the format has no "
                            "documented missing-sample marker")
                    iq[output_row, first:last] = part
                    del part
    finally:
        source._mmap.close()
    return PtmData(header, aspects, frequencies, iq)


def _uniform_axis(values: np.ndarray, name: str, *, periodic: bool = False) -> tuple[float, float]:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"PTM {name} axis must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"PTM {name} axis contains a non-finite value")
    working = np.rad2deg(np.unwrap(np.deg2rad(axis))) if periodic else axis
    if working.size == 1:
        return float(working[0]), 0.0
    step = float((working[-1] - working[0]) / (working.size - 1))
    expected = float(working[0]) + step * np.arange(working.size, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(working))))
    atol = max(1.0e-7, 8.0 * np.finfo(np.float32).eps * scale)
    if not np.allclose(working, expected, rtol=1.0e-7, atol=atol):
        max_error = float(np.max(np.abs(working - expected)))
        raise ValueError(
            f"PTM {name} axis must be uniformly spaced; maximum residual is {max_error:g}"
        )
    return float(working[0]), step


def _periodic_aspect_order(
    values: np.ndarray,
    preferred_header: PtmHeader | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return a uniform acquisition ordering for one set of aspect samples."""

    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("PTM aspect axis must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)):
        raise ValueError("PTM aspect axis contains a non-finite value")
    canonical = np.mod(axis + 180.0, 360.0) - 180.0
    sorted_values = np.sort(canonical, kind="stable")
    tol = max(
        1.0e-7,
        8.0 * np.finfo(np.float32).eps
        * max(1.0, float(np.max(np.abs(canonical)))),
    )
    if canonical.size > 1 and np.any(np.diff(sorted_values) <= tol):
        raise ValueError(
            "PTM aspect axis contains duplicate coordinates or a -180/180 seam alias"
        )

    def _sequence_result(order):
        ordered = canonical[np.asarray(order, dtype=int)]
        try:
            start, step = _uniform_axis(ordered, "aspect", periodic=True)
        except ValueError:
            return None
        if ordered.size > 1 and abs(step) <= tol:
            return None
        return np.asarray(order, dtype=int), ordered, start, step


    if (
        preferred_header is not None
        and int(preferred_header.num_aspects) == canonical.size
        and canonical.size > 0
    ):
        preferred_raw = (
            float(preferred_header.start_aspect)
            + np.arange(canonical.size, dtype=np.float64)
            * float(preferred_header.aspect_increment)
        )
        preferred = np.mod(preferred_raw + 180.0, 360.0) - 180.0
        order = []
        used = set()
        for target in preferred:
            matches = np.where(np.isclose(canonical, target, rtol=0.0, atol=tol))[0]
            matches = [int(index) for index in matches if int(index) not in used]
            if len(matches) != 1:
                order = []
                break
            used.add(matches[0])
            order.append(matches[0])
        if len(order) == canonical.size:
            result = _sequence_result(order)
            if result is not None:
                return (
                    result[0],
                    result[1],
                    float(preferred_header.start_aspect),
                    float(preferred_header.aspect_increment),
                )

    result = _sequence_result(np.arange(canonical.size, dtype=int))
    if result is not None:
        return result


    sorted_order = np.argsort(canonical, kind="stable")
    sorted_axis = canonical[sorted_order]
    gaps = np.diff(np.concatenate((sorted_axis, [sorted_axis[0] + 360.0])))
    rotation = (int(np.argmax(gaps)) + 1) % canonical.size
    rotated_order = np.concatenate((sorted_order[rotation:], sorted_order[:rotation]))
    result = _sequence_result(rotated_order)
    if result is None:
        raise ValueError("PTM aspect axis must be uniformly spaced around its scan arc")
    return result


def _header_blob(header: PtmHeader, num_frequencies: int) -> bytes:
    polarity = _validate_polarity(header.polarity)
    ints = tuple(int(getattr(header, name)) for name in _INT_FIELDS)
    floats = tuple(float(getattr(header, name)) for name in _FLOAT_FIELDS)
    if not np.all(np.isfinite(np.asarray(floats, dtype=float))):
        raise ValueError("PTM numeric header contains a non-finite float")

    blob = bytearray(struct.pack("<9i9f", *ints, *floats))
    blob.extend(_encode_ascii(polarity, 2, "polarity"))
    blob.extend(_encode_ascii(header.subject, 50, "subject"))
    blob.extend(_encode_ascii(header.configuration, 50, "configuration"))
    if header.embedded_num_frequencies:
        blob.extend(_encode_ascii(header.operator, 46, "operator"))
        blob.extend(struct.pack("<i", int(num_frequencies)))
    else:
        blob.extend(_encode_ascii(header.operator, 50, "operator"))
    blob.extend(_encode_filename(header.filename))
    blob.extend(_encode_ascii(header.date, 9, "date"))
    blob.extend(_encode_ascii(header.time, 8, "time"))
    if len(blob) != PTM_HEADER_SIZE:
        raise RuntimeError(f"PTM header width drift ({len(blob)} != {PTM_HEADER_SIZE})")
    return bytes(blob)


def write_ptm(
    path: os.PathLike[str] | str,
    aspects_deg,
    frequencies_ghz,
    iq,
    *,
    polarity: str,
    pitch_deg: float = 0.0,
    header: PtmHeader | None = None,
    _block_reader=None,
) -> str:
    """Write one uniformly sampled, complex, great-circle PTM cut.

    New files are always little-endian.  ``header`` supplies optional
    acquisition metadata; axes, dimensions, polarization, pitch, and output
    filename are authoritative arguments and replace those header fields.
    """

    path = os.fspath(path)
    if not path.lower().endswith(".ptm"):
        path = f"{path}.ptm"

    aspects = np.asarray(aspects_deg, dtype=np.float64)
    frequencies = np.asarray(frequencies_ghz, dtype=np.float64)
    base = header or PtmHeader()
    aspect_order, _, start_aspect, aspect_increment = (
        _periodic_aspect_order(aspects, base)
    )
    start_frequency, frequency_increment = _uniform_axis(
        frequencies, "frequency", periodic=False
    )
    if frequencies.size < 37:
        raise ValueError(
            "PTM requires at least 37 frequencies because its 291-byte header "
            "must fit inside one 8-byte-per-frequency block"
        )
    if start_frequency <= 0.0 or np.any(frequencies <= 0.0):
        raise ValueError("PTM frequencies must be positive GHz values")
    if frequencies.size > 1 and frequency_increment <= 0.0:
        raise ValueError("PTM frequency axis must be strictly increasing")

    expected_shape = (int(aspects.size), int(frequencies.size))
    sample_count = expected_shape[0] * expected_shape[1]
    # Validate addressable dimensions only; the writer streams bounded blocks,
    # so output size is limited by the destination disk rather than a cap.
    from GRIM_Backend.datasets.memory import _checked_dense_import_allocation
    _checked_dense_import_allocation(
        expected_shape, (np.complex64,), source=f"PTM export {path}",
        max_output_bytes=sys.maxsize)
    if _block_reader is None:
        complex_data = np.asarray(iq)
        if complex_data.shape != expected_shape:
            raise ValueError(f"PTM IQ shape {complex_data.shape} != {expected_shape}")
        if not np.iscomplexobj(complex_data):
            raise ValueError("PTM IQ data must be complex")
        _block_reader = lambda row, first, last: complex_data[row, first:last]

    now = datetime.now()
    center_frequency = float((frequencies[0] + frequencies[-1]) / 2.0)
    bandwidth = float(frequencies[-1] - frequencies[0])
    output_polarity = _validate_polarity(polarity)
    if output_polarity not in PTM_WRITABLE_POLARIZATIONS:
        supported = ", ".join(sorted(PTM_WRITABLE_POLARIZATIONS))
        raise ValueError(
            "PTM writer supports only the documented polarizations "
            f"{supported}; got {output_polarity!r}"
        )
    final_header = replace(
        base,
        num_aspects=int(aspects.size),
        start_aspect=start_aspect,
        aspect_increment=aspect_increment,
        center_frequency=center_frequency,
        bandwidth=bandwidth,
        pitch=float(pitch_deg),
        polarity=output_polarity,
        filename=os.path.basename(path),
        date=base.date or now.strftime("%d%b%Y").upper(),
        time=base.time or now.strftime("%H:%M:%S"),
        num_frequencies=int(frequencies.size),
        byte_order="little",
    )
    header_bytes = _header_blob(final_header, int(frequencies.size))
    block_size = 8 * int(frequencies.size)
    padding_size = block_size - PTM_HEADER_SIZE
    if padding_size < 0:
        raise RuntimeError("PTM internal framing error: negative header padding")

    expected_size = (int(aspects.size) + 1) * block_size
    destination_dir = os.path.dirname(os.path.abspath(path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = stream.name
            stream.write(header_bytes)
            from GRIM_Backend.io.binary_storage import BLOCK_CELLS
            from GRIM_Backend.datasets.work_control import checkpoint
            for first in range(0, padding_size, BLOCK_CELLS * 8):
                checkpoint(detail="Writing PTM header")
                stream.write(b"\x00" * min(padding_size - first, BLOCK_CELLS * 8))
            for output_row, source_row in enumerate(aspect_order):
                for first in range(0, expected_shape[1], BLOCK_CELLS):
                    checkpoint(output_row * expected_shape[1] + first,
                               sample_count, "Writing PTM samples")
                    last = min(expected_shape[1], first + BLOCK_CELLS)
                    source = np.asarray(_block_reader(int(source_row), first, last))
                    if source.shape != (last - first,):
                        raise ValueError("PTM IQ block shape does not match the frequency axis")
                    if not np.all(np.isfinite(source.real) & np.isfinite(source.imag)):
                        raise ValueError(
                            "PTM IQ data contains a non-finite sample; the format has no "
                            "documented missing-sample marker")
                    # Check before the float32 cast, which would silently
                    # turn out-of-range I/Q values into infinities.
                    overflow = (
                        (np.abs(source.real) > _PTM_FLOAT32_MAX)
                        | (np.abs(source.imag) > _PTM_FLOAT32_MAX)
                    )
                    if np.any(overflow):
                        raise ValueError(
                            "PTM stores I/Q as float32; "
                            f"{int(np.count_nonzero(overflow))} sample(s) at aspect "
                            f"row {int(source_row)} exceed the float32 maximum "
                            f"{_PTM_FLOAT32_MAX:.6g} in |I| or |Q|")
                    part = np.asarray(source, dtype=np.dtype("<c8"))
                    stream.write(part.tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        actual_size = os.path.getsize(temp_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"PTM write size {actual_size} != expected {expected_size}"
            )
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
    return path


def _ptm_conjugate_for_time_convention(grid) -> bool:
    """Return True when ``grid`` must be conjugated to become exp(+jwt).

    PTM has no time-convention field and its I/Q samples are read back as
    exp(+jwt) phasors.  An explicit exp(-jwt) declaration is honored by
    conjugating on write (with a warning); an unrecognized declaration is
    refused rather than guessed.
    """

    import warnings

    from GRIM_Backend.datasets.metadata import canonical_time_convention

    evidence = grid.inspect_scalar_metadata("time_convention")
    if evidence.malformed_sources or evidence.conflicting:
        raise ValueError(
            "save_ptm: dataset time_convention metadata is "
            f"{evidence.status}; PTM requires one exp(+jwt) or exp(-jwt) declaration")
    if not evidence.declarations:
        return False
    declared = evidence.declarations[0]
    sign = canonical_time_convention(declared)
    if sign == "+jwt":
        return False
    if sign != "-jwt":
        raise ValueError(
            f"save_ptm: unrecognized time_convention {declared!r}; PTM stores "
            "exp(+jwt) I/Q and cannot record another convention")
    warnings.warn(
        f"save_ptm: dataset declares time_convention {declared!r}; PTM has no "
        "time-convention field, so samples are conjugated to exp(+jwt) on write",
        UserWarning,
        stacklevel=3,
    )
    return True


def header_to_extra(header: PtmHeader) -> dict[str, object]:
    """Map PTM header metadata to pickle-free grid extras."""

    out: dict[str, object] = {
        "ptm_original_byte_order": header.byte_order,
        "ptm_original_num_aspects": int(header.num_aspects),
        "ptm_original_start_aspect": float(header.start_aspect),
        "ptm_original_aspect_increment": float(header.aspect_increment),
        "ptm_embedded_num_frequencies": bool(header.embedded_num_frequencies),
        "ptm_source_filename": header.filename,
    }
    for name in _INT_FIELDS[1:]:
        out[f"ptm_{name}"] = int(getattr(header, name))
    for name in ("roll", "tilt", "cal_dbsm", "attenuation"):
        out[f"ptm_{name}"] = float(getattr(header, name))
    for name in ("subject", "configuration", "operator", "date", "time"):
        out[f"ptm_{name}"] = str(getattr(header, name))
    return out


def _extra_scalar(extra: dict[str, object], key: str, default):
    value = extra.get(key, default)
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return default


def header_from_extra(extra: dict[str, object] | None) -> PtmHeader:
    """Build PTM header metadata from preserved grid extras."""

    values = dict(extra or {})
    kwargs: dict[str, object] = {}
    kwargs["num_aspects"] = int(
        _extra_scalar(values, "ptm_original_num_aspects", 0)
    )
    kwargs["start_aspect"] = float(
        _extra_scalar(values, "ptm_original_start_aspect", 0.0)
    )
    kwargs["aspect_increment"] = float(
        _extra_scalar(values, "ptm_original_aspect_increment", 0.0)
    )
    for name in _INT_FIELDS[1:]:
        kwargs[name] = int(_extra_scalar(values, f"ptm_{name}", 0))
    for name in ("roll", "tilt", "cal_dbsm", "attenuation"):
        kwargs[name] = float(_extra_scalar(values, f"ptm_{name}", 0.0))
    for name in ("subject", "configuration", "operator", "date", "time"):
        kwargs[name] = str(_extra_scalar(values, f"ptm_{name}", ""))
    kwargs["embedded_num_frequencies"] = bool(
        _extra_scalar(values, "ptm_embedded_num_frequencies", True)
    )
    return PtmHeader(**kwargs)


class PtmFormatMixin:
    """PTM binary parsing, writing, and RCS grid conversion."""

    @classmethod
    def load_ptm(cls, path, *, storage="memory", cache_dir=None, max_output_bytes=None):
        """Load one PTM RCS cut.

        The file supplies one polarization and elevation, uniformly spaced
        azimuth and GHz frequency axes, and complex float32 IQ samples. The
        output contains 3-D RCS power/phase arrays on plain azimuth/elevation
        axes; header roll/tilt are kept only as PTM metadata.
        """
        from GRIM_Backend.datasets.constants import _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.binary_storage import loaded_binary_grid
        parsed = read_ptm(path, storage=storage, cache_dir=cache_dir,
                          max_output_bytes=max_output_bytes)
        header = parsed.header
        header_extra = header_to_extra(header)
        complex_grid = parsed.iq[:, np.newaxis, :, np.newaxis]
        history = (
            f"Loaded PTM cut ({header.num_aspects} aspects, "
            f"{header.num_frequencies} freqs, {header.polarity}, "
            f"{header.byte_order}-endian): {path}"
        )
        return loaded_binary_grid(cls(
            parsed.aspects_deg,
            np.asarray([header.pitch], dtype=np.float32),
            parsed.frequencies_ghz,
            np.asarray([header.polarity]),
            rcs=complex_grid,
            rcs_domain="complex_amplitude",
            source_path=str(path),
            history=history,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
            },
            extra=header_extra,
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        ), storage)

    def save_ptm(self, path, *, el_idx=None, pol_idx=None):
        """Save one (elevation, polarization) slice as little-endian PTM.

        PTM is a complex 3-D RCS format.  It cannot represent 2-D scattering
        width, missing phase, nonuniform axes, multiple elevations, or multiple
        polarizations in one file.  Callers must select one slice when the grid
        contains more than one elevation or polarization.
        """
        from GRIM_Backend.io.ptm import _ptm_configuration_without_grim_gc_marker

        if self.linear_quantity() != "sigma_3d":
            raise ValueError(
                "save_ptm: PTM stores 3-D RCS (sigma_3d/dBsm); "
                f"dataset quantity is {self.linear_quantity()!r}"
            )
        if el_idx is None:
            if len(self.elevations) == 1:
                el_idx = 0
            else:
                raise ValueError(
                    f"save_ptm: el_idx required ({len(self.elevations)} elevations present)"
                )
        if pol_idx is None:
            if len(self.polarizations) == 1:
                pol_idx = 0
            else:
                raise ValueError(
                    f"save_ptm: pol_idx required "
                    f"({len(self.polarizations)} polarizations present)"
                )
        el_idx = int(el_idx)
        pol_idx = int(pol_idx)
        if not 0 <= el_idx < len(self.elevations):
            raise IndexError(f"save_ptm: el_idx {el_idx} is out of range")
        if not 0 <= pol_idx < len(self.polarizations):
            raise IndexError(f"save_ptm: pol_idx {pol_idx} is out of range")

        def _angle_axis_to_deg(values, unit_key):
            unit = str((self.units or {}).get(unit_key, "deg")).strip().lower()
            array = np.asarray(values, dtype=float)
            if unit in ("deg", "degree", "degrees", ""):
                return array
            if unit in ("rad", "radian", "radians"):
                return np.rad2deg(array)
            raise ValueError(f"save_ptm: unsupported {unit_key} unit {unit!r}")

        aspects_deg = _angle_axis_to_deg(self.azimuths, "azimuth")
        elevations_deg = _angle_axis_to_deg(self.elevations, "elevation")
        pitch_deg = float(elevations_deg[el_idx])
        frequencies_ghz = np.asarray(
            self._frequency_value_to_hz(self.frequencies), dtype=float
        ) / 1.0e9

        selected_polarity = str(self.polarizations[pol_idx]).strip().upper()

        # PTM I/Q samples are exp(+jwt) phasors and the header has no field
        # for another convention, so conjugate an exp(-jwt) dataset on write
        # instead of silently dropping its declaration.
        conjugate = _ptm_conjugate_for_time_convention(self)

        def read_block(row, first, last):
            selection = (row, el_idx, slice(first, last), pol_idx)
            power_slice, phase_slice = self.power_phase_slice(selection)
            power_missing = ~np.isfinite(power_slice)
            if np.any(power_missing):
                raise ValueError(
                    "save_ptm: PTM has no documented missing-sample marker; "
                    f"{int(np.count_nonzero(power_missing))} sample(s) lack finite power")
            phase_missing = (power_slice > 0.0) & ~np.isfinite(phase_slice)
            if np.any(phase_missing):
                raise ValueError(
                    "save_ptm: complex PTM export requires phase for every positive-power "
                    f"sample; {int(np.count_nonzero(phase_missing))} sample(s) lack phase")
            result = np.asarray(self.rcs_slice(selection))
            zero_without_phase = (power_slice == 0.0) & ~np.isfinite(phase_slice)
            if np.any(zero_without_phase):
                result = np.array(result, copy=True)
                result[zero_without_phase] = 0.0 + 0.0j
            return np.conj(result) if conjugate else result

        header = header_from_extra(self.extra)
        # Elevation is written as the PTM pitch without any coordinate-system
        # claim, so drop GRIM's old great-circle convention marker.
        header = replace(
            header,
            configuration=_ptm_configuration_without_grim_gc_marker(
                header.configuration
            ),
        )
        return write_ptm(
            path,
            aspects_deg,
            frequencies_ghz,
            None,
            _block_reader=read_block,
            polarity=selected_polarity,
            pitch_deg=pitch_deg,
            header=header,
        )
