"""OUT table loading into frequency and azimuth RCS grids."""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from GRIM_Backend.datasets.result_storage import allocate_result
from GRIM_Backend.datasets.work_control import checkpoint
from GRIM_Backend.io.text_storage import discover_coordinates, text_reader


_POLARIZATION_TOKENS = re.compile(r"(?<![A-Z0-9])(HH|VV|HV|VH)(?![A-Z0-9])")


def _out_records(path):
    """Yield ``(line_no, freq_ghz, azimuth_deg, rcs_dbke, phase_deg, channel)``.

    A full-line ``# polarization:`` comment yields a record whose numeric
    fields are ``None`` and whose ``channel`` is the declared label. Every
    message and the order in which problems are reported match the former
    single-pass reader: the first offending line wins, whatever its problem.
    """
    from GRIM_Backend.io.csv import open_text_table

    with open_text_table(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            if line_no % 1024 == 0:
                checkpoint(detail="Reading OUT rows")
            comment = raw_line.strip()
            channel = None
            if comment.lower().startswith("# polarization:"):
                text = comment.split(":", 1)[1].strip()
                try:
                    declared = json.loads(text)
                except (ValueError, TypeError):
                    # Accept ordinary legacy comments such as VV. Empty
                    # or malformed quoted comments remain ordinary comments.
                    declared = text if text and text[:1] not in {'"', '[', '{'} else None
                if isinstance(declared, str) and declared.strip():
                    channel = declared.strip()
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                if channel is not None:
                    yield line_no, None, None, None, None, channel
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"line {line_no}: expected exactly 4 columns "
                    "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                )
            try:
                freq_ghz = float(parts[0])
                azimuth_deg = float(parts[1])
                rcs_dbke = float(parts[2])
                phase_deg = float(parts[3])
            except ValueError as exc:
                raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc

            if not (freq_ghz > 0.0 and math.isfinite(freq_ghz)):
                raise ValueError(
                    f"line {line_no}: frequency_ghz must be positive and finite"
                )
            if not math.isfinite(azimuth_deg):
                raise ValueError(
                    f"line {line_no}: azimuth_deg must be finite"
                )
            if math.isnan(rcs_dbke) or rcs_dbke == math.inf:
                raise ValueError(
                    f"line {line_no}: rcs_dbke must be finite or -Inf"
                )
            if math.isinf(phase_deg):
                raise ValueError(
                    f"line {line_no}: phase_deg must be finite or NaN"
                )
            yield line_no, freq_ghz, azimuth_deg, rcs_dbke, phase_deg, channel


class OutFormatMixin:
    """OUT table loading into frequency and azimuth RCS grids."""

    @classmethod
    @text_reader
    def load_out(cls, path, *, max_output_bytes=None, _estimate_only=False):
        """Load whitespace-delimited `.out` data into an RcsGrid.

        Expected columns per non-comment line:
            frequency_ghz  azimuth_deg  rcs_dbke  phase_deg

        Parsing rules:
            - Lines starting with `#` (or text after `#`) are ignored.
            - Values are whitespace-delimited.
            - An optional `# polarization: "VV"` comment declares the channel.
            - Otherwise polarization is inferred from filename (`HH` or `VV`);
              if not present, polarization is `NA`.
            - The third column is interpreted as absolute dBke and converted to
              linear 2D scattering width using sigma_2d = (lambda / 2pi) * 10^(dBke/10).

        Output mapping:
            - azimuth axis   <- angle column
            - elevation axis <- single value [0.0]
            - frequency axis <- frequency_ghz column
            - polarization   <- inferred filename polarization
            - stored power   <- linear 2D scattering width (matches .grim storage)

        The table is read twice: coordinate discovery (reusable through a
        source-stamped discovery plan) and a bounded sample pass whose
        destinations come from the managed allocator, so Disk storage keeps
        the parsed arrays mapped instead of resident.
        """
        from GRIM_Backend.datasets.constants import C0, _ADOPT_CLEAN_ARRAYS_TOKEN
        from GRIM_Backend.io.cst import _vendor_estimate, _vendor_import_allocation
        from GRIM_Backend.io.samples import _cst_samples_equivalent

        file_name = os.path.basename(str(path))
        stem_upper = os.path.splitext(file_name)[0].upper()
        pol_matches = set(_POLARIZATION_TOKENS.findall(stem_upper))
        pol_label = next(iter(pol_matches)) if pol_matches else "NA"

        def _discover():
            frequency_values, azimuth_values = set(), set()
            record_count = 0
            declared = None
            for line_no, freq_ghz, azimuth_deg, _rcs, _phase, channel in _out_records(path):
                if channel is not None:
                    if declared is not None and declared != channel:
                        raise ValueError(f"line {line_no}: conflicting OUT polarization comments")
                    declared = channel
                if freq_ghz is None:
                    continue
                frequency_values.add(freq_ghz)
                azimuth_values.add(azimuth_deg)
                record_count += 1
            return frequency_values, azimuth_values, record_count, declared

        frequency_values, azimuth_values, record_count, declared_polarization = (
            discover_coordinates(path, "out", _discover))
        if not record_count:
            raise ValueError("OUT contains no data rows")
        if declared_polarization is None and len(pol_matches) > 1:
            raise ValueError(
                f"OUT filename {file_name!r} ambiguously declares "
                + " and ".join(sorted(pol_matches))
            )

        frequencies = np.asarray(sorted(frequency_values), dtype=float)
        azimuths = np.asarray(sorted(azimuth_values), dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        polarizations = np.asarray([declared_polarization or pol_label], dtype=object)
        del frequency_values, azimuth_values

        shape = (len(azimuths), 1, len(frequencies), 1)
        allocation = _vendor_import_allocation(
            shape, source=f"OUT import {path}", max_output_bytes=max_output_bytes,
        )
        if _estimate_only:
            return _vendor_estimate(allocation, "OUT")

        power = allocate_result(shape, np.float64, fill=np.nan)
        phase = allocate_result(shape, np.float64, fill=np.nan)
        first_line = allocate_result(shape, np.uint64, fill=0)
        az_index = {value: index for index, value in enumerate(azimuths.tolist())}
        f_index = {value: index for index, value in enumerate(frequencies.tolist())}

        # The dBke -> sigma_2d arithmetic is the former per-row expression on
        # Python floats, so the stored values are identical. Rows are applied
        # in file order: the first record of a cell is kept, a later record
        # may only supply the missing phase of a zero-width (-Inf dBke) cell
        # (a finite-width cell without phase conflicts with a phased repeat),
        # and a conversion failure is raised when its row is reached, after
        # conflicts on earlier rows.
        two_pi = 2.0 * np.pi
        for line_no, freq_ghz, azimuth_deg, rcs_dbke, phase_deg, _channel in _out_records(path):
            if freq_ghz is None:
                continue
            try:
                index = (az_index[azimuth_deg], 0, f_index[freq_ghz], 0)
            except KeyError as exc:
                raise ValueError("OUT axes changed while the file was being loaded") from exc
            if rcs_dbke == -math.inf:
                sigma_2d = 0.0
            else:
                lambda_m = C0 / (freq_ghz * 1.0e9)
                try:
                    sigma_2d = (lambda_m / two_pi) * (10.0 ** (rcs_dbke / 10.0))
                except OverflowError:
                    raise ValueError(
                        "OUT dBke magnitude overflows finite linear power at "
                        f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                    ) from None
                if not math.isfinite(sigma_2d):
                    raise ValueError(
                        "OUT dBke magnitude does not produce finite linear power at "
                        f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                    )
            incoming_phase = float(np.deg2rad(phase_deg))
            if first_line[index]:
                existing_phase = float(phase[index])
                if not _cst_samples_equivalent(
                    float(power[index]), existing_phase, sigma_2d, incoming_phase,
                ):
                    raise ValueError(
                        "conflicting duplicate OUT sample at "
                        f"frequency={freq_ghz:g} GHz, azimuth={azimuth_deg:g} deg"
                    )
                if not np.isfinite(existing_phase) and np.isfinite(incoming_phase):
                    phase[index] = incoming_phase
                continue
            first_line[index] = line_no
            power[index] = sigma_2d
            phase[index] = incoming_phase
        del first_line

        if not np.isfinite(power).any():
            raise ValueError("OUT parsed, but no finite RCS magnitude values were found")

        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded OUT (dBke -> linear sigma_2d): {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_log_unit": "dBke"},
            extra={
                "source_format": "OUT",
                "dense_import_allocation_bytes": allocation["dense_bytes"],
                "dense_import_peak_bytes": allocation["peak_bytes"],
                "dense_import_limit_bytes": allocation["limit_bytes"],
            },
            _adopt_clean_arrays=_ADOPT_CLEAN_ARRAYS_TOKEN,
        )
