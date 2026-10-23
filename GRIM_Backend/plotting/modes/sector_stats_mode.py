"""Azimuth sector statistics drawn as level segments over each sector.

Each dataset/frequency (or frequency band)/elevation cut becomes flat segments, one
per sector, on the same azimuth axis as Azimuth (Rect). Statistics use linear
power (a mean of dB values is not a mean level), including valid zero-power
samples. Every statistic is kept for Copy Sector Table; logarithmic zero
is represented by negative infinity, separately from missing data (NaN).
"""

from __future__ import annotations

import warnings

import numpy as np

from . import common
from GRIM_Backend.datasets.statistics_boxes import normalize_boxes

STATISTICS = (
    ("mean", "Mean"),
    ("median", "Median"),
    ("max", "Max"),
    ("min", "Min"),
    ("percentile", "Percentile"),
)


def _sector_linear_statistics(
    linear, members, percentile: float, interpolate: bool = False
) -> dict[str, float]:
    values = linear[members]
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "max": np.nan,
                "min": np.nan, "percentile": np.nan}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
        "percentile": float(np.percentile(
            values, percentile, method="linear" if interpolate else "inverted_cdf"
        )),
    }


def statistic_tag(statistic: str, percentile: float, interpolate: bool = False) -> str:
    if statistic != "percentile":
        return statistic
    suffix = "" if interpolate else " (nearest rank)"
    return f"P{percentile:g}{suffix}"


def render(self) -> None:
    self.last_plot_mode = "sector_stats"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    if self._button_checked(self.btn_phase):
        self.status.showMessage(
            "Sector statistics summarize magnitude levels. Turn off Phase to plot them."
        )
        return

    controls = getattr(self, "analysis_controls", None)
    boxes = controls.sector_boxes() if controls is not None else []
    use_selected_azimuths = controls.sector_use_selected_azimuths() if controls is not None else False
    az_values = np.asarray(sorted(
        self._selected_values(self.list_az) if use_selected_azimuths else reference.azimuths
    ), dtype=float)
    freq_values = np.asarray(sorted(reference.frequencies), dtype=float)
    elev_values = np.asarray(sorted(self._selected_values(self.list_elev)), dtype=float)
    for values, axis in ((az_values, "azimuths"), (freq_values, "frequencies"),
                         (elev_values, "elevations")):
        if values.size == 0:
            self.status.showMessage(f"Select one or more {axis} to plot.")
            return
    polarization = self._overlay_polarizations()
    if polarization is None:
        return

    statistic = controls.sector_statistic() if controls is not None else "mean"
    percentile = controls.sector_percentile() if controls is not None else 90.0
    interpolate = controls.sector_interpolate() if controls is not None else False
    period = float(common.convert_axis_values(
        [360.0], "azimuth", "deg", self._plot_axis_unit(reference, "azimuth")
    )[0])
    try:
        boxes = normalize_boxes(boxes)
        sectors = []
        for box in boxes:
            start, stop = common.convert_axis_values(
                box['azimuth_band'], 'azimuth', 'deg', self._plot_axis_unit(reference, 'azimuth'))
            sectors.append(common.parse_sectors(f'{start:.17g}:{stop:.17g}', az_values, period=period)[0])
    except ValueError as exc:
        self.status.showMessage(f"Sector Stats blocked: {exc}.")
        return

    plans, skipped = [], []
    for name, dataset in datasets:
        for pol in ([polarization] if isinstance(polarization, str) else polarization):
            selection = (
                self._overlay_axis_selection(reference, dataset, 'azimuth', az_values)
                if use_selected_azimuths else np.arange(len(dataset.azimuths)),
                self._axis_selection_for_dataset(reference, dataset, 'elevation', elev_values),
                np.arange(len(dataset.frequencies)),
                self._indices_for_values(dataset.polarizations, [pol], tol=0.0),
            )
            if any(indices is None or len(indices) == 0 for indices in selection):
                skipped.append(f'{name} | Pol {pol}')
            else:
                plans.append((name, dataset, selection))
    if not plans:
        self._show_plot_status("No compatible one-to-one coordinates for the selected plot.")
        return
    band_plans = []
    peak_cells = total_cells = series_count = 0
    try:
        for name, dataset, selection in plans:
            az_indices, elev_indices, freq_indices, _ = selection
            hz = common.convert_axis_values(dataset.frequencies[freq_indices], 'frequency',
                                            common.axis_unit(dataset, 'frequency'), 'Hz')
            groups = []
            azimuths = self._plot_axis_values(reference, dataset, 'azimuth', dataset.azimuths[az_indices])
            for box, sector in zip(boxes, sectors):
                lo, hi = box['frequency_band']
                eps = max(abs(lo), abs(hi), 1.) * 1e-12
                offsets = np.flatnonzero((hz >= lo - eps) & (hz <= hi + eps))
                if not len(offsets) or not sector.contains(azimuths).any():
                    skipped.append(f"{name} | {box['name']} (no samples inside box)")
                    continue
                groups.append((offsets, (lo, hi), [sector], box['name']))
            populated = []
            for offsets, bounds, group_sectors, box_name in groups:
                if not len(offsets):
                    skipped.append(f"{name} | frequency band {bounds[0]:g}–{bounds[1]:g} Hz (no selected samples)")
                    continue
                group_az_indices = np.asarray(az_indices)
                group_az_indices = group_az_indices[group_sectors[0].contains(azimuths)]
                populated.append((group_az_indices, np.asarray(freq_indices)[offsets],
                                  bounds, group_sectors, box_name))
                cells = len(group_az_indices) * len(offsets)
                peak_cells = max(peak_cells, cells)
                total_cells += cells * len(elev_indices) * len(group_sectors)
                series_count += len(elev_indices)
            band_plans.append((name, dataset, selection, populated))
        common.validate_synchronous_plot_workload(
            operation="Sector frequency-band statistics", peak_slice_cells=peak_cells,
            total_cells=total_cells,
        )
    except ValueError as exc:
        self._show_plot_status(f"Sector Stats blocked: {exc}.")
        return
    self._configure_line_budget(series_count)
    if not self._prepare_line_plot_axes("sector_stats", "rectilinear", reference, datasets):
        return
    previous_zero_note = getattr(self.plot_ax, "_grim_sector_zero_note", None)
    if previous_zero_note is not None:
        if previous_zero_note in self.plot_ax.texts:
            previous_zero_note.remove()
        self.plot_ax._grim_sector_zero_note = None

    tag = statistic_tag(statistic, percentile, interpolate)
    freq_unit = self._plot_axis_unit(reference, "frequency")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    elev_name = self._plot_axis_name(reference, "elevation")
    unit = self._display_unit(datasets)
    low, high = float(az_values[0]), float(az_values[-1])
    table = []
    row_units = []
    rendered = 0
    omitted = 0
    zero_levels = 0
    frequency_references = []
    row_box_names = []
    for name, dataset, selection, groups in band_plans:
        native_unit = self._display_unit([(name, dataset)])
        az_indices, elev_indices, freq_indices, pol_indices = selection
        pol_value = dataset.polarizations[pol_indices[0]]
        for az_indices, group_indices, bounds, group_sectors, box_name in groups:
            azimuths = self._plot_axis_values(reference, dataset, "azimuth", dataset.azimuths[az_indices])
            memberships = [sector.contains(azimuths) for sector in group_sectors]
            # dBke needs a reference frequency even after reducing frequency.
            # Use the mean of the acquired frequencies and retain it in tables.
            native_frequency = float(np.mean(dataset.frequencies[group_indices]))
            frequency = float(self._plot_axis_values(
                reference, dataset, "frequency", [native_frequency])[0])
            limits = (self._plot_axis_values(reference, dataset, "frequency",
                      dataset.frequencies[group_indices]) if bounds is None else
                      common.convert_axis_values(bounds, "frequency", "Hz", freq_unit))
            frequency_label = f"{np.min(limits):.12g}–{np.max(limits):.12g}"
            for elev_idx in elev_indices:
                if rendered >= common.MAX_LINE_SERIES:
                    omitted += 1
                    continue
                elevation = float(self._plot_axis_values(
                    reference, dataset, "elevation", [float(dataset.elevations[elev_idx])])[0])
                # Stored rcs_power is already linear. rcs_to_linear clamps
                # negative real inputs to zero, which would make invalid
                # samples indistinguishable from valid physical zeros here.
                linear = np.asarray(
                    dataset.power[np.ix_(az_indices, [elev_idx], group_indices, [pol_indices[0]])],
                    dtype=float,
                ).reshape(len(az_indices), len(group_indices))
                x_points, y_points = [], []
                for sector, members in zip(group_sectors, memberships):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        stats = _sector_linear_statistics(linear, members, percentile, interpolate)
                    keys = ("mean", "median", "max", "min", "percentile")
                    values = np.asarray([stats[key] for key in keys])
                    if not self._plot_scale_is_linear():
                        # The shared line-display helper omits zero. Tables
                        # must instead distinguish log(0)=-inf from missing
                        # statistics, and retain the complete population.
                        values = dataset.linear_to_default_db(
                            values, frequency_value=native_frequency, eps=0.0
                        )
                        if any(stats[key] == 0.0 for key in keys):
                            self._note_plot_render(
                                "Zero-power statistics are retained as -inf in the sector table."
                            )
                    shown = {key: float(value) for key, value in zip(keys, values)}
                    table.append((
                        name, str(pol_value), frequency_label,
                        elevation, sector.label(),
                        stats["count"], shown["mean"], shown["median"], shown["min"],
                        shown["max"], shown["percentile"],
                    ))
                    row_units.append(native_unit)
                    row_box_names.append(box_name)
                    frequency_references.append(frequency)
                    level = shown[statistic]
                    if np.isneginf(level):
                        zero_levels += 1
                    if not np.isfinite(level):
                        continue
                    for first, last in sector.display_pieces(low, high):
                        x_points += [first, last, np.nan]
                        y_points += [level, level, np.nan]
                if not x_points:
                    continue
                label = (
                    f"{name} | Pol {pol_value}, Freq {frequency_label} {freq_unit}, "
                    f"{elev_name} {elevation:.12g} {elev_unit}, sector {tag}"
                )
                if box_name:
                    label = f'{box_name} | {label}'
                if native_unit.lower() == "dbke":
                    label += f", dBke ref {frequency:.12g} {freq_unit}"
                trace_key = (
                    "sector", statistic, percentile,
                    interpolate if statistic == "percentile" else None,
                    tuple((s.start, s.width) for s in group_sectors), box_name,
                    tuple(float(v) for v in dataset.azimuths[az_indices]),
                    bounds, tuple(float(v) for v in dataset.frequencies[group_indices]),
                    float(dataset.elevations[elev_idx]), str(pol_value),
                )
                self._plot_bounded_line(
                    self.plot_ax, np.asarray(x_points), np.asarray(y_points),
                    label=label, dataset=dataset, trace_key=trace_key,
                    linewidth=2.5, solid_capstyle="butt",
                )
                rendered += 1

    self.plot_figure._grim_sector_table = {
        "unit": unit,
        "percentile": percentile,
        "interpolate": interpolate,
        "percentile_method": "linear" if interpolate else "inverted_cdf",
        "rows": table,
        "row_units": row_units,
        "frequency_unit": freq_unit,
        "frequency_references": frequency_references,
        "box_names": row_box_names,
    }
    if zero_levels:
        zero_message = (
            f"{zero_levels} zero-power sector level(s) (-inf on the dB scale). "
            "Use Linear to display these levels."
        )
        self._note_plot_render(zero_message)
        # Include the explanation in exported figures as well as the status
        # bar: there is no finite ordinate at which to draw these levels.
        self.plot_ax._grim_sector_zero_note = self.plot_ax.text(
            0.02, 0.02,
            f"{zero_levels} zero-power sector level(s): -inf on the dB scale\n"
            "Retained in the sector table; use Linear to display.",
            transform=self.plot_ax.transAxes, ha="left", va="bottom", fontsize="small",
        )
    if rendered == 0 and zero_levels == 0:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        self._show_plot_status(f"No finite levels inside the sectors.{detail}")
        return
    if omitted:
        self._note_plot_render(
            f"Displayed {rendered} series; {omitted} further cuts were not drawn. "
            "Narrow frequency/elevation selections to show the rest."
        )

    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(self._display_axis_label(datasets, tag=f" {tag}"))
    self._update_legend_visibility()
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    common.set_spin_value(self.spin_plot_xmin, low)
    common.set_spin_value(self.spin_plot_xmax, high)
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self._apply_plot_limits()
    status = f"Sector statistics ({len(sectors)} boxes, {tag}) plot updated."
    if skipped:
        status = f"{status[:-1]} Skipped: {', '.join(skipped)}."
    self._show_plot_status(status)


def table_text(table) -> str:
    """Tab-separated sector table for the clipboard."""
    unit = table["unit"]
    row_units = table.get("row_units", [])
    mixed_units = len(set(row_units)) > 1
    suffix = "" if mixed_units else f" ({unit})"
    percentile_tag = statistic_tag(
        "percentile", table["percentile"], table.get("interpolate", False)
    )
    header = (
        "Dataset", "Pol", f"Frequency band ({table['frequency_unit']})", "Elevation", "Azimuth band", "Samples",
        f"Mean{suffix}", f"Median{suffix}", f"Min{suffix}", f"Max{suffix}",
        f"{percentile_tag}{suffix}",
    )
    if mixed_units:
        header += ("Unit",)
    show_reference = any(str(value).lower() == "dbke" for value in row_units)
    if show_reference:
        header += (f"dBke reference frequency ({table['frequency_unit']})",)
    header += ('Box',)
    lines = ["\t".join(header)]
    for row_index, row in enumerate(table["rows"]):
        cells = [str(value) if not isinstance(value, float) else f"{value:.6g}" for value in row]
        if mixed_units:
            cells.append(row_units[row_index])
        if show_reference:
            cells.append(f"{table['frequency_references'][row_index]:.12g}"
                         if row_units[row_index].lower() == "dbke" else "")
        cells.append(table['box_names'][row_index])
        lines.append("\t".join(cells))
    return "\n".join(lines)
