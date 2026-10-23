"""Azimuth vs Down-Range image — partial ISAR.

For each selected azimuth, IFFT over frequency to build a range profile;
stack the profiles side by side. Unlike the ISAR mode this does NOT FFT
across azimuth, so the X axis stays in degrees rather than collapsing to
a spatial cross-range coordinate.

Useful for spotting which look-angles a particular scatterer lights up at,
diagnosing range-walk before doing a full ISAR, or quickly seeing target
extent without committing to a small azimuth window.
"""
from __future__ import annotations

import numpy as np

from GRIM_Backend.datasets.result_storage import allocate_result, operation_result_storage
from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest, PlotResult, check_cancel
from GRIM_Backend.plotting.image_grid import native_sample_width

from GRIM_Backend.datasets.phase_conventions import plot_phase_conventions
from . import common
from .isar_mode import (
    _MAX_INTERP_COMPLEX_CELLS,
    _length_unit,
    _uniform_resample_plan,
    _unit_to_hz_scale,
    _angle_values_to_degrees,
    _ifft,
)
from GRIM_Backend.isar.interpolation import resample_pair
from GRIM_Backend.isar.geometry import angular_bands, pool_image_peaks, attach_block_peak_probe


def _spin_band_bounds(lo_spin, hi_spin, lo: float, hi: float) -> tuple[float, float]:
    """Widen a spin-box band by half a display step and open it at range limits.

    The spins round their range and value to their display decimals, so an
    exact comparison can drop the dataset's endpoint frequency even when the
    band is left at its "full band" default.
    """
    tolerance = 0.0
    decimals = getattr(lo_spin, "decimals", None)
    if callable(decimals):
        tolerance = 0.5 * 10.0 ** -int(decimals())
    low, high = lo - tolerance, hi + tolerance
    minimum = getattr(lo_spin, "minimum", None)
    if callable(minimum) and lo <= float(minimum()):
        low = -np.inf
    maximum = getattr(hi_spin, "maximum", None)
    if callable(maximum) and hi >= float(maximum()):
        high = np.inf
    return low, high


def _prepare_uniform_frequency_history(
    frequency_hz: np.ndarray,
    complex_history: np.ndarray,
    *, plan=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Gap-aware range-processing grid and per-row measurement weights.

    Interpolation is allowed only between nearby acquired frequencies. A large
    missing band and any interpolation touching an unknown complex sample stay
    zero-weighted rather than becoming apparently measured phase history.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    history = np.asarray(complex_history)
    if history.ndim != 2 or history.shape[1] != frequency_hz.size:
        raise ValueError(
            "azimuth/range phase history must have shape (azimuth, frequency)"
        )
    maximum_frequency_samples = max(
        frequency_hz.size,
        _MAX_INTERP_COMPLEX_CELLS // max(history.shape[0], 1),
    )
    if plan is None:
        plan = _uniform_resample_plan(
            frequency_hz, max_output_samples=maximum_frequency_samples)
    finite = np.isfinite(history)
    clean = np.nan_to_num(
        history,
        copy=True,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.complex64, copy=False)
    uniform_history, interpolated_validity = resample_pair(
        frequency_hz, clean, finite.astype(np.float32), plan["target"],
        axis=1, support=plan["support"],
    )
    weights = (interpolated_validity >= 1.0 - 1.0e-6).astype(np.float32)
    uniform_history = np.asarray(uniform_history, dtype=np.complex64) * weights
    return (
        np.asarray(plan["target"], dtype=float),
        uniform_history,
        weights,
        dict(plan["info"]),
    )


# Zero-pad each range profile to this many times the sweep length while the
# padded display stays within the cell budget. Without padding an off-bin
# point reads up to 3.9 dB (rectangular) / 1.4 dB (Hann) low; 4x padding
# bounds that scalloping to about 0.22 dB / 0.09 dB, matching Range-Frequency.
_RANGE_PAD_FACTOR = 4
_RANGE_PAD_MAX_CELLS = 32 * 1024**2


def _range_fft_size(n_frequency: int, n_azimuth: int) -> int:
    """Padded range FFT length: 4x, else 2x, else 1x within the cell budget."""
    n_frequency, n_azimuth = max(int(n_frequency), 1), max(int(n_azimuth), 1)
    for factor in (_RANGE_PAD_FACTOR, 2):
        if n_azimuth * n_frequency * factor <= _RANGE_PAD_MAX_CELLS:
            return n_frequency * factor
    return n_frequency


def _range_scalloping_loss_db(window, n_fft: int) -> float:
    """Worst off-bin peak loss of the windowed profile on an n_fft grid."""
    taper = np.asarray(window, dtype=float)
    offset = 0.5 * taper.size / max(int(n_fft), 1)  # half a display bin, in native bins
    phase = np.exp(2j * np.pi * offset * np.arange(taper.size) / taper.size)
    gain = abs(np.sum(taper * phase)) / max(float(np.sum(taper)), np.finfo(float).tiny)
    return float(-20.0 * np.log10(max(gain, 1.0e-12)))


def _form_range_image(history, sample_weights, window, n_fft=None):
    """Single-precision FFT with double-precision coherent-gain accumulation.

    ``n_fft`` zero-pads each profile (default: the sweep length). The coherent
    gain uses the padded length, so an on-bin unit point reads 1 at any size.
    """
    taper = np.asarray(window, dtype=np.float32)
    weighted = np.asarray(history, dtype=np.complex64) * taper[None, :]
    weighted *= sample_weights
    size = weighted.shape[1] if n_fft is None else max(int(n_fft), weighted.shape[1])
    image = np.fft.fftshift(_ifft(weighted, n=size, axis=1), axes=1)
    gain = np.sum(sample_weights * taper[None, :], axis=1, dtype=np.float64) / size
    usable = gain > 0
    np.divide(image, gain[:, None], out=image, where=usable[:, None])
    image[~usable] = np.nan + 1j * np.nan
    return image, usable


def _range_display_values(dataset, magnitude: np.ndarray, *, linear: bool) -> np.ndarray:
    """Convert coherent range amplitude to generic image intensity.

    A frequency IFFT does not, by itself, preserve a calibrated physical RCS
    normalization. Keep the familiar amplitude-dB scaling without claiming
    the result is dBsm/dBke.
    """

    del dataset  # retained in the public helper signature for compatibility
    magnitude = np.asarray(magnitude, dtype=float)
    intensity = magnitude ** 2
    if linear:
        return intensity
    intensity = np.where(np.isfinite(intensity), intensity, np.nan)
    return 10.0 * np.log10(np.maximum(intensity, 1.0e-12))


def _range_display_grid(azimuths, ranges, image, *, azimuth_width, max_side, return_peaks=False, cancel=None):
    """Draw contiguous selected samples with shared midpoint cell boundaries.

    Normal spacing changes and regular selection strides are not missing
    sectors. Use ISAR's local-cadence gap detection to split real sectors, and
    preserve their actual cell edges during peak pooling. Native sampling
    width is only a fallback for a sector containing one sample.
    """
    azimuths = np.asarray(azimuths, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    image = np.asarray(image)
    width = float(azimuth_width)
    if (azimuths.ndim != 1 or not azimuths.size or not np.all(np.isfinite(azimuths))
            or np.any(np.diff(azimuths) <= 0)):
        raise ValueError("azimuth samples must be finite and strictly increasing")
    if (ranges.ndim != 1 or ranges.size < 2 or not np.all(np.isfinite(ranges))
            or np.any(np.diff(ranges) <= 0) or image.shape != (len(ranges), len(azimuths))):
        raise ValueError("range image must match its increasing range and azimuth axes")
    if not np.isfinite(width) or width <= 0 or max_side < 1:
        raise ValueError("display sampling width and size limit must be positive")

    dy = float(ranges[1] - ranges[0])
    y_edges = np.r_[ranges - dy / 2.0, ranges[-1] + dy / 2.0]
    y_starts = np.arange(0, len(ranges), max(1, int(np.ceil(len(ranges) / max_side))))
    y_edges = y_edges[np.r_[y_starts, len(ranges)]]

    sectors = angular_bands(range(len(azimuths)), azimuths)
    bounds = []
    for sector in sectors:
        centers = azimuths[sector]
        if len(centers) == 1:
            cell_edges = np.array([centers[0] - width/2, centers[0] + width/2])
        else:
            steps = np.diff(centers)
            cell_edges = np.r_[centers[0] - steps[0]/2,
                               centers[:-1] + steps/2, centers[-1] + steps[-1]/2]
        bounds.append(cell_edges)
    # A singleton's fallback width must not overlap an adjacent sector or
    # hide a gap detected from the surrounding local acquisition cadence.
    for i in range(len(sectors) - 1):
        last = azimuths[sectors[i][-1]]
        first = azimuths[sectors[i+1][0]]
        gap = first - last
        bounds[i][-1] = min(bounds[i][-1], last + gap/4)
        bounds[i+1][0] = max(bounds[i+1][0], first - gap/4)

    max_blocks = max_side - (len(sectors) - 1)
    if len(sectors) > max_blocks:
        raise ValueError("too many disconnected azimuth samples; select fewer angles")
    stride = max(1, int(np.ceil(len(azimuths) / max_blocks)))
    while sum((len(sector) + stride - 1)//stride for sector in sectors) > max_blocks:
        stride *= 2
    edges = [float(bounds[0][0])]
    column_blocks = []
    block_count = 0
    for i, (sector, cell_edges) in enumerate(zip(sectors, bounds)):
        if i:
            column_blocks.append(None)
            edges.append(float(cell_edges[0]))
        for first in range(0, len(sector), stride):
            end = min(first + stride, len(sector))
            column_blocks.append((sector[0] + first, sector[0] + end))
            edges.append(float(cell_edges[end]))
            block_count += 1
    reduced = len(y_starts) < len(ranges) or block_count < len(azimuths)
    if not reduced and all(block is not None for block in column_blocks):
        result = (np.asarray(edges), y_edges, image, False)
        return (*result, None) if return_peaks else result
    values, iy, ix = pool_image_peaks(image,
        max(1, int(np.ceil(len(ranges) / max_side))), column_blocks, cancel=cancel)
    result = (np.asarray(edges), y_edges, values, reduced)
    return (*result, (ix, iy) if reduced else None) if return_peaks else result


def _range_convention(dataset, *, flip_y=False, notes=None):
    """Resolve time normalization while leaving display flips to the user."""
    time_sign, range_sign = plot_phase_conventions(dataset, notes=notes)
    # Negative-time phasors are conjugated before the IFFT. A positive range
    # law in that convention therefore already becomes the canonical law.
    if range_sign == 1 and time_sign != "-jwt" and not flip_y and notes is not None:
        notes.append(
            "dataset explicitly declares S~exp(+j*2*k*R), while the default range "
            "axis assumes S~exp(-j*2*k*R); target positions may be mirrored. "
            "The selected Flip Y setting is preserved."
        )
    return time_sign


def render(self) -> None:
    self.last_plot_mode = "az_vs_range"
    self._start_plot_render()
    datasets = self._selected_datasets()
    if not datasets:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets(datasets)
    if reference is None:
        return
    flip_y_widget = getattr(self, "chk_isar_flip_y", None)
    flip_y = bool(flip_y_widget.isChecked()) if flip_y_widget is not None else False
    flip_x_widget = getattr(self, "chk_isar_flip_x", None)
    if flip_x_widget is not None and hasattr(flip_x_widget, "setEnabled"):
        flip_x_widget.setEnabled(False)
    az_indices = sorted(self._selected_indices(self.list_az))
    aperture = getattr(self, "chk_isar_aperture", None)
    if aperture is not None and aperture.isChecked() and az_indices:
        center, width = float(self.spin_isar_ap_center.value()), float(self.spin_isar_ap_width.value())
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            self.status.showMessage("Aperture center/width must be finite and width positive.")
            return
        degrees = _angle_values_to_degrees(reference, 'azimuth', reference.azimuths[az_indices])
        az_indices = [i for i, distance in zip(az_indices, abs((degrees-center+180) % 360-180)) if distance <= width/2+1e-9]
    if not az_indices:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_indices = sorted(self._selected_indices(self.list_freq))
    band = getattr(self, "chk_isar_freq_band", None)
    if band is not None and band.isChecked():
        lo, hi = float(self.spin_isar_freq_min.value()), float(self.spin_isar_freq_max.value())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            self.status.showMessage("Frequency band max must exceed min; both must be finite.")
            return
        band_lo, band_hi = _spin_band_bounds(self.spin_isar_freq_min, self.spin_isar_freq_max, lo, hi)
        freq_indices = [i for i in freq_indices if band_lo <= reference.frequencies[i] <= band_hi]
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for range processing.")
        return

    pol_values = self._selected_values(self.list_pol)
    elev_values = self._selected_values(self.list_elev)
    if not len(pol_values) or not len(elev_values):
        self.status.showMessage("Select one or more elevations and polarizations to plot.")
        return

    selected = {"azimuth": reference.azimuths[az_indices],
                "frequency": reference.frequencies[freq_indices], "elevation": elev_values}
    plans, skipped = [], []
    for name, dataset in datasets:
        indices = {axis: self._axis_selection_for_dataset(reference, dataset, axis, values)
                   for axis, values in selected.items()}
        if any(ids is None for ids in indices.values()):
            skipped.append(name)
            continue
        for axis, attr in (("azimuth", "azimuths"), ("frequency", "frequencies")):
            values = np.asarray(getattr(dataset, attr)[indices[axis]], dtype=float)
            order = np.argsort(values)
            if not np.all(np.isfinite(values)) or np.any(np.diff(values[order]) <= 0):
                self.status.showMessage(f"{name}: {axis} samples must be finite and strictly increasing.")
                return
            indices[axis] = np.asarray(indices[axis])[order]
        for pol in pol_values:
            pol_ids = self._indices_for_values(dataset.polarizations, [pol], tol=0.)
            if pol_ids is None:
                skipped.append(f"{name}/{pol}")
                continue
            for elev_idx in indices["elevation"]:
                plans.append(PlotPlan(name, dataset,
                    (indices["azimuth"], [elev_idx], indices["frequency"], pol_ids)))
    if not plans:
        self._show_plot_status("No compatible selected data for Az vs Down-Range. " + ", ".join(skipped))
        return
    if len(plans) > common.MAX_WATERFALL_PANELS:
        self.status.showMessage(f"Plot blocked: selection creates {len(plans)} panels; select fewer datasets, elevations, or polarizations.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(units_combo.currentText() if units_combo else "in")
    window_control = getattr(self, "combo_isar_window", None)
    window_options = {"window": window_control.currentText()} if window_control is not None else {
        "window_arrays": [self._isar_window(len(_frequency_plan(plan)["target"])) for plan in plans]}
    request = PlotRequest("az_vs_range", reference, tuple(plans),
        scale=self._plot_scale_mode(), options={**window_options,
            "unit_name": unit_name, "unit_scale": unit_scale,
            "flip_y": flip_y, "skipped": skipped})

    def publish(result):
        _display(self, result)

    return self._submit_plot_preparation(request,
        lambda cancel: prepare_azimuth_range(request, cancel), publish,
        cells=sum(len(plan.selection[0]) * len(plan.selection[2]) for plan in plans))


def _frequency_plan(plan):
    azimuth, _, frequency, _ = plan.selection
    freq_hz = np.asarray(plan.dataset.frequencies[frequency], dtype=float) * _unit_to_hz_scale(
        str(plan.dataset.units.get("frequency", "ghz")))
    return _uniform_resample_plan(freq_hz,
        max_output_samples=max(len(frequency), _MAX_INTERP_COMPLEX_CELLS // max(len(azimuth), 1)))


def prepare_azimuth_range(request, cancel=None, *, batch_bytes=8 * 1024**2):
    """Prepare every selected dataset/elevation/polarization in bounded batches."""
    if len(request.plans) > common.MAX_WATERFALL_PANELS:
        raise ValueError("select fewer datasets, elevations, or polarizations")
    frequency_plans, total_cells = [], 0
    maximum = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
    for plan in request.plans:
        check_cancel(cancel)
        sampling = _frequency_plan(plan)
        frequency_plans.append(sampling)
        n_az = len(plan.selection[0])
        n_fft = _range_fft_size(len(sampling["target"]), n_az)
        # Include synthetic gap columns in the conservative display bound.
        total_cells += min(maximum, 2 * n_az - 1) * min(maximum, n_fft)
    common.validate_aggregate_image_cells(total_cells, panel_count=len(request.plans),
                                          operation="Az vs Down-Range")
    panels = []
    for index, (plan, sampling) in enumerate(zip(request.plans, frequency_plans)):
        check_cancel(cancel)
        panels.append(_prepare_panel(request, plan, sampling, index, cancel, batch_bytes))
    return PlotResult(request, panels, skipped=request.options.get("skipped", []))


def _prepare_panel(request, plan, frequency_plan, index, cancel, batch_bytes):
    """Read and FFT one panel without changing the original RF math."""
    from .isar_mode import _window_array
    dataset = plan.dataset
    flip_y = bool(request.options.get("flip_y", False))
    metadata_notes = []
    time_sign = _range_convention(dataset, flip_y=flip_y, notes=metadata_notes)
    if dataset.phase_coverage() == "none":
        # Zero-power cells of power-only data read as 0+0j and would pass the
        # usable-row check below without any real phase history.
        raise ValueError("Range processing requires phase; this dataset stores power only.")
    azimuth, elevation, frequency, polarization = plan.selection
    az_values = common.values_for_display(request.reference, dataset, "azimuth", dataset.azimuths[azimuth])
    freq_hz = np.asarray(dataset.frequencies[frequency], dtype=float) * _unit_to_hz_scale(
        str(dataset.units.get("frequency", "ghz")))
    target = np.asarray(frequency_plan["target"], dtype=float)
    window = request.options.get("window_array")
    if "window_arrays" in request.options:
        window = request.options["window_arrays"][index]
    if window is None:
        window = _window_array(request.options["window"], len(target))
    n_fft = _range_fft_size(len(target), len(azimuth))
    # One full frequency vector is unavoidable; all additional azimuth rows
    # stay within this scratch target regardless of the total image size.
    rows = max(1, int(batch_bytes) // max(1, 64 * (len(frequency) + len(target)) + 24 * n_fft))
    any_usable, missing_profiles = False, 0
    check_cancel(cancel)
    with operation_result_storage("auto", ram_budget_bytes=16 * 1024**2):
        display = allocate_result((len(azimuth), n_fft), np.float64)
        for first in range(0, len(azimuth), rows):
            check_cancel(cancel)
            history = dataset.rcs_slice(np.ix_(azimuth[first:first + rows], elevation,
                                               frequency, polarization))[:, 0, :, 0]
            if time_sign == "-jwt":
                np.conjugate(history, out=history)
            _, uniform, weights, _ = _prepare_uniform_frequency_history(freq_hz, history, plan=frequency_plan)
            image, usable = _form_range_image(uniform, weights, window, n_fft=n_fft)
            any_usable = any_usable or bool(np.any(usable))
            missing_profiles += int(np.count_nonzero(~usable))
            magnitude = np.abs(image)
            display[first:first + rows] = magnitude
            del history, uniform, weights, image, magnitude
        if not any_usable:
            raise ValueError("No azimuth row has finite, supported phase history for range processing.")
        for first in range(0, len(azimuth), rows):
            check_cancel(cancel)
            # Retain formation precision before converting amplitude to intensity.
            magnitude = np.asarray(display[first:first + rows], dtype=np.float32)
            display[first:first + rows] = _range_display_values(dataset, magnitude,
                linear=request.scale == "linear")
        ranges = np.fft.fftshift(np.fft.fftfreq(n_fft, d=float(np.mean(np.diff(target)))))
        ranges *= 299_792_458.0 / 2.0 * request.options["unit_scale"]
        if flip_y:
            display = display[:, ::-1]
            ranges = -ranges[::-1]
        maximum = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
        xe, ye, drawn, reduced, peaks = _range_display_grid(az_values, ranges, display.T,
            azimuth_width=request.options.get("azimuth_width", native_sample_width(request.reference, dataset, "azimuth")), max_side=maximum,
            return_peaks=True, cancel=lambda: check_cancel(cancel))
        check_cancel(cancel)
        return dict(x_edges=xe, y_edges=ye, display_for_plot=drawn,
            display=None if reduced else display, decimated=reduced, peak_indices=peaks,
            range_axis=ranges, az_values=az_values, freq_hz=freq_hz,
            frequency_sampling=dict(frequency_plan["info"]), range_fft_size=n_fft,
            range_padding_factor=n_fft / len(target),
            azimuth_gap_count=len(angular_bands(range(len(az_values)), az_values)) - 1,
            missing_profile_count=missing_profiles,
            metadata_notes=metadata_notes,
            time_convention_conjugated=time_sign == "-jwt",
            range_scalloping_loss_db=_range_scalloping_loss_db(window, n_fft))


def _display(self, result):
    request, panels = result.request, result.items
    self._remove_colorbar()
    release_native = getattr(self, "_release_native_plot_samples", None)
    if callable(release_native):
        release_native()
    self.plot_figure.clear()
    self.plot_figure.set_facecolor(self._current_plot_bg())
    self.plot_figure.set_layout_engine("constrained")
    columns = min(3, int(np.ceil(np.sqrt(len(panels)))))
    slots = self.plot_figure.subplots(int(np.ceil(len(panels) / columns)), columns, squeeze=False).ravel()
    axes = list(slots[:len(panels)])
    for extra in slots[len(panels):]:
        self.plot_figure.delaxes(extra)
    self.plot_axes, self.plot_ax = axes, axes[0]
    shared = self.chk_colorbar_shared.isChecked()
    zmin, zmax = self.spin_plot_zmin.value(), self.spin_plot_zmax.value()
    limits = common.finite_data_limits(p["display_for_plot"] for p in panels) if shared else None
    vmin, vmax = (zmin, zmax) if zmin < zmax else (limits or (None, None))
    meshes = []
    for ax, plan, panel in zip(axes, request.plans, panels):
        self._style_axes(ax)
        meshes.append(_draw_panel(self, request, plan, panel, ax, vmin, vmax))
    # Retain the first artist for existing callers; discover all artists from
    # the current axes when changing shared limits or image styling.
    self._az_vs_range_mesh = meshes[0]
    if self.chk_colorbar.isChecked():
        self.plot_colorbars = ([self.plot_figure.colorbar(meshes[-1], ax=axes)] if shared else
            [self.plot_figure.colorbar(mesh, ax=ax) for mesh, ax in zip(meshes, axes)])
        for colorbar in self.plot_colorbars:
            self._apply_colorbar_ticks(colorbar)
            colorbar.set_label("Range image intensity (linear, a.u.)" if self._plot_scale_is_linear()
                               else "Range image intensity (dB re 1 a.u.)", color=self._current_plot_text())
            colorbar.ax.tick_params(colors=self._current_plot_text())
    for spin, value in ((self.spin_plot_xmin, min(p["x_edges"][0] for p in panels)),
                        (self.spin_plot_xmax, max(p["x_edges"][-1] for p in panels)),
                        (self.spin_plot_ymin, min(p["y_edges"][0] for p in panels)),
                        (self.spin_plot_ymax, max(p["y_edges"][-1] for p in panels))):
        blocked = spin.blockSignals(True)
        common.set_spin_value(spin, float(value))
        spin.blockSignals(blocked)
    self._apply_plot_limits()
    if any(p["decimated"] for p in panels):
        self._note_plot_render("Large range image was peak-preserving display-decimated for responsive "
            "interaction; narrow the selected axes for full display resolution.")
    notes = []
    for plan, panel in zip(request.plans, panels):
        sampling = panel["frequency_sampling"]
        details = list(panel.get("metadata_notes", ()))
        if float(sampling["non_uniformity"]) >= 1e-3:
            details.append(f"resampled frequency (Δ-spread {sampling['non_uniformity']*100:.1f}%)")
        if sampling.get("gap_count", 0):
            details.append(f"{sampling['gap_count']} missing frequency band(s) kept zero-weighted "
                           f"({100 * sampling.get('unsupported_fraction', 0):.1f}% unsupported grid)")
        if panel["azimuth_gap_count"]:
            details.append(f"{panel['azimuth_gap_count']} detected azimuth gap(s) shown blank")
        if panel["missing_profile_count"]:
            details.append(f"{panel['missing_profile_count']} azimuth profile(s) without usable phase history shown blank")
        if details:
            notes.append(f"{plan.name}: " + "; ".join(details))
    if request.options.get("flip_y", False):
        notes.append("range mirrored (Flip Y)")
    if result.skipped:
        notes.append("Skipped: " + ", ".join(result.skipped))
    self._last_az_vs_range_generation = getattr(self, "_plot_render_generation", 0)
    self._show_plot_status(f"Az vs Down-Range updated ({len(panels)} panel(s))." +
                           (" — " + " | ".join(dict.fromkeys(notes)) if notes else ""))


def _draw_panel(self, request, plan, panel, ax, vmin, vmax):
    dataset, reference = plan.dataset, request.reference
    elev_idx, pol_idx = int(plan.selection[1][0]), int(plan.selection[3][0])
    unit_name = request.options["unit_name"]
    source_name = plan.name
    x_edges, y_edges = panel["x_edges"], panel["y_edges"]
    display, display_for_plot = panel["display"], panel["display_for_plot"]
    decimated, peak_indices = panel["decimated"], panel["peak_indices"]
    range_axis, az_values, freq_hz = panel["range_axis"], panel["az_values"], panel["freq_hz"]
    color_options = dict(cmap=self._effective_colormap(), vmin=vmin, vmax=vmax)
    uniform_edges = all(np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-6, atol=0)
                        for edges in (x_edges, y_edges))
    if uniform_edges:
        mesh = ax.imshow(
            display_for_plot,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower", aspect="auto", interpolation="nearest", **color_options,
        )
    else:
        mesh = ax.pcolormesh(
            x_edges, y_edges, display_for_plot, shading="flat", rasterized=True,
            edgecolors="none", antialiased=False, snap=True,
            **color_options,
        )

    mesh._grim_rectilinear_data = (x_edges, y_edges, display_for_plot)
    from GRIM_Backend.plotting.image_analysis import ImageRecord, compact_metadata
    mesh._grim_image_record = ImageRecord(
        f"{source_name} | Azimuth–Down-Range | {dataset.polarizations[pol_idx]} | El {dataset.elevations[elev_idx]:g}",
        (x_edges[:-1]+x_edges[1:])/2 if decimated else az_values,
        (y_edges[:-1]+y_edges[1:])/2 if decimated else range_axis,
        display_for_plot if decimated else display.T,
        "azimuth", "range", self._plot_axis_unit(reference,"azimuth"), unit_name,
        "range_profile_intensity", "linear" if self._plot_scale_is_linear() else "dB",
        logarithmic=not self._plot_scale_is_linear(),
        notes=tuple(panel.get("metadata_notes", ())) + (
            ("Overview block maxima: cuts/ROI/differences use displayed blocks.",) if decimated else ()),
        metadata={"kind":"az_vs_range", "source_name":source_name,
            "polarization":str(dataset.polarizations[pol_idx]),
            "frequency_values_hz":freq_hz.tolist(),
            "remaining_selection":("elevation",float(dataset.elevations[elev_idx]),common.axis_unit(dataset,"elevation")),
            "source_units":compact_metadata(dataset.units),
            "source_metadata":compact_metadata(dataset.extra),
            "azimuth_gap_count":panel["azimuth_gap_count"],
            "missing_profile_count":panel["missing_profile_count"],
            "range_mirrored":bool(request.options.get("flip_y", False)),
            "time_convention_conjugated":panel["time_convention_conjugated"],
            "metadata_notes":panel.get("metadata_notes", [])})
    if peak_indices is not None:
        attach_block_peak_probe(mesh, x_edges, y_edges, az_values, range_axis, *peak_indices)
    ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    ax.set_ylabel(f"Down-Range ({unit_name})")
    elev_value = common.values_for_display(reference, dataset, "elevation", [dataset.elevations[elev_idx]])[0]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    pol_value = dataset.polarizations[pol_idx]
    ax.set_title(
        f"{source_name}\n"
        f"{elev_name} {elev_value:g} {elev_unit} | Pol {pol_value}",
        color=self._current_plot_text(),
    )

    return mesh
