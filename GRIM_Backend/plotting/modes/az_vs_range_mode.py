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

from GRIM_Backend.datasets.transforms import _declared_time_sign
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


def render(self) -> None:
    self.last_plot_mode = "az_vs_range"
    self._start_plot_render()
    if self.active_dataset is None:
        self.status.showMessage("Select a dataset before plotting.")
        return
    reference = self._preflight_plot_datasets([("Dataset", self.active_dataset)])
    if reference is None:
        return
    try:
        time_sign = _declared_time_sign(self.active_dataset)
    except ValueError as exc:
        self.status.showMessage(f"Az vs Down-Range blocked: {exc}")
        return

    az_indices = sorted(self._selected_indices(self.list_az))
    aperture = getattr(self, "chk_isar_aperture", None)
    if aperture is not None and aperture.isChecked() and az_indices:
        center, width = float(self.spin_isar_ap_center.value()), float(self.spin_isar_ap_width.value())
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            self.status.showMessage("Aperture center/width must be finite and width positive.")
            return
        degrees = _angle_values_to_degrees(self.active_dataset, 'azimuth', self.active_dataset.azimuths[az_indices])
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
        freq_indices = [i for i in freq_indices if band_lo <= self.active_dataset.frequencies[i] <= band_hi]
    if not freq_indices:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    if len(freq_indices) < 2:
        self.status.showMessage("Select at least 2 frequency samples for range processing.")
        return

    pol_idx = self._single_selection_index(self.list_pol, "polarization")
    if pol_idx is None:
        return
    elev_idx = self._single_selection_index(self.list_elev, "elevation")
    if elev_idx is None:
        return

    # Sort axes ascending; build the (n_az, n_freq) complex slice.
    az_values = self.active_dataset.azimuths[az_indices].astype(float)
    az_order = np.argsort(az_values)
    sorted_az_indices = [az_indices[i] for i in az_order]
    az_values = az_values[az_order]
    if not np.all(np.isfinite(az_values)) or np.any(np.diff(az_values) <= 0):
        self.status.showMessage(
            f"{self._plot_axis_name(reference, 'azimuth')} samples must be strictly increasing."
        )
        return

    freq_values = self.active_dataset.frequencies[freq_indices].astype(float)
    freq_order = np.argsort(freq_values)
    sorted_freq_indices = [freq_indices[i] for i in freq_order]
    freq_values = freq_values[freq_order]
    if np.any(np.diff(freq_values) <= 0):
        self.status.showMessage("Frequency samples must be strictly increasing.")
        return

    units_combo = getattr(self, "combo_isar_units", None)
    unit_name, unit_scale = _length_unit(units_combo.currentText() if units_combo else "in")
    pn_widget = getattr(self, "chk_isar_peak_normalize", None)
    peak_norm = bool(pn_widget.isChecked()) if pn_widget else False
    native_steps = np.diff(np.sort(np.unique(self.active_dataset.azimuths)))
    width = float(np.min(native_steps)) if native_steps.size else (
        float(np.deg2rad(1.0)) if self._plot_axis_unit(reference, "azimuth") == "rad" else 1.0)
    source_name = next((name for name, candidate in getattr(self, "_selected_datasets", lambda: [])()
                        if candidate is self.active_dataset), "Dataset")
    window_control = getattr(self, "combo_isar_window", None)
    window_options = {"window": window_control.currentText()} if window_control is not None else {
        "window_array": self._isar_window(len(_uniform_resample_plan(
            freq_values * _unit_to_hz_scale(str(self.active_dataset.units.get("frequency", "ghz"))),
            max_output_samples=max(len(freq_values), _MAX_INTERP_COMPLEX_CELLS // len(az_values)))["target"]))}
    request = PlotRequest("az_vs_range", reference,
        (PlotPlan(source_name, self.active_dataset,
                  (sorted_az_indices, [elev_idx], sorted_freq_indices, [pol_idx])),),
        scale=self._plot_scale_mode(), options={**window_options,
            "unit_name": unit_name, "unit_scale": unit_scale, "peak_norm": peak_norm,
            "azimuth_width": width, "time_sign": time_sign})

    def publish(result):
        _display(self, result)

    return self._submit_plot_preparation(request,
        lambda cancel: prepare_azimuth_range(request, cancel), publish,
        cells=len(sorted_az_indices) * len(sorted_freq_indices))


def prepare_azimuth_range(request, cancel=None, *, batch_bytes=8 * 1024**2):
    """Read and FFT bounded azimuth batches, retaining the original RF math."""
    from .isar_mode import _window_array
    plan = request.plans[0]
    dataset = plan.dataset
    if dataset.phase_coverage() == "none":
        # Zero-power cells of power-only data read as 0+0j and would pass the
        # usable-row check below without any real phase history.
        raise ValueError("Range processing requires phase; this dataset stores power only.")
    azimuth, elevation, frequency, polarization = plan.selection
    az_values = np.asarray(dataset.azimuths[azimuth], dtype=float)
    freq_hz = np.asarray(dataset.frequencies[frequency], dtype=float) * _unit_to_hz_scale(
        str(dataset.units.get("frequency", "ghz")))
    frequency_plan = _uniform_resample_plan(freq_hz,
        max_output_samples=max(len(frequency), _MAX_INTERP_COMPLEX_CELLS // max(len(azimuth), 1)))
    target = np.asarray(frequency_plan["target"], dtype=float)
    window = request.options.get("window_array")
    if window is None:
        window = _window_array(request.options["window"], len(target))
    n_fft = _range_fft_size(len(target), len(azimuth))
    # One full frequency vector is unavoidable; all additional azimuth rows
    # stay within this scratch target regardless of the total image size.
    rows = max(1, int(batch_bytes) // max(1, 64 * (len(frequency) + len(target)) + 24 * n_fft))
    any_usable, peak = False, 0.
    check_cancel(cancel)
    with operation_result_storage("auto", ram_budget_bytes=16 * 1024**2):
        display = allocate_result((len(azimuth), n_fft), np.float64)
        for first in range(0, len(azimuth), rows):
            check_cancel(cancel)
            history = dataset.rcs_slice(np.ix_(azimuth[first:first + rows], elevation,
                                               frequency, polarization))[:, 0, :, 0]
            if request.options["time_sign"] == "-jwt":
                np.conjugate(history, out=history)
            _, uniform, weights, _ = _prepare_uniform_frequency_history(freq_hz, history, plan=frequency_plan)
            image, usable = _form_range_image(uniform, weights, window, n_fft=n_fft)
            any_usable = any_usable or bool(np.any(usable))
            magnitude = np.abs(image)
            display[first:first + rows] = magnitude
            if np.any(usable):
                peak = max(peak, float(np.nanmax(magnitude)))
            del history, uniform, weights, image, magnitude
        if not any_usable:
            raise ValueError("No azimuth row has finite, supported phase history for range processing.")
        for first in range(0, len(azimuth), rows):
            check_cancel(cancel)
            # Match the original single-precision normalization before the
            # double-precision intensity conversion, including tiny values.
            magnitude = np.asarray(display[first:first + rows], dtype=np.float32)
            if request.options["peak_norm"] and peak > 0.:
                magnitude = magnitude / peak
            display[first:first + rows] = _range_display_values(dataset, magnitude,
                linear=request.scale == "linear")
        ranges = np.fft.fftshift(np.fft.fftfreq(n_fft, d=float(np.mean(np.diff(target)))))
        ranges *= 299_792_458.0 / 2.0 * request.options["unit_scale"]
        maximum = min(common.MAX_IMAGE_SIDE, int(np.sqrt(common.MAX_IMAGE_CELLS)))
        xe, ye, drawn, reduced, peaks = _range_display_grid(az_values, ranges, display.T,
            azimuth_width=request.options["azimuth_width"], max_side=maximum,
            return_peaks=True, cancel=lambda: check_cancel(cancel))
        check_cancel(cancel)
        return PlotResult(request, [dict(x_edges=xe, y_edges=ye, display_for_plot=drawn,
            display=None if reduced else display, decimated=reduced, peak_indices=peaks,
            range_axis=ranges, az_values=az_values, freq_hz=freq_hz,
            frequency_sampling=dict(frequency_plan["info"]), range_fft_size=n_fft,
            range_padding_factor=n_fft / len(target),
            range_scalloping_loss_db=_range_scalloping_loss_db(window, n_fft))])


def _display(self, result):
    request, panel = result.request, result.items[0]
    plan = request.plans[0]
    dataset, reference = plan.dataset, request.reference
    elev_idx, pol_idx = int(plan.selection[1][0]), int(plan.selection[3][0])
    unit_name, peak_norm = request.options["unit_name"], request.options["peak_norm"]
    source_name = plan.name
    x_edges, y_edges = panel["x_edges"], panel["y_edges"]
    display, display_for_plot = panel["display"], panel["display_for_plot"]
    decimated, peak_indices = panel["decimated"], panel["peak_indices"]
    range_axis, az_values, freq_hz = panel["range_axis"], panel["az_values"], panel["freq_hz"]
    frequency_sampling = panel["frequency_sampling"]
    fr_nonuniformity = float(frequency_sampling["non_uniformity"])
    if decimated:
        self._note_plot_render("Large range image was peak-preserving display-decimated for responsive "
            "interaction; narrow the selected axes for full display resolution.")
    # Build the figure.
    self._remove_colorbar()
    release_native = getattr(self, "_release_native_plot_samples", None)
    if callable(release_native):
        release_native()
    self.plot_figure.clear()
    self.plot_ax = self.plot_figure.add_subplot(111)
    self.plot_axes = None
    self._style_plot_axes()

    cmap = self._effective_colormap()
    zmin = self.spin_plot_zmin.value()
    zmax = self.spin_plot_zmax.value()
    use_clamp = zmin < zmax

    color_options = dict(cmap=cmap, vmin=zmin if use_clamp else None,
                         vmax=zmax if use_clamp else None)
    uniform_edges = all(np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-6, atol=0)
                        for edges in (x_edges, y_edges))
    if uniform_edges:
        mesh = self.plot_ax.imshow(
            display_for_plot,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower", aspect="auto", interpolation="nearest", **color_options,
        )
    else:
        mesh = self.plot_ax.pcolormesh(
            x_edges, y_edges, display_for_plot, shading="flat", rasterized=True,
            edgecolors="none", antialiased=False, snap=True,
            **color_options,
        )

    self._az_vs_range_mesh = mesh
    mesh._grim_rectilinear_data = (x_edges, y_edges, display_for_plot)
    from GRIM_Backend.plotting.image_analysis import ImageRecord, compact_metadata
    mesh._grim_image_record = ImageRecord(
        f"{source_name} | Azimuth–Down-Range | {dataset.polarizations[pol_idx]}",
        (x_edges[:-1]+x_edges[1:])/2 if decimated else az_values,
        (y_edges[:-1]+y_edges[1:])/2 if decimated else range_axis,
        display_for_plot if decimated else display.T,
        "azimuth", "range", self._plot_axis_unit(reference,"azimuth"), unit_name,
        "range_profile_intensity", "linear" if self._plot_scale_is_linear() else "dB",
        logarithmic=not self._plot_scale_is_linear(),
        notes=("Overview block maxima: cuts/ROI/differences use displayed blocks.",) if decimated else (),
        metadata={"kind":"az_vs_range", "source_name":source_name,
            "polarization":str(dataset.polarizations[pol_idx]),
            "frequency_values_hz":freq_hz.tolist(),
            "remaining_selection":("elevation",float(dataset.elevations[elev_idx]),common.axis_unit(dataset,"elevation")),
            "source_units":compact_metadata(dataset.units),
            "source_metadata":compact_metadata(dataset.extra), "peak_normalized":peak_norm})
    if peak_indices is not None:
        attach_block_peak_probe(mesh, x_edges, y_edges, az_values, range_axis, *peak_indices)
    self.plot_ax.set_xlabel(self._plot_axis_label(reference, "azimuth"))
    self.plot_ax.set_ylabel(f"Down-Range ({unit_name})")
    elev_value = dataset.elevations[elev_idx]
    elev_name = self._plot_axis_name(reference, "elevation")
    elev_unit = self._plot_axis_unit(reference, "elevation")
    pol_value = dataset.polarizations[pol_idx]
    self.plot_ax.set_title(
        f"{self._plot_axis_name(reference, 'azimuth')} vs Down-Range | "
        f"{elev_name} {elev_value:g} {elev_unit} | Pol {pol_value}",
        color=self._current_plot_text(),
    )

    if self.chk_colorbar.isChecked():
        colorbar = self.plot_figure.colorbar(mesh, ax=self.plot_ax)
        self.plot_colorbars = [colorbar]
        self._apply_colorbar_ticks(colorbar)
        if self._plot_scale_is_linear():
            colorbar.set_label(
                "Range image intensity (linear, a.u.)",
                color=self._current_plot_text(),
            )
        else:
            colorbar.set_label(
                "Range image intensity (dB re 1 a.u.)",
                color=self._current_plot_text(),
            )
        colorbar.ax.tick_params(colors=self._current_plot_text())
        for label in colorbar.ax.get_yticklabels():
            label.set_color(self._current_plot_text())

    # Update axis spinboxes to match the new view.
    self.spin_plot_xmin.blockSignals(True)
    self.spin_plot_xmax.blockSignals(True)
    self.spin_plot_ymin.blockSignals(True)
    self.spin_plot_ymax.blockSignals(True)
    self.spin_plot_xmin.setValue(float(x_edges[0]))
    self.spin_plot_xmax.setValue(float(x_edges[-1]))
    self.spin_plot_ymin.setValue(float(y_edges[0]))
    self.spin_plot_ymax.setValue(float(y_edges[-1]))
    self.spin_plot_xmin.blockSignals(False)
    self.spin_plot_xmax.blockSignals(False)
    self.spin_plot_ymin.blockSignals(False)
    self.spin_plot_ymax.blockSignals(False)

    self._apply_plot_limits()

    note = ""
    if fr_nonuniformity >= 1e-3:
        note = f" — resampled frequency (Δ-spread {fr_nonuniformity*100:.1f}%)"
    gap_count = int(frequency_sampling.get("gap_count", 0))
    if gap_count:
        unsupported = 100.0 * float(
            frequency_sampling.get("unsupported_fraction", 0.0)
        )
        note += (
            f" — {gap_count} missing frequency band(s) kept zero-weighted "
            f"({unsupported:.1f}% unsupported grid)"
        )
    self._last_az_vs_range_generation = getattr(self, "_plot_render_generation", 0)
    self._show_plot_status(f"Az vs Down-Range updated{note}.")
