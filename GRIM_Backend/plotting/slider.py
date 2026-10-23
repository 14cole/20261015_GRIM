"""Scrub one parameter axis and re-plot at each value.

The slider selects exactly one value in the chosen parameter list and
re-renders the current plot type, independent of Auto Plot. Under Hold the
curves and bands the slider added last time are replaced, so a scrubbed plot
can move over held PBP bands or reference curves without piling up.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
import numpy as np

from .sweep_frames import make_sweep_frames, FrameLabels

RANGE_MODES = {"isar_image", "isar_composite", "az_vs_range"}

# The axis each plot type sweeps along; the slider must scrub a different one.
SWEEP_AXIS = {
    "azimuth_rect": "azimuth",
    "azimuth_polar": "azimuth",
    "sector_stats": "azimuth",
    "frequency": "frequency",
    "elevation_sweep": "elevation",
    "range_freq": "frequency",
}


def slider_axis_error(mode, axis):
    if mode == "sector_stats" and axis == "frequency":
        return "Box statistics use each box's frequency band. Choose Elevation to play or export these statistics."
    if SWEEP_AXIS.get(mode) == axis:
        return f"This plot sweeps {axis}; choose another slider axis to scrub."
    return None


class PlotSliderMixin:
    def _slider_list(self, axis: str):
        return {
            "frequency": self.list_freq,
            "elevation": self.list_elev,
            "azimuth": self.list_az,
        }[axis]

    def _on_plot_slider_toggled(self, checked: bool) -> None:
        bar = getattr(self, "plot_slider", None)
        if bar is None:
            return
        bar.setVisible(bool(checked))
        if checked:
            self._refresh_plot_slider()
        else:
            bar.stop_play()

    def _schedule_plot_slider_refresh(self, *_args) -> None:
        """Coalesce list rebuilds (one signal per inserted row) into one refresh."""
        if getattr(self, "plot_slider", None) is None:
            return
        timer = getattr(self, "_plot_slider_refresh_timer", None)
        if timer is None:
            timer = self._plot_slider_refresh_timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(0)
            timer.timeout.connect(self._refresh_plot_slider)
        timer.start()

    def _refresh_plot_slider(self, *_args) -> None:
        """Match the slider range and position to the chosen parameter list."""
        bar = getattr(self, "plot_slider", None)
        if bar is None or bar.isHidden():
            return
        if getattr(self, "_sweep_export", None) is not None:
            return
        axis = bar.axis()
        widget = self._slider_list(axis)
        dataset = getattr(self, "active_dataset", None)
        mode = getattr(self, "last_plot_mode", None)
        reason = slider_axis_error(mode, axis)
        if reason:
            bar._frames = None
            bar._plan_key = None
            for control in (bar.band_label, bar.band_width, bar.step_mode, bar.advance):
                control.hide()
            bar.stop_interaction()
            bar.set_positions([], 0)
            bar.sequence_note.setText(reason)
            return
        band = mode in RANGE_MODES and axis in ("frequency", "azimuth")
        bar.band_label.setVisible(band)
        bar.band_width.setVisible(band)
        bar.step_mode.setVisible(band)
        bar.advance.setVisible(band and bar.step_mode.currentData() == "custom")
        bar.band_label.setText("Bandwidth" if axis == "frequency" else "Aperture width")
        if dataset is None or widget.count() == 0:
            bar._frames = None
            bar.set_positions([], 0)
            return
        native_values = getattr(dataset, {"frequency": "frequencies", "azimuth": "azimuths", "elevation": "elevations"}[axis])
        key = (id(dataset), id(native_values), axis, mode, widget.count(), str((dataset.units or {}).get(axis, "")))
        plan_key = (key, bar.band_width.value() if band else None,
                    bar.step_mode.currentData() if band else None,
                    bar.advance.value() if band and bar.step_mode.currentData() == "custom" else None)
        if plan_key == getattr(bar, "_plan_key", None) and bar._frames is not None:
            self._sync_slider_position(bar, widget, band)
            return
        values = np.asarray(native_values, dtype=float)
        unit = str((dataset.units or {}).get(axis, ""))
        if axis == "azimuth" and band:
            from .modes.isar_mode import _angle_values_to_degrees
            values = _angle_values_to_degrees(dataset, axis, values)
            unit = "deg"
        changed = getattr(bar, "_source_key", None) != key
        if changed:
            bar.stop_interaction()
            bar._source_key = key
        if band and len(values) >= 2:
            old = bar.band_width.blockSignals(True)
            extent = float(np.ptp(values))
            gaps = np.diff(np.unique(values))
            minimum = float(np.min(gaps)) if len(gaps) else 1.e-9
            decimals = max(9, min(15, int(np.ceil(-np.log10(max(minimum, 1.e-15))))+2))
            bar.band_width.setDecimals(decimals)
            bar.band_width.setRange(minimum, max(minimum, extent))
            bar.band_width.setSuffix(f" {unit}")
            bar.band_width.setSingleStep(minimum)
            if changed:
                bar.band_width.setValue(max(minimum, extent/4))
            bar.band_width.blockSignals(old)
            old = bar.advance.blockSignals(True)
            bar.advance.setDecimals(decimals)
            bar.advance.setSuffix(f" {unit}")
            bar.advance.setRange(minimum, max(minimum, extent))
            if changed:
                bar.advance.setValue(bar.band_width.value())
            bar.advance.blockSignals(old)
        try:
            frames = make_sweep_frames(
                values, width=bar.band_width.value() if band else None, unit=unit,
                step_by_width=band and bar.step_mode.currentData() == "width",
                advance=bar.advance.value() if band and bar.step_mode.currentData() == "custom" else None)
        except ValueError as exc:
            bar._frames = None
            bar.set_positions([], 0)
            bar.value_label.setText(str(exc))
            bar.sequence_note.clear()
            bar.stop_play()
            return
        bar._frames = frames
        bar._plan_key = (key, bar.band_width.value() if band else None,
                        bar.step_mode.currentData() if band else None,
                        bar.advance.value() if band and bar.step_mode.currentData() == "custom" else None)
        bar._labels = FrameLabels(frames)
        if band:
            preview = "; ".join(frames.label(i) for i in range(min(3, len(frames))))
            if len(frames) > 3:
                preview += "; …; " + frames.label(len(frames)-1)
            bar.sequence_note.setText(
                f"All dataset {axis} samples · {len(frames)} windows: {preview}. "
                "Endpoints are inclusive. Final windows may be shorter; duplicate sample sets and windows with fewer than two samples are omitted.")
        else:
            bar.sequence_note.setText(f"All dataset {axis} samples · {len(frames)} positions.")
        self._sync_slider_position(bar, widget, band)

    def _sync_slider_position(self, bar, widget, band):
        frames = bar._frames
        rows = [widget.row(item) for item in widget.selectedItems()] if not band else []
        current = bar.slider.value()
        if not band and rows:
            matches = np.flatnonzero(frames.rows == min(rows))
            current = int(matches[0]) if len(matches) else 0
        bar.set_positions(bar._labels, current)

    def _on_plot_slider_moved(self, row: int) -> None:
        bar = getattr(self, "plot_slider", None)
        if bar is None:
            return
        bar._debounce.stop()
        axis = bar.axis()
        widget = self._slider_list(axis)
        mode = getattr(self, "last_plot_mode", None)
        reason = slider_axis_error(mode, axis)
        if reason:
            bar.stop_play()
            self.status.showMessage(
                reason
            )
            self._slider_frame_finished(False)
            return
        if getattr(self, "_sweep_export", None) is None:
            self._refresh_plot_slider()
        frames = getattr(bar, "_frames", None)
        if frames is None or not 0 <= row < len(frames):
            self._slider_frame_finished(False)
            return
        bar.set_positions(bar._labels, row)
        self._slider_frame_finished(False)
        token = object()
        self._slider_request = (token, bar)
        self._slider_dispatching = True
        self._slider_async_token = None
        bar.render_started()
        blocked = widget.blockSignals(True)
        try:
            widget.clearSelection()
            for index in frames.selection(row):
                widget.item(index).setSelected(True)
            widget.scrollToItem(widget.item(frames.selection(row)[0]))
        finally:
            widget.blockSignals(blocked)
        if mode in RANGE_MODES and frames.band:
            # The list is the exact sample mask. Clear a previous settings
            # filter so it cannot silently clip the moving band a second time.
            control = self.chk_isar_freq_band if axis == "frequency" else self.chk_isar_aperture
            blocked = control.blockSignals(True)
            control.setChecked(False)
            control.blockSignals(blocked)
            settings = getattr(getattr(self, "settings_frame", None), "plot_controls", None)
            if settings is not None:
                settings.refresh_conditions()
        self._invalidate_isar_result()
        if mode is None:
            self._slider_dispatching = False
            self._slider_frame_finished(False, token)
            return
        hold = self._button_checked(getattr(self, "btn_hold", None))
        before = set()
        if hold:
            for ax in self.plot_figure.axes:
                for artist in (*ax.lines, *ax.collections):
                    if getattr(artist, "_grim_slider_trace", False):
                        self._release_artist_native_samples(artist)
                        artist.remove()
            before = {id(artist) for ax in self.plot_figure.axes
                      for artist in (*ax.lines, *ax.collections)}
        try:
            self._render_plot_mode(mode)
        except Exception as exc:
            self.status.showMessage(f"Sweep frame failed: {exc}")
            self._slider_frame_finished(False, token)
        finally:
            self._slider_dispatching = False
        if hold:
            if getattr(self, "_ordinary_plot_pending", False):
                # Numerical work may finish later. Tag only the artists from
                # that publication so the next scrub replaces its own traces.
                self._ordinary_plot_slider_before = before
            else:
                self._tag_plot_slider_artists(before)
        if self._slider_async_token is not token:
            self._slider_frame_finished("updated" in self.status.currentMessage().lower(), token)

    def _slider_frame_finished(self, ok, token=None):
        request = getattr(self, "_slider_request", None)
        if request is None or (token is not None and request[0] is not token):
            return
        self._slider_request = None
        request[1].render_finished(ok)
        exporter = getattr(self, "_sweep_export", None)
        if exporter is not None:
            exporter.frame_finished(ok)

    def _export_plot_sweep(self):
        from GRIM_Backend.ui.sweep_export import start_sweep_export
        start_sweep_export(self)

    def _tag_plot_slider_artists(self, before) -> None:
        # Keep pre-existing held bands, but replace this slider's own
        # previous curves and bands. Markers re-snap after every render.
        for ax in self.plot_figure.axes:
            for artist in (*ax.lines, *ax.collections):
                key = getattr(artist, "_grim_dataset_key", None)
                if id(artist) not in before and key is not None:
                    artist._grim_slider_trace = True
