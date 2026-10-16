"""Nonmodal frozen A/B image inspection with linked views, cuts and ROI."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QFileDialog, QScrollArea, QWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path as PlotPath

from GRIM_Backend.plotting.image_analysis import (records_from_figure, compare_records,
    summarize_roi, image_display, comparison_display, comparison_advisories)
from GRIM_Backend.execution.plot_jobs import PlotJobRunner
from GRIM_Backend.plotting.modes.common import coordinate_edges, decimate_line


class ImageComparisonDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner if hasattr(owner, "winId") else None)
        self.owner = owner
        selected_colormap = getattr(owner, "_effective_colormap", None)
        self.colormap = selected_colormap() if callable(selected_colormap) else "turbo"
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Images A/B — frozen images and linked inspection")
        from GRIM_Backend.ui.widgets import initial_window_size
        screen = self.screen()
        self.resize(initial_window_size(screen.availableGeometry().size(), QSize(1150,860)))
        self.a = self.b = None
        self.result = None
        self._display = self._summary_stats = None
        self._generation = 0
        self._closed = False
        self._runner = PlotJobRunner(self)
        self._summary_runner = PlotJobRunner(self)
        # One runner per side so capturing B never cancels a capture of A.
        self._capture_runners = {"a": PlotJobRunner(self), "b": PlotJobRunner(self)}
        self._capture_generation = {"a": 0, "b": 0}
        self.roi_bounds = None
        self._selectors = []
        outer = QVBoxLayout(self)
        controls = QWidget()
        layout = QVBoxLayout(controls)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        palette = getattr(owner,"application_palette",{})
        background = palette.get("panel_bg", "#ffffff")
        foreground = palette.get("text", "#111111")
        scroll.setStyleSheet(f"QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {background}; color: {foreground}; }} QLabel,QCheckBox {{ color: {foreground}; }}")
        scroll.setMaximumHeight(240)
        scroll.setMinimumHeight(140)
        outer.addWidget(scroll)
        source_row = QHBoxLayout()
        self.combo_source = QComboBox()
        self.combo_source.setMinimumWidth(200)
        source_row.addWidget(QLabel("Current image"))
        source_row.addWidget(self.combo_source, 1)
        for text, callback in (("Refresh panels", self.refresh_sources), ("Capture as A", lambda: self.capture("a")),
                ("Capture as B", lambda: self.capture("b")), ("Swap A/B", self.swap)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            source_row.addWidget(button)
        layout.addLayout(source_row)
        self.names = QLabel("Capture A, change the dataset or image settings, then capture B. Source data is preserved.")
        self.names.setWordWrap(True)
        layout.addWidget(self.names)
        options = QHBoxLayout()
        self.combo_policy = QComboBox()
        for label, value in (("Matching sample centers", "exact"), ("Nearest neighbor on overlap", "nearest"), ("Linear intensity on overlap", "linear")):
            self.combo_policy.addItem(label, value)
        self.combo_mode = QComboBox()
        for label, value in (("Physical A − B", "physical"), ("Relative to each peak", "relative"), ("Displayed numeric A − B", "displayed")):
            self.combo_mode.addItem(label, value)
        self.check_as_supplied = QCheckBox("Compare coordinates as supplied")
        self.check_as_supplied.setToolTip("For unlike axis meanings, use the numeric coordinates without assuming a physical frame conversion. Frame/recipe metadata differences are always shown as advisories.")
        for widget in (QLabel("Grid"), self.combo_policy, self.combo_mode, self.check_as_supplied):
            options.addWidget(widget)
        layout.addLayout(options)
        view_row = QHBoxLayout()
        self.combo_view = QComboBox()
        for label, value in (("A / B with signed difference", "side"), ("Blink A ↔ B", "blink"), ("Swipe A | B", "swipe")):
            self.combo_view.addItem(label, value)
        self.swipe = QSlider(Qt.Horizontal)
        self.swipe.setRange(0,100)
        self.swipe.setValue(50)
        self.check_roi = QCheckBox("Draw ROI on an image")
        clear_roi = QPushButton("Full image ROI")
        clear_roi.clicked.connect(self.clear_roi)
        export = QPushButton("Export ROI summary")
        export.clicked.connect(self.export_summary)
        for widget in (self.combo_view, self.swipe, self.check_roi, clear_roi, export):
            view_row.addWidget(widget)
        layout.addLayout(view_row)
        self.advisory = QLabel()
        self.advisory.setWordWrap(True)
        self.advisory.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.advisory)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(300,240)
        outer.addWidget(NavigationToolbar2QT(self.canvas, self))
        outer.addWidget(self.canvas, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.summary)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self._blink)
        self._blink_b = False
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        for combo in (self.combo_policy, self.combo_mode):
            combo.currentIndexChanged.connect(self.render)
        self.check_as_supplied.toggled.connect(self.render)
        self.combo_view.currentIndexChanged.connect(self._view_changed)
        self.swipe.valueChanged.connect(self._update_primary)
        self.check_roi.toggled.connect(self._roi_toggled)
        self.refresh_sources()
        if self.combo_source.count():
            self.capture("a")
        if self.combo_source.count() > 1:
            records = records_from_figure(self.owner)
            source_name = records[0].metadata.get("source_name")
            second = next((i for i,r in enumerate(records[1:],1) if r.metadata.get("source_name") != source_name),1)
            self.combo_source.setCurrentIndex(second)
            self.capture("b")
        if self.a is None and self.b is None:
            self.render()

    def refresh_sources(self):
        index = self.combo_source.currentIndex()
        self.combo_source.clear()
        for record in records_from_figure(self.owner):
            self.combo_source.addItem(record.name)
        if self.combo_source.count():
            self.combo_source.setCurrentIndex(max(0, min(index, self.combo_source.count()-1)))

    def capture(self, side):
        """Freeze the chosen image; large images copy on a worker thread."""
        index = max(0, self.combo_source.currentIndex())
        self.refresh_sources()
        records = records_from_figure(self.owner)
        if not records:
            self.advisory.setText("Render a Waterfall, ISAR image, or Azimuth–Down-Range image, then refresh and capture it here.")
            return
        source = records[min(index, len(records)-1)]
        self._capture_generation[side] += 1
        generation = self._capture_generation[side]
        # The lease keeps the live plot's native mapping open while the copy
        # runs, even if the main plot is re-rendered meanwhile.
        lease = source.lease()

        def compute(cancelled):
            try:
                return source.frozen(cancel=cancelled)
            finally:
                lease.close()

        def publish(record, error):
            if self._closed or generation != self._capture_generation[side]:
                if record is not None:
                    record.close()
                return
            if error is not None:
                self.advisory.setText(f"Image capture unavailable: {error}")
                return
            old = getattr(self, side)
            setattr(self, side, record)
            if old is not None:
                old.close()
            self.roi_bounds = None
            self.render()

        if np.size(source.values) <= getattr(self, "_async_threshold", 100_000):
            try:
                record, error = compute(lambda: False), None
            except (OSError, MemoryError, ValueError) as exc:
                record, error = None, exc
            publish(record, error)
            return
        self.advisory.setText(f"Capturing {side.upper()} off the UI thread…")
        self.canvas.draw_idle()
        self._capture_runners[side].submit(compute, publish)

    def swap(self):
        self.a, self.b = self.b, self.a
        self.roi_bounds = None
        self.render()

    @staticmethod
    def _mesh(ax, x, y, values, *, edges=None, **kwargs):
        xe, ye = (coordinate_edges(x), coordinate_edges(y)) if edges is None else edges
        return ax.pcolormesh(xe, ye, np.ma.masked_invalid(values), shading="flat", **kwargs)

    def _render_pending_captures(self, previews=None):
        """Show the first frozen image while the engineer prepares the second."""
        instructions = ("Render a Waterfall, ISAR image, or Azimuth–Down-Range image "
                        "in the main plot, then refresh and capture it here.")
        if self.a is None and self.b is None:
            ax = self.figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(.5, .6, "No image captured", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.text(.5, .4, "Render an image in the main plot.\n"
                    "Refresh panels, then Capture as A or Capture as B.",
                    transform=ax.transAxes, ha="center", va="center")
            self.advisory.setText(instructions)
            return
        for index, (side, record) in enumerate((("A", self.a), ("B", self.b)), 1):
            ax = self.figure.add_subplot(1, 2, index)
            if record is None:
                ax.set_axis_off()
                ax.set_title(f"{side}: not captured")
                ax.text(.5, .5, f"Render the next image in the main plot.\n"
                        f"Refresh panels, then Capture as {side}.",
                        transform=ax.transAxes, ha="center", va="center")
                continue
            ax.set_title(f"{side}: {record.name}", wrap=True)
            try:
                display = (previews or {}).get(side) or image_display(record)
                mesh = self._mesh(ax, display["x"], display["y"], display["a"],
                                  edges=(display["x_edges"], display["y_edges"]), cmap=self.colormap)
                self.figure.colorbar(mesh, ax=ax, label=record.value_unit)
                ax.set_xlabel(f"{record.x_axis} ({record.x_unit})")
                ax.set_ylabel(f"{record.y_axis} ({record.y_unit})")
            except (ValueError, MemoryError, OSError) as exc:
                ax.text(.5, .5, str(exc), transform=ax.transAxes, ha="center", wrap=True)
        missing = "A" if self.a is None else "B"
        self.advisory.setText(f"Capture {missing} to compare. The existing capture stays fixed "
                             "while you change the main image.")

    def render(self, *_args):
        """Prepare native comparisons off the UI thread for larger captures."""
        self._runner.cancel(); self._summary_runner.cancel()
        self._generation += 1
        generation = self._generation
        self.timer.stop()
        for selector in self._selectors:
            selector.disconnect_events()
        self._selectors = []
        self.result = self._display = self._summary_stats = None
        self.summary.clear()
        self.figure.clear()
        self.image_axes = []
        a, b = self.a, self.b
        self.names.setText(f"A: {a.name if a else 'not captured'}\nB: {b.name if b else 'not captured'}")
        policy, mode, supplied = self.combo_policy.currentData(), self.combo_mode.currentData(), self.check_as_supplied.isChecked()
        leases = [record.lease() for record in (a, b) if record is not None]
        bounds = self.roi_bounds

        def compute(cancelled):
            try:
                if a is None or b is None:
                    return {"previews": {side: image_display(record, cancel=cancelled)
                                         for side, record in (("A", a), ("B", b)) if record is not None}}
                try:
                    result = compare_records(a, b, policy=policy, mode=mode, as_supplied=supplied, cancel=cancelled)
                except ValueError as exc:
                    return {"comparison_error": str(exc), "previews": {
                        side: image_display(record, cancel=cancelled) for side, record in (("A", a), ("B", b))}}
                display = comparison_display(result, cancel=cancelled)
                try:
                    stats = summarize_roi(result, bounds, cancel=cancelled)
                except ValueError as exc:
                    stats = str(exc)
                return {"result": result, "display": display, "stats": stats}
            finally:
                for lease in leases:
                    lease.close()

        def publish(prepared, error):
            if self._closed or generation != self._generation:
                return
            self.figure.clear()
            if error is not None:
                self.advisory.setText(f"Comparison storage or allocation failed: {error}. Free memory or capture a smaller input selection and retry.")
                self.summary.setText("The frozen source captures are preserved.")
            elif "result" in prepared:
                self.result, self._display = prepared["result"], prepared["display"]
                self._summary_stats = prepared["stats"]
                self._render_result()
            else:
                self._render_pending_captures(prepared.get("previews"))
                if prepared.get("comparison_error"):
                    self.advisory.setText(f"Quantitative difference unavailable: {prepared['comparison_error']}\n" + " · ".join(comparison_advisories(a, b)))
                    self.summary.setText("Separate native axes/scales are shown until an explicit comparison policy is selected.")
            self.canvas.draw_idle()

        cells = sum(np.size(record.values) for record in (a, b) if record is not None)
        if cells <= getattr(self, "_async_threshold", 100_000):
            try:
                prepared, error = compute(lambda: False), None
            except Exception as exc:
                prepared, error = None, exc
            publish(prepared, error)
        else:
            self.advisory.setText("Preparing native image comparison…")
            self.canvas.draw_idle()
            self._runner.submit(compute, publish)

    def _render_result(self):
        result = self.result
        display = self._display
        x,y = result["x"], result["y"]
        dx,dy,a,b,delta = (display[k] for k in ("x","y","a","b","difference"))
        edges = (display["x_edges"], display["y_edges"])
        limits = display["limits"]
        if limits[0] == limits[1]:
            limits = (limits[0]-.5,limits[1]+.5)
        scale = display["difference_scale"]
        grid = self.figure.add_gridspec(2,3, height_ratios=[3,2])
        self.image_axes = [self.figure.add_subplot(grid[0,i]) for i in range(3)]
        for ax in self.image_axes[1:]:
            ax.sharex(self.image_axes[0]); ax.sharey(self.image_axes[0])
        self.mesh_a = self._mesh(self.image_axes[0],dx,dy,a,edges=edges,cmap=self.colormap,norm=Normalize(*limits))
        self.mesh_overlay = self._mesh(self.image_axes[0],dx,dy,b,edges=edges,cmap=self.colormap,norm=self.mesh_a.norm)
        self.mesh_overlay.set_visible(False)
        mesh_b = self._mesh(self.image_axes[1],dx,dy,b,edges=edges,cmap=self.colormap,norm=self.mesh_a.norm)
        mesh_d = self._mesh(self.image_axes[2],dx,dy,delta,edges=edges,cmap=self.colormap,norm=TwoSlopeNorm(vcenter=0.,vmin=-scale,vmax=scale))
        for ax,title in zip(self.image_axes,("A","B","Signed A − B")):
            ax.set_title(title)
            ax.set_xlabel(f"{self.a.x_axis} ({self.a.x_unit})")
            ax.set_ylabel(f"{self.a.y_axis} ({self.a.y_unit})")
            ax.set_xlim(coordinate_edges(x)[[0,-1]])
            ax.set_ylim(coordinate_edges(y)[[0,-1]])
        level_unit = ("relative level (dB)" if self.combo_mode.currentData() == "relative" else
            ("deg" if self.a.phase else result['level_unit']))
        if self.combo_mode.currentData() == "displayed":
            level_unit = f"supplied A: {self.a.value_unit}; B: {self.b.value_unit}"
        self.figure.colorbar(mesh_b,ax=self.image_axes[:2],label=level_unit,shrink=.85)
        self.figure.colorbar(mesh_d,ax=self.image_axes[2],label=result['unit'],shrink=.85)
        self.horizontal = self.figure.add_subplot(grid[1,:2])
        self.vertical = self.figure.add_subplot(grid[1,2])
        self.hlines = self.horizontal.plot([],[],label="A") + self.horizontal.plot([],[],label="B")
        self.vlines = self.vertical.plot([],[],label="A") + self.vertical.plot([],[],label="B")
        self.horizontal.legend(); self.vertical.legend()
        self.horizontal.set_xlabel(f"{self.a.x_axis} ({self.a.x_unit})")
        self.horizontal.set_ylabel(level_unit)
        self.vertical.set_ylabel(f"{self.a.y_axis} ({self.a.y_unit})")
        self.vertical.set_xlabel(level_unit)
        self.cursors = [(ax.axvline(x[len(x)//2], color="white",linewidth=.7), ax.axhline(y[len(y)//2], color="white",linewidth=.7)) for ax in self.image_axes]
        self._selectors = [RectangleSelector(ax,self._on_roi,button=[1],useblit=False,interactive=True) for ax in self.image_axes]
        self._roi_toggled(self.check_roi.isChecked())
        self.advisory.setText("Locked coordinates and A/B color scale on overlapping centers. " +
            (("Interpolation uses supplied numeric values. " if self.combo_mode.currentData() == "displayed" else "Interpolation uses physical linear intensity (unit phasors for phase); missing corners stay missing. ") if result['policy']=='linear' else "") +
            " · ".join(result["notes"]) + (" " + display["note"] if display["note"] else ""))
        self._last_cursor = None
        self.update_cuts(x[len(x)//2],y[len(y)//2])
        self._show_summary(self._summary_stats)
        self._view_changed()
        self.canvas.draw_idle()

    def _view_changed(self, *_args):
        self.timer.stop()
        mode = self.combo_view.currentData()
        self.swipe.setEnabled(mode == "swipe")
        if self.result is not None:
            if mode == "blink":
                self.timer.start()
            self._update_primary()

    def _blink(self):
        self._blink_b = not self._blink_b
        self._update_primary()

    def _update_primary(self, *_args):
        if self.result is None:
            return
        mode = self.combo_view.currentData()
        title = "A"
        self.mesh_a.set_visible(True)
        self.mesh_overlay.set_visible(False)
        for mesh in (self.mesh_a, self.mesh_overlay):
            mesh.set_clip_box(self.image_axes[0].bbox)
            mesh.set_clip_path(None)
        if mode == "blink":
            title = "Blink: B" if self._blink_b else "Blink: A"
            self.mesh_a.set_visible(not self._blink_b)
            self.mesh_overlay.set_visible(self._blink_b)
        elif mode == "swipe":
            xe, ye = self._display["x_edges"], self._display["y_edges"]
            # Sweep the full physical cell support, including singleton axes.
            # Center-to-center motion leaves half a cell visible at each end.
            split = xe[0] + (xe[-1]-xe[0])*self.swipe.value()/100.
            transform = self.image_axes[0].transData
            for mesh, left, right in ((self.mesh_a, xe[0], split), (self.mesh_overlay, split, xe[-1])):
                clip = PlotPath([(left, ye[0]), (right, ye[0]), (right, ye[-1]), (left, ye[-1]), (left, ye[0])],
                                [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.CLOSEPOLY])
                mesh.set_clip_path(clip, transform)
            self.mesh_overlay.set_visible(True)
            title = f"Swipe A | B at {split:.6g} {self.a.x_unit}"
        self.image_axes[0].set_title(title)
        self.canvas.draw_idle()

    def on_motion(self,event):
        if self.result is not None and event.inaxes in self.image_axes and event.xdata is not None and event.ydata is not None:
            self.update_cuts(event.xdata,event.ydata)

    def update_cuts(self,x,y):
        result = self.result
        ix,iy = int(np.argmin(abs(result['x']-x))),int(np.argmin(abs(result['y']-y)))
        if (ix,iy) == self._last_cursor:
            return
        self._last_cursor = ix,iy
        for index,key in enumerate(('a','b')):
            hx,hy,_ = decimate_line(result['x'],result[key][iy,:])
            vy,vx,_ = decimate_line(result['y'],result[key][:,ix])
            self.hlines[index].set_data(hx,hy)
            self.vlines[index].set_data(vx,vy)
        for v,h in self.cursors:
            v.set_xdata([result['x'][ix]]*2); h.set_ydata([result['y'][iy]]*2)
        for ax in (self.horizontal,self.vertical):
            ax.relim(); ax.autoscale_view()
        self.horizontal.set_title(f"Row at {self.a.y_axis} = {result['y'][iy]:.6g} {self.a.y_unit}")
        self.vertical.set_title(f"Column at {self.a.x_axis} = {result['x'][ix]:.6g} {self.a.x_unit}")
        self.canvas.draw_idle()

    def _roi_toggled(self,enabled):
        for selector in self._selectors:
            selector.set_active(enabled)

    def _on_roi(self,start,stop):
        if None in (start.xdata,start.ydata,stop.xdata,stop.ydata):
            return
        self.roi_bounds = (min(start.xdata,stop.xdata),max(start.xdata,stop.xdata),min(start.ydata,stop.ydata),max(start.ydata,stop.ydata))
        for selector in self._selectors:
            selector.extents = self.roi_bounds
        self.update_summary()
        self.canvas.draw_idle()

    def clear_roi(self):
        self.roi_bounds = None
        for selector in self._selectors:
            selector.set_visible(False)
        self.update_summary()
        self.canvas.draw_idle()

    def _show_summary(self, stats):
        if isinstance(stats, str):
            self.summary.setText(stats)
            return
        if stats is None:
            return
        self.summary.setText(f"ROI: {stats['pixels']:,} finite cells / {stats['selected_cells']:,}; {stats['excluded_cells']:,} excluded. "
            f"A−B {stats['mean_definition']} mean {stats['mean_delta']:.6g}; RMS {stats['rms_delta']:.6g}; "
            f"min {stats['minimum_delta']:.6g}; max {stats['maximum_delta']:.6g} {stats['unit']}. "
            "Cell statistics are not integrated RCS or area-weighted energy.")

    def update_summary(self):
        if self.result is None:
            return
        result, bounds, generation = self.result, self.roi_bounds, self._generation
        self._summary_stats = None
        self._summary_runner.cancel()
        def compute(cancelled):
            try:
                return summarize_roi(result, bounds, cancel=cancelled)
            except ValueError as exc:
                return str(exc)
        def publish(stats, error):
            if self._closed or self.result is not result or generation != self._generation:
                return
            self._summary_stats = str(error) if error is not None else stats
            self._show_summary(self._summary_stats)
        if result['difference'].size <= getattr(self, "_async_threshold", 100_000):
            publish(compute(lambda: False), None)
        else:
            self.summary.setText("Calculating native ROI statistics…")
            self._summary_runner.submit(compute, publish)

    def export_summary(self):
        if self.result is None:
            return
        stats = self._summary_stats
        if not isinstance(stats, dict):
            self.summary.setText(stats if isinstance(stats, str) else "ROI statistics are still being prepared.")
            return
        path,_ = QFileDialog.getSaveFileName(self,"Export image comparison ROI","image_comparison_roi.json","JSON (*.json)")
        if path:
            payload = {"a":self.a.name,"b":self.b.name,"bounds":self.roi_bounds,
                "x_axis":self.a.x_axis,"x_unit":self.a.x_unit,"y_axis":self.a.y_axis,"y_unit":self.a.y_unit,
                "comparison":self.combo_mode.currentData(),"resampling":self.result['policy'],
                "statistics":stats,"advisories":self.result['notes'], "a_source":self.a.metadata,"b_source":self.b.metadata}
            try:
                Path(path).write_text(json.dumps(payload,indent=2,default=lambda value:np.asarray(value).tolist()),encoding="utf-8")
            except OSError as exc:
                self.summary.setText(f"Export failed: {exc}")

    def done(self,result):
        self._dispose()
        super().done(result)

    def _dispose(self):
        self._closed = True
        self._generation += 1
        self._runner.close(); self._summary_runner.close()
        for runner in self._capture_runners.values():
            runner.close()
        self.timer.stop()
        for record in (self.a,self.b):
            if record is not None:
                record.close()
        self.a = self.b = None
        self.result = self._display = self._summary_stats = None
        for selector in self._selectors:
            selector.disconnect_events()
        self._selectors = []
        self.figure.clear()
        if getattr(self.owner,"_image_comparison_dialog",None) is self:
            self.owner._image_comparison_dialog = None

    def closeEvent(self,event):
        self._dispose()
        super().closeEvent(event)


def open_image_comparison(owner):
    dialog = getattr(owner,"_image_comparison_dialog",None)
    if dialog is None:
        dialog = ImageComparisonDialog(owner)
        owner._image_comparison_dialog = dialog
    else:
        dialog.refresh_sources()
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
