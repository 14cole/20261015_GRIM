"""Interactive native-resolution image cuts, published as independent datasets."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QLineEdit, QFileDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector, PolygonSelector

from GRIM_Backend.ui.no_wheel import NoWheelComboBox
from GRIM_Backend.execution.plot_jobs import PlotJobRunner
from GRIM_Backend.isar.masking import (finish_cut, save_image, load_image, comparison_cuts,
                                     MaskImageRequest, prepare_mask_source, mask_preview)


class MaskCutDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Mask Cut — ISAR / Composite datasets")
        self.resize(950, 800)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.sources = []
        self.vertices = None
        self.selector = None
        self._shapes = {}
        self.runner = PlotJobRunner(self)
        self.runner.publication_failed.connect(self._publication_error)
        if not hasattr(owner, "_mask_image_datasets"):
            owner._mask_image_datasets = []
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Image dataset"))
        self.source = NoWheelComboBox()
        row.addWidget(self.source, 1)
        self.refresh = QPushButton("Refresh plot sources")
        self.load = QPushButton("Open Image…")
        self.save = QPushButton("Save Image…")
        for button in (self.refresh, self.load, self.save):
            row.addWidget(button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.shape = NoWheelComboBox()
        self.shape.addItems(["Rectangle", "Polygon"])
        self.operation = NoWheelComboBox()
        self.operation.addItems(["Cut Include", "Cut Exclude"])
        self.operation.setToolTip("Include keeps the selected pixels; Exclude removes them. Discarded pixels become zero on the same coordinate grid.")
        self.name = QLineEdit()
        row.addWidget(self.shape)
        row.addWidget(self.operation)
        row.addWidget(QLabel("New dataset name"))
        row.addWidget(self.name, 1)
        layout.addLayout(row)
        self.instructions = QLabel("Drag a rectangle, or click polygon vertices and close the polygon at its first vertex. "
                                   "Finish Cut creates a new dataset. The source stays intact.")
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.PlainText)
        layout.addWidget(self.details)
        self.message = QLabel("Image datasets remain available here until GRIM closes. Use Save Image to keep them between sessions.")
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        layout.addWidget(self.message)
        row = QHBoxLayout()
        self.remove = QPushButton("Remove image from list")
        row.addWidget(self.remove)
        row.addStretch(1)
        self.finish = QPushButton("Finish Cut")
        self.finish.setEnabled(False)
        self.comparison = QPushButton("Create comparison set")
        self.comparison.setToolTip("Create Baseline, Include, and Exclude from this source using exactly the same boundary. Baseline retains any earlier cuts on this source.")
        self.comparison.setEnabled(False)
        self.close_button = QPushButton("Close")
        row.addWidget(self.finish)
        row.addWidget(self.comparison)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.source.currentIndexChanged.connect(self._show_source)
        self.shape.currentIndexChanged.connect(self._set_selector)
        self.refresh.clicked.connect(self._refresh_sources)
        self.finish.clicked.connect(self._finish)
        self.comparison.clicked.connect(self._finish_comparison)
        self.save.clicked.connect(self._save)
        self.load.clicked.connect(self._load)
        self.remove.clicked.connect(self._remove)
        self.close_button.clicked.connect(self.close)
        self._refresh_sources()

    def _append(self, image):
        self.sources.append(image)
        self.source.addItem(image.name)

    def _refresh_sources(self):
        self.source.blockSignals(True)
        self.source.clear()
        self.sources.clear()
        figure = getattr(self.owner, "plot_figure", None)
        if figure is not None:
            for axes in figure.axes:
                for artist in (*axes.images, *axes.collections):
                    image = getattr(artist, "_grim_mask_source", None)
                    if image is not None:
                        display = image.display if isinstance(image, MaskImageRequest) else image.metadata.setdefault('display', {})
                        display.update(cmap=artist.get_cmap().name, limits=list(artist.get_clim()))
                        self._append(image)
        for image in self.owner._mask_image_datasets:
            self._append(image)
        self.source.blockSignals(False)
        self._show_source()

    def _current(self):
        index = self.source.currentIndex()
        return self.sources[index] if 0 <= index < len(self.sources) else None

    def _show_source(self, *_args):
        if self.selector is not None:
            self.selector.disconnect_events()
            self.selector = None
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        image = self._current()
        self.save.setEnabled(image is not None)
        self.remove.setEnabled(image is not None)
        if image is None:
            self.details.setText("Plot ISAR or Composite, then Refresh plot sources; or open a saved Mask Cut image dataset.")
            self.vertices = None
            self.finish.setEnabled(False)
            self.comparison.setEnabled(False)
            self.canvas.draw_idle()
            return
        self._busy(True)
        self.details.setText("Preparing full-resolution cut source…" if isinstance(image, MaskImageRequest)
                             else "Preparing peak-preserving preview…")
        def prepare(cancel):
            resolved = prepare_mask_source(image, cancel) if isinstance(image, MaskImageRequest) else image
            return resolved, mask_preview(resolved, cancel)
        self.runner.submit(prepare, self._preview_ready)

    def _preview_ready(self, result, error):
        self._busy(False)
        if error is not None:
            self.vertices = None
            self.finish.setEnabled(False)
            self.comparison.setEnabled(False)
            self.save.setEnabled(False)
            self.details.setText(f"Could not prepare image: {error}")
            return
        image, preview = result
        self.sources[self.source.currentIndex()] = image
        limits = preview['limits'] or (None, None)
        mesh = self.ax.pcolormesh(preview['x'], preview['y'], preview['values'].T, shading="flat",
                                  cmap=preview['cmap'], vmin=limits[0], vmax=limits[1])
        self.figure.colorbar(mesh, ax=self.ax, label="Image intensity (linear)" if preview['linear'] else "Level (dB)")
        unit = image.metadata.get("unit", "m")
        self.ax.set(xlabel=f"Cross-range ({unit})", ylabel=f"Down-range ({unit})", title=image.name)
        self.ax.set_aspect("equal", adjustable="box")
        self.name.setText(image.name + " — cut")
        self.details.setText(image.reconstruction_reason)
        self._set_selector()
        self.canvas.draw_idle()

    def _set_selector(self, *_args):
        if self.selector is not None:
            self.selector.set_visible(False)
            self.selector.disconnect_events()
        self.selector = None
        self.vertices = None
        self.finish.setEnabled(False)
        self.comparison.setEnabled(False)
        if self._current() is None or isinstance(self._current(), MaskImageRequest) or self.runner.busy:
            return
        if self.shape.currentText() == "Polygon":
            self.selector = PolygonSelector(self.ax, self._polygon, useblit=True)
        else:
            self.selector = RectangleSelector(self.ax, self._rectangle, useblit=True,
                                               button=[1], interactive=True)
        saved = self._shapes.get((id(self._current()), self.shape.currentText()))
        if saved is not None:
            self.vertices = list(saved)
            if self.shape.currentText() == 'Polygon':
                self.selector.verts = self.vertices
            else:
                points = np.asarray(self.vertices)
                self.selector.extents = (points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max())
            self.finish.setEnabled(True)
            self.comparison.setEnabled(True)
        self.canvas.draw_idle()

    def _polygon(self, vertices):
        self.vertices = list(vertices)
        self._shapes[(id(self._current()), self.shape.currentText())] = list(vertices)
        self.finish.setEnabled(len(vertices) >= 3 and not self.runner.busy)
        self.comparison.setEnabled(len(vertices) >= 3 and not self.runner.busy)

    def _rectangle(self, first, last):
        if None in (first.xdata, first.ydata, last.xdata, last.ydata):
            return
        self._polygon([(first.xdata, first.ydata), (last.xdata, first.ydata),
                       (last.xdata, last.ydata), (first.xdata, last.ydata)])

    def _busy(self, busy):
        for widget in (self.source, self.shape, self.operation, self.name, self.refresh,
                       self.save, self.load, self.remove):
            widget.setEnabled(not busy)
        self.finish.setEnabled(not busy and self.vertices is not None)
        self.comparison.setEnabled(not busy and self.vertices is not None)
        if self.selector is not None:
            self.selector.set_active(not busy)

    def _publication_error(self, error):
        self._busy(False)
        self.message.setText(f"Could not publish Mask Cut result: {error}")

    def _finish(self):
        image = self._current()
        if image is None or self.vertices is None:
            return
        vertices = np.array(self.vertices, copy=True)
        include = self.operation.currentIndex() == 0
        name = self.name.text().strip() or image.name + " — cut"
        self._busy(True)
        self.message.setText("Applying mask and reconstructing radar data where available…")
        self.runner.submit(lambda cancel: finish_cut(image, vertices, include=include, name=name, cancel=cancel),
                           self._publish_cut)

    def _publish_cut(self, result, error):
        self._busy(False)
        if error is not None:
            self.message.setText(f"Mask Cut failed: {error}")
            return
        image, radar = result
        if radar is not None:
            self.owner._add_dataset_row(radar, image.name, "Mask Cut: reconstructed radar data")
        self.owner._mask_image_datasets.append(image)
        self._append(image)
        self.message.setText("Created reconstructed radar dataset in the main dataset table and an edited image here."
                             if radar is not None else "Created an image dataset here. Save Image keeps its pixels and coordinates for reopening.")
        self.message.setText(self.message.text() + " Source and boundary remain selected; choose the new image from the list to save it.")

    def _finish_comparison(self):
        source = self._current()
        if source is None or self.vertices is None:
            return
        vertices = np.array(self.vertices, copy=True)
        name = self.name.text().strip() or source.name
        self._busy(True)
        self.message.setText("Creating Baseline, Include, and Exclude from the same source…")
        self.runner.submit(lambda cancel: comparison_cuts(source, vertices, name=name, cancel=cancel),
                           self._publish_comparison)

    def _publish_comparison(self, results, error):
        if error is not None:
            self._publish_cut(None, error)
            return
        for result in results:
            self._publish_cut(result, None)
        self.message.setText("Created Baseline, Include, and Exclude. The original source and boundary remain selected.")

    def _save(self):
        image = self._current()
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image dataset", image.name + ".grim-image.npz",
                                             "GRIM image dataset (*.npz)")
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"
        self._busy(True)
        self.runner.submit(lambda cancel: save_image(path, image),
                           lambda result, error: self._saved(path, error))

    def _saved(self, path, error):
        self._busy(False)
        self.message.setText(f"Save failed: {error}" if error else f"Saved image dataset: {path}")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Mask Cut image dataset", "", "GRIM image dataset (*.npz)")
        if not path:
            return
        self._busy(True)
        self.runner.submit(lambda cancel: load_image(path, cancel), self._loaded)

    def _loaded(self, image, error):
        self._busy(False)
        if error is not None:
            self.message.setText(f"Open failed: {error}")
            return
        self.owner._mask_image_datasets.append(image)
        self._append(image)
        self.source.setCurrentIndex(len(self.sources)-1)
        self.message.setText("Opened image dataset. It can be cut again, including radar reconstruction when retained.")

    def _remove(self):
        image = self._current()
        if image is None:
            return
        self.owner._mask_image_datasets[:] = [item for item in self.owner._mask_image_datasets if item is not image]
        index = self.source.currentIndex()
        self.sources.pop(index)
        self.source.removeItem(index)
        self._show_source()

    def done(self, result):
        self.runner.close()
        super().done(result)

    def closeEvent(self, event):
        self.runner.close()
        super().closeEvent(event)


def open_mask_cut(owner):
    bar = getattr(owner, 'plot_slider', None)
    if bar is not None:
        bar.stop_interaction()
    previous = getattr(owner, "_mask_cut_dialog", None)
    if previous is not None:
        try:
            if previous.isVisible():
                previous.raise_()
                previous.activateWindow()
                return previous
        except RuntimeError:  # Qt already deleted a closed dialog.
            pass
    dialog = MaskCutDialog(owner)
    def forget_closed_dialog():
        if getattr(owner, "_mask_cut_dialog", None) is dialog:
            owner._mask_cut_dialog = None
    dialog.destroyed.connect(forget_closed_dialog)
    dialog.show()
    return dialog
