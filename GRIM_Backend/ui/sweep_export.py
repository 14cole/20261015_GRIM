"""Sequential plot sweep export; capture only successfully published frames."""
from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from PIL import Image, ImageDraw
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QLabel, QMessageBox, QProgressDialog, QVBoxLayout)


from GRIM_Backend.ui.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def video_command(executable, folder, seconds, output):
    return [str(executable), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-framerate", f"{1/float(seconds):.12g}", "-start_number", "1",
            "-i", str(Path(folder) / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
            "-video_track_timescale", "1000000", "-movflags", "+faststart", str(output)]


def gif_command(executable, folder, seconds, output):
    """Per-frame palettes keep encoding bounded; GIF timing is in centiseconds."""
    delay = round(float(seconds) * 100)
    return [str(executable), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-framerate", f"{100/delay:.12g}", "-start_number", "1",
            "-i", str(Path(folder) / "frame_%06d.png"),
            "-filter_complex", "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1",
            "-loop", "0", "-final_delay", str(delay), str(output)]


class SweepExportDialog(QDialog):
    def __init__(self, bar, parent):
        super().__init__(parent)
        self.setWindowTitle("Export parameter sweep")
        self.resize(460, 250)
        layout = QVBoxLayout(self)
        note = QLabel("Export every slider position in order. Each frame waits for its plot to finish. "
                      "Images include a timing and parameter manifest.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.format = NoWheelComboBox()
        self.format.addItem("PNG image sequence", "images")
        self.format.addItem("MP4 video", "video")
        self.format.addItem("Animated GIF", "gif")
        form.addRow("Format", self.format)
        self.seconds = NoWheelDoubleSpinBox()
        self.seconds.setRange(.05, 3600.)
        self.seconds.setDecimals(2)
        self.seconds.setSuffix(" s")
        self.seconds.setValue(bar.seconds.value())
        self.format.currentIndexChanged.connect(
            lambda: self.seconds.setMaximum(655.35 if self.format.currentData() == "gif" else 3600.))
        form.addRow("Seconds per frame", self.seconds)
        self.stride = NoWheelSpinBox()
        self.stride.setRange(1, max(1, len(bar._frames)))
        self.stride.setToolTip("Export every Nth slider position; always include the final position.")
        form.addRow("Export every Nth position", self.stride)
        self.summary = QLabel()
        form.addRow(self.summary)
        def update_summary():
            count = len(export_positions(len(bar._frames), self.stride.value()))
            self.summary.setText(f"{count:,} frames · {count*self.seconds.value():g} seconds")
        self.seconds.valueChanged.connect(update_summary)
        self.stride.valueChanged.connect(update_summary)
        update_summary()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Choose folder…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def export_positions(count, stride):
    positions = list(range(0, count, stride))
    if positions and positions[-1] != count-1:
        positions.append(count-1)
    return positions


def start_sweep_export(window):
    if getattr(window, "_sweep_export", None) is not None:
        return
    bar = window.plot_slider
    bar.stop_interaction()
    window._refresh_plot_slider()
    from GRIM_Backend.plotting.slider import slider_axis_error
    reason = slider_axis_error(window.last_plot_mode, bar.axis())
    if reason:
        window.status.showMessage(reason)
        return
    if window.last_plot_mode is None or not getattr(bar, "_frames", None):
        window.status.showMessage("Plot a dataset and choose valid slider positions before exporting.")
        return
    context = window._plot_contexts[window._active_plot_tab]
    if (getattr(window, "_slider_request", None) or getattr(window, "_isar_busy", False)
            or (context.session.runner is not None and context.session.runner.busy)):
        window.status.showMessage("Wait for the current plot to finish before exporting the sweep.")
        return
    if window._button_checked(getattr(window, "btn_hold", None)):
        window.status.showMessage("Turn off Hold before exporting a sweep so each frame has its own plot.")
        return
    dialog = SweepExportDialog(bar, window)
    if dialog.exec() != QDialog.Accepted:
        return
    kind = dialog.format.currentData()
    executable = ffmpeg_executable() if kind in ("video", "gif") else None
    if kind in ("video", "gif") and executable is None:
        QMessageBox.information(window, "Animation encoder unavailable",
            "MP4 and GIF export require FFmpeg or the imageio-ffmpeg package included in the GRIM setup requirements. "
            "PNG image export is available without a video encoder.")
        return
    parent = QFileDialog.getExistingDirectory(window, "Choose sweep export folder")
    if not parent:
        return
    folder = Path(parent) / f"sweep_{uuid.uuid4().hex[:12]}"
    try:
        exporter = SweepExporter(window, folder, kind=kind, seconds=dialog.seconds.value(),
                                 stride=dialog.stride.value(), executable=executable, notify=True)
        exporter.start()
    except Exception as exc:
        QMessageBox.warning(window, "Sweep export failed", str(exc))


class SweepExporter(QObject):
    def __init__(self, window, folder, *, kind="images", seconds=.35, stride=1, executable=None, notify=False):
        super().__init__(window)
        from GRIM_Backend.plotting.slider import slider_axis_error
        reason = slider_axis_error(window.last_plot_mode, window.plot_slider.axis())
        if reason:
            raise ValueError(reason)
        if kind not in ("images", "video", "gif") or not .05 <= seconds <= (655.35 if kind == "gif" else 3600) or stride < 1:
            raise ValueError("Invalid sweep export settings.")
        self.window, self.bar = window, window.plot_slider
        self.folder = Path(folder)
        self.kind, self.seconds, self.executable = kind, float(seconds), executable
        if kind == "gif":
            self.seconds = round(self.seconds * 100) / 100
        self.notify = notify
        self.frames = self.bar._frames
        self.positions = export_positions(len(self.frames), stride)
        self.axis, self.mode = self.bar.axis(), window.last_plot_mode
        self.figure = window.plot_figure
        self.dataset_names = [name for name, _dataset in window._selected_datasets()]
        self.selections = {axis: window._selected_indices(window._slider_list(axis))
                           for axis in ("azimuth", "frequency", "elevation")}
        self.filters = [(window.chk_isar_freq_band, window.chk_isar_freq_band.isChecked()),
                        (window.chk_isar_aperture, window.chk_isar_aperture.isChecked())]
        self.original_position = self.bar.slider.value()
        self.index = 0
        self.records = []
        self.finished = False
        self.process = None
        self.error_file = None
        self.progress = QProgressDialog("Preparing sweep…", "Cancel", 0, len(self.positions), window)
        self.progress.setWindowTitle("Export parameter sweep")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setAutoClose(False)
        self.progress.setAutoReset(False)
        self.progress.setMinimumDuration(0)
        self.progress.canceled.connect(self.cancel)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll_encoder)

    def start(self):
        self.folder.mkdir(parents=False, exist_ok=False)
        self.frame_folder = self.folder if self.kind == "images" else self.folder / "frames"
        if self.kind in ("video", "gif"):
            self.frame_folder.mkdir()
        self._write_manifest("running")
        self.window._sweep_export = self
        self.bar.stop_interaction()
        timer = getattr(self.window, "_autoplot_timer", None)
        if timer is not None:
            timer.stop()
        self.progress.show()
        QTimer.singleShot(0, self._next_frame)

    def _next_frame(self):
        if self.finished:
            return
        if self.window.plot_figure is not self.figure or self.window.last_plot_mode != self.mode:
            self._finish("failed", "The active plot changed during export.")
            return
        if self.index >= len(self.positions):
            if self.kind in ("video", "gif"):
                self._encode()
            else:
                self._finish("complete")
            return
        self.progress.setLabelText(f"Rendering frame {self.index+1} / {len(self.positions)}…")
        self.window._on_plot_slider_moved(self.positions[self.index])

    def frame_finished(self, ok):
        # Completion can be synchronous inside the renderer; defer capture
        # until limits, markers and pending draw calls have finished updating.
        detail = self.window.status.currentMessage()
        QTimer.singleShot(0, lambda: self._capture(ok, detail))

    def _capture(self, ok, detail):
        if self.finished:
            return
        if not ok:
            self._finish("failed", f"Frame {self.index+1}: {detail}")
            return
        try:
            position = self.positions[self.index]
            label = self.frames.label(position)
            data = io.BytesIO()
            self.figure.savefig(data, format="png", dpi=self.figure.dpi)
            data.seek(0)
            with Image.open(data) as source:
                image = Image.new("RGB", (source.width, source.height+32), "white")
                image.paste(source.convert("RGB"), (0, 0))
                ImageDraw.Draw(image).text((10, source.height+8),
                    f"{self.axis}: {label} | frame {self.index+1}/{len(self.positions)} | {self.seconds:g} s/frame",
                    fill="black")
                filename = f"frame_{self.index+1:06d}.png"
                image.save(self.frame_folder / filename)
            self.records.append({"file": str((self.frame_folder / filename).relative_to(self.folder)),
                "position": position, "label": label, "bounds": self.frames.bounds[position].tolist(),
                "sample_rows": self.frames.selection(position), "seconds": self.seconds})
            self.index += 1
            self.progress.setValue(self.index)
            QTimer.singleShot(0, self._next_frame)
        except Exception as exc:
            self._finish("failed", str(exc))

    def _encode(self):
        try:
            if not self.executable:
                raise ValueError("No FFmpeg encoder is available.")
            self.progress.setLabelText("Encoding animated GIF…" if self.kind == "gif" else "Encoding MP4 video…")
            self.progress.setRange(0, 0)
            self.error_file = open(self.folder / "encoder.log", "wb")
            command = gif_command if self.kind == "gif" else video_command
            output = "sweep.gif" if self.kind == "gif" else "sweep.mp4"
            self.process = subprocess.Popen(command(self.executable, self.frame_folder,
                self.seconds, self.folder / output), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=self.error_file,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.timer.start()
        except Exception as exc:
            self._finish("failed", str(exc))

    def _poll_encoder(self):
        code = self.process.poll()
        if code is not None:
            self._finish("complete" if code == 0 else "failed",
                         "" if code == 0 else "Animation encoding failed; see encoder.log. PNG frames are retained.")

    def _write_manifest(self, status, detail=""):
        manifest = {"format": "grim-plot-sweep", "version": 1, "status": status, "detail": detail,
            "plot_mode": self.mode, "axis": self.axis, "unit": self.frames.unit,
            "datasets": self.dataset_names,
            "initial_sample_rows": {key: sorted(rows) for key, rows in self.selections.items()},
            "band_width": self.bar.band_width.value() if self.frames.band else None,
            "band_step": ("custom" if self.frames.advance is not None else "bandwidth" if self.frames.step_by_width else "acquired") if self.frames.band else None,
            "window_advance": self.frames.advance if self.frames.advance is not None else (self.bar.band_width.value() if self.frames.step_by_width else None),
            "seconds_per_frame": self.seconds, "requested_positions": self.positions,
            "frames": self.records, "video": "sweep.mp4" if self.kind == "video" and status == "complete" else None,
            "gif": "sweep.gif" if self.kind == "gif" and status == "complete" else None}
        (self.folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def cancel(self):
        self._finish("canceled", "Export canceled; completed PNG frames are retained.")

    def _finish(self, status, detail=""):
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.error_file is not None:
            self.error_file.close()
        try:
            self._write_manifest(status, detail)
        except OSError as exc:
            status, detail = "failed", f"Could not save sweep manifest: {exc}"
        self.window._sweep_export = None
        self.window._slider_frame_finished(False)
        self.window._invalidate_isar_result()
        for axis, rows in self.selections.items():
            widget = self.window._slider_list(axis)
            blocked = widget.blockSignals(True)
            widget.clearSelection()
            for row in rows:
                if widget.item(row) is not None:
                    widget.item(row).setSelected(True)
            widget.blockSignals(blocked)
        for widget, value in self.filters:
            blocked = widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(blocked)
        self.bar.set_positions(self.bar._labels, self.original_position)
        self.progress.close()
        self.window._render_plot_mode(self.mode)
        self.window.status.showMessage(f"Sweep export {status}: {self.folder}. {detail}")
        if self.notify:
            message = QMessageBox(self.window)
            message.setWindowTitle(f"Sweep export {status}")
            message.setText(f"{len(self.records)} frames saved to:\n{self.folder}\n\n{detail}")
            message.setAttribute(Qt.WA_DeleteOnClose)
            message.open()
        self.progress.deleteLater()
        self.deleteLater()
