"""Plotting-tab analysis settings and the parameter slider bar."""

from __future__ import annotations

import math
from copy import deepcopy

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.ui.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSlider
from GRIM_Backend.ui.widgets import PlotNumberSpin


def _percent_spin(value: float, tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 100.0)
    spin.setDecimals(1)
    spin.setSingleStep(5.0)
    spin.setSuffix(" %")
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    spin.setToolTip(tooltip)
    return spin


class PlotAnalysisControls(QObject):
    """Sweep, CDF, sector-statistics, and range-frequency settings.

    ``changed`` carries which feature changed ("cdf", "sector",
    "range", "sweep") so only the matching plot type re-renders.
    """

    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_cdf = QComboBox()
        self.combo_cdf.addItem("Cumulative: samples at or below level", "cdf")
        self.combo_cdf.addItem("Exceedance: samples at or above level", "exceedance")

        self.combo_sweep_aggregation = QComboBox()
        for text, key in (("Individual azimuth cuts", "individual"),
                          ("Mean linear power", "mean"), ("Median (P50)", "median"),
                          ("Percentile of linear power", "percentile"), ("Worst case (maximum)", "max")):
            self.combo_sweep_aggregation.addItem(text, key)
        self.combo_sweep_aggregation.setCurrentIndex(self.combo_sweep_aggregation.findData("median"))
        self.combo_sweep_aggregation.setToolTip(
            "Frequency and Elevation Sweep reduce the selected azimuths at each point. "
            "Mean/median/percentile/max use valid linear power, including zero, before dB conversion. "
            "Individual cuts keep every azimuth separate. Legend n is valid/selected azimuth samples per point (a range when coverage varies). These are sample statistics, not coherent-field averages. "
            "In Phase, mean/median are circular; percentile/max show the phase of the nearest-rank/strongest power observation."
        )
        self.spin_sweep_percentile = _percent_spin(90., "Percentile across the selected azimuth samples, in linear power.")
        self.chk_sweep_interpolate = QCheckBox("Interpolate between percentile samples")
        self.chk_sweep_interpolate.setChecked(False)
        self.chk_sweep_interpolate.setToolTip(
            "Magnitude percentiles: checked interpolates between sorted linear powers; "
            "unchecked selects a nearest-rank power sample. Median always averages the "
            "middle two powers for an even sample count. Phase percentiles always show "
            "the phase of a nearest-rank power sample."
        )
        self.combo_sweep_aggregation.currentIndexChanged.connect(self._sweep_changed)
        self.spin_sweep_percentile.valueChanged.connect(lambda: self.changed.emit("sweep"))
        self.chk_sweep_interpolate.toggled.connect(lambda: self.changed.emit("sweep"))
        self._sweep_changed(emit=False)

        self.btn_cdf_settings = QPushButton("Population / threshold…")
        self._build_cdf_dialog(parent)
        self.btn_cdf_settings.clicked.connect(self._show_cdf_dialog)

        self._sector_statistic = "mean"
        self._sector_percentile = 90.0
        self._sector_interpolate = False
        self._sector_use_selected_azimuths = False
        self._sector_boxes = []  # Draft defaults are shown in the box editor before Apply.

        self.spin_range_subband = QDoubleSpinBox()
        self.spin_range_subband.setRange(5.0, 100.0)
        self.spin_range_subband.setDecimals(0)
        self.spin_range_subband.setSingleStep(5.0)
        self.spin_range_subband.setSuffix(" % of band")
        self.spin_range_subband.setValue(25.0)
        self.spin_range_subband.setKeyboardTracking(False)
        self.spin_range_subband.setToolTip(
            "Width of each sliding window within the frequencies you select. "
            "The plot shows each window's center frequency. Narrower windows "
            "show more frequency detail but coarser down-range resolution. "
            "100% uses the entire selection for one range profile."
        )
        self.combo_range_window = QComboBox()
        self.combo_range_window.addItems(
            ["Hanning", "Hamming", "Blackman", "Blackman-Harris", "Kaiser β=15", "Rectangular"]
        )
        self.combo_range_window.setToolTip("Window applied to each sub-band before the inverse FFT.")
        self.combo_range_unit = QComboBox()
        self.combo_range_unit.addItems(["m", "cm", "mm", "in", "ft"])
        self.combo_range_unit.setCurrentText("in")

        self.spin_range_subband.valueChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_window.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.combo_range_unit.currentIndexChanged.connect(lambda: self.changed.emit("range"))
        self.combo_cdf.currentIndexChanged.connect(self._cdf_options_changed)

    def cdf_exceedance(self) -> bool:
        return self.combo_cdf.currentData() == "exceedance"

    def _sweep_changed(self, *_args, emit=True):
        percentile = self.combo_sweep_aggregation.currentData() == "percentile"
        self.spin_sweep_percentile.setEnabled(percentile)
        self.chk_sweep_interpolate.setEnabled(percentile)
        if emit:
            self.changed.emit("sweep")

    def sweep_aggregation(self):
        return {"mode": self.combo_sweep_aggregation.currentData(),
                "percentile": float(self.spin_sweep_percentile.value()),
                "interpolate": self.chk_sweep_interpolate.isChecked()}

    def _build_cdf_dialog(self, parent):
        self.cdf_dialog = QDialog(parent if isinstance(parent, QWidget) else None)
        self.cdf_dialog.setWindowTitle("CDF population and threshold lookup")
        self.cdf_dialog.resize(840, 580)
        layout = QVBoxLayout(self.cdf_dialog)
        definition = QLabel(
            "Each dataset/run and polarization remains a separate population. Bands and sectors subdivide "
            "the selected coordinates; no runs or unlike units are pooled. Valid zero power remains included; "
            "missing, nonfinite and negative power is excluded.\n\n"
            "Equal samples gives every observation one vote. Quadrature weights approximate uniform measure "
            "over each group's selected coordinate hull (trapezoidal endpoint weights). This can bridge "
            "unsampled gaps; it is not a dwell-time, solid-angle or independent-trial probability. "
            "Singleton axes have unit weight. Coverage reports valid weight / selected weight."
        )
        definition.setWordWrap(True)
        layout.addWidget(definition)
        form = QFormLayout()
        self.combo_cdf_grouping = QComboBox()
        for text, key in (("Per run: all selected coordinates", "run"),
                          ("Per run and frequency band", "band"),
                          ("Per run and azimuth sector", "sector"),
                          ("Per run, frequency band and sector", "band_sector")):
            self.combo_cdf_grouping.addItem(text, key)
        form.addRow("Groups", self.combo_cdf_grouping)
        self.spin_cdf_bands = QSpinBox()
        self.spin_cdf_bands.setRange(1, 1000)
        self.spin_cdf_bands.setValue(4)
        self.spin_cdf_bands.setToolTip("Equal-width frequency bands across the selected reference-frequency extent; final band includes its upper edge.")
        form.addRow("Frequency bands", self.spin_cdf_bands)
        self.spin_cdf_sector_width = QDoubleSpinBox()
        self.spin_cdf_sector_width.setRange(.1, 360.)
        self.spin_cdf_sector_width.setDecimals(3)
        self.spin_cdf_sector_width.setValue(30.)
        self.spin_cdf_sector_width.setSuffix(" deg")
        self.spin_cdf_sector_width.setToolTip("Sector width measured from the lowest selected reference azimuth. Angular units are converted explicitly.")
        form.addRow("Azimuth sector width", self.spin_cdf_sector_width)
        self.combo_cdf_weighting = QComboBox()
        for text, key in (("Equal samples", "samples"), ("Uniform azimuth measure", "azimuth"),
                          ("Uniform frequency measure", "frequency"),
                          ("Uniform azimuth × frequency measure", "azimuth_frequency"),
                          ("Uniform azimuth × elevation × frequency measure", "all_axes")):
            self.combo_cdf_weighting.addItem(text, key)
        form.addRow("Weighting", self.combo_cdf_weighting)
        self.chk_cdf_threshold = QCheckBox("Show threshold and lookup results")
        form.addRow(self.chk_cdf_threshold)
        self.spin_cdf_threshold = QDoubleSpinBox()
        self.spin_cdf_threshold.setRange(-1.e12, 1.e12)
        self.spin_cdf_threshold.setDecimals(9)
        self.spin_cdf_threshold.setKeyboardTracking(False)
        self.spin_cdf_threshold.setToolTip("Numeric threshold in each curve's displayed unit. The table identifies that unit; mixed dB conventions are not physically interchangeable.")
        form.addRow("Threshold (current display unit)", self.spin_cdf_threshold)
        layout.addLayout(form)
        self.cdf_results = QPlainTextEdit()
        self.cdf_results.setReadOnly(True)
        self.cdf_results.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.cdf_results.setPlaceholderText("Plot a CDF to inspect population coverage and inclusive threshold probabilities.")
        layout.addWidget(self.cdf_results, 1)
        copy = QPushButton("Copy lookup table")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.cdf_results.toPlainText()))
        layout.addWidget(copy)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.cdf_dialog.hide)
        layout.addWidget(buttons)
        for widget in (self.combo_cdf_grouping, self.combo_cdf_weighting):
            widget.currentIndexChanged.connect(self._cdf_options_changed)
        for widget in (self.spin_cdf_bands, self.spin_cdf_sector_width, self.spin_cdf_threshold):
            widget.valueChanged.connect(self._cdf_options_changed)
        self.chk_cdf_threshold.toggled.connect(self._cdf_options_changed)
        self._cdf_options_changed(emit=False)

    def _show_cdf_dialog(self):
        self.cdf_dialog.show()
        self.cdf_dialog.raise_()

    def _cdf_options_changed(self, *_args, emit=True):
        self.cdf_results.clear()
        grouping = self.combo_cdf_grouping.currentData()
        self.spin_cdf_bands.setEnabled(grouping in ("band", "band_sector"))
        self.spin_cdf_sector_width.setEnabled(grouping in ("sector", "band_sector"))
        self.spin_cdf_threshold.setEnabled(self.chk_cdf_threshold.isChecked())
        if emit:
            self.changed.emit("cdf")

    def cdf_options(self):
        return {"exceedance": self.cdf_exceedance(), "grouping": self.combo_cdf_grouping.currentData(),
                "band_count": self.spin_cdf_bands.value(), "sector_width_deg": self.spin_cdf_sector_width.value(),
                "weighting": self.combo_cdf_weighting.currentData(),
                "threshold_enabled": self.chk_cdf_threshold.isChecked(), "threshold": self.spin_cdf_threshold.value()}

    def set_cdf_results(self, text):
        self.cdf_results.setPlainText(str(text))

    def analysis_options(self):
        return {"sweep": self.sweep_aggregation(), "cdf": self.cdf_options()}

    def restore_analysis_options(self, options):
        options = options if isinstance(options, dict) else {}
        sweep, cdf = options.get("sweep", {}), options.get("cdf", {})
        sweep = sweep if isinstance(sweep, dict) else {}
        cdf = cdf if isinstance(cdf, dict) else {}
        def number(mapping, key, default):
            try:
                value = float(mapping.get(key, default))
                return value if math.isfinite(value) else default
            except (ValueError, TypeError, OverflowError):
                return default
        blocked = self.blockSignals(True)
        try:
            for widget, value in ((self.combo_sweep_aggregation, sweep.get("mode", "median")),
                                  (self.combo_cdf, "exceedance" if cdf.get("exceedance", False) else "cdf"),
                                  (self.combo_cdf_grouping, cdf.get("grouping", "run")),
                                  (self.combo_cdf_weighting, cdf.get("weighting", "samples"))):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            self.spin_sweep_percentile.setValue(number(sweep, "percentile", 90.))
            self.chk_sweep_interpolate.setChecked(bool(sweep.get("interpolate", False)))
            self.spin_cdf_bands.setValue(int(number(cdf, "band_count", 4)))
            self.spin_cdf_sector_width.setValue(number(cdf, "sector_width_deg", 30.))
            self.chk_cdf_threshold.setChecked(bool(cdf.get("threshold_enabled", False)))
            self.spin_cdf_threshold.setValue(number(cdf, "threshold", 0.))
        finally:
            self.blockSignals(blocked)

    def sector_statistic(self) -> str:
        return self._sector_statistic

    def sector_percentile(self) -> float:
        return self._sector_percentile

    def sector_interpolate(self) -> bool:
        return self._sector_interpolate

    def sector_use_selected_azimuths(self) -> bool:
        return self._sector_use_selected_azimuths

    def sector_boxes(self):
        return deepcopy(self._sector_boxes)

    def set_sector_settings(self, *, boxes, statistic="mean", percentile=90.,
                            interpolate=False, use_selected_azimuths=False):
        """Apply one explicit set of paired azimuth/frequency boxes."""
        from GRIM_Backend.datasets.statistics_boxes import normalize_boxes
        if statistic not in ("mean", "median", "min", "max", "percentile"):
            raise ValueError("Unknown box statistic.")
        percentile = float(percentile)
        if not math.isfinite(percentile) or not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100.")
        values = (normalize_boxes(boxes), statistic, percentile, bool(interpolate),
                  bool(use_selected_azimuths))
        current = (self._sector_boxes, self._sector_statistic, self._sector_percentile,
                   self._sector_interpolate, self._sector_use_selected_azimuths)
        if values != current:
            (self._sector_boxes, self._sector_statistic, self._sector_percentile,
             self._sector_interpolate, self._sector_use_selected_azimuths) = values
            self.changed.emit("sector")

    def range_subband_percent(self) -> float:
        return float(self.spin_range_subband.value())

    def range_window(self) -> str:
        return self.combo_range_window.currentText()

    def range_unit(self) -> str:
        return self.combo_range_unit.currentText()


class PlotSliderBar(QFrame):
    """Axis choice, slider, step buttons, and Play for scrubbing one axis."""

    moved = Signal(int)
    axis_changed = Signal(str)
    shown = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("plotSliderBar")
        self.combo_axis = NoWheelComboBox()
        for label, axis in (("Frequency", "frequency"), ("Elevation", "elevation"),
                            ("Azimuth", "azimuth")):
            self.combo_axis.addItem(label, axis)
        self.combo_axis.setToolTip("Parameter list the slider steps through.")
        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setPageStep(1)
        self.btn_prev = QToolButton()
        self.btn_prev.setArrowType(Qt.LeftArrow)
        self.btn_prev.setToolTip("Previous value")
        self.btn_next = QToolButton()
        self.btn_next.setArrowType(Qt.RightArrow)
        self.btn_next.setToolTip("Next value")
        self.value_label = QLabel("--")
        self.value_label.setMinimumWidth(90)
        self.btn_play = QToolButton(text="Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setToolTip("Step through every value, wrapping at the end.")
        self.band_label = QLabel("Bandwidth")
        self.band_width = PlotNumberSpin()
        self.band_width.setDecimals(9)
        self.band_width.setRange(0.000000001, 1.e15)
        self.band_width.setKeyboardTracking(False)
        self.band_width.setToolTip("Width of the moving band in the displayed axis units. Uses acquired samples only; windows with fewer than two samples are omitted. The slider replaces the matching Plot Settings filter.")
        self.step_mode = NoWheelComboBox()
        self.step_mode.addItem("Adjacent windows (advance = width)", "width")
        self.step_mode.addItem("Custom window advance", "custom")
        self.step_mode.addItem("Advance one acquired sample", "acquired")
        self.advance = NoWheelDoubleSpinBox()
        self.advance.setRange(1.e-9, 1.e15)
        self.advance.setDecimals(9)
        self.advance.setKeyboardTracking(False)
        self.advance.setToolTip("Distance between window starts, in axis units. Smaller than width overlaps; larger leaves gaps.")
        self.sequence_note = QLabel()
        self.sequence_note.setWordWrap(True)
        self.sequence_note.setTextFormat(Qt.PlainText)
        self.seconds = NoWheelDoubleSpinBox()
        self.seconds.setRange(.05, 3600.)
        self.seconds.setDecimals(2)
        self.seconds.setValue(.35)
        self.seconds.setSuffix(" s")
        self.seconds.setKeyboardTracking(False)
        self.seconds.setToolTip("Minimum seconds between GUI frames. Slow calculations finish before advancing. Export timing is chosen separately.")
        self.btn_export = QPushButton("Export sweep…")
        self.btn_export.clicked.connect(self.export_requested)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        layout = QHBoxLayout()
        outer.addLayout(layout)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Slider"))
        layout.addWidget(self.combo_axis)
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.value_label)
        layout.addWidget(self.btn_play)
        options = QHBoxLayout()
        outer.addLayout(options)
        options.addWidget(self.band_label)
        options.addWidget(self.band_width)
        options.addWidget(self.step_mode)
        options.addWidget(self.advance)
        options.addStretch(1)
        options.addWidget(QLabel("Seconds / frame"))
        options.addWidget(self.seconds)
        options.addWidget(self.btn_export)
        self.band_label.hide()
        self.band_width.hide()
        self.step_mode.hide()
        self.advance.hide()
        outer.addWidget(self.sequence_note)
        self._busy = False

        self._labels: list[str] = []
        # Coalesce drags and key repeats into one render per settled value.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(lambda: self.moved.emit(self.slider.value()))
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(350)
        self._play_timer.timeout.connect(self._play_step)
        self.seconds.valueChanged.connect(lambda value: self._play_timer.setInterval(round(value * 1000)))
        self.band_width.valueChanged.connect(self._band_changed)
        self.step_mode.currentIndexChanged.connect(self._band_changed)
        self.advance.valueChanged.connect(self._band_changed)
        self.slider.valueChanged.connect(self._value_changed)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_play.toggled.connect(self._play_toggled)
        self.combo_axis.currentIndexChanged.connect(
            lambda: self.axis_changed.emit(self.axis())
        )
        self.hide()

    def axis(self) -> str:
        return str(self.combo_axis.currentData())

    def set_positions(self, labels: list[str], current: int) -> None:
        """Replace the slider positions without emitting a move."""
        self._labels = labels
        blocked = self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(self._labels) - 1))
        self.slider.setValue(max(0, min(int(current), len(self._labels) - 1)))
        self.slider.blockSignals(blocked)
        self.slider.setEnabled(len(self._labels) > 1)
        for button in (self.btn_prev, self.btn_next, self.btn_play):
            button.setEnabled(len(self._labels) > 1)
        self.btn_export.setEnabled(bool(len(self._labels)))
        self._show_label(self.slider.value())

    def step(self, delta: int, *, wrap: bool = False) -> None:
        count = self.slider.maximum() + 1
        if count <= 1:
            return
        value = self.slider.value() + int(delta)
        value = value % count if wrap else max(0, min(value, count - 1))
        self.slider.setValue(value)

    def stop_play(self) -> None:
        self.btn_play.setChecked(False)

    def stop_interaction(self) -> None:
        """End playback and any delayed scrub when its canvas is hidden."""
        self.stop_play()
        self._debounce.stop()

    def _play_step(self):
        if not self._busy and not self._debounce.isActive():
            self.step(1, wrap=True)

    def _band_changed(self):
        self.stop_play()
        self.axis_changed.emit(self.axis())
        self._debounce.start()

    def render_started(self):
        self._busy = True

    def render_finished(self, ok=True):
        self._busy = False
        if not ok:
            self.stop_play()
        elif self.btn_play.isChecked():
            self._play_timer.start()

    def _value_changed(self, value: int) -> None:
        self._show_label(value)
        self._debounce.start()

    def _show_label(self, value: int) -> None:
        self.value_label.setText(
            self._labels[value] if 0 <= value < len(self._labels) else "--"
        )

    def _play_toggled(self, checked: bool) -> None:
        self.btn_play.setText("Pause" if checked else "Play")
        if checked:
            self._play_timer.start()
        else:
            self._play_timer.stop()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Lists can change while the bar is hidden (another tab, another row).
        self.shown.emit()

    def hideEvent(self, event) -> None:
        self.stop_play()
        super().hideEvent(event)
