"""Edit azimuth sectors and frequency bands without live changes."""

from __future__ import annotations

import csv
import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.plotting.modes.sector_stats_mode import STATISTICS
from GRIM_Backend.ui.statistics_bands import StatisticsBandTable


def _number_text(value: float) -> str:
    return np.format_float_positional(float(value), unique=True, trim="-")


class SectorStatisticsDialog(QDialog):
    """Keep edits local until Apply or OK, then update all settings together."""

    def __init__(self, controls, azimuths, *, selected_azimuths=None,
                 unit="deg", frequency_unit="GHz", parent=None):
        super().__init__(parent)
        self.controls = controls
        self.unit = unit
        self.frequency_unit = frequency_unit
        self.azimuths = np.asarray(azimuths, dtype=float)
        self.selected_azimuths = np.asarray(
            azimuths if selected_azimuths is None else selected_azimuths, dtype=float
        )
        self.period = float(common.convert_axis_values([360], "azimuth", "deg", unit)[0])
        self.setWindowTitle("Sector Statistics")
        self.resize(650, 680)
        outer_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, 1)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Uniform sectors", "uniform")
        self.combo_mode.addItem("Custom sectors", "custom")
        mode_row = QFormLayout()
        mode_row.addRow("Sector layout", self.combo_mode)
        layout.addLayout(mode_row)

        self.check_selected_range = QCheckBox("Use the selected azimuth range")
        self.check_selected_range.setChecked(controls.sector_use_selected_azimuths())
        self.check_selected_range.setToolTip(
            "Unchecked: use all available azimuths. Checked: use only the azimuths "
            "selected in Plot Settings. Applies to plots and created datasets."
        )
        layout.addWidget(self.check_selected_range)

        self.combo_frequency_mode = QComboBox()
        self.combo_frequency_mode.addItem("Separate selected frequencies", "individual")
        self.combo_frequency_mode.addItem("Combine selected frequencies into one band", "selected")
        self.combo_frequency_mode.addItem("Custom frequency bands", "custom")
        self.combo_frequency_mode.setCurrentIndex(
            max(0, self.combo_frequency_mode.findData(controls.sector_frequency_mode()))
        )
        frequency_row = QFormLayout()
        frequency_row.addRow("Frequency selection", self.combo_frequency_mode)
        layout.addLayout(frequency_row)
        self.frequency_bands = StatisticsBandTable(f"Frequency bands ({frequency_unit})", checked=True)
        self.frequency_bands.setCheckable(False)
        self.frequency_bands.table.setHorizontalHeaderLabels(
            [f"Start ({frequency_unit})", f"Stop ({frequency_unit})"]
        )
        for start, stop in controls.sector_frequency_bands(unit=frequency_unit):
            self.frequency_bands.add_band(_number_text(start), _number_text(stop))
        self.frequency_bands.note.setText(
            "Add at least one band. Bands include both boundaries and may overlap."
        )
        layout.addWidget(self.frequency_bands)
        frequency_help = QLabel(
            "Bands use only frequencies selected in Plot Settings. Each band combines its "
            "frequency and sector samples into one linear-power population per elevation and polarization."
        )
        frequency_help.setWordWrap(True)
        layout.addWidget(frequency_help)

        file_row = QHBoxLayout()
        self.btn_load = QPushButton("Load sectors…")
        self.btn_save = QPushButton("Save sectors…")
        self.btn_load.setToolTip(
            "Load CSV start,stop columns in degrees into the custom table. Apply or OK uses the loaded sectors."
        )
        self.btn_save.setToolTip(
            "Save the current sector bounds as CSV start,stop columns in degrees, without applying changes."
        )
        file_row.addWidget(self.btn_load)
        file_row.addWidget(self.btn_save)
        file_row.addStretch()
        file_row.addWidget(QLabel("CSV angles use degrees."))
        layout.addLayout(file_row)
        self._csv_path = "sectors.csv"

        self.btn_uniform_setup = QPushButton("Fill from uniform sectors…")
        self.btn_uniform_setup.setCheckable(True)
        layout.addWidget(self.btn_uniform_setup)
        self.uniform_group = QGroupBox("Uniform sectors")
        uniform_layout = QFormLayout(self.uniform_group)
        self.edit_width = QLineEdit(_number_text(self.period / 12))
        self.check_custom_range = QCheckBox("Set explicit uniform start/stop bounds")
        finite = self.azimuths[np.isfinite(self.azimuths)]
        start, stop = (float(finite.min()), float(finite.max())) if finite.size else (-self.period / 2, self.period / 2)
        self.edit_start = QLineEdit(_number_text(start))
        self.edit_stop = QLineEdit(_number_text(stop))
        uniform_layout.addRow(f"Width ({unit})", self.edit_width)
        uniform_layout.addRow(self.check_custom_range)
        bounds = QHBoxLayout()
        bounds.addWidget(QLabel(f"Start ({unit})"))
        bounds.addWidget(self.edit_start)
        bounds.addWidget(QLabel(f"Stop ({unit})"))
        bounds.addWidget(self.edit_stop)
        uniform_layout.addRow(bounds)
        self.btn_fill = QPushButton("Fill custom table from these values")
        uniform_layout.addRow(self.btn_fill)
        layout.addWidget(self.uniform_group)

        self.custom_group = QGroupBox("Custom sectors")
        custom_layout = QVBoxLayout(self.custom_group)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([f"Start azimuth ({unit})", f"Stop azimuth ({unit})"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setMinimumHeight(160)
        custom_layout.addWidget(self.table)
        row_buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add sector")
        self.btn_remove = QPushButton("Remove selected")
        row_buttons.addWidget(self.btn_add)
        row_buttons.addWidget(self.btn_remove)
        row_buttons.addStretch()
        custom_layout.addLayout(row_buttons)
        help_label = QLabel(
            "Each row is independent: widths, gaps, and overlaps are allowed. "
            "A stop below the start wraps through the azimuth seam. "
            "Custom sectors include both boundaries, so adjacent sectors share their edge sample."
        )
        help_label.setWordWrap(True)
        custom_layout.addWidget(help_label)
        layout.addWidget(self.custom_group, 1)

        self.combo_statistic = QComboBox()
        for key, label in STATISTICS:
            self.combo_statistic.addItem(label, key)
        self.combo_statistic.setCurrentIndex(max(0, self.combo_statistic.findData(controls.sector_statistic())))
        self.combo_statistic.setToolTip("Statistic of linear power inside each sector, displayed in dB.")
        self.spin_percentile = QDoubleSpinBox()
        self.spin_percentile.setRange(0, 100)
        self.spin_percentile.setDecimals(3)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.setSuffix(" %")
        self.spin_percentile.setKeyboardTracking(False)
        self.spin_percentile.setValue(controls.sector_percentile())
        self.spin_percentile.setToolTip(
            "Used for the Percentile plot statistic and the percentile column in Copy Sector Table."
        )
        self.chk_interpolate = QCheckBox("Interpolate between percentile samples")
        self.chk_interpolate.setChecked(controls.sector_interpolate())
        self.chk_interpolate.setToolTip(
            "Checked: linearly interpolate between sorted linear-power samples. "
            "Unchecked: select an actual sample using the nearest-rank method. "
            "Applies to the Percentile plot and the percentile column in Copy Sector Table, "
            "even when another plot statistic is selected."
        )
        statistics_layout = QFormLayout()
        statistics_layout.addRow("Plot statistic", self.combo_statistic)
        statistics_layout.addRow("Percentile", self.spin_percentile)
        statistics_layout.addRow(self.chk_interpolate)
        layout.addLayout(statistics_layout)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c75050;")
        self.error_label.hide()
        outer_layout.addWidget(self.error_label)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        outer_layout.addWidget(self.button_box)
        self.btn_create_datasets = self.button_box.addButton('Create datasets', QDialogButtonBox.ActionRole)
        self.btn_create_datasets.setToolTip('Apply these settings and create one bounded dataset per sector, frequency band and selected input.')
        self.btn_create_datasets.clicked.connect(self.create_datasets)
        # Enter in a cell should finish that edit, never apply the whole dialog.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

        self._loaded_text = controls.sector_text(unit=unit)
        self._load_sectors(self._loaded_text)
        self._loaded_signature = self._sector_signature()
        self.combo_mode.currentIndexChanged.connect(self._update_mode)
        self.btn_uniform_setup.toggled.connect(self._update_uniform_visibility)
        self.check_selected_range.toggled.connect(self._update_range)
        self.check_custom_range.toggled.connect(self._update_range)
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_rows)
        self.btn_fill.clicked.connect(self._fill_table)
        self.btn_load.clicked.connect(self._load_csv)
        self.btn_save.clicked.connect(self._save_csv)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.table.itemChanged.connect(self._clear_error)
        self.frequency_bands.table.itemChanged.connect(self._clear_error)
        self.combo_frequency_mode.currentIndexChanged.connect(self._update_frequency_mode)
        for edit in (self.edit_width, self.edit_start, self.edit_stop):
            edit.textChanged.connect(self._clear_error)
        self._update_mode()
        self._update_range()
        self._update_frequency_mode()

    def _update_frequency_mode(self, *_args):
        self.frequency_bands.setVisible(self.combo_frequency_mode.currentData() == "custom")
        self._clear_error()

    def _load_sectors(self, text):
        parts = [part.strip() for part in str(text).replace(";", ",").split(",") if part.strip()]
        if len(parts) == 1 and ":" not in parts[0]:
            self.edit_width.setText(parts[0])
        elif len(parts) == 1 and parts[0].count(":") == 2:
            start, width, stop = parts[0].split(":")
            self.edit_start.setText(start.strip())
            self.edit_width.setText(width.strip())
            self.edit_stop.setText(stop.strip())
            self.check_custom_range.setChecked(True)
        else:
            self.combo_mode.setCurrentIndex(self.combo_mode.findData("custom"))
            for part in parts:
                bounds = part.split(":", 1)
                self._append_row(bounds[0].strip(), bounds[1].strip() if len(bounds) == 2 else "")
        if not self.table.rowCount():
            self._append_row("", "")

    def _update_mode(self, *_args):
        custom = self.combo_mode.currentData() == "custom"
        self.custom_group.setVisible(custom)
        self.btn_uniform_setup.setVisible(custom)
        self.btn_fill.setVisible(custom)
        self.uniform_group.setTitle("Fill table with uniform sectors" if custom else "Uniform sectors")
        self._update_uniform_visibility()
        self._clear_error()

    def _update_uniform_visibility(self, *_args):
        custom = self.combo_mode.currentData() == "custom"
        self.uniform_group.setVisible(not custom or self.btn_uniform_setup.isChecked())
        self.btn_uniform_setup.setText(
            "Hide uniform sector setup" if self.btn_uniform_setup.isChecked() else "Fill from uniform sectors…"
        )

    def _update_range(self, *_args):
        enabled = self.check_custom_range.isChecked()
        self.edit_start.setEnabled(enabled)
        self.edit_stop.setEnabled(enabled)
        if not enabled:
            values = self._population_azimuths()
            finite = values[np.isfinite(values)]
            start, stop = ((float(finite.min()), float(finite.max())) if finite.size
                           else (-self.period / 2, self.period / 2))
            self.edit_start.setText(_number_text(start))
            self.edit_stop.setText(_number_text(stop))
        self._clear_error()

    def _population_azimuths(self):
        return self.selected_azimuths if self.check_selected_range.isChecked() else self.azimuths

    def _append_row(self, start, stop):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(start)))
        self.table.setItem(row, 1, QTableWidgetItem(str(stop)))

    def _add_row(self):
        self._append_row("", "")
        self.table.setCurrentCell(self.table.rowCount() - 1, 0)
        self.table.editItem(self.table.currentItem())

    def _remove_rows(self):
        self._finish_editing()
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        if not rows and self.table.currentRow() >= 0:
            rows.add(self.table.currentRow())
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)
        self._clear_error()

    def _clear_error(self, *_args):
        self.error_label.clear()
        self.error_label.hide()

    def _show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()

    def _finish_editing(self):
        editor = QApplication.focusWidget()
        for table in (self.table, self.frequency_bands.table):
            if isinstance(editor, QLineEdit) and table.isAncestorOf(editor):
                table.commitData(editor)
                table.closeEditor(editor, QAbstractItemDelegate.NoHint)
        self.spin_percentile.interpretText()

    @staticmethod
    def _finite_number(text, label):
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label}: enter a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}: enter a finite number.")
        return value

    def _uniform_text(self):
        width = self.edit_width.text().strip()
        if self._finite_number(width, "Width") <= 0:
            raise ValueError("Width must be greater than zero.")
        if not self.check_custom_range.isChecked():
            return width
        start, stop = self.edit_start.text().strip(), self.edit_stop.text().strip()
        self._finite_number(start, "Start")
        self._finite_number(stop, "Stop")
        return f"{start}:{width}:{stop}"

    def _rows(self):
        return tuple(tuple(self.table.item(row, col).text().strip() if self.table.item(row, col) else ""
                           for col in range(2)) for row in range(self.table.rowCount()))

    def _sector_signature(self):
        if self.combo_mode.currentData() == "custom":
            return "custom", self._rows()
        explicit = self.check_custom_range.isChecked()
        return "uniform", self.edit_width.text(), explicit, ((self.edit_start.text(), self.edit_stop.text()) if explicit else ())

    def _custom_text(self):
        sectors = []
        for row, (start, stop) in enumerate(self._rows(), 1):
            if not start and not stop:
                continue
            first = self._finite_number(start, f"Row {row} start")
            last = self._finite_number(stop, f"Row {row} stop")
            if first == last:
                raise ValueError(f"Row {row}: start and stop must differ.")
            sectors.append(f"{start}:{stop}")
        if not sectors:
            raise ValueError("Add at least one sector with start and stop azimuths.")
        return ", ".join(sectors)

    def _fill_table(self):
        self._finish_editing()
        try:
            sectors = common.parse_sectors(self._uniform_text(), self._population_azimuths(), period=self.period)
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return
        self.table.setRowCount(0)
        for sector in sectors:
            self._append_row(_number_text(sector.start), _number_text(sector.start + sector.width))
        self.combo_mode.setCurrentIndex(self.combo_mode.findData("custom"))
        self._clear_error()

    @classmethod
    def _read_csv(cls, path):
        rows = []
        first_row = True
        with open(path, encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            for fields in reader:
                fields = [field.strip() for field in fields]
                if not any(fields):
                    continue
                if first_row and [field.lower() for field in fields] == ["start", "stop"]:
                    first_row = False
                    continue
                first_row = False
                line = reader.line_num
                if len(fields) != 2:
                    raise ValueError(f"CSV line {line}: expected two columns, start and stop.")
                start = cls._finite_number(fields[0], f"CSV line {line} start")
                stop = cls._finite_number(fields[1], f"CSV line {line} stop")
                if start == stop:
                    raise ValueError(f"CSV line {line}: start and stop must differ.")
                rows.append((start, stop))
        if not rows:
            raise ValueError("The CSV contains no sectors. Add start and stop values in degrees.")
        return rows

    @staticmethod
    def _convert_bounds(rows, from_unit, to_unit):
        with np.errstate(over="ignore", invalid="ignore"):
            values = common.convert_axis_values(rows, "azimuth", from_unit, to_unit)
        if not np.isfinite(values).all():
            raise ValueError(f"Sector bounds must remain finite when converted to {to_unit}.")
        return values

    def _load_csv(self):
        self._finish_editing()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load sectors (degrees)", self._csv_path, "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            rows = self._convert_bounds(self._read_csv(path), "deg", self.unit)
        except (OSError, UnicodeError, csv.Error, ValueError, OverflowError) as exc:
            self._show_error(f"Could not load sectors: {exc}")
            return
        # Replace the draft only after the whole file has been validated.
        self.table.setRowCount(0)
        for start, stop in rows:
            self._append_row(_number_text(start), _number_text(stop))
        self.combo_mode.setCurrentIndex(self.combo_mode.findData("custom"))
        self.btn_uniform_setup.setChecked(False)
        self._csv_path = path
        self._clear_error()

    def _save_csv(self):
        self._finish_editing()
        try:
            if self.combo_mode.currentData() == "custom":
                rows = [tuple(map(float, bounds.split(":"))) for bounds in self._custom_text().split(",")]
            else:
                sectors = common.parse_sectors(self._uniform_text(), self._population_azimuths(), period=self.period)
                rows = [(sector.start, sector.start + sector.width) for sector in sectors]
                if not rows:
                    raise ValueError("Add at least one sector.")
            rows = self._convert_bounds(rows, self.unit, "deg")
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sectors (degrees)", self._csv_path, "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as destination:
                writer = csv.writer(destination)
                writer.writerow(("start", "stop"))
                writer.writerows((_number_text(start), _number_text(stop)) for start, stop in rows)
        except (OSError, UnicodeError, csv.Error) as exc:
            self._show_error(f"Could not save sectors: {exc}")
            return
        self._csv_path = path
        self._clear_error()

    def apply_settings(self) -> bool:
        self._finish_editing()
        try:
            text = self._custom_text() if self.combo_mode.currentData() == "custom" else self._uniform_text()
            if self._sector_signature() == self._loaded_signature:
                text = self._loaded_text
            selected = self._population_azimuths()
            if (
                self.combo_mode.currentData() == "uniform"
                and not self.check_custom_range.isChecked()
                and not np.isfinite(selected).any()
            ):
                # A width can be configured before loading/selecting data.
                # Its range will come from all or selected azimuths when plotting.
                selected = [0.0]
            sectors = common.parse_sectors(text, selected, period=self.period)
            if not sectors:
                raise ValueError("Add at least one sector.")
            frequency_mode = self.combo_frequency_mode.currentData()
            frequency_bands = None
            if frequency_mode == "custom":
                bounds = self.frequency_bands.bounds()
                if not bounds:
                    raise ValueError("Add at least one frequency band with Start and Stop values.")
                frequency_bands = common.convert_axis_values(
                    bounds, "frequency", self.frequency_unit, "Hz"
                )
                if not np.isfinite(frequency_bands).all():
                    raise ValueError("Frequency band bounds must remain finite in Hz.")
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return False
        self.controls.set_sector_settings(
            text, str(self.combo_statistic.currentData()), float(self.spin_percentile.value()),
            unit=self.unit, interpolate=self.chk_interpolate.isChecked(),
            frequency_mode=frequency_mode, frequency_bands=frequency_bands,
            use_selected_azimuths=self.check_selected_range.isChecked(),
        )
        self._loaded_text = text
        self._loaded_signature = self._sector_signature()
        self._clear_error()
        return True

    def accept(self):
        if self.apply_settings():
            super().accept()

    def create_datasets(self):
        if self.apply_settings():
            callback = getattr(self.parent(), '_create_sector_statistics_datasets', None)
            if callback is not None:
                callback()
