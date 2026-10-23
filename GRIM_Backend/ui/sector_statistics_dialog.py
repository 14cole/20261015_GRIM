"""Edit explicit azimuth/frequency boxes without live changes."""
from __future__ import annotations

import csv
import numpy as np
from PySide6.QtWidgets import (
    QAbstractItemDelegate, QAbstractItemView, QApplication, QCheckBox,
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from GRIM_Backend.datasets.statistics_boxes import normalize_boxes
from GRIM_Backend.plotting.modes import common
from GRIM_Backend.plotting.modes.sector_stats_mode import STATISTICS
from GRIM_Backend.ui.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox

CSV_COLUMNS = ('name', 'azimuth_start_deg', 'azimuth_stop_deg',
               'frequency_start_hz', 'frequency_stop_hz')


def _number_text(value):
    return np.format_float_positional(float(value), unique=True, trim='-')


class SectorStatisticsDialog(QDialog):
    """Each table row creates one dataset per selected input."""

    def __init__(self, controls, azimuths, *, selected_azimuths=None,
                 frequencies=(), unit='deg', frequency_unit='GHz', parent=None):
        super().__init__(parent)
        self.controls = controls
        self.unit, self.frequency_unit = unit, frequency_unit
        self.azimuths = np.asarray(azimuths, dtype=float)
        self.selected_azimuths = np.asarray(
            azimuths if selected_azimuths is None else selected_azimuths, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.period = float(common.convert_axis_values([360], 'azimuth', 'deg', unit)[0])
        self._csv_path = 'statistics_boxes.csv'
        self.setWindowTitle('Sector Statistics — Boxes')
        self.resize(850, 620)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        note = QLabel(
            'Define one box per row: an azimuth band paired with a frequency band. '
            'Create datasets makes one dataset per box for each selected input. '
            'The statistic combines samples inside that box, separately for each elevation and polarization.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.check_selected_range = QCheckBox('Use the selected azimuth range')
        self.check_selected_range.setChecked(controls.sector_use_selected_azimuths())
        self.check_selected_range.setToolTip(
            'Unchecked: use all acquired azimuths inside each box. Checked: also restrict '
            'each box to the azimuths selected in Plot Settings.')
        layout.addWidget(self.check_selected_range)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            'Box name', f'Azimuth start ({unit})', f'Azimuth stop ({unit})',
            f'Frequency start ({frequency_unit})', f'Frequency stop ({frequency_unit})'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setMinimumHeight(200)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.btn_add = QPushButton('Add box')
        self.btn_remove = QPushButton('Remove selected boxes')
        self.btn_load = QPushButton('Load boxes…')
        self.btn_save = QPushButton('Save boxes…')
        for button in (self.btn_add, self.btn_remove, self.btn_load, self.btn_save):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        help_text = QLabel(
            'Each box uses all acquired frequencies within its own bounds, regardless of the '
            'Plot Settings frequency selection. Both boundaries are included; boxes may overlap. '
            'Azimuth stop below start wraps through the seam. Equal frequency bounds select a single frequency. '
            'Add box starts with the full available ranges (or selected azimuth range when checked). '
            'CSV files store complete boxes in degrees and Hz.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.combo_statistic = NoWheelComboBox()
        for key, label in STATISTICS:
            self.combo_statistic.addItem(label, key)
        self.combo_statistic.setCurrentIndex(max(0, self.combo_statistic.findData(controls.sector_statistic())))
        self.spin_percentile = NoWheelDoubleSpinBox()
        self.spin_percentile.setRange(0, 100)
        self.spin_percentile.setDecimals(3)
        self.spin_percentile.setSingleStep(5)
        self.spin_percentile.setSuffix(' %')
        self.spin_percentile.setKeyboardTracking(False)
        self.spin_percentile.setValue(controls.sector_percentile())
        self.spin_percentile.setToolTip('Used for the Percentile statistic and the copied table percentile column.')
        self.chk_interpolate = QCheckBox('Interpolate between percentile samples')
        self.chk_interpolate.setChecked(controls.sector_interpolate())
        self.chk_interpolate.setToolTip(
            'Unchecked: nearest rank selects an acquired sample. Checked: interpolate sorted linear-power values.')
        form = QFormLayout()
        form.addRow('Statistic', self.combo_statistic)
        form.addRow('Percentile', self.spin_percentile)
        form.addRow(self.chk_interpolate)
        layout.addLayout(form)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet('color: #c75050;')
        self.error_label.hide()
        outer.addWidget(self.error_label)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.btn_create_datasets = self.button_box.addButton('Create datasets', QDialogButtonBox.ActionRole)
        outer.addWidget(self.button_box)
        boxes = controls.sector_boxes()
        if boxes:
            self._load_boxes(boxes)
        else:
            self._add_row()
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_rows)
        self.btn_load.clicked.connect(self._load_csv)
        self.btn_save.clicked.connect(self._save_csv)
        self.table.itemChanged.connect(self._clear_error)
        self.check_selected_range.toggled.connect(self._clear_error)
        self.btn_create_datasets.clicked.connect(self.create_datasets)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _clear_error(self, *_args):
        self.error_label.clear()
        self.error_label.hide()

    def _show_error(self, message):
        self.error_label.setText(message)
        self.error_label.show()

    def _finish_editing(self):
        editor = QApplication.focusWidget()
        if isinstance(editor, QLineEdit) and self.table.isAncestorOf(editor):
            self.table.commitData(editor)
            self.table.closeEditor(editor, QAbstractItemDelegate.NoHint)
        self.spin_percentile.interpretText()

    def _append_row(self, values):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))

    def _add_row(self, *_args):
        az = self.selected_azimuths if self.check_selected_range.isChecked() else self.azimuths
        az, freq = az[np.isfinite(az)], self.frequencies[np.isfinite(self.frequencies)]
        az_bounds = [float(az.min()), float(az.max())] if az.size else [-self.period / 2, self.period / 2]
        if az_bounds[0] == az_bounds[1]:
            az_bounds[1] += self.period
        freq_bounds = [_number_text(freq.min()), _number_text(freq.max())] if freq.size else ['', '']
        names = {self.table.item(row, 0).text().strip() for row in range(self.table.rowCount())
                 if self.table.item(row, 0) is not None}
        index = 1
        while f'Box {index}' in names:
            index += 1
        self._append_row([f'Box {index}', *map(_number_text, az_bounds), *freq_bounds])
        self._clear_error()

    def _remove_rows(self):
        self._finish_editing()
        for row in sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True):
            self.table.removeRow(row)
        self._clear_error()

    def _boxes(self):
        boxes = []
        for row in range(self.table.rowCount()):
            cells = [self.table.item(row, col).text().strip() if self.table.item(row, col) else ''
                     for col in range(5)]
            try:
                az = common.convert_axis_values([float(v) for v in cells[1:3]], 'azimuth', self.unit, 'deg')
                freq = common.convert_axis_values([float(v) for v in cells[3:5]], 'frequency', self.frequency_unit, 'Hz')
            except (ValueError, OverflowError):
                raise ValueError(f'Box row {row + 1}: enter numeric azimuth and frequency bounds.') from None
            boxes.append(dict(name=cells[0], azimuth_band=az, frequency_band=freq))
        return normalize_boxes(boxes)

    def _load_boxes(self, boxes):
        boxes = normalize_boxes(boxes)
        self.table.setRowCount(0)
        for box in boxes:
            az = common.convert_axis_values(box['azimuth_band'], 'azimuth', 'deg', self.unit)
            freq = common.convert_axis_values(box['frequency_band'], 'frequency', 'Hz', self.frequency_unit)
            self._append_row([box['name'], *map(_number_text, az), *map(_number_text, freq)])

    @staticmethod
    def _read_csv(path):
        with open(path, encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source, strict=True)
            if reader.fieldnames != list(CSV_COLUMNS):
                raise ValueError('Expected CSV columns: ' + ', '.join(CSV_COLUMNS))
            boxes = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'CSV line {reader.line_num}: expected five columns.')
                boxes.append(dict(name=row['name'],
                                  azimuth_band=[row[key] for key in CSV_COLUMNS[1:3]],
                                  frequency_band=[row[key] for key in CSV_COLUMNS[3:5]]))
        return normalize_boxes(boxes)

    def _load_csv(self):
        self._finish_editing()
        path, _ = QFileDialog.getOpenFileName(self, 'Load boxes (degrees / Hz)', self._csv_path, 'CSV files (*.csv)')
        if not path:
            return
        try:
            self._load_boxes(self._read_csv(path))
        except (OSError, UnicodeError, csv.Error, ValueError, OverflowError) as exc:
            self._show_error(f'Could not load boxes: {exc}')
            return
        self._csv_path = path
        self._clear_error()

    def _save_csv(self):
        self._finish_editing()
        try:
            boxes = self._boxes()
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save boxes (degrees / Hz)', self._csv_path, 'CSV files (*.csv)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8', newline='') as destination:
                writer = csv.writer(destination)
                writer.writerow(CSV_COLUMNS)
                for box in boxes:
                    writer.writerow([box['name'], *map(_number_text, box['azimuth_band']),
                                     *map(_number_text, box['frequency_band'])])
        except (OSError, UnicodeError, csv.Error) as exc:
            self._show_error(f'Could not save boxes: {exc}')
            return
        self._csv_path = path
        self._clear_error()

    def apply_settings(self):
        self._finish_editing()
        try:
            boxes = self._boxes()
        except (ValueError, OverflowError) as exc:
            self._show_error(str(exc))
            return False
        self.controls.set_sector_settings(
            self.controls.sector_text(), str(self.combo_statistic.currentData()), self.spin_percentile.value(),
            interpolate=self.chk_interpolate.isChecked(),
            use_selected_azimuths=self.check_selected_range.isChecked(), boxes=boxes)
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
