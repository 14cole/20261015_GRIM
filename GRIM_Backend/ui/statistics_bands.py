"""Numeric band tables for statistics dialogs."""
import numpy as np
from PySide6.QtWidgets import (
    QAbstractItemView, QGroupBox, QHeaderView, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


class StatisticsBandTable(QGroupBox):
    def __init__(self, title, *, wrap=False, checked=False, parent=None):
        super().__init__(title, parent)
        self.wrap = wrap
        self.setCheckable(True)
        self.setChecked(checked)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Start', 'Stop'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(95)
        self.table.setMaximumHeight(115)
        layout.addWidget(self.table)
        bar = QHBoxLayout()
        add = QPushButton('Add band')
        remove = QPushButton('Remove selected')
        add.setAutoDefault(False)
        remove.setAutoDefault(False)
        bar.addWidget(add)
        bar.addWidget(remove)
        layout.addLayout(bar)
        self.note = note = QLabel('No bands listed: use the entire axis.' +
                      (' Stop below start wraps through 360°.' if wrap else ''))
        note.setWordWrap(True)
        layout.addWidget(note)
        add.clicked.connect(self.add_band)
        remove.clicked.connect(self.remove_selected)

    def add_band(self, start='', stop=''):
        # clicked(bool) must not become a start coordinate.
        if isinstance(start, bool):
            start = ''
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((start, stop)):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.setCurrentCell(row, 0)

    def remove_selected(self):
        for row in sorted({item.row() for item in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(row)

    def bounds(self):
        if self.isCheckable() and not self.isChecked():
            return []
        bands = []
        for row in range(self.table.rowCount()):
            try:
                start, stop = [float(self.table.item(row, col).text()) for col in (0, 1)]
            except (ValueError, AttributeError):
                raise ValueError(f'{self.title()}, band {row + 1}: enter numeric Start and Stop values.') from None
            if not np.isfinite([start, stop]).all() or start == stop or (not self.wrap and stop < start):
                raise ValueError(f'{self.title()}, band {row + 1}: enter finite, distinct bounds' +
                                 ('.' if self.wrap else ' with Stop above Start.'))
            bands.append([start, stop])
        return bands
