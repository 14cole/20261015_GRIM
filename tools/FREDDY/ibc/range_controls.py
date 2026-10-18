"""Shared Fixed/Vary controls for layer design ranges in every workspace."""
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QCheckBox, QLineEdit


class RangeControls(QObject):
    def __init__(self, parent, values, unit):
        super().__init__(parent)
        self.toggle = QCheckBox('Vary')
        self.toggle.setChecked(values[0] not in (None, '') or values[1] not in (None, ''))
        self.toggle.setToolTip('Vary evaluates every defined value. Unchecked means Fixed at the nominal value; range fields are ignored.')
        self.fields = [QLineEdit('' if value is None else str(value)) for value in values]
        for field, name in zip(self.fields, ('Minimum', 'Maximum', 'Step')):
            field.setToolTip(f'{name} ({unit}). Values are minimum + k × step, without exceeding maximum. An off-step maximum is excluded. Equal bounds define one value.')
        self.toggle.toggled.connect(self.sync)
        self.sync()

    def sync(self, *_):
        for field in self.fields: field.setEnabled(self.toggle.isChecked())

    def row(self):
        return dict(vary=self.toggle.isChecked(), **dict(zip(('minimum','maximum','step'), (f.text() for f in self.fields))))
