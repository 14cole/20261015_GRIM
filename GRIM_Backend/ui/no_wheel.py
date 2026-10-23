"""Editable controls whose values never change from wheel/trackpad scrolling."""
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QSlider


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()  # Let a parent scroll area scroll, even while focused.


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()
