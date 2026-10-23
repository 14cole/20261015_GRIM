"""Reusable presentation widgets; no dataset or solver operations."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QSize, Signal, QEvent, QTimer
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QFrame, QLabel, QAbstractSpinBox, QComboBox, QAbstractSlider, QTabBar,
    QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


from GRIM_Backend.ui.no_wheel import NoWheelDoubleSpinBox


def initial_window_size(available: QSize, preferred: QSize = QSize(1550, 900)) -> QSize:
    """Reserve room for native window borders and the title bar.

    QScreen.availableGeometry already excludes desktop taskbars/docks. Qt
    expresses both sizes in logical pixels, so no manual DPI scaling is used.
    """
    return QSize(
        min(preferred.width(), max(1, available.width() - 32)),
        min(preferred.height(), max(1, available.height() - 64)),
    )


class PlotNumberSpin(NoWheelDoubleSpinBox):
    """Compact display without reducing the precision of stored plot limits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self._edited_text = None
        self.lineEdit().installEventFilter(self)
        self.lineEdit().textEdited.connect(self._remember_edit)
        self.editingFinished.connect(self._commit_edit)

    def _remember_edit(self, text):
        self._edited_text = text

    def _commit_edit(self):
        text, self._edited_text = self._edited_text, None
        if text is None:
            return
        try:
            value = float(text.removeprefix(self.prefix()).removesuffix(self.suffix()).strip())
        except ValueError:
            return
        if math.isfinite(value) and self.minimum() <= value <= self.maximum():
            # Qt can reuse its cached value when a user explicitly types the
            # same compact text. Honor that edit, including intentional rounding.
            self.setValue(value)

    def textFromValue(self, value):
        return format(value, ".6g")

    def stepBy(self, steps):
        self._commit_edit()
        super().stepBy(steps)

    def valueFromText(self, text):
        text = text.removeprefix(self.prefix()).removesuffix(self.suffix()).strip()
        # Merely entering/leaving the compact field must not round its value.
        if self._edited_text is None and text == self.textFromValue(self.value()):
            return self.value()
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, position):
        value = text.removeprefix(self.prefix()).removesuffix(self.suffix()).strip()
        try:
            parsed = float(value)
            state = QValidator.Acceptable if math.isfinite(parsed) and self.minimum() <= parsed <= self.maximum() else QValidator.Intermediate
        except ValueError:
            state = QValidator.Intermediate if all(c in "+-.eE0123456789" for c in value) else QValidator.Invalid
        return state, text, position

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.selectAll()
        if event.reason() == Qt.MouseFocusReason:
            # The line editor handles the click after focus is assigned.
            QTimer.singleShot(0, self.selectAll)

    def eventFilter(self, obj, event):
        if obj is self.lineEdit() and event.type() == QEvent.MouseButtonPress and not obj.hasFocus():
            QTimer.singleShot(0, self.selectAll)
        return super().eventFilter(obj, event)


class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        else:
            super().mouseDoubleClickEvent(event)


class PlotSettingsPopup(QFrame):
    """Scrollable top-level popup for one plot context.

    Closing the window via its title bar untoggles the bound button so the
    button state mirrors visibility.  ``content_widget`` deliberately remains
    a plain widget: callers can keep using their existing grid layout while
    the popup owns the scroll-area plumbing.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "Plot Settings",
    ) -> None:
        super().__init__(parent)
        self._toggle_button: QToolButton | None = None
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        # Keep the popup useful on a compact display.  Larger contents scroll
        # in either direction instead of forcing the window beyond the screen.
        self.setMinimumSize(420, 280)
        self.resize(820, 540)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("plotSettingsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_widget.setObjectName("plotSettingsContent")
        self.scroll_area.setWidget(self.content_widget)
        outer.addWidget(self.scroll_area, 1)

    def set_toggle_button(self, button: QToolButton) -> None:
        self._toggle_button = button

    def protect_controls(self):
        """Wheel gestures scroll the popup; they never edit control values."""
        for widget in self.findChildren(QWidget):
            if isinstance(widget, QAbstractSpinBox):
                widget.setKeyboardTracking(False)
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            widget = obj
            while isinstance(widget, QWidget) and widget is not self:
                if isinstance(widget, (QAbstractSpinBox, QComboBox, QTabBar)) or (
                        isinstance(widget, QAbstractSlider) and not widget.inherits("QScrollBar")):
                    bar = self.scroll_area.verticalScrollBar()
                    delta = event.pixelDelta().y() or event.angleDelta().y() / 120 * bar.singleStep() * 3
                    bar.setValue(bar.value() - round(delta))
                    event.accept()
                    return True
                widget = widget.parentWidget()
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        controls = getattr(self, "plot_controls", None)
        if controls is not None:
            controls.refresh_from_canvas()
        self.protect_controls()
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        if self._toggle_button is not None and self._toggle_button.isChecked():
            self._toggle_button.setChecked(False)
        super().closeEvent(event)


class CollapsibleSection(QWidget):
    """A titled panel whose body collapses when its header is clicked.

    Purely presentational — it organizes the control dock into Datasets /
    Parameters / Operations / Plot Tools groups while holding the exact same
    widgets the app has always used.
    """

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True) -> None:
        super().__init__(parent)
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._body = QWidget()
        self._body.setObjectName("sectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(6)

        outer.addWidget(self.header)
        outer.addWidget(self._body)

        self.header.toggled.connect(self._sync)
        self._sync(expanded)

    def _sync(self, on: bool) -> None:
        self.header.setText(("▾  " if on else "▸  ") + self._title)
        self._body.setVisible(on)

    def addWidget(self, widget, stretch: int = 0) -> None:
        self._body_layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch: int = 0) -> None:
        self._body_layout.addLayout(layout, stretch)
