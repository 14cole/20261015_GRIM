"""Waterfall axis controls and linked native row/column inspection."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QComboBox, QCheckBox, QLabel, QFrame, QHBoxLayout, QVBoxLayout, QSizePolicy, QToolButton, QAbstractSpinBox
import numpy as np
from matplotlib.ticker import MaxNLocator

from GRIM_Backend.plotting.image_analysis import AXES, cuts
from GRIM_Backend.plotting.modes.common import decimate_line
from GRIM_Backend.plotting.image_grid import line_with_gaps


from GRIM_Backend.ui.no_wheel import NoWheelDoubleSpinBox


class SampleCoordinateSpinBox(NoWheelDoubleSpinBox):
    """Type physical coordinates; arrow keys step through available samples."""
    samples = None
    sampleStepped = Signal(float)
    _native_value = None
    _display_value = None

    def set_sample(self, value):
        self._native_value = float(value)
        self.setValue(float(value))
        self._display_value = self.value()

    def coordinate(self):
        # QDoubleSpinBox rounds to its configured decimals. Keep the actual
        # coordinate until the user enters a different value.
        if self._native_value is not None and self.value() == self._display_value:
            return self._native_value
        return self.value()

    def stepBy(self, steps):
        self.interpretText()
        if self.samples is None or not len(self.samples):
            return
        if steps == 0:
            return
        current = self.coordinate()
        if steps > 0:
            index = np.searchsorted(self.samples, current, side="right") + steps - 1
        else:
            index = np.searchsorted(self.samples, current, side="left") + steps
        value = float(self.samples[int(np.clip(index, 0, len(self.samples)-1))])
        blocked = self.blockSignals(True)
        try:
            self.set_sample(value)
        finally:
            self.blockSignals(blocked)
        # Emit the exact coordinate even when two samples share a rounded
        # displayed value and Qt emits no valueChanged signal.
        if value != current:
            self.sampleStepped.emit(value)

    def stepEnabled(self):
        flags = QAbstractSpinBox.StepEnabledFlag.StepNone
        if self.samples is not None and len(self.samples):
            value = self.coordinate()
            if value > self.samples[0]:
                flags |= QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
            if value < self.samples[-1]:
                flags |= QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
        return flags

    def textFromValue(self, value):
        text = super().textFromValue(value)
        decimal = self.locale().decimalPoint()
        return text.rstrip("0").rstrip(decimal) if decimal in text else text


class ImageAnalysisControls(QObject):
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_x, self.combo_y = QComboBox(), QComboBox()
        for widget in (self.combo_x, self.combo_y):
            for axis in AXES:
                widget.addItem(axis.capitalize(), axis)
        self.combo_y.setCurrentIndex(2)
        self.combo_aggregation = QComboBox()
        self.combo_aggregation.addItem("Separate remaining-axis cuts", "panels")
        self.combo_aggregation.addItem("Mean over remaining axis", "mean_power")
        self.combo_aggregation.setToolTip("All selected cuts are included. Level means use linear display units; phase uses a circular mean. Each polarization remains separate.")
        self._position = None
        self._coordinate_frame = None
        self._anchor_name = None
        self._bound_record = None
        self.controller = None
        self.cut_bar = QFrame()
        self.cut_bar.setObjectName("waterfallCutControls")
        self.cut_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer = QVBoxLayout(self.cut_bar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        layout = QHBoxLayout()
        outer.addLayout(layout)
        self.check_horizontal = QCheckBox("Horizontal cut")
        self.check_vertical = QCheckBox("Vertical cut")
        self.check_horizontal.setToolTip("Show the horizontal slice at the selected Y coordinate.")
        self.check_vertical.setToolTip("Show the vertical slice at the selected X coordinate.")
        for widget in (self.check_horizontal, self.check_vertical):
            widget.setChecked(True)
            layout.addWidget(widget)
            widget.toggled.connect(lambda: self.changed.emit("heatmap"))
        layout.addStretch(1)
        layout = QHBoxLayout()
        layout.setSpacing(4)
        outer.addLayout(layout)
        self.spin_x, self.spin_y = SampleCoordinateSpinBox(), SampleCoordinateSpinBox()
        self._step_buttons = {}
        self._coordinate_labels = {}
        for label, spin in (("X", self.spin_x), ("Y", self.spin_y)):
            spin.setObjectName("waterfallCut" + label)
            spin.setDecimals(9)
            spin.setRange(-1.e100, 1.e100)
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setMinimumWidth(110)
            spin.setMaximumWidth(165)
            spin.valueChanged.connect(self._position_edited)
            # Enter/focus-out also snaps a typed value that rounds to the
            # existing spin value and therefore emits no valueChanged.
            spin.editingFinished.connect(self._position_edited)
            spin.sampleStepped.connect(lambda value, axis=label: self._coordinate_stepped(axis, value))
            self._coordinate_labels[label] = QLabel(label)
            layout.addWidget(self._coordinate_labels[label])
            previous, following = QToolButton(), QToolButton()
            for button, direction, arrow in ((previous, -1, Qt.DownArrow), (following, 1, Qt.UpArrow)):
                button.setArrowType(arrow)
                button.setObjectName(f'waterfallCut{label}' + ('Previous' if direction < 0 else 'Next'))
                button.setMinimumSize(26, 26)
                button.setFocusPolicy(Qt.NoFocus)
                button.setAutoRepeat(True)
                button.setAutoRepeatDelay(350)
                button.setAutoRepeatInterval(100)
                button.clicked.connect(lambda _checked=False, widget=spin, delta=direction: widget.stepBy(delta))
            self._step_buttons[label] = (previous, following)
            layout.addWidget(previous)
            layout.addWidget(spin, 1)
            layout.addWidget(following)
        layout.addStretch(1)
        self.cut_bar.hide()
        self.combo_x.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_x, self.combo_y))
        self.combo_y.currentIndexChanged.connect(lambda: self._axes_changed(self.combo_y, self.combo_x))
        self.combo_aggregation.currentIndexChanged.connect(lambda: self.changed.emit("heatmap"))

    def _axes_changed(self, current, other):
        if current.currentData() == other.currentData():
            blocked = other.blockSignals(True)
            other.setCurrentIndex(next(i for i in range(3) if other.itemData(i) != current.currentData()))
            other.blockSignals(blocked)
        self._position = None
        self._coordinate_frame = None
        self.changed.emit("heatmap")

    def axes(self):
        return self.combo_x.currentData(), self.combo_y.currentData()

    def aggregation(self):
        return self.combo_aggregation.currentData()

    def linked_cuts(self):
        return self.horizontal_cuts() or self.vertical_cuts()

    def horizontal_cuts(self):
        return self.check_horizontal.isChecked()

    def vertical_cuts(self):
        return self.check_vertical.isChecked()

    def cut_position(self):
        return self._position

    def bind(self, controller):
        self.controller = controller
        records = [panel[2] for panel in controller.panels]
        record = records[0]
        frame = (record.x_axis, record.y_axis, record.x_unit, record.y_unit)
        if self._coordinate_frame is not None and self._coordinate_frame != frame:
            self._position = None
            self._anchor_name = None
        self._coordinate_frame = frame
        controller.anchor = next((i for i, r in enumerate(records) if r.name == self._anchor_name), 0)
        record = records[controller.anchor]
        self._bound_record = None
        controller.on_selected = self._selected
        controller.on_disconnect = self._disconnected
        position = self._position or (record.x[len(record.x)//2], record.y[len(record.y)//2])
        controller.update(*position, notify=False)
        self._selected(*controller.position, emit=False)
        self.cut_bar.show()

    def _sync_anchor(self):
        record = self.controller.panels[self.controller.anchor][2]
        if record is self._bound_record:
            return
        self._bound_record = record
        self._anchor_name = record.name
        for label, spin, axis, unit, values in (
            ('X', self.spin_x, record.x_axis, record.x_unit, record.x),
            ('Y', self.spin_y, record.y_axis, record.y_unit, record.y),
        ):
            blocked = spin.blockSignals(True)
            spin.samples = np.unique(values)
            steps = np.diff(spin.samples)
            # Display enough decimals to distinguish unusually close samples.
            spin.setDecimals(max(9, min(17, int(np.ceil(-np.log10(np.min(steps)))) + 2)) if steps.size else 9)
            spin.setRange(float(np.min(values)), float(np.max(values)))
            spin.setSuffix(" " + unit if unit else "")
            spin.setSingleStep(float(np.min(steps)) if steps.size else 1.)
            spin.setToolTip(f"{axis.capitalize()} ({unit}). Type a coordinate and press Enter to snap to the nearest sample. Down/up buttons step through {record.name}. Click a panel to use its sample sequence.")
            spin.blockSignals(blocked)
            self._coordinate_labels[label].setText(axis.capitalize())
            for button, direction in zip(self._step_buttons[label], ('Previous', 'Next')):
                button.setToolTip(f'{direction} {axis} sample in {record.name}')

    def _selected(self, x, y, *, emit=True):
        self._sync_anchor()
        position = (float(x), float(y))
        changed = position != self._position
        self._position = position
        for spin, value in ((self.spin_x, x), (self.spin_y, y)):
            blocked = spin.blockSignals(True)
            spin.set_sample(value)
            spin.blockSignals(blocked)
        for label, spin in (('X', self.spin_x), ('Y', self.spin_y)):
            previous, following = self._step_buttons[label]
            previous.setEnabled(bool(spin.coordinate() > spin.samples[0]))
            following.setEnabled(bool(spin.coordinate() < spin.samples[-1]))
        if emit and changed:
            self.changed.emit("heatmap_position")

    def _position_edited(self, *_):
        if self.controller is not None:
            self.controller.update(self.spin_x.coordinate(), self.spin_y.coordinate())

    def _coordinate_stepped(self, axis, value):
        if self.controller is not None:
            x, y = self._position
            self.controller.update(value if axis == 'X' else x, value if axis == 'Y' else y)

    def _disconnected(self):
        self.controller = None
        self._bound_record = None
        self.cut_bar.hide()

    def options(self):
        return {"axes": list(self.axes()), "aggregation": self.aggregation(), "linked_cuts": self.linked_cuts(),
                "horizontal_cuts": self.horizontal_cuts(), "vertical_cuts": self.vertical_cuts(),
                "cut_position": list(self._position) if self._position is not None else None}

    def restore_options(self, options):
        if not isinstance(options, dict):
            return
        widgets = (self.combo_x, self.combo_y, self.combo_aggregation, self.check_horizontal, self.check_vertical)
        states = [w.blockSignals(True) for w in widgets]
        try:
            axes = options.get("axes", self.axes())
            if isinstance(axes,(list,tuple)) and len(axes) == 2 and all(isinstance(a,str) and a in AXES for a in axes) and len(set(axes)) == 2:
                for widget, value in zip(widgets[:2], axes):
                    widget.setCurrentIndex(widget.findData(value))
            index = self.combo_aggregation.findData(options.get("aggregation", "panels"))
            if index >= 0:
                self.combo_aggregation.setCurrentIndex(index)
            legacy = bool(options.get("linked_cuts", True))
            self.check_horizontal.setChecked(bool(options.get("horizontal_cuts", legacy)))
            self.check_vertical.setChecked(bool(options.get("vertical_cuts", legacy)))
            self._position = None
            position = options.get("cut_position")
            try:
                if isinstance(position, (list, tuple)) and len(position) == 2 and np.all(np.isfinite(np.asarray(position, float))):
                    self._position = tuple(map(float, position))
            except (ValueError, TypeError, OverflowError):
                pass
            self._coordinate_frame = None
        finally:
            for widget, state in zip(widgets, states):
                widget.blockSignals(state)
        self.changed.emit("heatmap")


class LinkedImageCuts:
    """Linked cuts at actual sample centers, backed by native image records."""
    def __init__(self, canvas, panels, *, can_select=None):
        self.canvas, self.panels = canvas, panels
        self.can_select = can_select
        self.on_selected = self.on_disconnect = None
        self.position = None
        self.anchor = 0
        self.artists = []
        units_by_axes = {}
        records_by_axes = {}
        shared_horizontal = shared_vertical = None
        for _image, cut_axes, record in panels:
            horizontal, vertical = cut_axes
            # Compare each panel on the same value scale while its coordinate
            # axis remains linked to the corresponding heatmap.
            if horizontal is not None:
                if shared_horizontal is None:
                    shared_horizontal = horizontal
                elif horizontal is not shared_horizontal:
                    horizontal.sharey(shared_horizontal)
            if vertical is not None:
                if shared_vertical is None:
                    shared_vertical = vertical
                elif vertical is not shared_vertical:
                    vertical.sharex(shared_vertical)
            for ax in cut_axes:
                if ax is not None:
                    units_by_axes.setdefault(ax, set()).add(record.value_unit)
                    records_by_axes[ax] = records_by_axes.get(ax, 0) + 1
        for index,(image_ax, (horizontal, vertical), record) in enumerate(panels,1):
            hline = vline = hcursor = vcursor = None
            if horizontal is not None:
                label = f"{index} ({record.value_unit})" if len(units_by_axes[horizontal]) > 1 else str(index)
                hline, = horizontal.plot([], [], linewidth=1,label=label)
                hcursor = image_ax.axhline(record.y[len(record.y)//2], color="white", linewidth=.7)
                unit = " / ".join(sorted(units_by_axes[horizontal]))
                horizontal.set_ylabel(unit, fontsize=8)
                horizontal.yaxis.set_major_locator(MaxNLocator(3))
            if vertical is not None:
                label = f"{index} ({record.value_unit})" if len(units_by_axes[vertical]) > 1 else str(index)
                vline, = vertical.plot([], [], linewidth=1,label=label)
                vcursor = image_ax.axvline(record.x[len(record.x)//2], color="white", linewidth=.7)
                unit = " / ".join(sorted(units_by_axes[vertical]))
                vertical.set_xlabel(unit, fontsize=8)
                vertical.yaxis.set_major_locator(MaxNLocator(3))
                vertical.xaxis.set_major_locator(MaxNLocator(3))
            self.artists.append((hline, vline, hcursor, vcursor))
        for legend_ax, count in records_by_axes.items():
            if count > 1:
                legend = legend_ax.legend(ncol=min(4,count),fontsize=7,loc="upper right")
                for text in legend.get_texts():
                    text.set_color(legend_ax.xaxis.label.get_color())
                legend.get_frame().set_facecolor(legend_ax.get_facecolor())
        self._last = None
        self.connection = canvas.mpl_connect("button_press_event", self.on_click)
        if panels:
            record = panels[0][2]
            self.update(record.x[len(record.x)//2], record.y[len(record.y)//2])

    def disconnect(self):
        self.canvas.mpl_disconnect(self.connection)
        self.panels = []
        self.artists = []
        if self.on_disconnect is not None:
            self.on_disconnect()
        self.on_selected = self.on_disconnect = self.can_select = None

    def on_click(self, event):
        axes = [p[0] for p in self.panels]
        if event.button != 1 or event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        # Cleared figures can still own this controller until their next
        # heatmap; never read a disposed native snapshot from a stale axes.
        if event.inaxes not in self.canvas.figure.axes:
            return
        if (getattr(getattr(self.canvas, "toolbar", None), "mode", "")
                or self.canvas.widgetlock.locked()
                or (self.can_select is not None and not self.can_select())):
            return
        self.update(event.xdata, event.ydata, anchor=axes.index(event.inaxes))

    def update(self, x, y, *, anchor=None, notify=True):
        if not self.panels or not np.all(np.isfinite([x, y])):
            return
        # Keep typing and stepping on the last clicked panel. Other panels
        # snap independently to that panel's physical coordinate.
        if anchor is not None:
            self.anchor = anchor
        source = self.panels[self.anchor][2]
        x = float(source.x[int(np.argmin(abs(source.x-x)))])
        y = float(source.y[int(np.argmin(abs(source.y-y)))])
        self.position = (x, y)
        positions = tuple((int(np.argmin(abs(p[2].x-x))), int(np.argmin(abs(p[2].y-y)))) for p in self.panels)
        if notify and self.on_selected is not None:
            self.on_selected(x, y)
        if positions == self._last:
            return
        self._last = positions
        cut_selections = {}
        for (image_ax, (horizontal, vertical), record), artists in zip(self.panels, self.artists):
            xs, row, ys, column, px, py = cuts(record, x, y)
            for ax, value, direction, axis, unit in (
                (horizontal, py, "Horizontal", record.y_axis, record.y_unit),
                (vertical, px, "Vertical", record.x_axis, record.x_unit),
            ):
                if ax is not None:
                    cut_selections.setdefault((ax, direction, axis, unit), []).append(value)
            hline, vline, hcursor, vcursor = artists
            if hline is not None:
                hx,hy,_ = decimate_line(*line_with_gaps(xs,row,breaks=record.metadata.get("x_cut_breaks")))
                hline.set_data(hx, hy)
                hcursor.set_ydata([py,py])
            if vline is not None:
                vy,vx,_ = decimate_line(*line_with_gaps(ys,column,breaks=record.metadata.get("y_cut_breaks")))
                vline.set_data(vx, vy)
                vcursor.set_xdata([px,px])
        # Refresh every sibling's bounds before autoscaling the shared value
        # axes, so previous cuts cannot influence the new common range.
        for ax, _direction, _axis, _unit in cut_selections:
            ax.relim()
        for (ax, direction, axis, unit), values in cut_selections.items():
            ax.autoscale_view(scalex=direction == "Vertical", scaley=direction == "Horizontal")
            label = f"{min(values):.6g}" if min(values) == max(values) else f"{min(values):.6g}–{max(values):.6g} (per panel)"
            separator = "\n" if direction == "Vertical" else " · "
            ax.set_title(f"{direction} cuts{separator}{axis} = {label} {unit}", fontsize=8,
                         color=ax.xaxis.label.get_color(),pad=2)
        self.canvas.draw_idle()
