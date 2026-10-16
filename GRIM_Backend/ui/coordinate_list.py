"""Model-backed coordinate lists that populate in constant time.

The dataset sidebar shows four coordinate lists (polarization, frequency,
elevation, azimuth). Filling a ``QListWidget`` with one item per coordinate
took seconds for axes with 100,000 samples. These views keep the coordinates
in a small model and hand out lightweight item proxies that implement the
subset of the ``QListWidgetItem`` API the application and its tests use:
``text``/``setText``, ``data``/``setData`` with ``Qt.UserRole`` (value) and
``Qt.UserRole + 1`` (axis index), ``setSelected``/``isSelected``, flags and
tool tips. The view offers ``count``, ``item``, ``row``, ``selectedItems``,
``setCurrentItem``, ``currentItem``, ``scrollToItem``, ``clear`` and the
``itemSelectionChanged``/``itemChanged`` signals, plus bulk helpers that
avoid per-row Python work on large axes.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import (
    QAbstractListModel, QItemSelection, QItemSelectionModel, QModelIndex, Qt, Signal,
)
from PySide6.QtWidgets import QAbstractItemView, QListView


class CoordinateListModel(QAbstractListModel):
    """Coordinates of one axis, an optional index subset, and per-row edits."""

    item_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = np.zeros(0)
        self._indices = np.zeros(0, dtype=np.intp)
        self._value_overrides = {}
        self._text_overrides = {}
        self._tooltips = {}

    def reset(self, values, indices=None):
        """Replace every row at once; the view never materializes items."""
        self.beginResetModel()
        try:
            # Copy so later in-place axis edits elsewhere cannot change the list.
            self._values = np.array(values if values is not None else (), copy=True)
            if indices is None:
                self._indices = np.arange(len(self._values), dtype=np.intp)
            else:
                self._indices = np.asarray(list(indices), dtype=np.intp).reshape(-1)
            self._value_overrides.clear()
            self._text_overrides.clear()
            self._tooltips.clear()
        finally:
            self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else int(self._indices.size)

    def axis_index(self, row):
        return int(self._indices[row])

    def axis_indices(self):
        return self._indices.copy()

    def value(self, row):
        if row in self._value_overrides:
            return self._value_overrides[row]
        value = self._values[int(self._indices[row])]
        return value.item() if isinstance(value, np.generic) and not isinstance(value, np.str_) else value

    def values(self):
        if not self._value_overrides:
            selected = self._values[self._indices]
            return [item.item() if isinstance(item, np.generic) and not isinstance(item, np.str_) else item
                    for item in selected]
        return [self.value(row) for row in range(self.rowCount())]

    def text(self, row):
        override = self._text_overrides.get(row)
        if override is not None:
            return override
        return str(self._values[int(self._indices[row])] if row not in self._value_overrides
                   else self._value_overrides[row])

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.text(row)
        if role == Qt.UserRole:
            return self.value(row)
        if role == Qt.UserRole + 1:
            return self.axis_index(row)
        if role == Qt.ToolTipRole:
            return self._tooltips.get(row)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        return self.set_role(index.row(), role, value)

    def set_role(self, row, role, value):
        if role in (Qt.DisplayRole, Qt.EditRole):
            self._text_overrides[row] = str(value)
        elif role == Qt.UserRole:
            self._value_overrides[row] = value
        elif role == Qt.UserRole + 1:
            self._indices[row] = int(value)
        elif role == Qt.ToolTipRole:
            self._tooltips[row] = value
        else:
            return False
        model_index = self.index(row, 0)
        self.dataChanged.emit(model_index, model_index, [role])
        self.item_changed.emit(row)
        return True


class CoordinateItem:
    """A ``QListWidgetItem``-like handle for one row of a ``CoordinateListView``."""

    __slots__ = ("_view", "_row")

    def __init__(self, view, row):
        self._view = view
        self._row = int(row)

    def _index(self):
        return self._view.model().index(self._row, 0)

    def listWidget(self):
        return self._view

    def text(self):
        return self._view.model().text(self._row)

    def setText(self, text):
        self._view.model().set_role(self._row, Qt.EditRole, str(text))

    def data(self, role=Qt.UserRole):
        return self._view.model().data(self._index(), role)

    def setData(self, role, value):
        self._view.model().set_role(self._row, role, value)

    def toolTip(self):
        return self._view.model().data(self._index(), Qt.ToolTipRole) or ""

    def setToolTip(self, text):
        self._view.model().set_role(self._row, Qt.ToolTipRole, str(text))

    def flags(self):
        return self._view.model().flags(self._index())

    def setFlags(self, flags):
        # Every coordinate row is selectable and editable; nothing to store.
        return None

    def isSelected(self):
        return self._view.selectionModel().isSelected(self._index())

    def setSelected(self, selected):
        command = QItemSelectionModel.Select if selected else QItemSelectionModel.Deselect
        self._view.selectionModel().select(self._index(), command)

    def __eq__(self, other):
        return (isinstance(other, CoordinateItem) and other._view is self._view
                and other._row == self._row)

    def __hash__(self):
        return hash((id(self._view), self._row))

    def __repr__(self):
        return f"CoordinateItem(row={self._row}, text={self.text()!r})"


class CoordinateListView(QListView):
    """A list view over ``CoordinateListModel`` with a ``QListWidget``-style API."""

    itemSelectionChanged = Signal()
    itemChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = CoordinateListModel(self)
        self.setModel(self._model)
        self.setUniformItemSizes(True)
        self.selectionModel().selectionChanged.connect(self._emit_selection_changed)
        self._model.item_changed.connect(self._emit_item_changed)

    def _emit_selection_changed(self, *_args):
        self.itemSelectionChanged.emit()

    def _emit_item_changed(self, row):
        self.itemChanged.emit(CoordinateItem(self, row))

    # -- population -------------------------------------------------------
    def set_coordinates(self, values, indices=None):
        """Show ``values[indices]`` (all values when ``indices`` is None)."""
        self._model.reset(values, indices)

    def clear(self):
        self._model.reset((), None)

    # -- QListWidget-style access -----------------------------------------
    def count(self):
        return self._model.rowCount()

    def item(self, row):
        row = int(row)
        if 0 <= row < self._model.rowCount():
            return CoordinateItem(self, row)
        return None

    def row(self, item):
        return -1 if item is None else int(item._row)

    def selected_rows(self):
        return sorted(index.row() for index in self.selectionModel().selectedIndexes())

    def selectedItems(self):
        return [CoordinateItem(self, row) for row in self.selected_rows()]

    def currentItem(self):
        index = self.currentIndex()
        return CoordinateItem(self, index.row()) if index.isValid() else None

    def setCurrentItem(self, item, command=None):
        index = self._model.index(self.row(item), 0)
        if command is None:
            self.setCurrentIndex(index)
        else:
            self.selectionModel().setCurrentIndex(index, command)

    def scrollToItem(self, item, hint=QAbstractItemView.EnsureVisible):
        self.scrollTo(self._model.index(self.row(item), 0), hint)

    # -- bulk helpers for large axes ------------------------------------------
    def coordinate_values(self):
        """Displayed coordinate values in row order."""
        return self._model.values()

    def displayed_axis_indices(self):
        return self._model.axis_indices()

    def selected_axis_indices(self):
        return [self._model.axis_index(row) for row in self.selected_rows()]

    def selected_values(self):
        return [self._model.value(row) for row in self.selected_rows()]

    def select_rows(self, rows, *, clear=False):
        """Select many rows with one selection command, merging contiguous runs."""
        rows = sorted({int(row) for row in rows if 0 <= int(row) < self.count()})
        selection = QItemSelection()
        start = previous = None
        for row in rows:
            if start is None:
                start = previous = row
            elif row == previous + 1:
                previous = row
            else:
                selection.select(self._model.index(start, 0), self._model.index(previous, 0))
                start = previous = row
        if start is not None:
            selection.select(self._model.index(start, 0), self._model.index(previous, 0))
        command = QItemSelectionModel.Select
        if clear:
            command |= QItemSelectionModel.Clear
        if rows or clear:
            self.selectionModel().select(selection, command)
