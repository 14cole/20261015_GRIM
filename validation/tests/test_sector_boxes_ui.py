"""Offscreen Qt regression checks; skipped when PySide6 is not installed."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if QT_AVAILABLE:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit
    # Only the sector renderer is needed; avoid loading unrelated plot modes.
    from test_sector_frequency_bands import mode, package
    with patch.dict('sys.modules', {'GRIM_Backend.plotting.modes': package,
                                  'GRIM_Backend.plotting.modes.sector_stats_mode': mode}):
        from GRIM_Backend.ui.sector_statistics_dialog import SectorStatisticsDialog


@unittest.skipUnless(QT_AVAILABLE, 'PySide6 is required for desktop widget tests')
class SectorBoxDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controls = SimpleNamespace(
            sector_boxes=lambda: None, sector_use_selected_azimuths=lambda: False,
            sector_statistic=lambda: 'mean', sector_percentile=lambda: 90.,
            sector_interpolate=lambda: False, sector_text=lambda: '30', set_sector_settings=Mock())
        self.dialog = SectorStatisticsDialog(self.controls, [-180., 0., 180.],
                                             selected_azimuths=[0.], frequencies=[1., 2., 3.])
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()

    def test_initial_box_uses_all_azimuths_and_frequencies(self):
        self.assertFalse(self.dialog.check_selected_range.isChecked())
        self.assertEqual(self.dialog.table.columnCount(), 5)
        self.assertEqual(self.dialog._boxes(), [dict(name='Box 1', azimuth_band=[-180., 180.],
                                                    frequency_band=[1e9, 3e9])])

    def test_wheel_never_changes_statistic_or_percentile_even_with_focus(self):
        for widget in (self.dialog.combo_statistic, self.dialog.spin_percentile):
            for focused in (False, True):
                widget.setFocus() if focused else widget.clearFocus()
                self.app.processEvents()
                before = (self.dialog.combo_statistic.currentData(), self.dialog.spin_percentile.value())
                for delta in (-120, 120):
                    event = QWheelEvent(QPointF(widget.rect().center()), QPointF(widget.mapToGlobal(widget.rect().center())),
                                        QPoint(), QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                                        Qt.NoScrollPhase, False)
                    self.app.sendEvent(widget, event)
                    self.assertEqual(before, (self.dialog.combo_statistic.currentData(),
                                              self.dialog.spin_percentile.value()))

    def test_invalid_box_does_not_apply_partial_settings(self):
        self.dialog.table.item(0, 4).setText('not a frequency')
        self.assertFalse(self.dialog.apply_settings())
        self.controls.set_sector_settings.assert_not_called()

    def test_apply_commits_active_cell_editor_and_keeps_bands_paired(self):
        self.dialog.table.setCurrentCell(0, 3)
        self.dialog.table.editItem(self.dialog.table.item(0, 3))
        self.app.processEvents()
        editor = self.app.focusWidget()
        self.assertIsInstance(editor, QLineEdit)
        editor.setText('2')
        self.assertTrue(self.dialog.apply_settings())
        self.assertEqual(self.controls.set_sector_settings.call_args.kwargs['boxes'][0]['frequency_band'],
                         [2e9, 3e9])

    def test_cancel_does_not_apply(self):
        self.dialog.table.item(0, 3).setText('2')
        self.dialog.reject()
        self.controls.set_sector_settings.assert_not_called()

    def test_csv_round_trip_preserves_box_pair_and_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'boxes.csv')
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(path, '')):
                self.dialog._save_csv()
            self.assertEqual(self.dialog._read_csv(path), self.dialog._boxes())


if __name__ == '__main__':
    unittest.main()
