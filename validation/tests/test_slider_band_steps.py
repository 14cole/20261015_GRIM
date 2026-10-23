"""Adjacent slider bands, acquired sample masks, and desktop integration."""
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from GRIM_Backend.plotting.sweep_frames import make_sweep_frames


class SweepBandStepTests(unittest.TestCase):
    def test_requested_two_ghz_bands_share_endpoints(self):
        frames = make_sweep_frames(np.arange(1., 7.5, .5), width=2., unit='GHz', step_by_width=True)
        np.testing.assert_array_equal(frames.bounds, [[1, 3], [3, 5], [5, 7]])
        self.assertEqual([frames.selection(i) for i in range(3)],
                         [list(range(5)), list(range(4, 9)), list(range(8, 13))])
        self.assertTrue(frames.step_by_width)
        self.assertEqual(frames.label(1), '3 – 5 GHz')

    def test_default_preserves_overlapping_acquired_positions(self):
        frames = make_sweep_frames(np.arange(1., 7.5, .5), width=2.)
        np.testing.assert_array_equal(frames.bounds[:, 0], np.arange(1., 5.5, .5))
        self.assertFalse(frames.step_by_width)

    def test_final_partial_band_does_not_slide_backward(self):
        frames = make_sweep_frames(np.arange(1., 6.5, .5), width=2., step_by_width=True)
        np.testing.assert_array_equal(frames.bounds, [[1, 3], [3, 5], [5, 6]])
        self.assertEqual(frames.selection(2), [8, 9, 10])

    def test_sparse_bands_are_omitted_without_moving_remaining_boundaries(self):
        frames = make_sweep_frames([1., 2., 7., 8.], width=2., step_by_width=True)
        np.testing.assert_array_equal(frames.bounds, [[1, 3], [7, 8]])
        self.assertEqual(frames.selection(1), [2, 3])
        with self.assertRaisesRegex(ValueError, 'No band contains'):
            make_sweep_frames([1., 4., 7.], width=2., step_by_width=True)

    def test_irregular_unsorted_coordinates_are_not_used_as_band_starts(self):
        values = np.array([5.5, 1., 6., 2., 3.5, 4.])
        frames = make_sweep_frames(values, width=2., step_by_width=True)
        np.testing.assert_array_equal(frames.bounds, [[1, 3], [3, 5], [5, 6]])
        self.assertEqual(frames.selection(0), [1, 3])
        self.assertEqual(frames.selection(1), [4, 5])
        self.assertEqual(frames.selection(2), [0, 2])

    def test_hz_mhz_and_ghz_choose_the_same_samples(self):
        for scale in (1., 1000., 1e9):
            with self.subTest(scale=scale):
                frames = make_sweep_frames(np.arange(1., 7.5, .5) * scale,
                                          width=2. * scale, step_by_width=True)
                np.testing.assert_allclose(frames.bounds / scale, [[1, 3], [3, 5], [5, 7]])
                self.assertEqual(frames.selection(1), list(range(4, 9)))

    def test_decimal_steps_do_not_add_spurious_tail_or_drop_shared_edges(self):
        frames = make_sweep_frames(np.linspace(.1, .7, 7), width=.2, step_by_width=True)
        np.testing.assert_allclose(frames.bounds, [[.1, .3], [.3, .5], [.5, .7]])
        self.assertEqual([frames.selection(i) for i in range(3)], [[0, 1, 2], [2, 3, 4], [4, 5, 6]])

    def test_large_empty_gaps_do_not_allocate_empty_band_positions(self):
        frames = make_sweep_frames([0., 1e-6, 1e6, 1e6 + 1e-6], width=1e-6, step_by_width=True)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames.selection(1), [2, 3])

    def test_full_extent_and_single_coordinate_modes(self):
        frames = make_sweep_frames([1., 2., 3.], width=2., step_by_width=True)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames.selection(0), [0, 1, 2])
        single = make_sweep_frames([3., 1., 2.], step_by_width=True)
        self.assertFalse(single.band)
        self.assertFalse(single.step_by_width)
        self.assertEqual(single.selection(0), [1])


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
QT_AVAILABLE = importlib.util.find_spec('PySide6') is not None
if QT_AVAILABLE:
    from PySide6.QtWidgets import QApplication, QCheckBox, QListWidget, QAbstractItemView
    from PySide6.QtTest import QTest
    from GRIM_Backend.ui.analysis_controls import PlotSliderBar
    from GRIM_Backend.plotting.slider import PlotSliderMixin
    from GRIM_Backend.ui.sweep_export import SweepExporter

    class SliderHarness(PlotSliderMixin):
        def __init__(self, mode):
            self.active_dataset = SimpleNamespace(frequencies=np.arange(1., 7.5, .5),
                azimuths=np.arange(0., 7.), elevations=np.array([0., 10.]),
                units={'frequency': 'GHz', 'azimuth': 'deg', 'elevation': 'deg'})
            self.last_plot_mode = mode
            self.plot_slider = PlotSliderBar()
            self.widgets = [self.plot_slider]
            for axis, values in (('freq', self.active_dataset.frequencies),
                                 ('az', self.active_dataset.azimuths),
                                 ('elev', self.active_dataset.elevations)):
                widget = QListWidget()
                widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
                widget.addItems([str(value) for value in values])
                setattr(self, 'list_' + axis, widget)
                self.widgets.append(widget)
            self.chk_isar_freq_band = QCheckBox()
            self.chk_isar_aperture = QCheckBox()
            self.widgets.extend([self.chk_isar_freq_band, self.chk_isar_aperture])
            self.status = SimpleNamespace(showMessage=lambda _: None, currentMessage=lambda: 'updated')
            self.rendered = []
            self.plot_slider.axis_changed.connect(self._refresh_plot_slider)
            self.plot_slider.moved.connect(self._on_plot_slider_moved)
            self.plot_slider.show()
            self._refresh_plot_slider()
            self.plot_slider.band_width.setValue(2.)
            self.plot_slider.stop_interaction()

        def _invalidate_isar_result(self): pass
        def _button_checked(self, button): return False
        def _render_plot_mode(self, mode):
            self.rendered.append([self.list_freq.row(item) for item in self.list_freq.selectedItems()])


@unittest.skipUnless(QT_AVAILABLE, 'PySide6 is required for desktop widget tests')
class SliderBandWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self):
        if hasattr(self, 'harness'):
            self.harness.plot_slider.stop_interaction()
            for widget in self.harness.widgets:
                widget.close()
                widget.deleteLater()
            del self.harness
        self.app.processEvents()

    def test_checkbox_updates_both_range_modes_and_arrows_select_adjacent_band(self):
        for mode in ('isar_image', 'az_vs_range'):
            with self.subTest(mode=mode):
                self.harness = SliderHarness(mode)
                bar = self.harness.plot_slider
                self.assertFalse(bar.chk_step_by_bandwidth.isChecked())
                self.assertFalse(bar.chk_step_by_bandwidth.isHidden())
                bar.chk_step_by_bandwidth.setChecked(True)
                QTest.qWait(65)
                np.testing.assert_array_equal(bar._frames.bounds, [[1, 3], [3, 5], [5, 7]])
                self.assertEqual(self.harness.rendered[-1], list(range(5)))
                bar.btn_next.click()
                QTest.qWait(65)
                self.assertEqual(bar.value_label.text(), '3 – 5 GHz')
                self.assertEqual(self.harness.rendered[-1], list(range(4, 9)))
                self.assertFalse(self.harness.chk_isar_freq_band.isChecked())
                self.tearDown()

    def test_playback_uses_same_positions_and_setting_does_not_change_azimuth(self):
        self.harness = SliderHarness('isar_image')
        bar = self.harness.plot_slider
        bar.chk_step_by_bandwidth.setChecked(True)
        QTest.qWait(65)
        bar._play_step()
        QTest.qWait(65)
        self.assertEqual(self.harness.rendered[-1], list(range(4, 9)))
        bar.combo_axis.setCurrentIndex(2)
        self.assertTrue(bar.chk_step_by_bandwidth.isHidden())
        self.assertFalse(bar._frames.step_by_width)
        bar.combo_axis.setCurrentIndex(1)
        self.assertFalse(bar._frames.band)
        bar.combo_axis.setCurrentIndex(0)
        self.assertTrue(bar._frames.step_by_width)
        self.harness.last_plot_mode = 'azimuth_rect'
        self.harness._refresh_plot_slider()
        self.assertTrue(bar.chk_step_by_bandwidth.isHidden())
        self.assertFalse(bar._frames.band)

    def test_export_manifest_records_captured_band_step(self):
        self.harness = SliderHarness('isar_image')
        bar = self.harness.plot_slider
        bar.chk_step_by_bandwidth.setChecked(True)
        frames = bar._frames
        # Export freezes the frame plan; the manifest must describe that plan.
        bar.chk_step_by_bandwidth.setChecked(False)
        with TemporaryDirectory() as temp:
            exporter = SimpleNamespace(mode='isar_image', axis='frequency', frames=frames,
                dataset_names=['Example'], selections={}, bar=bar, seconds=.35,
                positions=[0, 1, 2], records=[], kind='images', folder=Path(temp))
            SweepExporter._write_manifest(exporter, 'running')
            manifest = json.loads((Path(temp) / 'manifest.json').read_text())
        self.assertEqual(manifest['band_step'], 'bandwidth')
        self.assertEqual(manifest['band_width'], 2.)
        self.assertEqual(manifest['requested_positions'], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
