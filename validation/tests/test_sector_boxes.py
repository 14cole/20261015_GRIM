"""Box population, plot/export agreement, and dataset metadata regressions."""
import json
import unittest

import numpy as np

from test_sector_frequency_bands import Harness, mode
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.statistics_bands import band_statistics_dataset
from GRIM_Backend.datasets.statistics_boxes import box_specs, normalize_boxes


def box(name, azimuth, frequency):
    return dict(name=name, azimuth_band=azimuth, frequency_band=frequency)


class SectorBoxTests(unittest.TestCase):
    def setUp(self):
        self.dataset = RcsGrid([0., 10.], [0.], [1., 2., 3.], ['HH'],
                              rcs_power=np.array([[1., 10., 100.], [3., 30., 300.]]).reshape(2, 1, 3, 1),
                              units={'frequency': 'GHz'})
        self.boxes = [box('Low/front', [-1, 5], [1e9, 2e9]),
                      box('High/rear', [5, 15], [2e9, 3e9])]

    def outputs(self, boxes=None, dataset=None, **kwargs):
        return [band_statistics_dataset(dataset or self.dataset, statistic='mean',
                                        axes=['azimuth', 'frequency'], **spec, **kwargs)
                for spec in box_specs(self.boxes if boxes is None else boxes)]

    def render(self, boxes=None, dataset=None, **kwargs):
        app = Harness(dataset or self.dataset, **kwargs)
        app.analysis_controls.sector_boxes = lambda: self.boxes if boxes is None else boxes
        return app, app.render()

    def test_two_paired_boxes_create_two_bounded_datasets(self):
        outputs = self.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual([out.shape for out in outputs], [(1, 1, 2, 1)] * 2)
        np.testing.assert_array_equal(outputs[0].azimuths, [0.])
        np.testing.assert_array_equal(outputs[0].frequencies, [1., 2.])
        np.testing.assert_array_equal(outputs[1].azimuths, [10.])
        np.testing.assert_array_equal(outputs[1].frequencies, [2., 3.])
        np.testing.assert_allclose(np.asarray(outputs[0].power), 5.5)
        np.testing.assert_allclose(np.asarray(outputs[1].power), 165.)

    def test_plot_matches_export_without_cross_combining_boxes(self):
        app, table = self.render(selected=(1.,))
        self.assertEqual(len(table['rows']), 2)
        self.assertEqual(len(app.lines), 2)
        self.assertEqual(table['box_names'], ['Low/front', 'High/rear'])
        self.assertEqual([row[5:7] for row in table['rows']], [(2, 5.5), (2, 165.)])
        self.assertIn('Low/front', mode.table_text(table))
        self.assertTrue(mode.table_text(table).splitlines()[0].endswith('\tBox'))

    def test_metadata_preserves_box_name_and_exact_paired_bounds(self):
        for original, out in zip(self.boxes, self.outputs()):
            self.assertEqual(out.extra['statistics_box_name'], original['name'])
            meta = json.loads(out.extra['statistics_bands_json'])
            self.assertEqual(meta['azimuth_bounds_deg'], original['azimuth_band'])
            self.assertEqual(meta['frequency_bounds_hz'], original['frequency_band'])

    def test_single_frequency_box(self):
        boxes = [box('Single', [-1, 15], [2e9, 2e9])]
        out = self.outputs(boxes)[0]
        self.assertEqual(out.shape, (2, 1, 1, 1))
        np.testing.assert_allclose(np.asarray(out.power), 20.)
        _, table = self.render(boxes)
        self.assertEqual(table['rows'][0][5:7], (2, 20.))

    def test_wrapped_azimuth_box_in_radians_and_mhz(self):
        ds = RcsGrid(np.deg2rad([-175., 0., 175.]), [0.], [1000., 2000.], ['HH'],
                     rcs_power=np.array([[1., 3.], [100., 200.], [5., 7.]]).reshape(3, 1, 2, 1),
                     units={'azimuth': 'rad', 'frequency': 'MHz'})
        out = self.outputs([box('Seam', [170, -170], [1e9, 2e9])], ds)[0]
        np.testing.assert_allclose(np.asarray(out.power), 4.)
        np.testing.assert_allclose(out.azimuths, np.deg2rad([-175., 175.]))
        _, table = self.render([box('Seam', [170, -170], [1e9, 2e9])], ds, selected=(1.,))
        self.assertEqual(table['rows'][0][5:7], (4, 4.))

    def test_empty_box_is_reported_and_other_boxes_still_plot(self):
        app, table = self.render(self.boxes + [box('Empty', [50, 60], [1e9, 2e9])])
        self.assertEqual(len(table['rows']), 2)
        self.assertIn('Empty (no samples inside box)', app.messages[-1])
        with self.assertRaisesRegex(ValueError, 'No samples'):
            self.outputs([box('Empty', [50, 60], [1e9, 2e9])])

    def test_overlapping_boxes_are_independent_and_include_edges(self):
        outputs = self.outputs([box('A', [0, 10], [1e9, 2e9]), box('B', [0, 10], [2e9, 3e9])])
        np.testing.assert_allclose(np.asarray(outputs[0].power), 11.)
        np.testing.assert_allclose(np.asarray(outputs[1].power), 110.)

    def test_elevations_and_polarizations_stay_separate_within_one_dataset(self):
        power = np.arange(24., dtype=float).reshape(2, 2, 3, 2)
        ds = RcsGrid([0., 10.], [0., 5.], [1., 2., 3.], ['HH', 'VV'], rcs_power=power)
        out = self.outputs([box('All', [0, 10], [1e9, 3e9])], ds)[0]
        self.assertEqual(out.shape, ds.shape)
        expected = np.broadcast_to(power.mean(axis=(0, 2), keepdims=True), ds.shape)
        np.testing.assert_allclose(np.asarray(out.power), expected)

    def test_optional_azimuth_selection_intersects_box(self):
        boxes = [box('All', [-1, 15], [1e9, 3e9])]
        _, table = self.render(boxes, selected_azimuths=[0.], use_selected_azimuths=True)
        out = self.outputs(boxes, selection={'azimuths': [0.]})[0]
        self.assertEqual(table['rows'][0][5:7], (3, 37.))
        np.testing.assert_allclose(np.asarray(out.power), 37.)

    def test_boxes_use_each_inputs_native_sampling(self):
        self.dataset.frequencies = np.array([1.1, 1.8, 2.8])
        _, table = self.render(selected=())
        self.assertEqual([row[5:7] for row in table['rows']], [(2, 5.5), (1, 300.)])

    def test_invalid_boxes_are_rejected(self):
        cases = [[], [box('A', [0, 0], [1, 2])], [box('A', [0, 10], [2, 1])],
                 [box('A', [0, 10], [-1, 1])], [box('A', [0, 10], [1, np.inf])],
                 [box('A', [0, 10], [1, 2]), box('A', [10, 20], [2, 3])]]
        for boxes in cases:
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                normalize_boxes(boxes)


if __name__ == '__main__':
    unittest.main()
