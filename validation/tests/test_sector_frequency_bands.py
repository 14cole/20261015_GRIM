"""Headless regressions: python -m unittest discover -s validation/tests -v.

Load the renderer without modes/__init__.py's unrelated image-mode imports.
The renderer, common helpers, NumPy indexing and statistics are real; only
the application/widget boundary is replaced by a small test harness.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
package = ModuleType('_sector_test_modes')
package.__path__ = [str(ROOT / 'GRIM_Backend/plotting/modes')]
sys.modules[package.__name__] = package
mode = importlib.import_module('_sector_test_modes.sector_stats_mode')
common = mode.common


class Dataset:
    def __init__(self, power, frequency_unit='GHz', log_unit='dBsm'):
        self.azimuths = np.array([0., 10.])
        self.elevations = np.array([0.])
        self.frequencies = common.convert_axis_values([1., 2., 3.], 'frequency', 'GHz', frequency_unit)
        self.polarizations = np.array(['HH'])
        self.units = dict(azimuth='deg', elevation='deg', frequency=frequency_unit)
        self.power = np.asarray(power, dtype=float).reshape(2, 1, 3, 1)
        self.log_unit = log_unit

    def linear_to_default_db(self, values, frequency_value=None, eps=0.):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = 10 * np.log10(values)
        if self.log_unit == 'dBke':
            hz = common.convert_axis_values(frequency_value, 'frequency', self.units['frequency'], 'Hz')
            result += 10 * np.log10(2 * np.pi * hz / 299792458.)
        return result


class Harness:
    def __init__(self, dataset, frequency_mode='individual', bands=(), selected=(1., 2., 3.),
                 linear=True, percentile=50., interpolate=False, selected_azimuths=None,
                 use_selected_azimuths=False):
        self.dataset = dataset
        self.linear = linear
        self.list_az, self.list_freq, self.list_elev = 'az', 'freq', 'elev'
        self.selected = {'az': dataset.azimuths, 'freq': common.convert_axis_values(
            selected, 'frequency', 'GHz', dataset.units['frequency']), 'elev': dataset.elevations}
        if selected_azimuths is not None:
            self.selected['az'] = np.asarray(selected_azimuths, dtype=float)
        self.analysis_controls = SimpleNamespace(
            sector_statistic=lambda: 'mean', sector_percentile=lambda: percentile,
            sector_interpolate=lambda: interpolate, sector_text=lambda **kwargs: '0:10',
            sector_frequency_mode=lambda: frequency_mode, sector_frequency_bands=lambda: bands,
            sector_use_selected_azimuths=lambda: use_selected_azimuths)
        self.status = SimpleNamespace(showMessage=self._show_plot_status)
        self.btn_phase = None
        self.plot_ax = Mock(texts=[])
        self.plot_figure = SimpleNamespace()
        self.spin_plot_xmin = Mock()
        self.spin_plot_xmax = Mock()
        self.spin_plot_xmin.decimals.return_value = self.spin_plot_xmax.decimals.return_value = 6
        for spin in (self.spin_plot_xmin, self.spin_plot_xmax):
            spin.minimum.return_value, spin.maximum.return_value = -360., 360.
        self.messages, self.lines = [], []

    def _selected_datasets(self): return [('test', self.dataset)]
    def _preflight_plot_datasets(self, datasets): return self.dataset
    def _button_checked(self, button): return False
    def _selected_values(self, widget): return self.selected[widget]
    def _overlay_polarizations(self): return ['HH']
    def _plot_axis_unit(self, dataset, axis): return common.axis_unit(dataset, axis)
    def _plot_axis_name(self, dataset, axis): return axis.title()
    def _plot_axis_label(self, dataset, axis): return axis.title()
    def _display_unit(self, datasets): return 'linear' if self.linear else self.dataset.log_unit
    def _display_axis_label(self, datasets, tag=''): return tag
    def _plot_scale_is_linear(self): return self.linear
    def _plot_axis_values(self, reference, dataset, axis, values):
        return common.values_for_display(reference, dataset, axis, values)
    def _axis_selection_for_dataset(self, reference, dataset, axis, values):
        coordinates = getattr(dataset, {'azimuth': 'azimuths', 'elevation': 'elevations',
                                        'frequency': 'frequencies'}[axis])
        return np.array([np.flatnonzero(coordinates == value)[0] for value in values])
    _overlay_axis_selection = _axis_selection_for_dataset
    def _indices_for_values(self, values, requested, tol=0.): return np.array([0])
    def _show_plot_status(self, message): self.messages.append(message)
    _note_plot_render = _show_plot_status
    def _start_plot_render(self): pass
    def _configure_line_budget(self, count): self.budget = count
    def _prepare_line_plot_axes(self, *args): return True
    def _plot_bounded_line(self, ax, x, y, **kwargs): self.lines.append((x, y, kwargs))
    def _update_legend_visibility(self): pass
    def _apply_plot_limits(self): pass

    def render(self):
        mode.render(self)
        return self.plot_figure._grim_sector_table


class SectorFrequencyTests(unittest.TestCase):
    def test_all_azimuths_are_used_by_default(self):
        app = Harness(Dataset([[1, 10, 100], [3, 30, 300]]), 'selected', selected_azimuths=[0.])
        row = app.render()['rows'][0]
        self.assertEqual(row[5], 6)
        self.assertEqual(row[6], 74.)

    def test_checked_selection_limits_azimuth_population(self):
        app = Harness(Dataset([[1, 10, 100], [3, 30, 300]]), 'selected',
                      selected_azimuths=[0.], use_selected_azimuths=True)
        row = app.render()['rows'][0]
        self.assertEqual(row[5], 3)
        self.assertEqual(row[6], 37.)

    def test_empty_azimuth_selection_does_not_limit_default(self):
        app = Harness(Dataset(np.ones((2, 3))), 'selected', selected_azimuths=[])
        self.assertEqual(app.render()['rows'][0][5], 6)

    def test_checked_empty_azimuth_selection_reports_missing_selection(self):
        app = Harness(Dataset(np.ones((2, 3))), 'selected',
                      selected_azimuths=[], use_selected_azimuths=True)
        mode.render(app)
        self.assertEqual(app.messages[-1], 'Select one or more azimuths to plot.')
        self.assertFalse(app.lines)

    def test_default_keeps_frequencies_separate(self):
        table = Harness(Dataset([[1, 10, 100], [3, 30, 300]])).render()
        self.assertEqual([row[2] for row in table['rows']], [1., 2., 3.])
        self.assertEqual([row[5] for row in table['rows']], [2, 2, 2])
        self.assertEqual([row[6] for row in table['rows']], [2., 20., 200.])
        self.assertIn('Frequency\tElevation', mode.table_text(table))

    def test_joint_population_excludes_invalid_but_includes_zero(self):
        app = Harness(Dataset([[0, 2, np.nan], [4, -1, 8]]), 'selected')
        row = app.render()['rows'][0]
        self.assertEqual(row[5:], (4, 3.5, 3., 0., 8., 2.))
        self.assertEqual(len(app.lines), 1)

    def test_percentile_interpolation_uses_joint_samples(self):
        table = Harness(Dataset([[0, 2, np.nan], [4, -1, 8]]), 'selected', interpolate=True).render()
        self.assertEqual(table['rows'][0][-1], 3.)

    def test_custom_overlapping_bands_include_shared_edge(self):
        app = Harness(Dataset([[1, 10, 100], [3, 30, 300]]), 'custom', [(1e9, 2e9), (2e9, 3e9)])
        rows = app.render()['rows']
        self.assertEqual([row[5] for row in rows], [4, 4])
        self.assertEqual([row[6] for row in rows], [11., 110.])

    def test_bands_do_not_include_unselected_frequencies(self):
        row = Harness(Dataset([[1, 1000, 5], [3, 1000, 7]]), 'custom',
                      [(1e9, 3e9)], selected=(1., 3.)).render()['rows'][0]
        self.assertEqual(row[5:7], (4, 4.))

    def test_hz_and_ghz_storage_choose_same_samples(self):
        for unit in ('Hz', 'kHz', 'MHz', 'GHz'):
            with self.subTest(unit=unit):
                row = Harness(Dataset([[1, 10, 100], [3, 30, 300]], unit), 'custom',
                              [(1e9, 2e9)]).render()['rows'][0]
                self.assertEqual(row[5:7], (4, 11.))

    def test_empty_band_is_reported_and_does_not_hide_valid_band(self):
        app = Harness(Dataset([[1, 10, 100], [3, 30, 300]]), 'custom', [(4e9, 5e9), (1e9, 2e9)])
        self.assertEqual(len(app.render()['rows']), 1)
        self.assertIn('no selected samples', app.messages[-1])

    def test_all_empty_bands_report_no_levels(self):
        app = Harness(Dataset(np.ones((2, 3))), 'custom', [(4e9, 5e9)])
        self.assertEqual(app.render()['rows'], [])
        self.assertIn('No finite levels', app.messages[-1])

    def test_invalid_custom_bounds_fail(self):
        for bands in ([], [(2, 1)], [(1, 1)], [(0, np.inf)], [(np.nan, 2)]):
            with self.subTest(bands=bands), self.assertRaises(ValueError):
                mode.frequency_groups([1, 2], 'GHz', 'custom', bands)

    def test_log_conversion_happens_after_linear_aggregation(self):
        table = Harness(Dataset([[1, 1, 1], [100, 100, 100]]), 'selected', linear=False).render()
        self.assertAlmostEqual(table['rows'][0][6], 10 * np.log10(50.5))

    def test_zero_and_missing_populations_remain_distinct(self):
        zero = Harness(Dataset(np.zeros((2, 3))), 'selected', linear=False).render()['rows'][0]
        missing = Harness(Dataset(np.full((2, 3), np.nan)), 'selected', linear=False).render()['rows'][0]
        self.assertEqual(zero[5], 6)
        self.assertTrue(np.isneginf(zero[6]))
        self.assertEqual(missing[5], 0)
        self.assertTrue(np.isnan(missing[6]))

    def test_dbke_reference_is_explicit_in_legend_and_table(self):
        app = Harness(Dataset(np.ones((2, 3)), log_unit='dBke'), 'selected', linear=False)
        table = app.render()
        self.assertEqual(table['frequency_references'], [2.])
        self.assertIn('dBke ref 2 GHz', app.lines[0][2]['label'])
        self.assertIn('dBke reference frequency (GHz)', mode.table_text(table))
        self.assertIn('Frequency band (GHz)', mode.table_text(table))


if __name__ == '__main__':
    unittest.main()
