"""Read-only imaging trusts user selections, with provenance notes rather than vetoes."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.phase_conventions import declared_time_sign
from GRIM_Backend.isar.artifact import build_isar_manifest, save_isar_artifact, load_isar_artifact
from GRIM_Backend.isar.comparison import hydrate_band
from GRIM_Backend.isar.quality import reconstruction_advisories
from GRIM_Backend.plotting.modes import isar_mode as isar
from GRIM_Backend.plotting.modes import isar_render, az_vs_range_mode as azrange, range_freq_mode as rangefreq
from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest


def grid(*, units=None, extra=None, conjugate=False, elevation=30., power=False):
    azimuth = np.linspace(-3., 3., 9)
    frequency = np.linspace(9., 11., 16)
    distance = (.3 * np.cos(np.deg2rad(azimuth[:, None]))
                + .15 * np.sin(np.deg2rad(azimuth[:, None])))
    samples = np.exp(-1j * 4 * np.pi * frequency[None, :] * 1e9 / 299792458 * distance)
    if conjugate:
        samples = samples.conj()
    payload = {'rcs_power': abs(samples[:, None, :, None])**2} if power else {'rcs': samples[:, None, :, None]}
    return RcsGrid(azimuth, [elevation], frequency, ['HH'], **payload,
                   units={'frequency': 'GHz', **(units or {})}, extra=extra)


def form(dataset, **kwargs):
    return isar.form_isar(dataset, retain_complex=True, **kwargs)[0][0]


def request(dataset, mode, **options):
    selection = (range(9), [0], range(16), [0])
    return PlotRequest(mode, dataset, (PlotPlan('Example', dataset, selection),),
                       scale='linear', options={'unit': 'm', 'unit_scale': 1.,
                                              'window': 'Hanning', 'peak_norm': False,
                                              'percent': 50., **options})


class IsarMetadataTests(unittest.TestCase):
    def test_sentri_tag_does_not_change_selected_angles_or_image(self):
        for method in ('fast', 'accurate'):
            for units, extra in (
                ({'elevation_coordinate_convention': isar._SENTRI_NATIVE_ELEVATION}, {}),
                ({}, {'sentri_elevation_convention': isar._SENTRI_NATIVE_ELEVATION}),
            ):
                with self.subTest(method=method, units=units):
                    source = grid(units=units, extra=extra)
                    original_units, original_extra = deepcopy(source.units), deepcopy(source.extra)
                    original_samples = source.rcs_slice(np.s_[:, :, :, :]).copy()
                    actual = form(source, reconstruction=method)
                    baseline = form(grid(), reconstruction=method)
                    np.testing.assert_allclose(actual['complex_image'], baseline['complex_image'])
                    np.testing.assert_array_equal(actual['x_range'], baseline['x_range'])
                    np.testing.assert_array_equal(actual['y_range'], baseline['y_range'])
                    self.assertIn('without automatic SENTRi', ' '.join(actual['isar_metadata_notes']))
                    np.testing.assert_array_equal(source.elevations, [30.])
                    np.testing.assert_array_equal(source.rcs_slice(np.s_[:, :, :, :]), original_samples)
                    self.assertEqual(source.units, original_units)
                    self.assertEqual(source.extra, original_extra)

    def test_other_metadata_vetoes_become_notes(self):
        cases = [
            ({}, {'measurement_geometry': 'bistatic'}),
            ({}, {'measurement_domain': 'near field'}),
            ({}, {'measurement_geometry': 'quasi monostatic'}),
            ({}, {'complex_field_domain': 'Fresnel'}),
            ({}, {'fixed_incident_azimuth_deg': 20.}),
            ({}, {'motion_compensation': False}),
            ({}, {'phase_center_motion': True}),
            ({}, {'phase_reference': 'moving center'}),
            ({}, {'range_phase_convention': 'S~exp(+j*2*k*R)'}),
            ({'elevation_coordinate_convention': 'zenith theta'}, {}),
            ({'elevation_coordinate_convention': isar._GRIM_ELEVATION},
             {'sentri_elevation_convention': isar._SENTRI_NATIVE_ELEVATION}),
            ({'measurement_geometry': 'monostatic'}, {'measurement_geometry': 'bistatic'}),
            ({}, {'phase_reference': ['origin', 'moving center']}),
            ({}, {'motion_compensation': [True, False]}),
            ({'time_convention': '+jwt'}, {'time_convention': '-jwt'}),
            ({}, {'time_convention': ['+jwt', '-jwt']}),
            ({}, {'range_phase_convention': 'exp(+j2kR) and exp(-j2kR)'}),
        ]
        baseline = form(grid())
        for units, extra in cases:
            with self.subTest(units=units, extra=extra):
                actual = form(grid(units=units, extra=extra))
                self.assertTrue(actual['isar_metadata_notes'])
                np.testing.assert_allclose(actual['complex_image'], baseline['complex_image'])
                self.assertTrue(actual['isar_contract_user_assumed'])

    def test_declared_negative_time_still_normalizes_without_modifying_source(self):
        source = grid(units={'time_convention': '-jwt'}, conjugate=True)
        original = source.rcs_slice(np.s_[:, :, :, :]).copy()
        actual = form(source)
        np.testing.assert_allclose(actual['complex_image'], form(grid())['complex_image'])
        self.assertTrue(actual['time_convention_conjugated'])
        np.testing.assert_array_equal(source.rcs_slice(np.s_[:, :, :, :]), original)

    def test_positive_range_law_honors_each_flip_combination(self):
        source = grid(extra={'range_phase_convention': 'exp(+j2kR)'})
        baseline = form(source)
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            with self.subTest(flip_x=flip_x, flip_y=flip_y):
                actual = form(source, flip_x=flip_x, flip_y=flip_y)
                ix, iy = slice(None, None, -1 if flip_x else 1), slice(None, None, -1 if flip_y else 1)
                np.testing.assert_array_equal(actual['complex_image'], baseline['complex_image'][ix, iy])
                np.testing.assert_array_equal(actual['x_range'], (-1 if flip_x else 1) * baseline['x_range'][ix])
                np.testing.assert_array_equal(actual['y_range'], (-1 if flip_y else 1) * baseline['y_range'][iy])

    def test_metadata_cache_does_not_reuse_wrong_time_normalization(self):
        source = grid(units={'time_convention': '-jwt'}, conjugate=True)
        normalized = form(source)
        source.extra['time_convention'] = '+jwt'
        ambiguous = form(source)
        baseline = form(grid(conjugate=True))
        np.testing.assert_allclose(ambiguous['complex_image'], baseline['complex_image'])
        self.assertFalse(ambiguous['time_convention_conjugated'])
        self.assertFalse(np.allclose(normalized['complex_image'], ambiguous['complex_image']))

    def test_strict_parsers_remain_strict_for_dataset_operations(self):
        source = grid(units={'time_convention': '+jwt'}, extra={'time_convention': '-jwt'})
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            declared_time_sign(source)

    def test_numerical_prerequisites_still_report_errors(self):
        for source, options, reason in (
            (grid(power=True), {}, 'phase-aware'),
            (grid(), {'frequency_indices': [0]}, 'at least two'),
            (grid(), {'azimuth_indices': [0, 0]}, 'duplicates'),
            (grid(elevation=90.), {}, 'degenerate'),
            (grid(), {'memory_limit_bytes': 1}, 'working set|working-set'),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                form(source, **options)
        source = grid()
        source.units.pop('frequency')
        with self.assertRaisesRegex(ValueError, 'frequency units are missing'):
            form(source)

    def test_gui_submits_sentri_selection_and_artifact_retains_notes(self):
        source = grid(extra={'sentri_elevation_convention': isar._SENTRI_NATIVE_ELEVATION,
                            'measurement_geometry': 'bistatic'})
        messages, submitted = [], []
        combo = lambda text: SimpleNamespace(currentText=lambda: text)
        app = SimpleNamespace(active_dataset=source, plot_figure=object(),
            status=SimpleNamespace(showMessage=messages.append),
            _start_plot_render=lambda: None, _preflight_plot_datasets=lambda _: source,
            list_az=list(range(9)), list_freq=list(range(16)), list_pol=[0], list_elev=[0],
            _selected_indices=lambda values: values, _single_selection_index=lambda values, _: values[0],
            combo_isar_window=combo('Hanning'), combo_isar_units=combo('m'),
            _isar_submit=submitted.append)
        isar_render.render(app)
        self.assertEqual(len(submitted), 1, messages)
        params = submitted[0]
        self.assertEqual(params['elevation_deg'], 30.)
        self.assertTrue(params['isar_metadata_notes'])
        params['retain_complex'] = True
        bands, elapsed = isar.compute_bands(params)
        self.assertTrue(reconstruction_advisories(bands))
        manifest = build_isar_manifest(source, params, bands, elapsed)
        with TemporaryDirectory() as temp:
            path = save_isar_artifact(Path(temp) / 'sentri', bands, manifest)
            saved, images = load_isar_artifact(path)
            hydrated = hydrate_band(images[0], saved)
        self.assertEqual(saved['formation']['isar_metadata_notes'], bands[0]['isar_metadata_notes'])
        self.assertEqual(hydrated['isar_metadata_notes'], bands[0]['isar_metadata_notes'])
        np.testing.assert_array_equal(images[0]['complex_image'], bands[0]['complex_image'])


class RangeMetadataTests(unittest.TestCase):
    def test_azimuth_range_positive_phase_does_not_require_flip(self):
        baseline = azrange.prepare_azimuth_range(request(grid(), 'az_vs_range')).items[0]
        source = grid(extra={'range_phase_convention': 'exp(+j2kR)'})
        actual = azrange.prepare_azimuth_range(request(source, 'az_vs_range')).items[0]
        np.testing.assert_array_equal(actual['display'], baseline['display'])
        self.assertIn('Flip Y setting is preserved', ' '.join(actual['metadata_notes']))
        mirrored = azrange.prepare_azimuth_range(request(source, 'az_vs_range', flip_y=True)).items[0]
        np.testing.assert_array_equal(mirrored['display'], actual['display'][:, ::-1])
        np.testing.assert_array_equal(mirrored['range_axis'], -actual['range_axis'][::-1])

    def test_azimuth_range_ambiguous_time_uses_stored_samples(self):
        source = grid(units={'time_convention': '+jwt'}, extra={'time_convention': '-jwt'})
        actual = azrange.prepare_azimuth_range(request(source, 'az_vs_range')).items[0]
        baseline = azrange.prepare_azimuth_range(request(grid(), 'az_vs_range')).items[0]
        np.testing.assert_array_equal(actual['display'], baseline['display'])
        self.assertFalse(actual['time_convention_conjugated'])
        self.assertTrue(actual['metadata_notes'])

    def test_range_frequency_ambiguous_metadata_does_not_skip_dataset(self):
        source = grid(units={'time_convention': '+jwt'}, extra={'time_convention': '-jwt',
                      'range_phase_convention': 'exp(+j2kR) and exp(-j2kR)'})
        actual = rangefreq.prepare_range_frequency(request(source, 'range_frequency'))
        baseline = rangefreq.prepare_range_frequency(request(grid(), 'range_frequency'))
        self.assertEqual(len(actual.items), 1, actual.skipped)
        self.assertTrue(actual.notes)
        np.testing.assert_array_equal(actual.items[0]['image'], baseline.items[0]['image'])


if __name__ == '__main__':
    unittest.main()
