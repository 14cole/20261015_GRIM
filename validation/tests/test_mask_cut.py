"""Mask geometry, calibrated inverse FFT, native phase history, and Qt publication."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.isar.masking import (MaskImage, image_from_band, cut_image, finish_cut,
                                      inverse_spectrum, reconstruct_radar, save_image, load_image)
from GRIM_Backend.plotting.modes import isar_mode as isar, isar_render


def grid(negative=False):
    az = np.linspace(-3, 3, 9)
    freq = np.linspace(9, 11, 16)
    phase = -4*np.pi * freq[None, :] * 1e9 / 299792458 * (.2 + .1*np.sin(np.deg2rad(az[:, None])))
    field = np.exp(1j*phase).astype(np.complex64)
    return RcsGrid(az, [0.], freq, ['HH'], rcs=(field.conj() if negative else field)[:, None, :, None],
                   units={'frequency': 'GHz', 'time_convention': '-jwt' if negative else '+jwt'})


def form(source=None, **options):
    source = source or grid()
    band = isar.form_isar(source, retain_complex=True, length_unit='m', **options)[0][0]
    return image_from_band(band, {'dataset': source, 'unit_name': 'm'})


def rectangle(x0, x1, y0, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def all_pixels(image):
    return rectangle(image.x[0]-1, image.x[-1]+1, image.y[0]-1, image.y[-1]+1)


class MaskNumericalTests(unittest.TestCase):
    def test_inverse_normalization_matches_weighted_windowed_fourier_input(self):
        rng = np.random.default_rng(42)
        data = (rng.normal(size=(7, 11)) + 1j*rng.normal(size=(7, 11))).astype(np.complex64)
        weights = rng.uniform(.2, 1, size=data.shape).astype(np.float32)
        theta = np.linspace(-.03, .03, 7)
        freq = np.linspace(9e9, 11e9, 11)
        for window in ('Rectangular', 'Hanning', 'Hamming', 'Blackman'):
            with self.subTest(window=window):
                values, x, y = isar._compute_band_polar_format(window, data, theta, freq, np.diff(freq).mean(), 1.,
                                           sample_weights=weights, apply_sample_weights=True)
                wa = isar._window_array(window, 7).astype(np.float32)
                wf = isar._window_array(window, 11).astype(np.float32)
                expected = data*weights*wa[:, None]*wf
                gain = sum(np.sum(weights[i]*wa[i]*wf, dtype=np.float64) for i in range(7))
                context = {'full_x': x, 'full_y': y, 'u': theta, 'v': freq, 'gain': gain}
                image = MaskImage('Example', x, y, abs(values), values, context)
                actual = inverse_spectrum(image)
                np.testing.assert_allclose(actual, expected, atol=6e-7)
                # Fully included image is exactly the same operation as no cut.
                cut = cut_image(image, all_pixels(image), include=True, name='Full')
                np.testing.assert_array_equal(cut.complex_image, values)

    def test_polygon_boundaries_unknown_pixels_and_source_immutability(self):
        values = np.ones((5, 5), dtype=np.float32)
        values[0, 0] = np.nan
        source = MaskImage('Composite', np.arange(5.), np.arange(5.), values, metadata={'kind': 'composite'})
        vertices = [(1, 1), (3, 1), (1, 3)]
        include, radar = finish_cut(source, vertices, include=True, name='Inside')
        exclude = cut_image(source, vertices[::-1], include=False, name='Outside')
        self.assertIsNone(radar)
        self.assertEqual(include.magnitude[1, 1], 1)
        self.assertEqual(include.magnitude[3, 3], 0)
        np.testing.assert_allclose(include.magnitude + exclude.magnitude, source.magnitude)
        self.assertTrue(np.isnan(include.magnitude[0, 0]))
        np.testing.assert_allclose(source.magnitude, values)
        self.assertFalse(np.shares_memory(include.magnitude, source.magnitude))
        self.assertEqual(source.metadata, {'kind': 'composite'})
        with self.assertRaisesRegex(ValueError, 'no image pixel'):
            cut_image(source, rectangle(20, 21, 20, 21), include=True, name='Empty')
        with self.assertRaisesRegex(ValueError, 'empty'):
            cut_image(source, rectangle(1, 1, 1, 3), include=True, name='Empty')

    def test_include_and_exclude_reconstruct_complementary_phase_history(self):
        for method in ('fft', 'accurate'):
            with self.subTest(method=method):
                source = grid()
                image = form(source, reconstruction=method)
                original = source.rcs_slice(np.s_[:, :, :, :]).copy()
                vertices = rectangle(-.3, .3, 0., .4)
                included, inside = finish_cut(image, vertices, include=True, name='Inside')
                excluded, outside = finish_cut(image, vertices, include=False, name='Outside')
                baseline = reconstruct_radar(image)
                np.testing.assert_allclose(inside.rcs + outside.rcs, baseline.rcs, atol=5e-7)
                np.testing.assert_allclose(included.complex_image + excluded.complex_image, image.complex_image)
                np.testing.assert_array_equal(inside.azimuths, source.azimuths)
                np.testing.assert_array_equal(inside.frequencies, source.frequencies)
                self.assertEqual(inside.units, source.units)
                self.assertIn('Approximate reconstructed', inside.history)
                self.assertIn('mask_cut_json', inside.extra)
                np.testing.assert_array_equal(source.rcs_slice(np.s_[:, :, :, :]), original)

    def test_mirrors_and_scene_crops_use_original_fourier_coordinates(self):
        baseline = form(reconstruction='accurate')
        expected = reconstruct_radar(baseline).rcs
        for flip_x, flip_y in ((True, False), (False, True), (True, True)):
            mirrored = form(reconstruction='accurate', flip_x=flip_x, flip_y=flip_y)
            np.testing.assert_allclose(reconstruct_radar(mirrored).rcs, expected, atol=1e-6)
        cropped = form(reconstruction='accurate', flip_x=True, scene_half_extent_m=(.4, .4))
        full_mirrored = form(reconstruction='accurate', flip_x=True)
        dx, dy = np.diff(cropped.x)[0], np.diff(cropped.y)[0]
        vertices = rectangle(cropped.x[0]-dx/4, cropped.x[-1]+dx/4,
                             cropped.y[0]-dy/4, cropped.y[-1]+dy/4)
        cut = cut_image(full_mirrored, vertices, include=True, name='Crop')
        np.testing.assert_allclose(reconstruct_radar(cropped).rcs, reconstruct_radar(cut).rcs, atol=1e-6)

    def test_time_convention_missing_samples_and_cartesian_support(self):
        positive = form(grid(), reconstruction='accurate')
        negative = form(grid(True), reconstruction='accurate')
        np.testing.assert_allclose(reconstruct_radar(negative).rcs,
                                   reconstruct_radar(positive).rcs.conj(), atol=1e-6)
        source = grid()
        data = source.rcs.copy()
        data[4, 0, 7, 0] = complex(np.nan, np.nan)
        missing = RcsGrid(source.azimuths, source.elevations, source.frequencies, source.polarizations,
                          rcs=data, units=source.units)
        radar = reconstruct_radar(form(missing, reconstruction='accurate'))
        self.assertTrue(np.isnan(radar.rcs[4, 0, 7, 0]))
        self.assertTrue(np.isnan(radar.rcs[0, 0, 0, 0]))  # Outside inscribed Cartesian support.

    def test_saved_complex_cut_reopens_with_reconstruction_and_composite_without_phase(self):
        for image in (form(), form(aperture_mode='composite', composite_side=32)):
            with self.subTest(kind=image.metadata['kind']), TemporaryDirectory() as temp:
                result, radar = finish_cut(image, all_pixels(image), include=True, name='Saved cut')
                path = Path(temp)/'cut.npz'
                save_image(path, result)
                reopened = load_image(path)
                np.testing.assert_allclose(reopened.magnitude, result.magnitude)
                self.assertEqual(reopened.metadata, result.metadata)
                if radar is None:
                    self.assertIsNone(reopened.complex_image)
                    self.assertIsNone(reopened.inverse)
                else:
                    np.testing.assert_allclose(reconstruct_radar(reopened).rcs, radar.rcs)
                    radar_path = Path(temp)/'radar.grim'
                    radar.save(radar_path)
                    radar_reopened = RcsGrid.load(radar_path)
                    np.testing.assert_allclose(radar_reopened.rcs, radar.rcs)
                    self.assertEqual(radar_reopened.extra['mask_cut_json'], radar.extra['mask_cut_json'])

    def test_reconstructed_metadata_preserves_phase_origin_but_not_solver_certification(self):
        source = grid()
        source.extra.update(phase_center_xyz_m=[1., 2., 3.], solver_certification='certified',
                            assembly_base_response_sha256='original-sample-hash')
        radar = reconstruct_radar(form(source))
        self.assertEqual(radar.extra['phase_center_xyz_m'], [1., 2., 3.])
        self.assertNotIn('solver_certification', radar.extra)
        self.assertNotIn('assembly_base_response_sha256', radar.extra)
        self.assertEqual(radar.extra['assembly_source_base_response_sha256'], 'original-sample-hash')
        self.assertEqual(source.extra['solver_certification'], 'certified')

    def test_cancel_does_not_modify_source(self):
        from GRIM_Backend.plotting.requests import PlotCancelled
        image = form()
        original = image.magnitude.copy()
        with self.assertRaises(PlotCancelled):
            finish_cut(image, all_pixels(image), include=True, name='Cancelled', cancel=lambda: True)
        np.testing.assert_array_equal(image.magnitude, original)

    def test_headless_replay_keeps_explicit_plot_identity(self):
        from GRIM_Backend.scripting.plotting import plot_datasets
        source = grid()
        for mode, name in (('isar_image', 'ISAR'), ('isar_composite', 'Composite')):
            figure = plot_datasets([('Example', source)], mode=mode,
                azimuths=source.azimuths, elevations=source.elevations,
                frequencies=source.frequencies, polarization='HH',
                isar_options={'aperture_mode': 'auto', 'composite_side': 32})
            self.assertTrue(figure.axes[0].get_title().startswith(name + ' Image'))
            figure.clear()

    def test_display_decimation_does_not_change_cut_resolution(self):
        source = grid()
        with patch.object(isar.common, 'MAX_IMAGE_SIDE', 32):
            bands, _ = isar.compute_bands({
                'dataset': source, 'bands': [list(range(9))], 'freq_indices_sorted': list(range(16)),
                'elev_idx': 0, 'pol_idx': 0, 'elevation_deg': 0., 'freq_hz': source.frequencies*1e9,
                'df': np.mean(np.diff(source.frequencies))*1e9, 'unit_scale': 1.,
                'az_target_deg': None, 'window_name': 'Hanning', 'recon': 'fft',
                'retain_complex': True, 'retain_mask_source': True, 'aperture_mode': 'coherent'})
        band = bands[0]
        self.assertEqual(band['magnitude'].shape, (32, 32))
        image = image_from_band(band, {'unit_name': 'm'})
        self.assertEqual(image.magnitude.shape, (256, 256))
        cut = cut_image(image, all_pixels(image), include=True, name='Native cut')
        np.testing.assert_array_equal(cut.complex_image, image.complex_image)

    def test_gui_requests_are_distinct_and_keep_native_mask_sources(self):
        source = grid()
        messages, submitted = [], []
        combo = lambda text: SimpleNamespace(currentText=lambda: text)
        app = SimpleNamespace(active_dataset=source, plot_figure=object(),
            status=SimpleNamespace(showMessage=messages.append),
            _start_plot_render=lambda: None, _preflight_plot_datasets=lambda _: source,
            list_az=list(range(9)), list_freq=list(range(16)), list_pol=[0], list_elev=[0],
            _selected_indices=lambda values: values, _single_selection_index=lambda values, _: values[0],
            combo_isar_window=combo('Hanning'), combo_isar_units=combo('m'),
            _isar_submit=submitted.append)
        for composite in (False, True, False):
            isar_render.render(app, composite=composite)
            params = submitted[-1]
            self.assertEqual(params['plot_mode'], 'isar_composite' if composite else 'isar_image')
            self.assertEqual(params['aperture_mode'], 'composite' if composite else 'coherent')
            params['composite_side'] = 32
            bands, _ = isar.compute_bands(params)
            self.assertEqual(bool(bands[0].get('composite')), composite)
            self.assertEqual('complex_image' in bands[0], not composite)
            self.assertEqual(bands[0]['mask_magnitude'].shape,
                             (len(bands[0]['x_range']), len(bands[0]['y_range'])))
        self.assertFalse(messages)


class MaskDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_finish_creates_both_datasets_and_reopen_keeps_cut(self):
        from PySide6.QtWidgets import QWidget
        from matplotlib.figure import Figure
        from GRIM_Backend.ui.mask_cut import MaskCutDialog
        owner = QWidget()
        owner.plot_figure = Figure()
        image = form()
        artist = owner.plot_figure.add_subplot(111).imshow(image.magnitude.T)
        artist._grim_mask_source = image
        outputs = []
        owner._add_dataset_row = lambda *args: outputs.append(args)
        dialog = MaskCutDialog(owner)
        self.assertEqual(dialog.source.count(), 1)
        dialog._polygon(all_pixels(image))
        dialog._finish()
        deadline = time.monotonic()+10
        while dialog.runner.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertFalse(dialog.runner.busy)
        self.assertEqual(len(outputs), 1, dialog.message.text())
        self.assertEqual(len(owner._mask_image_datasets), 1)
        self.assertEqual(dialog.source.count(), 2)
        self.assertFalse(dialog.finish.isEnabled())  # A new selection is required.
        dialog.close()
        reopened = MaskCutDialog(owner)
        self.assertEqual(reopened.source.count(), 2)
        reopened.close()
        owner.close()

    def test_main_window_buttons_render_distinct_images_and_open_mask_cut(self):
        from PySide6.QtCore import QSettings
        from GRIM_Backend.ui.app import GrimCutWindow
        with TemporaryDirectory() as folder:
            window = GrimCutWindow(settings=QSettings(str(Path(folder)/'settings.ini'), QSettings.IniFormat))
            try:
                window._add_dataset_row(grid(), 'Mask Cut test source', '', dirty=False)
                window.table.selectRow(0)
                window._activate_plot_tab('isar')
                for widget in (window.list_az, window.list_freq, window.list_elev, window.list_pol):
                    widget.selectAll()
                window.isar_advanced.side.setValue(32)
                controls = window._plot_controls_by_tab['isar']
                for mode in ('isar_image', 'isar_composite', 'isar_image'):
                    controls[mode].click()
                    deadline = time.monotonic()+10
                    while window._isar_busy and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(.01)
                    self.assertFalse(window._isar_busy)
                    self.assertEqual(window.last_plot_mode, mode)
                    self.assertTrue(window._isar_figure_is_current(), window.status.currentMessage())
                    image = window._isar_meshes[0]._grim_mask_source
                    self.assertEqual(image.metadata['kind'], 'composite' if mode == 'isar_composite' else 'isar')
                    self.assertEqual(window.isar_advanced.mode.currentData(),
                                     'composite' if mode == 'isar_composite' else 'coherent')
                controls['mask_cut'].click()
                self.assertEqual(window._mask_cut_dialog.source.count(), 1)
                window._mask_cut_dialog.close()
            finally:
                window._python_clean_script = window.python_recorder.script
                window.close()

    def test_close_without_finishing_creates_nothing(self):
        from PySide6.QtWidgets import QWidget
        from GRIM_Backend.ui.mask_cut import MaskCutDialog
        owner = QWidget()
        dialog = MaskCutDialog(owner)
        self.assertFalse(dialog.finish.isEnabled())
        dialog.close()
        self.assertEqual(owner._mask_image_datasets, [])
        owner.close()


if __name__ == '__main__':
    unittest.main()
