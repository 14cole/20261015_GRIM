"""Statistics on bounded populations, retained on their original coordinates."""
from __future__ import annotations

import json
import numpy as np

from .constants import _ANGLE_UNITS, _FREQUENCY_UNITS
from .memory import _bounded_grid_selections
from .operations import dataset_operation
from .result_storage import copy_result
from .work_control import checkpoint


def _bounds(values, name, *, allow_equal=False):
    if values is None:
        return None
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f'{name} bounds must contain two finite numbers.')
    if bounds[0] == bounds[1] and not allow_equal:
        raise ValueError(f'{name} bounds must span a nonzero interval.')
    return bounds


def band_selection(dataset, *, azimuth_band=None, frequency_band=None,
                   azimuth_closed=True, azimuth_modular=True, selection=None):
    """Resolve physical bounds (degrees/Hz) and optional native plot selections."""
    selection = selection or {}
    if set(selection) - {'azimuths', 'elevations', 'frequencies', 'polarizations'}:
        raise ValueError('Unknown selection axis.')
    selected = {}
    for axis in ('azimuths', 'elevations', 'frequencies', 'polarizations'):
        values = np.asarray(getattr(dataset, axis))
        requested = selection.get(axis)
        if requested is not None:
            indices = dataset._indices_for_axis_values(
                values, requested, tol=0.0 if axis == 'polarizations' else 1e-6)
            if indices is None or not len(indices):
                raise ValueError(f'No matching selected {axis}.')
            values = values[indices]
        selected[axis] = values
    az = _bounds(azimuth_band, 'Azimuth')
    freq = _bounds(frequency_band, 'Frequency', allow_equal=True)
    if az is not None:
        unit = dataset._supported_unit('azimuth', _ANGLE_UNITS, 'deg')
        values = selected['azimuths']
        degrees = np.rad2deg(values) if unit == 'rad' else values
        eps = 360e-9
        width = az[1] - az[0]
        if azimuth_modular:
            width_mod = width % 360.
            width = 360. if abs(width) >= 360. - eps and min(width_mod, 360. - width_mod) <= eps else width_mod
            offset = (degrees - az[0] + eps) % 360. - eps
        else:
            if width <= 0:
                raise ValueError('Non-wrapping azimuth bands need increasing bounds.')
            offset = degrees - az[0]
        mask = (np.isfinite(degrees) & (offset >= -eps) &
                (offset <= width + eps if azimuth_closed else offset < width - eps))
        selected['azimuths'] = values[mask]
    if freq is not None:
        if freq[1] < freq[0]:
            raise ValueError('Frequency bounds must be increasing.')
        unit = dataset._supported_unit('frequency', _FREQUENCY_UNITS, 'GHz')
        hz = selected['frequencies'] * {'Hz': 1., 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9}[unit]
        eps = max(abs(freq).max(), 1.) * 1e-12
        selected['frequencies'] = selected['frequencies'][(hz >= freq[0] - eps) & (hz <= freq[1] + eps)]
    empty = [axis for axis, values in selected.items() if not len(values)]
    if empty:
        raise ValueError('No samples inside the requested bounds: ' + ', '.join(empty))
    return selected


@dataset_operation('band_statistics_dataset')
def band_statistics_dataset(dataset, *, statistic='percentile', percentile=90.,
                            axes=('azimuth',), azimuth_band=None, frequency_band=None,
                            interpolate=False, azimuth_closed=True,
                            azimuth_modular=True, selection=None, box_name=None):
    """Return one bounded dataset with the statistic repeated within the band.

    If both axes are reduced, their samples form one joint population for each
    elevation/polarization. Adjacent explicit bands share boundary samples;
    sector callers can request half-open membership for uniformly tiled sectors.
    """
    axes = tuple(axes)
    if not axes or set(axes) - {'azimuth', 'frequency'}:
        raise ValueError('Reduce azimuth, frequency, or both.')
    selected = band_selection(
        dataset, azimuth_band=azimuth_band, frequency_band=frequency_band,
        azimuth_closed=azimuth_closed, azimuth_modular=azimuth_modular, selection=selection)
    checkpoint()
    cropped = dataset.axis_crop(**selected)
    # Match the sector plot's finite, nonnegative linear-power population,
    # including physical zeros and excluding unknown or invalid samples.
    power = copy_result(cropped.power)
    for block in _bounded_grid_selections(power.shape, 65536):
        checkpoint()
        values = power[block]
        values[~np.isfinite(values) | (values < 0)] = np.nan
    scalar = cropped._new_grid(
        cropped.azimuths, cropped.elevations, cropped.frequencies, cropped.polarizations,
        rcs_power=power,
        extra=cropped._derived_response_extra(operation='band-statistics', coherent=False),
        _adopt_clean_arrays=cropped._adopt_owned_samples(rcs_power=power),
    )
    result = scalar.statistics_dataset(
        statistic=statistic, percentile=percentile, axes=axes,
        domain='magnitude', interpolate=interpolate, broadcast_reduced=True)
    result.extra['statistics_bands_json'] = json.dumps({
        'schema': 'grim.statistics-bands.v1',
        'azimuth_bounds_deg': None if azimuth_band is None else list(map(float, azimuth_band)),
        'frequency_bounds_hz': None if frequency_band is None else list(map(float, frequency_band)),
        'azimuth_closed': bool(azimuth_closed), 'azimuth_modular': bool(azimuth_modular),
        'coordinate_semantics': 'original coordinates inside the requested bounds; repeated aggregate',
    }, allow_nan=False, sort_keys=True)
    if box_name is not None:
        result.extra['statistics_box_name'] = str(box_name)
    return result
