"""Explicit paired azimuth/frequency populations for sector statistics."""
from __future__ import annotations

import numpy as np


def normalize_boxes(boxes):
    """Validate independent boxes in physical degrees/Hz, preserving row order."""
    if not boxes:
        raise ValueError('Add at least one box with azimuth and frequency bounds.')
    if len(boxes) > 720:
        raise ValueError('Use at most 720 boxes.')
    result, names = [], set()
    for index, box in enumerate(boxes, 1):
        name = str(box.get('name', '')).strip() or f'Box {index}'
        if name in names:
            raise ValueError(f'Box {index}: choose a unique box name ({name!r} is repeated).')
        if any(char in name for char in '\r\n\t'):
            raise ValueError(f'Box {index}: the name must be a single line without tabs.')
        names.add(name)
        bounds = {}
        for axis in ('azimuth', 'frequency'):
            try:
                pair = np.asarray(box.get(f'{axis}_band'), dtype=float)
            except (ValueError, TypeError):
                raise ValueError(f'{name}: enter numeric {axis} Start and Stop values.') from None
            if pair.shape != (2,) or not np.isfinite(pair).all():
                raise ValueError(f'{name}: enter two finite {axis} bounds.')
            if axis == 'azimuth' and pair[0] == pair[1]:
                raise ValueError(f'{name}: azimuth Start and Stop must differ (use 0 to 360 for a full turn).')
            if axis == 'frequency' and (pair[0] < 0 or pair[1] < pair[0]):
                raise ValueError(f'{name}: frequency bounds must be nonnegative with Stop at or above Start.')
            bounds[f'{axis}_band'] = pair.tolist()
        result.append(dict(name=name, **bounds))
    return result


def box_specs(boxes):
    """One operation spec per row; never form a product of separate band lists."""
    return [dict(azimuth_band=box['azimuth_band'], frequency_band=box['frequency_band'],
                 azimuth_closed=True, azimuth_modular=True, box_name=box['name'])
            for box in normalize_boxes(boxes)]
