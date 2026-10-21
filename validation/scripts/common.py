"""Reproducible fixtures and an Excel-independent comparison specification.

GRIM creates the actual results. Expected values are Excel formulas supplied by
the fixture generators; the workbook builder never calls GRIM to compute them.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'validation'
sys.path.insert(0, str(ROOT))
import numpy as np
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.csv import write_flat_csv


def scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 'missing' if math.isnan(value) else ('-inf' if value < 0 else 'inf')
    return value


def grid(power=None, phase=None, *, field=None, az=None, el=(0.,), freq=(10.,), pol=('VV',), units=None, extra=None):
    if az is None:
        az = list(range(len(power if power is not None else field)))
    shape = (len(az), len(el), len(freq), len(pol))
    metadata = dict(azimuth='deg', elevation='deg', frequency='GHz',
                    rcs_linear_quantity='sigma_3d', rcs_log_unit='dBsm',
                    time_convention='exp(+jwt)', polarization_basis='conic local V/H',
                    phase_reference='synthetic common origin', phase_wrap='-180_180')
    metadata.update(units or {})
    kwargs = {'rcs': np.asarray(field, complex).reshape(shape)} if field is not None else {
        'rcs_power': np.asarray(power, float).reshape(shape),
        'rcs_phase': None if phase is None else np.deg2rad(np.asarray(phase, float).reshape(shape))}
    return RcsGrid(az, el, freq, pol, units=metadata, extra=dict(extra or {}),
                   history='Independent deterministic validation fixture; values defined in validation/scripts.', **kwargs)


def save_grid(slug, name, dataset):
    folder = VALIDATION / 'datasets' / slug
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + '.grim')
    dataset.save(path)
    # Exercise the native archive reader before exporting every actual result.
    loaded = RcsGrid.load(path, storage='memory')
    csv_path = folder / (name + '.csv')
    write_flat_csv(loaded, csv_path, scale='both', include_phase=True)
    with csv_path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    return rows, path.relative_to(VALIDATION).as_posix(), csv_path.relative_to(VALIDATION).as_posix()


def number(row, key):
    value = row.get(key, '')
    if value == '':
        return 'missing'
    try:
        return scalar(float(value))
    except ValueError:
        return value


class Spec:
    def __init__(self, slug, title, description, *, settings='', scope='', category='Dataset operation', atol=1e-10, rtol=1e-9):
        self.data = dict(slug=slug, title=title, description=description, settings=settings,
                         scope=scope or 'Numerical behavior for the supplied fixtures; not all possible inputs.',
                         category=category, atol=atol, rtol=rtol, inputs=[], calculations=[], checks=[], files=[], notes=[])

    def input(self, key, label, value, unit=''):
        self.data['inputs'].append(dict(id=key, label=label, value=scalar(value), unit=unit))
        return '@' + key

    def calc(self, key, label, formula, unit=''):
        self.data['calculations'].append(dict(id=key, label=label, formula=formula, unit=unit))
        return '@' + key

    def save(self, name, dataset):
        rows, native, csv_path = save_grid(self.data['slug'], name, dataset)
        self.data['files'].append(dict(name=name, native=native, csv=csv_path, rows=len(rows)))
        return rows

    def check(self, case, metric, formula, actual, unit='', source='', *, tolerance=None):
        self.data['checks'].append(dict(case=case, metric=metric, formula=formula,
                                       actual=scalar(actual), unit=unit, source=source, tolerance=tolerance))

    def exported(self, case, metric, formula, rows, row, column, unit='', file='result', *, tolerance=None):
        if not unit and column in ('azimuth', 'elevation', 'frequency'):
            unit = rows[row].get(column + '_unit', '')
        self.check(case, metric, formula, number(rows[row], column), unit,
                   f'{file}.csv row {row + 2}: {column}', tolerance=tolerance)

    def reject(self, case, function, *, contains=''):
        try:
            function()
            actual = 'ACCEPTED'
        except ValueError as exc:
            actual = 'REJECTED' if not contains or contains.casefold() in str(exc).casefold() else 'WRONG ERROR'
            self.data['notes'].append(f'{case}: {exc}')
        self.check(case, 'Invalid input handling', '="REJECTED"', actual, source='Recorded GRIM operation response')

    def finish(self):
        target = VALIDATION / 'specs' / (self.data['slug'] + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.data, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
        return self.data
