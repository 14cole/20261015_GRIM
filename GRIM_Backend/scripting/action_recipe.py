"""Portable, declarative dataset workflows. Recipe files never execute Python."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import math

import numpy as np

from GRIM_Backend.datasets import transforms
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.datasets.statistics_bands import band_statistics_dataset
from GRIM_Backend.datasets.work_control import checkpoint, WorkCancelled


FUNCTIONS = {
    name: getattr(transforms, name) for name in (
        'crop_dataset', 'decimate_axis', 'duplicate_dataset', 'flip_phase',
        'offset_db', 'regrid_axis', 'shift_dataset', 'time_gate',
        'translate_phase_center', 'convert_extrusion', 'wedge_to_conic',
        'medianize_azimuth', 'wrap_phase',
    )
}
METHODS = {
    name: getattr(RcsGrid, name) for name in (
        'edit_axis_value', 'statistics_dataset', 'mirror_about_azimuth',
        'wrap_azimuth', 'wrap_phase', 'round_axes', 'swap_elevation_azimuth',
        'convert_sentri_elevation_to_grim', 'combine_elevation_pair_to_azimuth_360',
        'convert_axis_units',
    )
}


def regrid_resolved(dataset, axis, *, start, step, count, max_gap=None,
                    support='interpolate'):
    if type(count) is not int or not 1 <= count <= transforms._MAX_EXPLICIT_AXIS_POINTS:
        raise ValueError('Regrid sample count is outside the supported range.')
    return transforms.regrid_axis(
        dataset, axis, values=start + step * np.arange(count, dtype=float),
        max_gap=max_gap, support=support,
    )


FUNCTIONS['regrid_resolved'] = regrid_resolved
FUNCTIONS['band_statistics_dataset'] = band_statistics_dataset
LABELS = {
    'band_statistics_dataset': 'Band / sector statistics',
    'crop_dataset': 'Crop / Slice', 'offset_db': 'Offset (dB)',
    'regrid_resolved': 'Regrid', 'regrid_axis': 'Regrid',
    'statistics_dataset': 'Statistics', 'mirror_about_azimuth': 'Mirror azimuth',
    'convert_extrusion': 'dBke / dBsm conversion',
}


def plain(value):
    """Copy numeric parameters into the JSON value vocabulary."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: plain(item) for key, item in value.items()}
    raise ValueError('This action has parameters that cannot be saved in a recipe.')


def validate_step(step):
    if not isinstance(step, dict) or set(step) != {'kind', 'name', 'args', 'kwargs'}:
        raise ValueError('Invalid action step.')
    registry = {'function': FUNCTIONS, 'method': METHODS}.get(step['kind'], {})
    if step['name'] not in registry:
        raise ValueError(f"Unsupported recipe action: {step['name']}")
    if not isinstance(step['args'], list) or not isinstance(step['kwargs'], dict):
        raise ValueError('Invalid action parameters.')
    result = plain(step)
    inspect.signature(registry[result['name']]).bind(
        None, *result['args'], **result['kwargs'])
    if result['name'] == 'regrid_resolved':
        count = result['kwargs'].get('count')
        if type(count) is not int or not 1 <= count <= transforms._MAX_EXPLICIT_AXIS_POINTS:
            raise ValueError('Regrid sample count is outside the supported range.')
    return result


def step_label(step):
    return LABELS.get(step['name'], step['name'].replace('_', ' ').capitalize())


def axis_units(dataset):
    # Resolved GUI coordinates are stored in native units. Do not reinterpret
    # e.g. a GHz crop as Hz when using a recipe on another file.
    return {axis: str((dataset.units or {}).get(axis, ''))
            for axis in ('azimuth', 'elevation', 'frequency')}


@dataclass
class ActionRecipe:
    name: str = 'Recorded actions'
    units: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)

    def dumps(self):
        return json.dumps({'format': 'grim-actions', 'version': 1,
                           'name': self.name, 'units': self.units,
                           'steps': self.steps}, indent=2, allow_nan=False)

    @classmethod
    def loads(cls, text):
        if len(text) > 10_000_000:
            raise ValueError('The recipe exceeds the 10 MB limit.')
        data = json.loads(text)
        if not isinstance(data, dict) or data.get('format') != 'grim-actions' or data.get('version') != 1:
            raise ValueError('This is not a supported GRIM action recipe.')
        name, units, steps = data.get('name'), data.get('units'), data.get('steps')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('The recipe needs a name.')
        if not isinstance(units, dict) or set(units) != {'azimuth', 'elevation', 'frequency'} or not all(isinstance(v, str) for v in units.values()):
            raise ValueError('The recipe is missing its source axis units.')
        if not isinstance(steps, list) or not 1 <= len(steps) <= 1000:
            raise ValueError('A recipe must contain between 1 and 1,000 steps.')
        return cls(name.strip(), units, [validate_step(step) for step in steps])

    def run(self, dataset, progress=None):
        if axis_units(dataset) != self.units:
            raise ValueError('Axis units differ from the recording. Convert the input axis units first.')
        current = dataset
        for index, raw in enumerate(self.steps):
            checkpoint()
            step = validate_step(raw)
            try:
                registry = FUNCTIONS if step['kind'] == 'function' else METHODS
                current = registry[step['name']](current, *step['args'], **step['kwargs'])
                if not isinstance(current, RcsGrid):
                    raise ValueError('The action did not produce a dataset.')
            except WorkCancelled:
                raise
            except Exception as exc:
                raise ValueError(f"Step {index + 1} ({step_label(step)}): {exc}") from exc
            if progress:
                progress(index + 1, len(self.steps), step_label(step))
        checkpoint()
        return current

    def expression(self, variable):
        """Preserve the existing Python export for a completed GUI replay."""
        from GRIM_Backend.scripting.recorder import _Code, _call_text
        for step in self.steps:
            args, kwargs = step['args'], dict(step['kwargs'])
            name = step['name']
            if name == 'regrid_resolved':
                start, increment, count = (kwargs.pop(k) for k in ('start', 'step', 'count'))
                kwargs['values'] = _Code(f'{start!r} + {increment!r} * np.arange({count}, dtype=float)')
                name = 'regrid_axis'
            if step['kind'] == 'method':
                variable = _call_text(f'{variable}.{name}', args, kwargs)
            else:
                variable = _call_text(name, (_Code(variable), *args), kwargs)
        return variable


class ActionCapture:
    """Follow one demonstrated dataset chain, independent of file paths."""
    def __init__(self, on_change=None):
        self.recipe = ActionRecipe()
        self.recording = False
        self.current_id = None
        self.error = ''
        self.on_change = on_change

    def changed(self):
        if self.on_change:
            self.on_change()

    def start(self, reference, dataset, name='Recorded actions'):
        self.recipe = ActionRecipe(name, axis_units(dataset))
        self.current_id = reference.dataset_id
        self.error = ''
        self.recording = True
        self.changed()

    def stop(self):
        self.recording = False
        self.changed()

    def capture(self, outputs, sources, steps, label):
        if not self.recording:
            return
        try:
            if len(outputs) != 1 or len(sources) != 1:
                raise ValueError('Actions combining multiple datasets are not supported by per-dataset recipes.')
            if sources[0].dataset_id != self.current_id:
                raise ValueError('The action used a different input. Continue each recording from its latest result.')
            if not steps:
                raise ValueError(f'{label} is not supported by action recipes.')
            validated = [validate_step(plain(step)) for step in steps]
            if len(self.recipe.steps) + len(validated) > 1000:
                raise ValueError('A recipe can contain at most 1,000 steps.')
            candidate = ActionRecipe(self.recipe.name, self.recipe.units,
                                     self.recipe.steps + validated)
            if len(candidate.dumps()) > 10_000_000:
                raise ValueError('The recipe exceeds the 10 MB limit.')
            self.recipe.steps.extend(validated)
            self.current_id = outputs[0].dataset_id
        except (ValueError, TypeError, KeyError) as exc:
            self.error = f'Recording stopped: {exc} Only the steps listed below were recorded.'
            self.recording = False
        self.changed()
