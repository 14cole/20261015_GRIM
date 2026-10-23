"""Acquired-sample sweep plans. Planning never interpolates source data."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SweepFrames:
    rows: np.ndarray
    spans: np.ndarray
    bounds: np.ndarray
    unit: str
    band: bool

    def __len__(self):
        return len(self.spans)

    def selection(self, index):
        start, stop = self.spans[index]
        return self.rows[start:stop].tolist()

    def label(self, index):
        lo, hi = self.bounds[index]
        value = f"{lo:.9g} – {hi:.9g}" if self.band else f"{lo:.9g}"
        return f"{value} {self.unit}".strip()


def make_sweep_frames(values, *, width=None, unit="", minimum_samples=2):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("The sweep axis needs finite acquired coordinates.")
    order = np.argsort(values, kind="stable")
    coordinates = values[order]
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("The sweep axis must have unique coordinates.")
    if width is None:
        positions = np.arange(len(values))
        return SweepFrames(order, np.column_stack((positions, positions+1)),
                           np.column_stack((coordinates, coordinates)), unit, False)
    width = float(width)
    extent = float(coordinates[-1] - coordinates[0])
    if not np.isfinite(width) or width <= 0 or width > extent * (1 + 1.e-12):
        raise ValueError("Band width must be positive and no wider than the acquired axis.")
    width = min(width, extent)
    # One start per acquired coordinate, plus a final band ending exactly at
    # the upper bound. Searchsorted keeps planning linear in retained storage.
    last = coordinates[-1] - width
    starts = np.unique(np.r_[coordinates[coordinates < last], last])
    ends = starts + width
    tolerance = max(np.max(np.abs(coordinates)), width) * 8 * np.finfo(float).eps
    lower = np.searchsorted(coordinates, starts-tolerance, side="left")
    upper = np.searchsorted(coordinates, ends+tolerance, side="right")
    valid = upper-lower >= minimum_samples
    spans = np.column_stack((lower[valid], upper[valid]))
    bounds = np.column_stack((starts[valid], ends[valid]))
    if not len(spans):
        raise ValueError(f"No band contains at least {minimum_samples} acquired samples; increase the width.")
    distinct = np.r_[True, np.any(np.diff(spans, axis=0) != 0, axis=1)]
    return SweepFrames(order, spans[distinct], bounds[distinct], unit, True)
