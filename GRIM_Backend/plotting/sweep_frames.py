"""Acquired-sample sweep plans. Planning never interpolates source data."""
from dataclasses import dataclass
import numpy as np


class FrameLabels:
    """Format only the position being displayed, not every planned window."""
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.frames.label(index)


@dataclass
class SweepFrames:
    rows: np.ndarray
    spans: np.ndarray
    bounds: np.ndarray
    unit: str
    band: bool
    step_by_width: bool = False
    advance: float | None = None

    def __len__(self):
        return len(self.spans)

    def selection(self, index):
        start, stop = self.spans[index]
        return self.rows[start:stop].tolist()

    def label(self, index):
        lo, hi = self.bounds[index]
        value = f"{lo:.9g} – {hi:.9g}" if self.band else f"{lo:.9g}"
        return f"{value} {self.unit}".strip()


def make_sweep_frames(values, *, width=None, unit="", minimum_samples=2, step_by_width=False, advance=None):
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
    tolerance = max(np.max(np.abs(coordinates)), width) * 8 * np.finfo(float).eps
    if advance is not None:
        advance = float(advance)
        if not np.isfinite(advance) or advance <= 0:
            raise ValueError("Window advance must be finite and positive.")
        count = np.ceil(extent / advance)
        if not np.isfinite(count) or count > 100000:
            raise ValueError("More than 100,000 windows requested; increase the advance.")
        starts = coordinates[0] + np.arange(int(count)) * advance
        starts = starts[starts < coordinates[-1] - tolerance]
        ends = np.minimum(starts + width, coordinates[-1])
    elif step_by_width:
        # Anchor adjacent bands to the first coordinate. Consider only bins
        # near acquired samples, so large gaps cannot allocate billions of
        # empty positions. Include the previous bin for shared endpoints.
        bins = np.floor((coordinates - coordinates[0]) / width)
        bins = np.unique(np.r_[bins, bins - 1])
        bins = bins[bins >= 0]
        starts = coordinates[0] + bins * width
        keep = starts < coordinates[-1] - tolerance
        starts, bins = starts[keep], bins[keep]
        ends = np.minimum(coordinates[0] + (bins + 1) * width, coordinates[-1])
    else:
        # One start per acquired coordinate, plus a final full-width band
        # ending at the upper bound. Preserve the existing overlapping sweep.
        last = coordinates[-1] - width
        starts = np.unique(np.r_[coordinates[coordinates < last], last])
        ends = starts + width
    lower = np.searchsorted(coordinates, starts-tolerance, side="left")
    upper = np.searchsorted(coordinates, ends+tolerance, side="right")
    valid = upper-lower >= minimum_samples
    spans = np.column_stack((lower[valid], upper[valid]))
    bounds = np.column_stack((starts[valid], ends[valid]))
    if not len(spans):
        raise ValueError(f"No band contains at least {minimum_samples} acquired samples; increase the width.")
    distinct = np.r_[True, np.any(np.diff(spans, axis=0) != 0, axis=1)]
    return SweepFrames(order, spans[distinct], bounds[distinct], unit, True, bool(step_by_width), advance)
