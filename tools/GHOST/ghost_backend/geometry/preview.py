"""Geometry-only helpers for readable preview overlays.

Material shading uses the solver's region analysis (geometry.regions); this
module only places screen-space overlays. No Qt or Matplotlib is needed here.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence


def visible_segment_midpoint(start, end, bounds):
    """Midpoint of the visible part of a screen-space segment, or None.

    Clipping keeps long boundaries inspectable when their original midpoint
    lies outside a zoomed viewport. Canonical endpoint order makes the anchor
    unchanged when the segment direction is reversed.
    """
    a, b = sorted((tuple(map(float, start)), tuple(map(float, end))))
    x0, y0, x1, y1 = map(float, bounds)
    if not all(math.isfinite(v) for v in (*a, *b, x0, y0, x1, y1)):
        return None
    if x0 > x1 or y0 > y1:
        return None
    dx, dy = b[0] - a[0], b[1] - a[1]
    low, high = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            low = max(low, t)
        else:
            high = min(high, t)
        if low > high:
            return None
    t = (low + high) / 2
    return (min(x1, max(x0, a[0] + t * dx)),
            min(y1, max(y0, a[1] + t * dy)))

def select_normal_samples(
    candidates: Iterable[dict],
    *,
    min_spacing: float = 36.0,
    bounds: Sequence[float] | None = None,
) -> list[dict]:
    """Keep deterministic, spaced normal candidates in display coordinates.

    Every candidate provides ``midpoint=(x_px, y_px)`` and may provide ``row``,
    ``priority`` (smaller first), and ``footprint=(x0, y0, x1, y1)`` covering the
    arrow and rear tick. The optional bounds rectangle clips anchors to the
    viewport. Other candidate fields are retained for the caller. Equal-row
    candidates are sorted by position so reversing a chain retains anchors.
    """
    spacing = max(float(min_spacing), 1.0)
    valid = []
    for order, candidate in enumerate(candidates):
        x, y = candidate["midpoint"]
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if bounds is not None and not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]):
            continue
        valid.append((candidate.get("priority", 2), candidate.get("row", 0), x, y, order, candidate))
    valid.sort(key=lambda item: item[:-1])
    accepted = []
    grid = defaultdict(list)
    footprint_grid = defaultdict(list)

    def cells(rect):
        for ix in range(math.floor(rect[0] / spacing), math.floor(rect[2] / spacing) + 1):
            for iy in range(math.floor(rect[1] / spacing), math.floor(rect[3] / spacing) + 1):
                yield (ix, iy)

    for _priority, _row, x, y, _order, candidate in valid:
        cell = (math.floor(x / spacing), math.floor(y / spacing))
        crowded = any((x - px) ** 2 + (y - py) ** 2 < spacing * spacing
                      for ix in range(cell[0] - 1, cell[0] + 2)
                      for iy in range(cell[1] - 1, cell[1] + 2)
                      for px, py in grid.get((ix, iy), ()))
        if crowded:
            continue
        footprint = candidate.get("footprint", (x, y, x, y))
        if not all(math.isfinite(value) for value in footprint):
            continue
        rectangle = (min(footprint[0], footprint[2]), min(footprint[1], footprint[3]),
                     max(footprint[0], footprint[2]), max(footprint[1], footprint[3]))
        footprint_cells = list(cells(rectangle))
        nearby = {index for key in footprint_cells for index in footprint_grid.get(key, ())}
        if any(not (rectangle[2] < existing[0] or existing[2] < rectangle[0]
                    or rectangle[3] < existing[1] or existing[3] < rectangle[1])
               for index in nearby for existing in [accepted[index][1]]):
            continue
        index = len(accepted)
        accepted.append((candidate, rectangle))
        grid[cell].append((x, y))
        for key in footprint_cells:
            footprint_grid[key].append(index)
    return [candidate for candidate, _rectangle in accepted]
