"""Whole-geometry material regions of a 2-D cross section or a BoR meridian.

Every primitive declares the medium on each side; the left of travel is the
user-facing normal side (GEOMETRY_INPUT_CHEATSHEET.md, "Geometry normal and
winding"; the solver's `_apply_user_convention_flip`):

    TYPE 1  air | air                 TYPE 4  dielectric pos | conductor
    TYPE 2  air | conductor           TYPE 5  dielectric pos | dielectric neg
    TYPE 3  air | dielectric pos

The primitives exactly as the solver meshes them (straight point pairs; a
drawn arc is a chain of them) are welded into a planar graph whose faces are
traced with half-edges. Two rules follow:

A. Closure: the oriented boundary of each dielectric balances at every
   vertex. A free end or a junction whose labels disagree leaves it open.
B. Regions: every boundary around a face declares the same medium, and the
   unbounded face is air. Dangling TYPE 1 sheets and TYPE 2 plates have air
   on both sides there (a plate's conductor side encloses nothing).

A BoR meridian is closed by the rotation axis x = 0: unlabeled axis edges
join consecutive axis vertices and axis vertices need not balance.

Thin separations stay distinct. Points weld only within a tolerance far below
any drawn feature (a cluster never spans more than it), angular order at a
vertex and every inside test use orientation signs that fall back to exact
rational arithmetic when the floating-point value is within its error bound,
and a hole is attached by testing one of its own vertices against the
candidate face polygon -- never a probe point offset by a distance scaled to
the geometry, which can land across a 25 um layer on a 500 inch body.

Pure Python and NumPy; shared by the solver preflight and the GUI.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cmp_to_key
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Medium = Tuple[str, int]
Point = Tuple[float, float]

AIR: Medium = ("air", 0)
CONDUCTOR: Medium = ("conductor", 0)

# Relative bound on the rounding error of a 2x2 orientation determinant built
# from coordinate differences (about four unit roundoffs, padded).
_ORIENT_ERROR = 1.0e-15
# A dangling end "almost meets" another primitive within this fraction of its
# own primitive's length (and only roughly along its extension).
NEAR_MISS_FRACTION = 0.05
AXIS_EDGE = -1


def numerical_bor_axis_tolerance(points, meters_scale: float = 1.0) -> float:
    """Numerical zero in radius, expressed in the input coordinate units.

    Axial extent and translation along z cannot turn a positive aperture
    into the rotation axis. The absolute floor is one picometer.
    """
    meters_scale = float(meters_scale)
    if not math.isfinite(meters_scale) or meters_scale <= 0.0:
        raise ValueError("meters_scale must be positive and finite.")
    points = np.asarray(points, dtype=float)
    radius = float(np.max(np.abs(points[:, 0]))) if points.size else 0.0
    return 1.0e-12 * max(1.0 / meters_scale, radius)


def dielectric(flag: int) -> Medium:
    return ("dielectric", int(flag))


def medium_name(medium: Optional[Medium]) -> str:
    if medium is None:
        return "unknown"
    return f"dielectric {medium[1]}" if medium[0] == "dielectric" else medium[0]


def side_media(seg_type: int, pos_mat: int = 0, neg_mat: int = 0, *,
               sheet_layout: bool = False) -> Optional[Tuple[Medium, Medium]]:
    """(left, right) media of a primitive as drawn; None for an invalid TYPE.

    ``sheet_layout`` is the BoR sheet route (only TYPE 1 and 2 segments),
    where pure PEC portions are zero-impedance sheets with air on both sides.
    """

    if seg_type == 2 and sheet_layout:
        return AIR, AIR
    return {
        1: (AIR, AIR),
        2: (AIR, CONDUCTOR),
        3: (AIR, dielectric(pos_mat)),
        4: (dielectric(pos_mat), CONDUCTOR),
        5: (dielectric(pos_mat), dielectric(neg_mat)),
    }.get(int(seg_type))


def boundary_for(left: Medium, right: Medium) -> Optional[Tuple[int, int, int, bool]]:
    """(TYPE, pos_mat, neg_mat, reversed) of the boundary with these sides, else None."""

    for flipped, (a, b) in ((False, (left, right)), (True, (right, left))):
        if a == AIR and b == AIR:
            return 1, 0, 0, False
        if a == AIR and b == CONDUCTOR:
            return 2, 0, 0, flipped
        if a == AIR and b[0] == "dielectric":
            return 3, b[1], 0, flipped
        if a[0] == "dielectric" and b == CONDUCTOR:
            return 4, a[1], 0, flipped
        if a[0] == "dielectric" and b[0] == "dielectric" and a != b:
            return 5, a[1], b[1], flipped
    return None


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _at(point: Point) -> str:
    return f"({_fmt(point[0])}, {_fmt(point[1])})"


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)


@dataclass(frozen=True)
class RegionPrimitive:
    """One straight primitive with the media its segment declares on each side."""

    row: int
    name: str
    seg_type: int
    pos_mat: int
    neg_mat: int
    start: Point
    end: Point
    left: Medium
    right: Medium


def _flag(token: Any, default: int = 0) -> int:
    text = str("" if token is None else token).strip().lower()
    if text.startswith("mat."):
        text = text[4:]
    try:
        value = float(text)
    except ValueError:
        return default
    return int(value) if math.isfinite(value) and value.is_integer() else default


def _build_primitives(rows, mode: str) -> List[RegionPrimitive]:
    rows = list(rows)
    kinds = {seg_type for _, _, seg_type, _, _, _ in rows}
    sheet_layout = str(mode).lower() == "bor" and kinds in ({1}, {1, 2})
    out: List[RegionPrimitive] = []
    for row, name, seg_type, pos_mat, neg_mat, pairs in rows:
        sides = side_media(seg_type, pos_mat, neg_mat, sheet_layout=sheet_layout)
        if sides is None:
            continue
        for x1, y1, x2, y2 in pairs:
            try:
                a, b = (float(x1), float(y1)), (float(x2), float(y2))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in (*a, *b)):
                out.append(RegionPrimitive(row, name, seg_type, pos_mat, neg_mat,
                                           a, b, sides[0], sides[1]))
    return out


def primitives_from_snapshot_segments(segments: Sequence[Dict[str, Any]], *,
                                      mode: str = "2d") -> List[RegionPrimitive]:
    """Primitives of solver-snapshot segments (``point_pairs`` form)."""

    rows = []
    for row, seg in enumerate(segments):
        props = list(seg.get("properties", []) or []) + [""] * 5
        type_token = props[0] if str(props[0]).strip() else seg.get("seg_type", 0)
        pairs = [(pair.get("x1"), pair.get("y1"), pair.get("x2"), pair.get("y2"))
                 for pair in list(seg.get("point_pairs", []) or [])]
        rows.append((row, str(seg.get("name", f"segment_{row + 1}")), _flag(type_token, -1),
                     _flag(props[3]), _flag(props[4]), pairs))
    return _build_primitives(rows, mode)


def primitives_from_editor_segments(segments: Sequence[Any], *, mode: str = "2d") -> List[RegionPrimitive]:
    """Primitives of editor ``Segment`` objects (paired x/y coordinate arrays)."""

    rows = []
    for row, seg in enumerate(segments):
        props = list(seg.properties or []) + [""] * 5
        xs, ys = list(seg.x), list(seg.y)
        count = min(len(xs), len(ys)) // 2
        pairs = [(xs[2 * i], ys[2 * i], xs[2 * i + 1], ys[2 * i + 1]) for i in range(count)]
        rows.append((row, seg.name or f"segment_{row + 1}", _flag(props[0], -1),
                     _flag(props[3]), _flag(props[4]), pairs))
    return _build_primitives(rows, mode)


# ---------------------------------------------------------------- results


@dataclass
class NearMiss:
    """A free end of a must-close segment that stops just short of another primitive."""

    name: str
    row: int
    seg_type: int
    point: Point
    other_name: str
    other_row: int
    other_point: Point
    gap: float

    @property
    def message(self) -> str:
        return (f"Segments '{self.name}' and '{self.other_name}' almost meet at "
                f"{_at(self.point)}: gap {self.gap:.3g}.")


@dataclass
class ClosureFailure:
    """A vertex where a dielectric's oriented boundary does not balance."""

    media: List[Medium]
    point: Point
    names: List[str]
    rows: List[int]
    dangling: bool
    seg_types: List[int] = field(default_factory=list)
    near_miss: Optional[NearMiss] = None
    other_ends: List[Point] = field(default_factory=list)
    explained_by: Optional[int] = None
    message: str = ""
    summary: str = ""


@dataclass
class RegionConflict:
    """A face whose boundaries declare different media."""

    region: int
    media: Dict[Medium, List[str]]
    expected: Optional[Medium]
    suspects: List[str]
    suspect_rows: List[int]
    rows: List[int]
    location: Point
    alternative: Optional[Medium] = None
    explained: bool = False
    message: str = ""
    summary: str = ""


@dataclass
class Region:
    """One face of the drawing: a medium and its fillable outline with holes."""

    index: int
    medium: Optional[Medium]
    outer: Optional[List[Point]]
    holes: List[List[Point]] = field(default_factory=list)
    media: Dict[Medium, List[str]] = field(default_factory=dict)
    rows: List[int] = field(default_factory=list)
    conflict: Optional[int] = None
    area: float = math.inf

    @property
    def unbounded(self) -> bool:
        return self.outer is None


@dataclass
class ErrorItem:
    row: int
    rows: List[int]
    message: str
    summary: str


@dataclass
class RegionReport:
    regions: List[Region]
    conflicts: List[RegionConflict]
    closure_failures: List[ClosureFailure]
    near_misses: List[NearMiss]
    tolerance: float
    mode: str

    def error_items(self) -> List[ErrorItem]:
        """Conflicts and closure failures not already explained by another finding.

        ``rows`` are the rows most likely at fault (the minority labels of a
        conflict, the segments at an open vertex). Conflicts blaming the same
        segments (one swapped TYPE 5 label mislabels both of its sides) are
        one item.
        """

        items = []
        for failure in self.closure_failures:
            if failure.explained_by is None:
                items.append(ErrorItem(failure.rows[0] if failure.rows else -1,
                                       list(failure.rows), failure.message, failure.summary))
        by_suspects: Dict[Tuple[int, ...], ErrorItem] = {}
        for conflict in self.conflicts:
            if conflict.explained:
                continue
            rows = conflict.suspect_rows or conflict.rows
            key = tuple(conflict.suspect_rows) if conflict.alternative is None else ()
            if key and key in by_suspects:
                item = by_suspects[key]
                item.message += f" The same segment(s) also mislabel another region: {conflict.summary}."
                continue
            item = ErrorItem(rows[0] if rows else -1, list(rows), conflict.message, conflict.summary)
            if key:
                by_suspects[key] = item
            items.append(item)
        return items

    def errors(self) -> List[str]:
        return [item.message for item in self.error_items()]

    def near_misses_of_type(self, *types: int) -> List[NearMiss]:
        return [miss for miss in self.near_misses if miss.seg_type in types]


# ------------------------------------------------------------- predicates


def _exact_cross(ox, oy, ax, ay, bx, by) -> int:
    value = ((Fraction(ax) - Fraction(ox)) * (Fraction(by) - Fraction(oy))
             - (Fraction(ay) - Fraction(oy)) * (Fraction(bx) - Fraction(ox)))
    return (value > 0) - (value < 0)


def _orient(o: Point, a: Point, b: Point) -> int:
    """Sign of cross(a - o, b - o); exact when the float value is not conclusive."""

    lhs = (a[0] - o[0]) * (b[1] - o[1])
    rhs = (a[1] - o[1]) * (b[0] - o[0])
    det = lhs - rhs
    if abs(det) > _ORIENT_ERROR * (abs(lhs) + abs(rhs)):
        return 1 if det > 0 else -1
    return _exact_cross(o[0], o[1], a[0], a[1], b[0], b[1])


def _half_plane(o: Point, p: Point) -> int:
    # Float subtraction keeps the exact sign, so this split is exact.
    dx, dy = p[0] - o[0], p[1] - o[1]
    return 0 if dy > 0 or (dy == 0 and dx > 0) else 1


def _winding(point: Point, xs: np.ndarray, ys: np.ndarray) -> Optional[int]:
    """Winding number of the closed vertex cycle (xs, ys) around ``point``.

    None when the point lies on the cycle. Vectorized in floats; the edges
    whose orientation is within its error bound are redone exactly.
    """

    px, py = float(point[0]), float(point[1])
    x0, y0 = xs, ys
    x1, y1 = np.roll(xs, -1), np.roll(ys, -1)
    up = (y0 <= py) & (y1 > py)
    down = (y0 > py) & (y1 <= py)
    flat = (y0 == py) & (y1 == py) & (np.minimum(x0, x1) <= px) & (px <= np.maximum(x0, x1))
    if flat.any():
        return None
    lhs = (x1 - x0) * (py - y0)
    rhs = (px - x0) * (y1 - y0)
    det = lhs - rhs
    sign = np.sign(det)
    unsure = (up | down) & (np.abs(det) <= _ORIENT_ERROR * (np.abs(lhs) + np.abs(rhs)))
    for k in np.flatnonzero(unsure):
        sign[k] = _exact_cross(x0[k], y0[k], x1[k], y1[k], px, py)
    if ((up | down) & (sign == 0)).any():
        return None
    return int(np.count_nonzero(up & (sign > 0)) - np.count_nonzero(down & (sign < 0)))


def _closest_on_segment(p: Point, a: Point, b: Point) -> Tuple[float, Point]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    t = min(1.0, max(0.0, t))
    q = (a[0] + t * dx, a[1] + t * dy)
    return math.hypot(q[0] - p[0], q[1] - p[1]), q


# ------------------------------------------------------------------- weld


def default_tolerance(primitives: Sequence[RegionPrimitive]) -> float:
    """min(1e-9 * diagonal, 1e-3 * shortest primitive), above coordinate resolution."""

    xs = [c for p in primitives for c in (p.start[0], p.end[0])]
    ys = [c for p in primitives for c in (p.start[1], p.end[1])]
    if not xs:
        return 0.0
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    lengths = [math.hypot(p.end[0] - p.start[0], p.end[1] - p.start[1]) for p in primitives]
    shortest = min((value for value in lengths if value > 0), default=diag)
    resolution = 8.0 * math.ulp(max(max(map(abs, xs)), max(map(abs, ys)), 1e-300))
    return max(min(1e-9 * diag, 1e-3 * shortest), resolution)


class PointWelder:
    """Incremental, diameter-bounded vertex identity shared by geometry and meshes.

    A point joins a cluster only within ``tol`` of every registered member.
    Grid cells accelerate candidate lookup; they never imply identity. Exact
    repeats are cached, and a lookup does not enlarge the registered cluster.
    """

    def __init__(self, tol: float):
        self.tol = float(tol)
        if not math.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError("Point welding tolerance must be non-negative and finite.")
        self.points: List[Point] = []
        # Most vertices have one exactly repeated coordinate. Store extra
        # member lists only for the uncommon genuinely near-coincident case.
        self._members: Dict[int, List[Point]] = {}
        self._exact: Dict[Point, int] = {}
        self._grid: Dict[Tuple[int, int], Any] = {}

    def _nearest(self, p: Point) -> Optional[int]:
        if self.tol == 0.0:
            return None
        cx, cy = math.floor(p[0] / self.tol), math.floor(p[1] / self.tol)
        best, best_distance = None, math.inf
        seen = set()
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                entry = self._grid.get((i, j))
                if entry is None:
                    continue
                for cluster in entry if isinstance(entry, list) else (entry,):
                    if cluster in seen:
                        continue
                    seen.add(cluster)
                    members = self._members.get(cluster)
                    if members is None:
                        q = self.points[cluster]
                        distance = math.hypot(q[0] - p[0], q[1] - p[1])
                        if distance <= self.tol and distance < best_distance:
                            best, best_distance = cluster, distance
                        continue
                    nearest = math.inf
                    for q in members:
                        distance = math.hypot(q[0] - p[0], q[1] - p[1])
                        if distance > self.tol:
                            break
                        nearest = min(nearest, distance)
                    else:
                        if nearest < best_distance:
                            best, best_distance = cluster, nearest
        return best

    def lookup(self, point: Point) -> Optional[int]:
        p = (float(point[0]), float(point[1]))
        known = self._exact.get(p)
        return known if known is not None else self._nearest(p)

    def key(self, point: Point) -> int:
        p = (float(point[0]), float(point[1]))
        known = self._exact.get(p)
        if known is not None:
            return known
        if not all(math.isfinite(value) for value in p):
            raise ValueError("Point coordinates must be finite.")
        cluster = self._nearest(p)
        if cluster is None:
            cluster = len(self.points)
            self.points.append(p)
        elif cluster in self._members:
            self._members[cluster].append(p)
        else:
            self._members[cluster] = [self.points[cluster], p]
        self._exact[p] = cluster
        if self.tol > 0.0:
            cell = (math.floor(p[0] / self.tol), math.floor(p[1] / self.tol))
            entry = self._grid.get(cell)
            if entry is None:
                self._grid[cell] = cluster
            elif isinstance(entry, list):
                if cluster not in entry:
                    entry.append(cluster)
            elif entry != cluster:
                self._grid[cell] = [entry, cluster]
        return cluster


def _weld(points: Sequence[Point], tol: float) -> Tuple[List[int], List[Point]]:
    """Cluster ids and representatives using the solver's diameter-bounded rule."""
    welder = PointWelder(tol)
    return [welder.key(p) for p in points], welder.points


# --------------------------------------------------------------- analysis


def analyze_regions(
    primitives: Sequence[RegionPrimitive],
    *,
    mode: str = "2d",
    tol: Optional[float] = None,
    axis_tol: Optional[float] = None,
    meters_scale: float = 1.0,
    near_miss_types: Optional[Tuple[int, ...]] = None,
    checkpoint=None,
) -> RegionReport:
    """Region map, conflicts, dielectric closure failures and near-miss gaps.

    ``tol`` welds drawn vertices (default :func:`default_tolerance`); the
    solver passes its own mesh node tolerance so both see one topology.
    ``axis_tol`` (BoR) marks vertices on the rotation axis, defaulting to
    :func:`numerical_bor_axis_tolerance` in input units (``meters_scale``
    meters per input unit). Near misses
    are searched from free ends of ``near_miss_types`` segments (TYPE 2 only
    in 2-D: a BoR shell may end anywhere).
    """

    mode = str(mode).strip().lower()
    bor = mode == "bor"
    if checkpoint is not None:
        checkpoint()
    if near_miss_types is None:
        near_miss_types = (3, 4, 5) if bor else (2, 3, 4, 5)
    prims = list(primitives)
    if tol is None:
        tol = default_tolerance(prims)
    tol = float(tol)
    if not prims:
        return RegionReport([], [], [], [], tol, mode)
    if axis_tol is None:
        axis_tol = numerical_bor_axis_tolerance(
            [q for p in prims for q in (p.start, p.end)], meters_scale)

    ids, verts = _weld([q for p in prims for q in (p.start, p.end)], tol)

    # ---- edges: one per distinct welded vertex pair (duplicates are refused
    # by the intersection checks; the first copy keeps the graph planar).
    edges: List[Tuple[int, int, int]] = []
    seen: Dict[Tuple[int, int], int] = {}
    for index in range(len(prims)):
        u, v = ids[2 * index], ids[2 * index + 1]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen[key] = len(edges)
        edges.append((u, v, index))
    real_degree = [0] * len(verts)
    for u, v, _ in edges:
        real_degree[u] += 1
        real_degree[v] += 1
    on_axis = [bor and abs(verts[k][0]) <= axis_tol for k in range(len(verts))]
    if bor:
        axis = sorted((k for k in range(len(verts)) if on_axis[k] and real_degree[k]),
                      key=lambda k: verts[k][1])
        for a, b in zip(axis, axis[1:]):
            if (min(a, b), max(a, b)) not in seen:
                seen[(min(a, b), max(a, b))] = len(edges)
                edges.append((a, b, AXIS_EDGE))
    if not edges:
        return RegionReport([], [], [], [], tol, mode)

    # ---- half-edges: 2e runs u->v with the left medium, 2e+1 runs v->u with the right.
    count = 2 * len(edges)
    tail = [0] * count
    head = [0] * count
    medium: List[Optional[Medium]] = [None] * count
    for e, (u, v, index) in enumerate(edges):
        tail[2 * e], head[2 * e] = u, v
        tail[2 * e + 1], head[2 * e + 1] = v, u
        if index != AXIS_EDGE:
            medium[2 * e], medium[2 * e + 1] = prims[index].left, prims[index].right
    outgoing: List[List[int]] = [[] for _ in verts]
    for h in range(count):
        outgoing[tail[h]].append(h)

    def by_angle(origin: Point):
        def compare(a: int, b: int) -> int:
            pa, pb = verts[head[a]], verts[head[b]]
            ha, hb = _half_plane(origin, pa), _half_plane(origin, pb)
            if ha != hb:
                return ha - hb
            turn = _orient(origin, pa, pb)
            return -turn if turn else (a > b) - (a < b)
        return cmp_to_key(compare)

    position = [0] * count
    for k, hs in enumerate(outgoing):
        if len(hs) > 2:
            hs.sort(key=by_angle(verts[k]))
        for slot, h in enumerate(hs):
            position[h] = slot
    successor = [0] * count
    for h in range(count):
        around = outgoing[head[h]]
        # The face on the left: turn to the outgoing edge just clockwise of the twin.
        successor[h] = around[(position[h ^ 1] - 1) % len(around)]

    cycle_of = [-1] * count
    cycles: List[List[int]] = []
    for first in range(count):
        if cycle_of[first] >= 0:
            continue
        cycle, h = [], first
        while cycle_of[h] < 0:
            cycle_of[h] = len(cycles)
            cycle.append(h)
            h = successor[h]
        cycles.append(cycle)

    # Components (union-find over edges) keep a component's own faces from
    # being taken as the container of its outer boundary.
    parent = list(range(len(verts)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for u, v, _ in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    cycle_xy, cycle_area, cycle_box = [], [], []
    for cycle in cycles:
        cx = np.asarray([verts[tail[h]][0] for h in cycle], dtype=float)
        cy = np.asarray([verts[tail[h]][1] for h in cycle], dtype=float)
        ox, oy = cx[0], cy[0]
        terms = ((cx - ox) * (np.roll(cy, -1) - oy) - (np.roll(cx, -1) - ox) * (cy - oy)).tolist()
        # A walk around a tree cancels term by term, exactly, under fsum.
        area2 = math.fsum(terms)
        bound = 8.0 * np.finfo(float).eps * math.fsum(abs(t) for t in terms)
        cycle_xy.append((cx, cy))
        cycle_area.append(0.5 * area2 if abs(area2) > bound else 0.0)
        cycle_box.append((cx.min(), cx.max(), cy.min(), cy.max()))

    bounded = [c for c in range(len(cycles)) if cycle_area[c] > 0.0]
    face_of_cycle: Dict[int, int] = {}
    regions: List[Region] = []
    for c in bounded:
        face_of_cycle[c] = len(regions)
        cx, cy = cycle_xy[c]
        regions.append(Region(len(regions), None, list(zip(cx.tolist(), cy.tolist())),
                              area=cycle_area[c]))
    unbounded = Region(len(regions), None, None)
    regions.append(unbounded)
    if bounded:
        boxes = np.asarray([cycle_box[c] for c in bounded], dtype=float)
        areas = np.asarray([cycle_area[c] for c in bounded], dtype=float)
        components = np.asarray([find(tail[cycles[c][0]]) for c in bounded])
    for c, cycle in enumerate(cycles):
        if c in face_of_cycle:
            continue
        container = unbounded.index
        if bounded:
            component = find(tail[cycle[0]])
            cx, cy = cycle_xy[c]
            for k in range(len(cycle)):
                px, py = float(cx[k]), float(cy[k])
                candidates = np.flatnonzero((boxes[:, 0] <= px) & (px <= boxes[:, 1])
                                            & (boxes[:, 2] <= py) & (py <= boxes[:, 3])
                                            & (components != component))
                decided = True
                container = unbounded.index
                for j in candidates[np.argsort(areas[candidates], kind="stable")]:
                    winding = _winding((px, py), *cycle_xy[bounded[j]])
                    if winding is None:
                        decided = False     # this vertex touches the candidate: try another
                        break
                    if winding:
                        container = face_of_cycle[bounded[j]]
                        break
                if decided:
                    break
        face_of_cycle[c] = container
        if container != unbounded.index:
            cx, cy = cycle_xy[c]
            regions[container].holes.append(list(zip(cx.tolist(), cy.tolist())))
            regions[container].area += cycle_area[c]      # net fillable area

    # ---- B. media declared around each face
    declared: Dict[int, Dict[Medium, Dict[str, Any]]] = defaultdict(dict)
    open_rows: set = set()
    for h in range(count):
        index = edges[h >> 1][2]
        if index == AXIS_EDGE:
            continue
        prim = prims[index]
        m = medium[h]
        bridge = cycle_of[h] == cycle_of[h ^ 1]
        if bridge and prim.seg_type in (1, 2) and m != AIR:
            continue        # a sheet or plate encloses nothing on its conductor side
        if bridge and prim.seg_type in (3, 4, 5):
            open_rows.add(prim.row)
        face = face_of_cycle[cycle_of[h]]
        entry = declared[face].setdefault(m, {"names": {}, "rows": set(), "length": 0.0,
                                              "half": h, "halves": []})
        entry["names"].setdefault(prim.name, None)
        entry["rows"].add(prim.row)
        entry["halves"].append(h)
        entry["length"] += math.hypot(prim.end[0] - prim.start[0], prim.end[1] - prim.start[1])

    def midpoint(h: int) -> Point:
        a, b = verts[tail[h]], verts[head[h]]
        return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))

    def only_set_by(twin: Region, h: int) -> bool:
        """Is the bounded face across half-edge h declared by h's segment alone?"""
        if twin.unbounded or twin.index == face_of_cycle[cycle_of[h]]:
            return False
        rows = {r for e in declared.get(twin.index, {}).values() for r in e["rows"]}
        return rows <= {prims[edges[h >> 1][2]].row}

    def feasible(face: int, media, candidate: Medium) -> bool:
        """Can every other-labeled boundary of the face be relabeled to ``candidate``?"""
        for m, entry in media.items():
            if m == candidate:
                continue
            for h in entry["halves"]:
                twin = regions[face_of_cycle[cycle_of[h ^ 1]]]
                if twin.index != face and twin.medium is None and twin.index in declared:
                    continue        # the other side is itself in question: any medium may do
                other = twin.medium if twin.index != face else medium[h ^ 1]
                # When this segment alone sets the other side, it may swap its
                # two media (a TYPE 5 contour with its labels exchanged).
                options = [other] + ([m] if only_set_by(twin, h) else [])
                if not any(o is not None and boundary_for(*((candidate, o) if h % 2 == 0
                                                            else (o, candidate))) is not None
                           for o in options):
                    return False
        return True

    for face, media in declared.items():
        region = regions[face]
        region.media = {m: list(entry["names"]) for m, entry in media.items()}
        region.rows = sorted({r for entry in media.values() for r in entry["rows"]})
        if len(media) == 1 and (not region.unbounded or AIR in media):
            region.medium = next(iter(media))
    if not declared.get(unbounded.index) or set(declared[unbounded.index]) == {AIR}:
        unbounded.medium = AIR

    # Which label is wrong: the medium whose rival declarers are cheapest to
    # blame, where a segment in k conflicting regions costs 1/k in each (one
    # swapped TYPE 5 label mislabels both of its sides, so it is the likely
    # culprit of both), then the medium most of the boundary declares. Only
    # media every rival boundary could be relabeled to are candidates: an outer
    # shell naming the wrong dielectric is longer than the TYPE 5 interface
    # inside it, but that interface cannot have one dielectric on both sides.
    # Equal cost and boundary lengths within 10% (two layers 0.001 apart) are
    # ambiguous: both readings are reported.
    conflicted = [face for face, media in declared.items() if regions[face].medium is None]
    involvement: Dict[int, int] = defaultdict(int)
    for face in conflicted:
        for row in {r for entry in declared[face].values() for r in entry["rows"]}:
            involvement[row] += 1
    conflicts: List[RegionConflict] = []
    for face in conflicted:
        media, region = declared[face], regions[face]
        ranked = [AIR] if region.unbounded else list(media)
        candidates = [m for m in ranked if feasible(face, media, m)] or ranked

        def cost(m: Medium) -> float:
            return sum(1.0 / involvement[r] for k, e in media.items() if k != m for r in e["rows"])

        def length(m: Medium) -> float:
            return media[m]["length"] if m in media else 0.0

        candidates.sort(key=lambda m: (round(cost(m), 9), -length(m), -len(media.get(m, {}).get("names", ()))))
        expected, alternative = candidates[0], None
        if len(candidates) > 1:
            rival = candidates[1]
            longest = max(length(expected), length(rival))
            if (math.isclose(cost(expected), cost(rival), rel_tol=1e-9)
                    and longest > 0 and min(length(expected), length(rival)) >= 0.9 * longest):
                alternative = rival
        blamed = [m for m in media if m != expected]
        if alternative is not None:
            blamed = list(media)
        suspects = list(dict.fromkeys(n for m in blamed for n in media[m]["names"]))
        suspect_rows = sorted({r for m in blamed for r in media[m]["rows"]})
        wrong = [m for m in media if m != expected] or list(media)
        conflict = RegionConflict(face, {m: list(e["names"]) for m, e in media.items()}, expected,
                                  suspects, suspect_rows, region.rows, midpoint(media[wrong[0]]["half"]),
                                  alternative=alternative)
        region.conflict = len(conflicts)
        conflicts.append(conflict)
    # An ambiguous region is settled when one of its readings only blames
    # segments that an unambiguous region already blames.
    blamed = {r for c in conflicts if c.alternative is None for r in c.suspect_rows}
    for conflict in conflicts:
        if conflict.alternative is None:
            continue
        media = declared[conflict.region]
        rows_if = {m: {r for k, e in media.items() if k != m for r in e["rows"]}
                   for m in (conflict.expected, conflict.alternative)}
        settled = [m for m, rows in rows_if.items() if rows and rows <= blamed]
        if len(settled) == 1:
            conflict.expected, conflict.alternative = settled[0], None
            conflict.suspects = list(dict.fromkeys(
                n for k, e in media.items() if k != settled[0] for n in e["names"]))
            conflict.suspect_rows = sorted(rows_if[settled[0]])
            wrong = [m for m in media if m != settled[0]]
            conflict.location = midpoint(media[wrong[0]]["half"])

    # ---- A. closure of every dielectric's oriented boundary
    balance: Dict[int, Dict[Medium, int]] = defaultdict(lambda: defaultdict(int))
    for h in range(count):
        m = medium[h]
        if m is None or m[0] != "dielectric":
            continue
        balance[tail[h]][m] += 1
        balance[head[h]][m] -= 1
    at_vertex: Dict[int, List[int]] = defaultdict(list)
    for e, (u, v, index) in enumerate(edges):
        if index != AXIS_EDGE:
            at_vertex[u].append(index)
            at_vertex[v].append(index)
    failures: List[ClosureFailure] = []
    for k in sorted(balance, key=lambda k: verts[k]):
        if on_axis[k]:
            continue
        unbalanced = sorted(m for m, n in balance[k].items() if n)
        if not unbalanced:
            continue
        rows_here = list(dict.fromkeys(prims[i].row for i in at_vertex[k]))
        names_here = list(dict.fromkeys(prims[i].name for i in at_vertex[k]))
        failures.append(ClosureFailure(unbalanced, verts[k], names_here, rows_here,
                                       real_degree[k] == 1,
                                       [prims[i].seg_type for i in at_vertex[k]]))

    # ---- near misses from free ends of must-close segments
    dangling = [k for k in range(len(verts)) if real_degree[k] == 1 and not on_axis[k]]
    near_misses: List[NearMiss] = []
    if dangling:
        near_misses = _near_misses(dangling, verts, edges, at_vertex, prims,
                                   near_miss_types, checkpoint=checkpoint)
    miss_at = {miss.point: miss for miss in near_misses}
    for miss in near_misses:
        miss_at.setdefault(miss.other_point, miss)

    # ---- explanations and messages. A conflict that only arises because a
    # dielectric boundary is open is reported once, as the open end; a
    # junction whose labels disagree is reported once, as the conflict of the
    # region it leaves mislabeled (an unbalanced interior vertex always
    # borders such a region).
    dangling_rows = {r for f in failures if f.dangling for r in f.rows}
    for conflict in conflicts:
        rows = set(conflict.suspect_rows or conflict.rows)
        conflict.explained = bool(dangling_rows) and bool(rows) and rows <= (dangling_rows | open_rows)
    for conflict in conflicts:
        _describe_conflict(conflict, regions, conflicts, declared[conflict.region], prims, edges,
                           face_of_cycle, cycle_of, medium)
    reported_gaps = set()
    first_free: Dict[str, ClosureFailure] = {}
    for failure in failures:
        failure.near_miss = miss_at.get(failure.point)
        if failure.near_miss is not None:
            if id(failure.near_miss) in reported_gaps:
                failure.explained_by = -1       # one gap, two free ends: report it once
            reported_gaps.add(id(failure.near_miss))
        elif failure.dangling:
            # The free ends of one segment (a missing primitive) are one finding.
            first = first_free.setdefault(failure.names[0], failure)
            if first is not failure:
                first.other_ends.append(failure.point)
                failure.explained_by = -1
        if not failure.dangling:
            for index, conflict in enumerate(conflicts):
                if not conflict.explained and set(failure.rows) & set(conflict.rows):
                    failure.explained_by = index
                    break
    for failure in failures:
        _describe_closure(failure, prims)
    return RegionReport(regions, conflicts, failures, near_misses, tol, mode)


def _near_misses(dangling, verts, edges, at_vertex, prims, types,
                 checkpoint=None) -> List[NearMiss]:
    """Nearest primitive within NEAR_MISS_FRACTION of each free end's primitive length.

    Only gaps roughly along the end's extension (within 60 degrees), or two
    free ends facing each other, count: two parallel walls 0.001 apart are a
    deliberate separation, not a near miss.
    """

    real = [(u, v, i) for u, v, i in edges if i != AXIS_EDGE]
    lengths = [math.hypot(verts[v][0] - verts[u][0], verts[v][1] - verts[u][1]) for u, v, _ in real]
    if not real:
        return []
    cell = max(float(np.median(lengths)), 1e-300)
    grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    wide: List[int] = []
    for e, (u, v, _) in enumerate(real):
        if checkpoint is not None and e % 256 == 0:
            checkpoint()
        a, b = verts[u], verts[v]
        x0, x1 = sorted((math.floor(a[0] / cell), math.floor(b[0] / cell)))
        y0, y1 = sorted((math.floor(a[1] / cell), math.floor(b[1] / cell)))
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 4096:
            wide.append(e)
            continue
        for i in range(x0, x1 + 1):
            for j in range(y0, y1 + 1):
                grid[(i, j)].append(e)
    free = set(dangling)
    edge_at: Dict[int, List[int]] = defaultdict(list)
    for e, (u, v, _) in enumerate(real):
        edge_at[u].append(e)
        edge_at[v].append(e)

    def outward(e: int, k: int) -> Point:
        u, v, _ = real[e]
        other = v if u == k else u
        dx, dy = verts[k][0] - verts[other][0], verts[k][1] - verts[other][1]
        length = math.hypot(dx, dy)
        return (dx / length, dy / length) if length else (0.0, 0.0)

    found = []
    for k in dangling:
        if checkpoint is not None:
            checkpoint()
        e = edge_at[k][0]
        u, v, index = real[e]
        prim = prims[index]
        if prim.seg_type not in types:
            continue
        other_end = v if u == k else u
        radius = NEAR_MISS_FRACTION * lengths[e]
        p, t = verts[k], outward(e, k)
        skip = set(edge_at[k]) | set(edge_at[other_end])
        x0, x1 = math.floor((p[0] - radius) / cell), math.floor((p[0] + radius) / cell)
        y0, y1 = math.floor((p[1] - radius) / cell), math.floor((p[1] + radius) / cell)
        # A long free end among small primitives can cover billions of empty
        # cells. Scanning the actual edges then costs less and remains exact.
        if (x1 - x0 + 1) * (y1 - y0 + 1) > min(4096, 4 * len(real)):
            candidates = range(len(real))
        else:
            candidates = set(wide)
            for i in range(x0, x1 + 1):
                for j in range(y0, y1 + 1):
                    candidates.update(grid.get((i, j), ()))
        best = None
        for checked, c in enumerate(candidates):
            if checkpoint is not None and checked % 256 == 0:
                checkpoint()
            if c in skip:
                continue
            a_k, b_k, c_index = real[c]
            gap, q = _closest_on_segment(p, verts[a_k], verts[b_k])
            if gap > radius or gap <= 0.0 or (best is not None and gap >= best[0]):
                continue
            along = ((q[0] - p[0]) * t[0] + (q[1] - p[1]) * t[1]) >= 0.5 * gap
            facing = False
            for end in (a_k, b_k):
                if end in free and q == verts[end]:
                    s = outward(c, end)
                    facing = t[0] * s[0] + t[1] * s[1] <= -0.5
            if along or facing:
                best = (gap, q, c_index)
        if best is not None:
            other = prims[best[2]]
            if any(miss.point == best[1] and miss.other_point == p for miss in found):
                continue        # the same gap seen from its other end
            found.append(NearMiss(prim.name, prim.row, prim.seg_type, p, other.name, other.row,
                                  best[1], best[0]))
    return found


def _fix_for(prim: RegionPrimitive, conflicting_side_left: bool, want: Medium, other: Medium) -> str:
    """Plain suggestion that makes ``prim`` declare ``want`` on its conflicting side."""

    left, right = (want, other) if conflicting_side_left else (other, want)
    target = boundary_for(left, right)
    if target is None:
        return (f"no boundary TYPE has {medium_name(want)} on one side and "
                f"{medium_name(other)} on the other; check the neighboring regions.")
    seg_type, pos, neg, flipped = target
    if seg_type == 5 and flipped:
        # A TYPE 5 boundary reversed is the same boundary with swapped labels.
        pos, neg, flipped = neg, pos, False
    labels = {3: f"pos_mat={pos}", 4: f"pos_mat={pos}",
              5: f"pos_mat={pos}, neg_mat={neg}"}.get(seg_type, "")
    if seg_type != prim.seg_type:
        return (f"make it TYPE {seg_type}" + (f" with {labels}" if labels else "")
                + (", drawn the other way round." if flipped else "."))
    same = seg_type in (1, 2) or (pos, neg if seg_type == 5 else 0) == (
        prim.pos_mat, prim.neg_mat if seg_type == 5 else 0)
    if seg_type == 5 and (prim.pos_mat, prim.neg_mat) == (neg, pos):
        return "swap its pos_mat and neg_mat (or reverse its direction)."
    if flipped:
        return "reverse its direction." if same else f"set {labels} and reverse its direction."
    return f"set {labels}."


def _mid(prim: RegionPrimitive) -> Point:
    return (0.5 * (prim.start[0] + prim.end[0]), 0.5 * (prim.start[1] + prim.end[1]))


def _describe_conflict(conflict, regions, conflicts, media, prims, edges, face_of_cycle, cycle_of, medium):
    region = regions[conflict.region]
    parts = "; ".join(f"{medium_name(m)} by {_quoted(names)}" for m, names in conflict.media.items())
    if region.unbounded:
        conflict.summary = f"the unbounded exterior (which must be air) is declared as {parts}"
        message = f"Material regions disagree: {conflict.summary}."
    else:
        conflict.summary = f"the region next to {_at(conflict.location)} is declared as {parts}"
        message = f"Material regions disagree: one region is declared as {parts}."

    def fixes(expected: Medium) -> str:
        out = []
        for m, entry in media.items():
            if m == expected:
                continue
            h = entry["half"]
            prim = prims[edges[h >> 1][2]]
            twin = regions[face_of_cycle[cycle_of[h ^ 1]]]
            other = twin.medium
            if other is None and twin.conflict is not None and twin is not region:
                other = conflicts[twin.conflict].expected
            other = other or medium[h ^ 1]
            pair = (expected, other) if h % 2 == 0 else (other, expected)
            if (boundary_for(*pair) is None and twin is not region and not twin.unbounded
                    and set(twin.rows) <= {prim.row}):
                other = m       # only this segment sets that side: swap its two media
            labels = (f", pos_mat={prim.pos_mat}" if prim.seg_type in (3, 4, 5) else "") + (
                f", neg_mat={prim.neg_mat}" if prim.seg_type == 5 else "")
            out.append(f"'{prim.name}' (TYPE {prim.seg_type}{labels}) near {_at(_mid(prim))}: "
                       + _fix_for(prim, h % 2 == 0, expected, other))
        return " Also check ".join(out)

    if region.unbounded:
        message += f" It must be air, so check {fixes(conflict.expected)}"
    elif conflict.alternative is None:
        longest = max(media, key=lambda m: media[m]["length"])
        lead = (f"Most of its boundary says {medium_name(conflict.expected)}"
                if longest == conflict.expected or media.get(conflict.expected, {}).get("length") ==
                media[longest]["length"] else
                f"Only {medium_name(conflict.expected)} fits all of its neighboring regions")
        message += f" {lead}, so check " + fixes(conflict.expected)
    else:
        message += (" Its boundaries are about equally long, so one of two readings applies. "
                    f"If it is {medium_name(conflict.expected)}, check {fixes(conflict.expected)} "
                    f"If it is {medium_name(conflict.alternative)}, check {fixes(conflict.alternative)}")
    conflict.message = message


def _describe_closure(failure: ClosureFailure, prims) -> None:
    media = ", ".join(medium_name(m) for m in failure.media)
    if failure.dangling:
        ends = " and ".join(_at(p) for p in [failure.point] + failure.other_ends)
        failure.summary = (f"the {media} boundary is open: '{failure.names[0]}' (TYPE "
                           f"{failure.seg_types[0]}) ends at {ends} without "
                           "meeting another primitive")
        message = f"Open material boundary: {failure.summary}."
        if failure.near_miss is not None:
            miss = failure.near_miss
            message += (f" {miss.message} Move the endpoint onto '{miss.other_name}' "
                        f"at {_at(miss.other_point)} so the boundary closes.")
        else:
            message += " Close the contour or stitch it to the neighboring segment."
        failure.message = message
        return
    failure.summary = (f"the {media} boundary does not close at the junction "
                       f"{_at(failure.point)} of {_quoted(failure.names)}")
    failure.message = (f"Material regions disagree: {failure.summary}; the media these "
                       "segments declare around that vertex do not match. Check their TYPE, "
                       "pos_mat/neg_mat and direction.")


def merge_orientation_findings(
    orientation: Sequence[Tuple[str, int, str]],
    report: RegionReport,
) -> List[Tuple[str, int, str]]:
    """Old contour-by-contour findings plus the region errors, without duplicates.

    A region error whose likely culprit the orientation check already flags
    (a reversed winding is also a region conflict) is appended to that
    message as a short confirmation instead of being listed again; every
    other region error becomes its own ERROR.
    """

    out = list(orientation)
    flagged = {}
    for k, (severity, row, _message) in enumerate(out):
        if severity == "ERROR" and row is not None and row >= 0:
            flagged.setdefault(row, k)
    for item in report.error_items():
        hit = next((flagged[row] for row in item.rows if row in flagged), None)
        if hit is None:
            out.append(("ERROR", item.row, item.message))
        else:
            severity, row, message = out[hit]
            out[hit] = (severity, row, f"{message} The region check agrees: {item.summary}.")
    return out
