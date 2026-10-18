"""Graded meridian quadrature for every touching BoR element pair.

Self, adjacent and junction pairs (two surfaces sharing an end point) are
integrated here. The fixed graded-cell rules they used (``solver._graded_cells``:
same-surface depth 4, order 4; junctions depth 7, order 10) refine toward the
diagonal or the touching corner of the unit square only, and their block
error does not shrink with the mesh. Against converged references (k = 2 pi /
0.3 m, largest block change over the largest block of the pair, modes 0..8
unless stated): a sphere's self block 3e-3 (1e-2 at m_max = 30), its pole
element's 0.2 (0.4); 90..150 degree corners 3e-3..5e-3 (MFIE); cone tips and
disc centres 0.2..0.3 (0.6 at m_max = 30); rotated-PV junction blocks 3e-5 at
90 degrees, 8e-5 at 135, 3e-4 next to an element of L/8; a thin junction
1e-6..6e-5. Sharp pairs were worse:

* elements meeting at a small included angle (a thin rim, a feathered
  coating junction): the kernel's logarithm is near-singular along s = s' at
  a distance proportional to the angle, all along both elements (adjacent
  pair of a rim: 30 degrees 9.5e-3 MFIE, 15 degrees 2.5e-2, 1 degree 0.5);
* steep elements whose ring diameter 2 rho is small against their length (a
  needle or cone tip, a thin rod): the modal kernel changes from log(d) to
  about 1/(2 d) at meridian separations d ~ 2 rho, inside one graded cell
  (self pair of a 2 degree needle's fourth element 0.33).

A near-refinement level at most halves these, so the optional base/refined
comparison cannot see them.

The pairs are integrated in polar coordinates about the singular point
(Duffy triangles) with composite Gauss rules graded dyadically toward each
near-singular feature down to its own scale: an adjacent pair's angular
variable toward the complex pole of its separation (as in the 2-D acute
touching rule), the radial and separation variables toward the ring scale
rho (deeper for high modes, RING_MODES). Below that scale the kernel is a
logarithm plus a smooth part, which the innermost piece integrates through
x = h t**4. A self pair whose ring is at least an element length away uses
the two triangles about the diagonal in |s - s'| instead (cheaper near
kernels). Against references with 32 points per piece and every stop four
times deeper, every category is within 1e-7, mostly 1e-9..3e-8 (modes to
60, lossy and high-contrast media, 10 and 40 elements per wavelength): see
test_bor_touching_pairs.py and NUMERICAL_METHODS.md. A sampled self-check
(CHECK_*) runs on every solve.
"""
import math
from functools import lru_cache

import numpy as np

from ghost_backend.bor.kernels import cached_leggauss

# Classification (reporting and self-check probes only: every touching pair
# uses the graded rule). Touching pairs below this included angle are
# 'acute'.
SHARP_ANGLE_DEGREES = 60.0
# Steep elements (|d rho| below THIN_SLOPE element lengths: within 30 degrees
# of the axis) whose ring diameter 2 rho_max is below THIN_RING_RATIO element
# lengths are 'thin', for their self pair and every touching pair, junction
# pairs included. Every other pair is 'regular'.
THIN_RING_RATIO = 8.0
THIN_SLOPE = 0.5 * (1.0 - 1e-9)
# Gauss points per graded piece at the base near depth; each level of the
# near-refinement option adds four. Self pairs carry the logarithm along the
# whole diagonal and need more: with 12 their blocks were within 5e-8 at
# m <= 8 but 7e-7 at an axis pole (2e-8 / 3e-8 with 16).
TOUCHING_POINTS = 12
SELF_POINTS = 16
BASE_NEAR_DEPTH = 4
# The self-check rule adds CHECK_POINTS points per piece and takes every
# graded variable CHECK_DEPTH dyadic levels further. Its largest relative
# change of a pair's blocks is reported; above SELF_CHECK_LIMIT it is a
# warning. Production measures at most 3e-8.
CHECK_POINTS = 6
CHECK_DEPTH = 1
SELF_CHECK_LIMIT = 1.0e-5
# Mode m resolves azimuthal arcs ~ rho / m, so the ring-scale stops move a
# dyadic level deeper per doubling of m_max above RING_MODES (self pairs) or
# TOUCHING_RING_MODES (adjacent and junction pairs). With stops at rho itself
# (and 12 points) a 2 degree needle's blocks lost accuracy from 1.3e-7
# (m_max = 8) to 3e-6 (30) and 5.5e-5 (60). At m_max = 30 the adjacent pairs
# of a 30 degree cone tip are within 1e-7 with 16, 3e-9 with 8 and 1e-11
# with 4 (1.0, 1.3 and 1.7 times the points); its self pairs within 2e-8,
# 1e-8 and 6e-9 at 1.0, 1.8 and 2.9 times the points, so self pairs keep 16.
RING_MODES = 16.0
TOUCHING_RING_MODES = 8.0
# Regular pairs (ordinary geometry) are probed by the self-check every
# REGULAR_CHECK_STRIDE-th time the sampling stride would probe a sharp pair,
# plus the most severe regular pair: their error is uniform (2e-9..3e-8)
# and the check rule costs 2.4 times the production points.
REGULAR_CHECK_STRIDE = 4
# Pole closeness at which an angular variable is graded (Bernstein ellipse
# parameter on [0, 1], as in the 2-D touching rule).
ANGULAR_RHO = 2.0
_POLY = 4


def _unit_gauss(n):
    x, w = cached_leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def _toward_zero(stop, n, deeper=1.0):
    """[0, 1]: dyadic pieces toward 0 down to ``stop``, innermost x = h t**4.

    A graded variable (``stop`` < 1) goes ``deeper`` times further."""
    stop = min(1.0, float(stop))
    stop = max(stop * deeper if stop < 1.0 else stop, 2.0**-50)
    return _pieces(max(0, int(math.ceil(math.log2(1.0 / stop) - 1e-12))), int(n))


@lru_cache(maxsize=512)
def _pieces(pieces, n):
    t, w = _unit_gauss(n)
    lower = 2.0**-np.arange(1, pieces + 1, dtype=float)
    h = 2.0**-pieces
    x = np.concatenate([(lower[:, None] * (1.0 + t[None, :])).ravel(), h * t**_POLY])
    wx = np.concatenate([(lower[:, None] * w[None, :]).ravel(), h * _POLY * t**(_POLY - 1) * w])
    x.setflags(write=False); wx.setflags(write=False)
    return x, wx


def _bernstein_rho(c):
    z = 2.0 * complex(c) - 1.0
    root = np.sqrt(z - 1.0) * np.sqrt(z + 1.0)
    return max(abs(z + root), abs(z - root))


def _toward_pole(c, n, deeper=1.0):
    """[0, 1] Gauss, or composite graded toward Re c down to |Im c| (times ``deeper``)."""
    t, w = _unit_gauss(n)
    if _bernstein_rho(c) >= ANGULAR_RHO:
        return t, w
    a = min(max(float(c.real), 0.0), 1.0)
    reach = max(abs(complex(c) - a) * deeper, 1e-15)
    edges = [0.0, a, 1.0]
    for side, sign in ((a, -1.0), (1.0 - a, 1.0)):
        step = side
        while step > reach:
            step *= 0.5
            edges.append(a + sign * step)
    edges = np.unique(np.clip(edges, 0.0, 1.0))
    width = np.diff(edges)
    return ((edges[:-1, None] + width[:, None] * t[None, :]).ravel(),
            (width[:, None] * w[None, :]).ravel())


def _floor(points, length):
    """Smallest kept separation in units of ``length``.

    Kernel samples take coordinates (z relative to the touching point, see
    GradedRule), so separations below a few dozen roundings of them carry no
    information, and equal points are a non-integrable sample. Near a needle
    tip the dropped band is not negligible: a floor of 1e-11 cost 3e-6 of the
    tip element's block, 1e-13 costs 4e-8."""
    points = np.asarray(points, float)
    scale = max(float(np.max(np.abs(points[:, 0]))), float(np.ptp(points[:, 1])))
    return max(1e-13, 64.0 * np.finfo(float).eps * scale / float(length))


def _angle_degrees(d_o, d_s):
    return math.degrees(math.atan2(abs(float(d_o[0] * d_s[1] - d_o[1] * d_s[0])), float(d_o @ d_s)))


def _thin(r0, r1):
    length = float(np.hypot(*(r1 - r0)))
    return (abs(float(r1[0] - r0[0])) < THIN_SLOPE * length
            and 2.0 * max(float(r0[0]), float(r1[0])) < THIN_RING_RATIO * length)


def touching_class(other_e, corner, other_f):
    """'acute', 'thin' or 'regular' for elements (other_e, corner) and (corner, other_f).

    Same-surface and junction pairs are classified alike."""
    if _angle_degrees(other_e - corner, other_f - corner) < SHARP_ANGLE_DEGREES:
        return 'acute'
    if _thin(other_e, corner) or _thin(corner, other_f):
        return 'thin'
    return 'regular'


def _ring_factor(m_max, ring_modes=None):
    """Dyadic factor of the ring-scale stops for modes up to ``m_max``."""
    ring_modes = RING_MODES if ring_modes is None else ring_modes
    modes = max(0, int(m_max or 0))
    return 2.0**-max(0, int(math.ceil(math.log2(modes / ring_modes) - 1e-12))) if modes > ring_modes else 1.0


def _touching_rule(other_e, corner, other_f, e_end, f_end, n, deeper=1.0, ring=1.0):
    """(s, s', w) of a pair touching at test end ``e_end`` / source end ``f_end``.

    ``deeper`` scales the stops of graded variables (self-check), ``ring`` the
    ring scale (see RING_MODES)."""
    d_o, d_s = other_e - corner, other_f - corner
    l_o, l_s = float(np.hypot(*d_o)), float(np.hypot(*d_s))
    dot = float(d_o @ d_s)
    cross = abs(float(d_o[0] * d_s[1] - d_o[1] * d_s[0]))
    # A touching point on the axis has no ring scale: the kernel is
    # homogeneous about it, as at the apex of a self pair.
    ring_scale = float(corner[0]) / (l_o + l_s)
    u, wu = _toward_zero(ring * ring_scale if ring_scale > 1e-12 else 1.0, n, deeper)
    floor = _floor(np.vstack((other_e, corner, other_f)), min(l_o, l_s)) * min(l_o, l_s)
    xs, ys, ws = [], [], []
    for pole, first in (((dot + 1j * cross) / (l_s * l_s), True),
                        ((dot + 1j * cross) / (l_o * l_o), False)):
        # First triangle y = x v (x = u); second x = y v (y = u).
        v, wv = _toward_pole(pole, n, deeper)
        U, V = np.repeat(u, v.size), np.tile(v, u.size)
        x, y = (U, U * V) if first else (U * V, U)
        xs.append(x); ys.append(y)
        ws.append(np.outer(wu * u, wv).ravel())
    x, y, w = map(np.concatenate, (xs, ys, ws))
    keep = np.hypot(x * d_o[0] - y * d_s[0], x * d_o[1] - y * d_s[1]) >= floor
    x, y, w = x[keep], y[keep], w[keep]
    # x, y: distances from the touching point in units of each element.
    return (1.0 - x if e_end else x), (1.0 - y if f_end else y), w


def _diagonal_rule(n, floor):
    """(s, s', w) of a self pair without a ring scale: the two triangles about
    the diagonal in u = |s - s'| (graded toward 0, innermost x = h t**4) and
    Gauss in s along it. Its samples sit at tiny separations less often than
    the apex-centred triangles', whose separation is a product of two graded
    variables: the near kernels cost 40 % less at the same accuracy (1e-9 at
    m <= 8, 1e-8 at 30, 512 points)."""
    u, wu = _toward_zero(1.0, n)
    keep = u >= floor
    u, wu = u[keep], wu[keep]
    t, wt = _unit_gauss(n)
    U = np.repeat(u, t.size)
    big = U + (1.0 - U) * np.tile(t, u.size)
    weight = np.repeat(wu, t.size) * np.tile(wt, u.size) * (1.0 - U)
    return np.r_[big, big - U], np.r_[big - U, big], np.r_[weight, weight]


def _self_rule(r0, r1, n, deeper=1.0, ring=1.0):
    """(s, s', w) of a self pair: triangles about the diagonal from the smaller
    ring, graded toward the ring scale; ``_diagonal_rule`` when no variable
    would be graded (the ring at least an element length away)."""
    length = float(np.hypot(*(r1 - r0)))
    flip = r1[0] < r0[0]
    apex, far = (r1, r0) if flip else (r0, r1)
    stop = ring * float(apex[0]) / length if apex[0] > 1e-12 * length else 1.0
    floor = _floor(np.vstack((r0, r1)), length)
    if stop >= 1.0 and apex[0] > 1e-12 * length:
        return _diagonal_rule(n, floor)
    t, wt = _toward_zero(stop, n, deeper)
    ss, ps, ws = [], [], []
    for ti, wi in zip(t, wt):
        if ti < floor:
            continue
        rho = float(apex[0] + ti * (far[0] - apex[0]))
        v, wv = _toward_zero(ring * rho / (ti * length), n, deeper)
        keep = ti * v >= floor
        v, wv = v[keep], wv[keep]
        big = np.full(v.size, ti)
        small = ti - ti * v
        weight = (wi * ti) * wv
        ss += [big, small]; ps += [small, big]; ws += [weight, weight]
    s, sp, w = map(np.concatenate, (ss, ps, ws))
    return (1.0 - s, 1.0 - sp, w) if flip else (s, sp, w)


class GradedRule(tuple):
    """(s, s', weights) of a graded rule, with the axial coordinate its kernel
    samples are taken relative to (``z_reference``: the touching point or the
    smaller-ring end). Modal kernels depend on z - z' only, and relative
    coordinates keep separations far below the body's size resolvable: with
    absolute ones a 0.25 degree needle tip 0.15 m from z = 0 lost 2e-5 of its
    block."""

    def __new__(cls, values, z_reference):
        rule = super().__new__(cls, values)
        rule.z_reference = float(z_reference)
        return rule

    def __reduce__(self):
        return GradedRule, (tuple(self), self.z_reference)


@lru_cache(maxsize=4096)
def _cached_rule(key, ends, n, deeper, ring):
    points = np.array(key, float).reshape(-1, 2)
    if ends is None:
        out = _self_rule(*points, n, deeper, ring)
        reference = points[1 if points[1, 0] < points[0, 0] else 0, 1]
    else:
        out = _touching_rule(*points, *ends, n, deeper, ring)
        reference = points[1, 1]
    for value in out:
        value.setflags(write=False)
    return GradedRule(out, reference)


def points_per_piece(depth, check=False, self_pair=False):
    """Gauss points per graded piece for a near depth (4 + near_refinement)."""
    base = SELF_POINTS if self_pair else TOUCHING_POINTS
    n = max(8, base + 4 * (int(depth) - BASE_NEAR_DEPTH))
    return n + CHECK_POINTS if check else n


def _rule(points, ends, depth, m_max, check):
    key = tuple(float(v) for v in np.asarray(points, float).ravel())
    return _cached_rule(key, ends, points_per_piece(depth, check, ends is None),
                        2.0**-CHECK_DEPTH if check else 1.0,
                        _ring_factor(m_max, RING_MODES if ends is None else TOUCHING_RING_MODES))


def _ring_ratio(r0, r1):
    return 2.0 * max(float(r0[0]), float(r1[0])) / float(np.hypot(*(r1 - r0)))


def _same_surface_geometry(gen, e, f):
    """(class, rule points, ends) of a same-surface self/adjacent pair."""
    nodes = gen.nodes
    if e == f:
        return ('thin' if _thin(nodes[e], nodes[e + 1]) else 'regular'), nodes[e:e + 2], None
    if f == e + 1:
        points, ends = nodes[[e, e + 1, e + 2]], (1, 0)
    else:
        points, ends = nodes[[e + 1, e, e - 1]], (0, 1)
    return touching_class(*points), points, ends


def pair_points(gen, e, f, depth, m_max=None, check=False):
    """(s, s', weights) of the graded rule of a same-surface self/adjacent pair.

    ``m_max`` is the largest mode the points serve (None: RING_MODES or less)."""
    _, points, ends = _same_surface_geometry(gen, e, f)
    return _rule(points, ends, depth, m_max, check)


def _junction_geometry(gp, e, gq, f, cell):
    a, b = int(cell[6]), int(cell[7])
    return np.array([gp.nodes[e + 1 - a], gp.nodes[e + a], gq.nodes[f + 1 - b]]), (a, b)


def junction_points(gp, e, gq, f, cell, refinement, m_max=None, check=False):
    """(s, s', weights) of the graded rule of a junction pair.

    ``cell`` is the pair's ``cornerAB`` (touching at s = A, s' = B)."""
    points, ends = _junction_geometry(gp, e, gq, f, cell)
    return _rule(points, ends, BASE_NEAR_DEPTH + int(refinement), m_max, check)


def sharp_report(pairs):
    """Counts and extremes of the sharp ones of [(class, angle, ring ratio)] records, or None.

    ``min_angle_deg`` is over the acute pairs, ``min_ring_ratio`` (2 rho_max
    over the length of the thinner element) over the thin ones."""
    angles = [angle for kind, angle, _ in pairs if kind == 'acute']
    rings = [ring for kind, _, ring in pairs if kind == 'thin']
    if not angles and not rings:
        return None
    return dict(pairs=len(angles) + len(rings), acute=len(angles), thin=len(rings),
                min_angle_deg=min(angles) if angles else None,
                min_ring_ratio=min(rings) if rings else None)


CLASSES = ('acute', 'thin', 'regular')


def severity(record):
    """Sort key of a record: the sharpest angle or thinnest ring first.

    Regular pairs order by ring ratio: an axis pole first."""
    kind, angle, ring = record
    return (0, angle) if kind == 'acute' else (CLASSES.index(kind), ring)


def _record(kind, points, ends):
    if ends is None:
        return kind, None, _ring_ratio(*points)
    return (kind, _angle_degrees(points[0] - points[1], points[2] - points[1]),
            min(_ring_ratio(points[0], points[1]), _ring_ratio(points[1], points[2])))


def same_surface_pair(gen, e, f):
    """(class, included angle or None for a self pair, ring ratio) of a same-surface pair."""
    return _record(*_same_surface_geometry(gen, e, f))


def junction_pair(gp, e, gq, f, cell):
    """(class, included angle, ring ratio) of a junction pair."""
    points, ends = _junction_geometry(gp, e, gq, f, cell)
    return _record(touching_class(*points), points, ends)


def check_probes(records, stride):
    """Indices of the pairs a self-check integrates again: every ``stride``-th
    sharp pair and every (REGULAR_CHECK_STRIDE * ``stride``)-th regular one
    (0: every pair), and the most severe pair of each class (``severity``)."""
    probe = set()
    for kind in CLASSES:
        members = [i for i, record in enumerate(records) if record[0] == kind]
        step = stride * (REGULAR_CHECK_STRIDE if kind == 'regular' else 1)
        probe.update(members[::step] if step > 0 else members)
        if members:
            probe.add(min(members, key=lambda i: severity(records[i])))
    return probe


def relative_change(production, refined):
    """Largest |refined - production| over max |refined| per family, of one pair."""
    worst = 0.0
    for kind, value in refined.items():
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(production[kind])):
            return float('inf')
        scale = float(np.max(np.abs(value)))
        difference = float(np.max(np.abs(value - production[kind])))
        if scale > 0.0:
            worst = max(worst, difference / scale)
        elif difference > 0.0:
            return float('inf')
    return worst
