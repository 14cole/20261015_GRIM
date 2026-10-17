"""Graded meridian quadrature for sharp touching BoR element pairs.

The fixed graded-cell rules (``solver._graded_cells``: same-surface depth 4,
order 4; junctions depth 7, order 10) refine toward the diagonal or the
touching corner of the unit square only, so two kinds of touching pair keep
an error that does not shrink with the mesh:

* elements meeting at a small included angle (a thin rim, a feathered
  coating junction): the kernel's logarithm is near-singular along s = s' at
  a distance proportional to the angle, all along both elements;
* steep elements whose ring diameter 2 rho is small against their length (a
  needle or cone tip, a thin rod): the modal kernel changes from log(d) to
  about 1/(2 d) at meridian separations d ~ 2 rho, inside one graded cell.

Against converged references (k = 2 pi / 0.3 m, modes 0..8, largest block
change over the largest block of the pair) the fixed rules' errors are the
same at 10 and 160 elements per wavelength. Adjacent pair at a rim: 30
degrees 1.4e-6 (EFIE) / 9.5e-3 (MFIE), 15 degrees 1.5e-5 / 2.5e-2, 5 degrees
7e-4 / 0.14, 1 degree 5e-3 / 0.5 (a 90 degree corner: 5e-7 / 2.7e-3). Self
pair of a needle's fourth element: 10 degrees 7.6e-2 / 6.8e-3, 2 degrees
0.33 / 2.3e-2; its tip element 0.65 and 0.91 of the EFIE block. Junction
pairs (rotated-PV): 12.5 degrees 1.2e-3, 5 degrees 7e-3, 1 degree 0.26. A
near-refinement level at most halves these, and at 1 degree it hardly changes
them (0.50, 0.47, 0.45 of the rim's MFIE block at depths 4, 5, 6), so the
optional base/refined comparison cannot see them there.

Those pairs are integrated in polar coordinates about the singular point
(Duffy triangles) with composite Gauss rules graded dyadically toward each
near-singular feature down to its own scale: an adjacent pair's angular
variable toward the complex pole of its separation (as in the 2-D acute
touching rule), the radial and separation variables toward the ring scale
rho. Below that scale the kernel is a logarithm plus a smooth part, which the
innermost piece integrates through x = h t**4. Against references with 32
points per piece and every stop four times deeper, the production blocks are
within 6e-8 for rims to 0.25 degree, 3e-10 for junctions to 1 degree and
7e-7 for needles to 0.25 degree (modes to 60), the same at 10 and 160
elements per wavelength. Every other pair keeps its fixed rule, so ordinary
bodies are unchanged.
"""
import math
from functools import lru_cache

import numpy as np

from ghost_backend.bor.kernels import cached_leggauss

# Touching pairs below this included angle use the angle-graded rule.
SHARP_ANGLE_DEGREES = 60.0
# Steep elements (|d rho| below THIN_SLOPE element lengths: within 30
# degrees of the axis) whose ring diameter 2 rho_max is below
# THIN_RING_RATIO element lengths use the ring-graded rule, for their self
# pair and both adjacent pairs. End caps and the pole elements of round
# bodies are not steep and keep the fixed rule; a steep element with a wider
# ring has the fixed rule's ordinary error (2.4e-3 of its self block at a
# ratio of 8, as on a sphere).
THIN_RING_RATIO = 8.0
THIN_SLOPE = 0.5 * (1.0 - 1e-9)
# Gauss points per graded piece at the base near depth; each level of the
# near-refinement option adds four.
SHARP_POINTS = 12
BASE_NEAR_DEPTH = 4
# The self-check rule adds CHECK_POINTS points per piece and takes every
# graded variable CHECK_DEPTH dyadic levels further. Its largest relative
# change of a sharp pair's blocks is reported; above SELF_CHECK_LIMIT it is a
# warning. Production measures at most 7e-7.
CHECK_POINTS = 6
CHECK_DEPTH = 1
SELF_CHECK_LIMIT = 1.0e-5
# Mode m resolves azimuthal arcs ~ rho / m, so the ring-scale stops move a
# dyadic level deeper per doubling of m_max above RING_MODES. With stops at
# rho itself the blocks of a 2 degree needle lose accuracy from 1.3e-7
# (m_max = 8) to 3e-6 (30) and 5.5e-5 (60); with these stops needles to
# 0.25 degree stay within 7e-7.
RING_MODES = 16.0
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
    t, w = _unit_gauss(n)
    stop = min(1.0, float(stop))
    stop = max(stop * deeper if stop < 1.0 else stop, 2.0**-50)
    pieces = max(0, int(math.ceil(math.log2(1.0 / stop) - 1e-12)))
    lower = 2.0**-np.arange(1, pieces + 1, dtype=float)
    x = [(lower[:, None] * (1.0 + t[None, :])).ravel()]
    wx = [(lower[:, None] * w[None, :]).ravel()]
    h = 2.0**-pieces
    x.append(h * t**_POLY)
    wx.append(h * _POLY * t**(_POLY - 1) * w)
    return np.concatenate(x), np.concatenate(wx)


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


def touching_class(other_e, corner, other_f, junction=False):
    """'acute', 'thin' or None for elements (other_e, corner) and (corner, other_f).

    Junction pairs (two surfaces) are classified by their angle only."""
    if _angle_degrees(other_e - corner, other_f - corner) < SHARP_ANGLE_DEGREES:
        return 'acute'
    if not junction and (_thin(other_e, corner) or _thin(corner, other_f)):
        return 'thin'
    return None


def _ring_factor(m_max):
    """Dyadic factor of the ring-scale stops for modes up to ``m_max``."""
    modes = max(0, int(m_max or 0))
    return 2.0**-max(0, int(math.ceil(math.log2(modes / RING_MODES) - 1e-12))) if modes > RING_MODES else 1.0


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


def _self_rule(r0, r1, n, deeper=1.0, ring=1.0):
    """(s, s', w) of a self pair: triangles about the diagonal from the smaller ring."""
    length = float(np.hypot(*(r1 - r0)))
    flip = r1[0] < r0[0]
    apex, far = (r1, r0) if flip else (r0, r1)
    t, wt = _toward_zero(ring * float(apex[0]) / length if apex[0] > 1e-12 * length else 1.0, n, deeper)
    floor = _floor(np.vstack((r0, r1)), length)
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


def points_per_piece(depth, check=False):
    """Gauss points per graded piece for a near depth (4 + near_refinement)."""
    n = max(8, SHARP_POINTS + 4 * (int(depth) - BASE_NEAR_DEPTH))
    return n + CHECK_POINTS if check else n


def _rule(points, ends, depth, m_max, check):
    key = tuple(float(v) for v in np.asarray(points, float).ravel())
    return _cached_rule(key, ends, points_per_piece(depth, check),
                        2.0**-CHECK_DEPTH if check else 1.0, _ring_factor(m_max))


def _ring_ratio(r0, r1):
    return 2.0 * max(float(r0[0]), float(r1[0])) / float(np.hypot(*(r1 - r0)))


def _same_surface_geometry(gen, e, f):
    """(class, rule points, ends) of a same-surface self/adjacent pair."""
    nodes = gen.nodes
    if e == f:
        return ('thin' if _thin(nodes[e], nodes[e + 1]) else None), nodes[e:e + 2], None
    if f == e + 1:
        points, ends = nodes[[e, e + 1, e + 2]], (1, 0)
    else:
        points, ends = nodes[[e + 1, e, e - 1]], (0, 1)
    return touching_class(*points), points, ends


def sharp_points(gen, e, f, depth, m_max=None, check=False):
    """(s, s', weights) of the graded rule for a sharp same-surface pair, or None.

    ``m_max`` is the largest mode the points serve (None: RING_MODES or less)."""
    kind, points, ends = _same_surface_geometry(gen, e, f)
    return None if kind is None else _rule(points, ends, depth, m_max, check)


def _junction_geometry(gp, e, gq, f, cell):
    a, b = int(cell[6]), int(cell[7])
    return np.array([gp.nodes[e + 1 - a], gp.nodes[e + a], gq.nodes[f + 1 - b]]), (a, b)


def junction_points(gp, e, gq, f, cell, refinement, m_max=None, check=False):
    """(s, s', weights) of the graded rule for an acute junction pair, or None.

    ``cell`` is the pair's ``cornerAB`` (touching at s = A, s' = B)."""
    points, ends = _junction_geometry(gp, e, gq, f, cell)
    if touching_class(*points, junction=True) is None:
        return None
    return _rule(points, ends, BASE_NEAR_DEPTH + int(refinement), m_max, check)


def sharp_report(pairs):
    """Counts and extremes of [(class, angle, ring ratio)] records, or None.

    ``min_angle_deg`` is over the acute pairs, ``min_ring_ratio`` (2 rho_max
    over the length of the thinner element) over the thin ones."""
    if not pairs:
        return None
    angles = [angle for kind, angle, _ in pairs if kind == 'acute']
    rings = [ring for kind, _, ring in pairs if kind == 'thin']
    return dict(pairs=len(pairs), acute=len(angles), thin=len(rings),
                min_angle_deg=min(angles) if angles else None,
                min_ring_ratio=min(rings) if rings else None)


def severity(record):
    """Sort key of a record: the sharpest angle or thinnest ring first."""
    kind, angle, ring = record
    return (0, angle) if kind == 'acute' else (1, ring)


def same_surface_pair(gen, e, f):
    """(class, included angle, ring ratio) of a sharp same-surface pair, or None."""
    kind, points, ends = _same_surface_geometry(gen, e, f)
    if kind is None:
        return None
    if ends is None:
        return kind, None, _ring_ratio(*points)
    return (kind, _angle_degrees(points[0] - points[1], points[2] - points[1]),
            min(_ring_ratio(points[0], points[1]), _ring_ratio(points[1], points[2])))


def junction_pair(gp, e, gq, f, cell):
    """(class, included angle, None) of an acute junction pair, or None."""
    points, _ = _junction_geometry(gp, e, gq, f, cell)
    kind = touching_class(*points, junction=True)
    return None if kind is None else (kind, _angle_degrees(points[0] - points[1], points[2] - points[1]), None)


def relative_change(production, refined):
    """Largest |refined - production| over max |refined| per family, of one pair."""
    worst = 0.0
    for kind, value in refined.items():
        scale = float(np.max(np.abs(value)))
        if scale > 0.0:
            worst = max(worst, float(np.max(np.abs(value - production[kind]))) / scale)
    return worst
