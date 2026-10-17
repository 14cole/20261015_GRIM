"""Material-region geometry checks shared by BoR preview and direct solves."""
import numpy as np
from scipy.spatial import cKDTree


# Included angles are computed from rounded coordinates: a corner drawn at the
# threshold must not be rejected by the last bits of an arccos.
CORNER_ANGLE_TOLERANCE_DEGREES = 1.0e-6


def _included_angles(points):
    sides = np.diff(np.asarray(points, float), axis=0)
    lengths = np.linalg.norm(sides, axis=1)
    valid = (lengths[:-1] > 0) & (lengths[1:] > 0)
    cosines = np.full(len(sides)-1, -1.)
    cosines[valid] = (-np.sum(sides[:-1]*sides[1:], axis=1)[valid]
                      / (lengths[:-1]*lengths[1:])[valid])
    return np.degrees(np.arccos(np.clip(cosines, -1., 1.)))


# Smallest included meridian corner accepted. Touching element pairs below
# 60 degrees use the angle-graded polar rule (bor.sharp_pairs), whose blocks
# are within 6e-8 of refined references at every angle measured and whose
# self-check runs on every solve; biconical rims of 15, 5, 1, 0.5 and 0.25
# degrees change by at most 0.006 dB between near_refinement 0 and 1 (the
# fixed cell rule: 0.15 dB at 15 degrees). Below this angle no evidence exists.
MINIMUM_CORNER_DEGREES = 0.25


def require_resolved_corners(points, minimum_degrees=MINIMUM_CORNER_DEGREES):
    """Guard acute corners below the validated graded-corner rule.

    Angles at or above the threshold (within a rounding tolerance) pass. This
    is a minimum input safeguard, not a claim of mesh convergence: the current
    at a thin rim is singular, so the mesh still sets the accuracy there. The
    two on-axis end points are not corners of this kind; sharp tips use the
    ring-graded rule of bor.sharp_pairs (needles measured to 0.25 degrees).
    """
    points = np.asarray(points, float)
    if len(points) < 3:
        return
    angles = _included_angles(points)
    bad = np.flatnonzero(angles < minimum_degrees - CORNER_ANGLE_TOLERANCE_DEGREES)
    if len(bad):
        index = int(bad[0])
        raise ValueError(f'BoR corner at node {index+1} has included angle {angles[index]:.6g} degrees; '
            f'the graded corner quadrature is not validated below {minimum_degrees:g} degrees. '
            'Use a resolved rounded geometry or a validated singular-corner formulation; '
            'splitting straight panels alone does not change this corner angle.')


def require_containment(outer, inner, context='BoR coating'):
    """Require a separated closed inner meridian inside a closed outer one.

    Reflecting the outer meridian closes the physical cross-section without
    treating the rotation axis as a material interface. Crossings are checked
    independently of point inclusion, including concave outer profiles.
    """
    from ghost_backend.bor.solver import _segments_intersect_2d
    outer, inner = np.asarray(outer, float), np.asarray(inner, float)
    scale = max(float(np.ptp(np.vstack((outer, inner)), axis=0).max()), 1e-15)
    tol = max(1e-14, 1e-10 * scale)
    if any(abs(points[i, 0]) > tol for points in (outer, inner) for i in (0, -1)):
        raise ValueError(f'{context}: containment requires closed axis-to-axis surfaces.')
    mids = (outer[:-1] + outer[1:]) * .5
    half = np.linalg.norm(np.diff(outer, axis=0), axis=1) * .5
    tree = cKDTree(mids)
    for a, b in zip(inner[:-1], inner[1:]):
        center = (a + b) * .5
        radius = np.linalg.norm(b-a) * .5 + half.max() + tol
        for j in tree.query_ball_point(center, radius):
            if _segments_intersect_2d(a, b, outer[j], outer[j+1], tol):
                raise ValueError(f'{context}: inner and outer surfaces touch or intersect.')
    polygon = np.vstack((outer, outer[-2:0:-1] * [-1., 1.]))
    a, b = polygon, np.roll(polygon, -1, axis=0)
    for rho, z in inner:
        crossing = (a[:, 1] > z) != (b[:, 1] > z)
        aa, bb = a[crossing], b[crossing]
        x = aa[:, 0] + (z-aa[:, 1]) * (bb[:, 0]-aa[:, 0]) / (bb[:, 1]-aa[:, 1])
        if np.count_nonzero(x > rho) % 2 != 1:
            raise ValueError(f'{context}: the inner surface must be strictly inside the outer surface.')
