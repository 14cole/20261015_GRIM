"""Local finite-slab comparisons for the first-order collapsed layer.

These are admission checks, not error bounds for finite/curved scatterers.
The exact reference propagates the two bulk traces through a centered slab;
it does not reuse the first-order jump coefficients. Surface-mode checks
cover the sheet poles inside its small-transverse-phase domain, not all slab
modes. Poles outside that domain are deliberately not used for mesh sizing.
"""

import cmath
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import newton


PLANAR_RT_ERROR_LIMIT = .01
GUIDED_INDEX_ERROR_LIMIT = .01
TRANSVERSE_PHASE_LIMIT = .15
# Resolve the ordinary angular range and approach grazing logarithmically.
# This is a sampled comparison; its evidence must not imply an all-angle bound.
_ANGLES_DEG = np.unique(np.r_[np.linspace(0., 89., 179), 89.5, 89.9,
                             89.95, 89.99, 89.995, 89.999])


def _propagator(alpha, beta, electrical_thickness, sine_squared):
    """Exact bulk propagation of (u, (du/dz)/(beta*k0))."""
    phase = electrical_thickness * np.sqrt(complex(alpha * beta) - sine_squared)
    factor = np.sinc(phase / np.pi)
    matrix = np.zeros((len(sine_squared), 2, 2), complex)
    matrix[:, 0, 0] = matrix[:, 1, 1] = np.cos(phase)
    matrix[:, 0, 1] = electrical_thickness * beta * factor
    matrix[:, 1, 0] = -electrical_thickness * (alpha - sine_squared / beta) * factor
    return matrix


def planar_scattering(epsilon, permeability, thickness_m, k0, polarization,
                      angles_deg):
    """Complex (r,t) for a first-order sheet and exact centered bulk slab.

    Under exp(+j*omega*t), an incident forward plane wave has traces
    (1,-j*cos(theta)). Bulk propagation is exp(d*G), where
    G/k0=[[0,beta],[-(alpha-sin(theta)^2/beta),0]]. Air propagation by -d/2
    on either side moves both reference planes to the sheet midsurface.
    The implemented jump law is (I-J/2)*trace_right=(I+J/2)*trace_left,
    where J=d*(G-G0). Solve it directly, including when its equivalent
    transfer matrix has a removable singularity.
    """
    alpha, beta = ((epsilon, permeability) if polarization == 'TM'
                   else (permeability, epsilon))
    delta = float(k0) * float(thickness_m)
    angles = np.deg2rad(np.asarray(angles_deg, float).reshape(-1))
    cosine = np.cos(angles)
    if np.any(cosine <= 0) or np.any(angles < 0) or not np.all(np.isfinite(angles)):
        raise ValueError('Planar layer reference requires angles in [0, 90) degrees.')
    s2 = np.sin(angles) ** 2
    air = _propagator(1., 1., -delta / 2, s2)
    exact = air @ _propagator(alpha, beta, delta, s2) @ air
    jump = np.zeros_like(exact)
    jump[:, 0, 1] = delta * (beta - 1.)
    jump[:, 1, 0] = -delta * ((alpha - 1.) + s2 * (1. - 1. / beta))
    identity = np.eye(2, dtype=complex)
    incoming = np.column_stack((np.ones_like(cosine), -1j * cosine))
    backward = incoming.conj()

    def scattering(transfer):
        left = (transfer @ backward[:, :, None])[:, :, 0]
        rhs = -(transfer @ incoming[:, :, None])[:, :, 0]
        return np.linalg.solve(np.stack((left, -incoming), axis=2), rhs[:, :, None])[:, :, 0]

    left = ((identity + jump / 2) @ backward[:, :, None])[:, :, 0]
    right = ((identity - jump / 2) @ incoming[:, :, None])[:, :, 0]
    rhs = -((identity + jump / 2) @ incoming[:, :, None])[:, :, 0]
    sheet_rt = np.linalg.solve(np.stack((left, -right), axis=2), rhs[:, :, None])[:, :, 0]
    return sheet_rt, scattering(exact)


def _sheet_poles(alpha, beta, delta):
    """Dimensionless air decay x=kappa*d of the sheet's two parity laws."""
    a = 1. - 1. / beta
    c = delta * delta * (alpha - 1. / beta)
    # a*x^2-2*x+c=0. Stable quadratic evaluation also retains the small root.
    if a == 0:
        roots = [c / 2]
    else:
        discriminant = cmath.sqrt(1. - a * c)
        q = 1. + discriminant if abs(1. + discriminant) >= abs(1. - discriminant) else 1. - discriminant
        roots = [q / a, c / q]
    candidates = [('even', complex(value)) for value in roots]
    if beta != 1:
        candidates.append(('odd', -2. / (beta - 1.)))
    return candidates


def _slab_dispersion(x, alpha, beta, delta, parity):
    """Exact even/odd slab dispersion, scaled by beta*d.

    Exterior u decays as exp(-kappa*|z|); interior u is cosh/sinh of
    w*z/d, where w^2=x^2+(k0*d)^2*(1-alpha*beta). Continuity of u and
    (du/dz)/beta gives w*tanh(w/2)+beta*x=0 (even), or
    w*coth(w/2)+beta*x=0 (odd). The products have finite limits at w=0.
    """
    w = cmath.sqrt(x * x + delta * delta * (1. - alpha * beta))
    if abs(w) < 1.e-7:
        interior = (w * w / 2. - w**4 / 24. if parity == 'even'
                    else 2. + w * w / 6. - w**4 / 360.)
    else:
        interior = w * cmath.tanh(w / 2.) if parity == 'even' else w / cmath.tanh(w / 2.)
    return interior + beta * x, abs(interior) + abs(beta * x)


def _transverse_phase(x, alpha, beta, delta):
    return max(abs(x), abs(cmath.sqrt(delta * delta + x * x)),
               abs(cmath.sqrt(x * x + delta * delta * (1. - alpha * beta))))


@lru_cache(maxsize=256)
def _local_reference(epsilon, permeability, thickness_m, k0, polarization):
    alpha, beta = ((epsilon, permeability) if polarization == 'TM'
                   else (permeability, epsilon))
    if not all(np.isfinite(value) for value in (alpha, beta, thickness_m, k0)) or beta == 0:
        raise ValueError('Thin-layer local slab comparison requires finite nonzero constitutive parameters.')
    delta = k0 * thickness_m
    sheet, bulk = planar_scattering(epsilon, permeability, thickness_m, k0,
                                    polarization, _ANGLES_DEG)
    error = float(np.max(np.abs(sheet - bulk)))
    if not math.isfinite(error) or error > PLANAR_RT_ERROR_LIMIT:
        raise ValueError(
            f'Thin-layer local bulk-slab reflection/transmission mismatch {error:.4g} '
            f'exceeds {PLANAR_RT_ERROR_LIMIT:g} of incident amplitude. Use explicit bulk geometry.')
    mode_count = 0
    excluded_count = 0
    worst_mode_error = 0.
    worst_phase_error = 0.
    max_index = 0.
    for parity, x in _sheet_poles(alpha, beta, delta):
        index = cmath.sqrt(1. + (x / delta) ** 2)
        if x.real <= 0 or abs(index.imag) > abs(index.real):
            continue
        if _transverse_phase(x, alpha, beta, delta) > TRANSVERSE_PHASE_LIMIT:
            excluded_count += 1
            continue
        # Root search starts at each in-domain sheet pole. A low normalized
        # residual and a nearby same-parity root are both mandatory: an
        # unrelated root reached by iteration must not silently qualify it.
        function = lambda value: _slab_dispersion(value, alpha, beta, delta, parity)[0]
        try:
            exact = complex(newton(function, x, x1=x * (1. + 1.e-4) + 1.e-8,
                                   tol=1.e-13, maxiter=50))
            residual, scale = _slab_dispersion(exact, alpha, beta, delta, parity)
            exact_index = cmath.sqrt(1. + (exact / delta) ** 2)
            matched = (exact.real > 0 and abs(exact_index.imag) <= abs(exact_index.real)
                       and abs(residual) <= 1.e-9 * max(scale, 1.e-30)
                       and abs(exact - x) <= .1 * max(abs(x), delta)
                       and _transverse_phase(exact, alpha, beta, delta) <= TRANSVERSE_PHASE_LIMIT)
        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            matched = False
        if not matched:
            raise ValueError(
                'Thin-layer in-domain guided mode has no matching exact bulk-slab mode; '
                'use explicit bulk geometry.')
        relative = abs(index - exact_index) / max(1., abs(exact_index))
        phase_relative = abs(abs(index.real) - abs(exact_index.real)) / max(1., abs(exact_index.real))
        if not math.isfinite(relative) or relative > GUIDED_INDEX_ERROR_LIMIT:
            raise ValueError(
                f'Thin-layer guided-mode propagation index differs from the exact bulk slab by '
                f'{relative:.3%}, exceeding {GUIDED_INDEX_ERROR_LIMIT:.1%}. '
                'Use explicit bulk geometry.')
        mode_count += 1
        worst_mode_error = max(worst_mode_error, relative)
        worst_phase_error = max(worst_phase_error, phase_relative)
        max_index = max(max_index, abs(index.real), abs(exact_index.real))
    return dict(scope='sampled_planar_propagating_waves_and_low_transverse_phase_sheet_modes',
        passed=True, maximum_absolute_rt_error=error,
        absolute_rt_error_limit=PLANAR_RT_ERROR_LIMIT, angles_sampled=len(_ANGLES_DEG),
        maximum_angle_deg=float(_ANGLES_DEG[-1]),
        maximum_guided_index_relative_error=float(worst_mode_error),
        guided_index_relative_error_limit=GUIDED_INDEX_ERROR_LIMIT,
        maximum_guided_phase_index_relative_error=float(worst_phase_error),
        guided_mode_count=mode_count, maximum_guided_phase_index=float(max_index),
        excluded_out_of_domain_sheet_mode_count=excluded_count,
        transverse_phase_limit=TRANSVERSE_PHASE_LIMIT,
        all_bulk_modes_searched=False, global_scattering_error_bounded=False)


def local_bulk_reference(epsilon, permeability, thickness_m, k0, polarization):
    """Fresh evidence for a cached, bounded local qualification calculation."""
    pol = str(polarization).upper()
    if pol not in {'TM', 'TE'}:
        raise ValueError('Thin layer polarization must be TM or TE.')
    return dict(_local_reference(complex(epsilon), complex(permeability),
                                float(thickness_m), float(k0), pol))


def thin_layer_surface_wave_index(epsilon, permeability, thickness_m, k0,
                                  polarization=None):
    selected = ('TM', 'TE') if polarization is None else (str(polarization).upper(),)
    return max(local_bulk_reference(epsilon, permeability, thickness_m, k0, pol)
               ['maximum_guided_phase_index'] for pol in selected)
