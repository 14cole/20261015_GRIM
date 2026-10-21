"""Local planar dispersion prerequisites for isotropic BoR impedance surfaces.

The two tangential current families are coupled in a BoR Fourier mode. Neither
VV nor HH is one of the scalar 2-D polarizations, so both dispersion branches
must be checked. These checks resolve a possible guided wave; they do not
bound the physical error of a curved surface or replace mesh convergence.
"""
import math
import numpy as np


def surface_wave_indices(impedances, *, transmitting_sheet=False):
    """Maximum phase index of both local branches, retaining the input shape."""
    from ghost_backend.twod.geometry import _bound_surface_wave_index
    values = np.asarray(impedances, dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ValueError('BoR surface-wave sampling requires finite impedance values.')
    # Constant laws commonly span thousands of elements. Evaluate a law once
    # instead of repeating the medium validation for every element.
    unique, inverse = np.unique(values, return_inverse=True)
    indices = np.array([_bound_surface_wave_index(
        z, transmitting_sheet=transmitting_sheet) for z in unique])
    if not np.all(np.isfinite(indices)):
        raise ValueError('BoR surface-wave phase index exceeds the numerical range.')
    return indices[inverse].reshape(values.shape)


def direct_surface_wave_sampling(points, impedances, frequency_hz, *,
                                 transmitting_sheet=False):
    """Actual P1 meridian sampling and local ``beta*rho`` modal bandwidth.

    A mode varies exactly as exp(j*m*phi); no azimuthal element spacing is
    involved. A locally propagating guided mode can have |m| up to beta*rho.
    Starting the measured tail after this band avoids stopping at two quiet
    incident-wave modes before a possible later surface-wave resonance. This
    is a conservative local-planar prerequisite, not a universal spectral bound.
    """
    from ghost_backend.twod.geometry import (
        C0, SURFACE_WAVE_MIN_SAMPLES_PER_WAVELENGTH,
    )
    pts = np.asarray(points, dtype=float)
    frequency_hz = float(frequency_hz)
    if (pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2
            or not np.all(np.isfinite(pts))):
        raise ValueError('BoR surface-wave sampling needs finite (N, 2) meridian points.')
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError('BoR surface-wave sampling frequency must be positive and finite.')
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError('BoR surface-wave sampling requires positive element lengths.')
    values = np.asarray(impedances, complex)
    if values.ndim == 0:
        values = np.full(lengths.shape, complex(values))
    if values.shape != lengths.shape:
        raise ValueError('BoR surface impedance must have one value per meridian element.')
    indices = surface_wave_indices(values, transmitting_sheet=transmitting_sheet)
    active = indices > 0.0
    minimum = SURFACE_WAVE_MIN_SAMPLES_PER_WAVELENGTH
    if not np.any(active):
        return dict(resolved=True, phase_index=0.0, surface_wave_mode_bandwidth=0.0,
                    samples_per_wavelength=None,
                    required_samples_per_wavelength=float(minimum))
    wavelength0 = C0 / float(frequency_hz)
    samples = wavelength0 / (indices[active] * lengths[active])
    radii = np.maximum(pts[:-1, 0], pts[1:, 0])
    bandwidth = float(2.0 * math.pi / wavelength0 * np.max(indices * radii))
    if not math.isfinite(bandwidth):
        raise ValueError('BoR surface-wave modal bandwidth exceeds the numerical range.')
    return dict(resolved=bool(np.min(samples) >= minimum * (1.0 - 1.e-9)),
                phase_index=float(np.max(indices)),
                surface_wave_mode_bandwidth=bandwidth,
                samples_per_wavelength=float(np.min(samples)),
                required_samples_per_wavelength=float(minimum))


def require_resolved_surface_waves(report, phase='base'):
    """Reject before assembly when a certified mesh cannot resolve its law."""
    for record in report.get('segments', ()):
        if not record['resolved']:
            raise ValueError(
                "Certified BoR solve has unresolved bound surface-wave sampling "
                f"on the {phase} mesh: segment '{record['segment']}' at "
                f"{report['frequency_ghz']:g} GHz has "
                f"{record['samples_per_wavelength']:.3g} meridional intervals "
                "per bound wavelength; needs at least "
                f"{record['required_samples_per_wavelength']:.3g}. "
                "Increase that segment's N before certification. Automatic "
                f"density is capped at {report['automatic_density_factor_limit']:g} "
                "times the material-wave density; a small two-mesh difference "
                "cannot establish resolution of this surface wave.")
