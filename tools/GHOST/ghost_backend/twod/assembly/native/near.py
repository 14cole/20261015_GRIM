"""Native close-pair box and thin-gap arithmetic; the NumPy expressions remain the fallback.

``near.c`` repeats the NumPy operations of ``operators._integrate_linear_boxes_sk``
and ``thin_gap._inner`` one for one, while every transcendental value (Hankel
and Bessel functions, logarithms, arctangents, kernel tables) is still computed
by NumPy/SciPy and passed in, so each native result is bitwise the NumPy one.
Every function returns None when the library, or a NumPy complex-product mode
it can reproduce, is unavailable; callers then run their NumPy code.
"""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import numpy as np

# Tests and diagnostics switch the native route off to compare the two.
ENABLED = True


@lru_cache(None)
def _dll():
    path = Path(__file__).with_name('ghost_near.dll' if os.name == 'nt' else 'libghost_near.so')
    try:
        dll = ct.CDLL(str(path))
        functions = (dll.ghost_near_box_arguments, dll.ghost_near_box_blocks, dll.ghost_thin_analytic,
                     dll.ghost_thin_squares, dll.ghost_thin_remainder, dll.ghost_near_has_fused,
                     dll.ghost_thin_contract)
    except (OSError, AttributeError):
        return None
    p, i64, i32, f64 = ct.c_void_p, ct.c_int64, ct.c_int, ct.c_double
    functions[0].argtypes = [i64, i32] + [p]*7 + [f64, f64, p, p, i32]
    functions[1].argtypes = [i64, i32] + [p]*11 + [f64, f64] + [p]*4 + [i32]
    functions[2].argtypes = [i64, i32, i32] + [p]*19 + [i32]
    functions[3].argtypes = [i64, i32] + [p]*6
    functions[4].argtypes = [i64, i32, i32, i32] + [p]*13 + [f64, f64] + [p]*4 + [i32]
    functions[6].argtypes = [i64, i32, i32, p, p, p]
    for function in functions:
        function.restype = ct.c_int
    return dll


@lru_cache(None)
def _mode():
    """The fused-product flag to pass, or None when native results would differ."""
    dll = _dll()
    if dll is None:
        return None
    from ghost_backend.twod.assembly.native.far import numpy_fused_product
    fused = numpy_fused_product()
    if fused is None or (fused and not dll.ghost_near_has_fused()):
        return None
    return int(fused)


def available():
    return ENABLED and _mode() is not None


def _f64(value):
    return np.ascontiguousarray(value, dtype=np.float64)


def _pointer(array):
    return None if array is None else array.ctypes.data


def _check(status, name):
    if status:
        raise RuntimeError(f'Native {name} rejected its inputs (status {status}).')


def _complex_pairs(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.complex128)).view(np.float64)


def box_arguments(geometry, pair, intervals, qt, k0):
    """(kr, real_box) of `_integrate_linear_boxes_sk` boxes, or None for the NumPy route.

    kr is complex (n, q, q) with the 1e-12 floor applied; real_box marks boxes
    whose arguments all take the real Bessel branch.
    """
    if not available():
        return None
    obs_p0, obs_seg, _, src_p0, src_seg, _, _ = geometry
    pair = np.ascontiguousarray(pair, dtype=np.int64)
    intervals, qt = _f64(intervals), _f64(qt)
    n, q = len(pair), len(qt)
    kr = np.empty((n, q, q), dtype=np.complex128)
    real_box = np.empty(n, dtype=np.uint8)
    k0 = complex(k0)
    arrays = [_f64(a) for a in (obs_p0, obs_seg, src_p0, src_seg)]
    status = _dll().ghost_near_box_arguments(
        n, q, pair.ctypes.data, intervals.ctypes.data, qt.ctypes.data, *(a.ctypes.data for a in arrays),
        k0.real, k0.imag, kr.ctypes.data, real_box.ctypes.data, _mode())
    if status == 2:
        return None
    _check(status, 'box arguments')
    return kr, real_box.astype(bool)


def box_blocks(geometry, pair, intervals, qt, qw, coefficient, h0, h1):
    """S and K (n, 2, 2) blocks of `_integrate_linear_boxes_sk` from its Hankel values.

    ``coefficient`` is the derivative kernel's (-+0.25j k); h0 / h1 may be None
    for an operator that is not wanted (its blocks are then zero).
    """
    obs_p0, obs_seg, obs_length, src_p0, src_seg, src_length, normal = geometry
    pair = np.ascontiguousarray(pair, dtype=np.int64)
    intervals, qt, qw = _f64(intervals), _f64(qt), _f64(qw)
    n, q = len(pair), len(qt)
    hankel = [None if h is None else np.ascontiguousarray(h, dtype=np.complex128) for h in (h0, h1)]
    for h in hankel:
        if h is not None and h.shape != (n, q, q):
            raise ValueError('Invalid native box layout.')
    s = np.empty((n, 2, 2), dtype=np.complex128)
    k = np.empty((n, 2, 2), dtype=np.complex128)
    arrays = [_f64(a) for a in (obs_p0, obs_seg, obs_length, src_p0, src_seg, src_length, normal)]
    coefficient = complex(coefficient)
    status = _dll().ghost_near_box_blocks(
        n, q, pair.ctypes.data, intervals.ctypes.data, qt.ctypes.data, qw.ctypes.data,
        *(a.ctypes.data for a in arrays), coefficient.real, coefficient.imag,
        *(_pointer(h) for h in hankel), s.ctypes.data, k.ctypes.data, _mode())
    _check(status, 'box blocks')
    return s, k


def thin_analytic(degree, t, h, delta, d2, u0, u1, angle, l0, l1, length, t3, l3, m0, m1, sigma,
                  s_series, v_series, want_s, want_k):
    """Closed-form part of ``thin_gap._inner`` (S, K of shape (P, degree + 1)), or None.

    t3 and l3 are NumPy's t**3 and length**3 (None below degree 3).
    """
    if not available():
        return None
    count = len(t)
    columns = [_f64(a) for a in (t, h, delta, d2, u0, u1, angle, l0, l1, length)]
    cubes = [None if a is None else _f64(a) for a in (t3, l3)]
    if degree == 3 and any(a is None or a.shape != (count,) for a in cubes):
        raise ValueError('Invalid native thin-gap layout.')
    rest = [_f64(a) for a in (m0, m1, sigma)]
    s_out = np.empty((count, degree + 1), dtype=np.complex128) if want_s else None
    k_out = np.empty((count, degree + 1), dtype=np.complex128) if want_k else None
    series = [_complex_pairs(values) for values in (s_series, v_series)]
    status = _dll().ghost_thin_analytic(
        count, degree, len(s_series), *(a.ctypes.data for a in columns), *(_pointer(a) for a in cubes),
        *(a.ctypes.data for a in rest), *(a.ctypes.data for a in series),
        _pointer(s_out), _pointer(k_out), _mode())
    _check(status, 'thin-gap closed form')
    return s_out, k_out


def thin_squares(t, length, d2, qt, qw):
    """Squared distances (P, 2 len(qt)) to the remainder nodes of ``thin_gap._inner``, or None."""
    if not available():
        return None
    t, length, d2, qt, qw = (_f64(a) for a in (t, length, d2, qt, qw))
    r2 = np.empty((len(t), 2 * len(qt)), dtype=np.float64)
    _check(_dll().ghost_thin_squares(len(t), len(qt), t.ctypes.data, length.ctypes.data, d2.ctypes.data,
                                     qt.ctypes.data, qw.ctypes.data, r2.ctypes.data), 'thin-gap nodes')
    return r2


def thin_remainder(degree, t, length, d2, h, m0, m1, sigma, qt, qw, r, log_r2, green, derivative,
                   kk, s_series, v_series, want_s, want_k):
    """Smooth-remainder part of ``thin_gap._inner`` from its kernel values (see thin_squares)."""
    count, nq = len(t), len(qt)
    columns = [_f64(a) for a in (t, length, d2, h, m0, m1, sigma, qt, qw, r, log_r2)]
    kernels = [None if v is None else np.ascontiguousarray(v, dtype=np.complex128) for v in (green, derivative)]
    for array in columns[-2:] + [v for v in kernels if v is not None]:
        if array.shape != (count, 2 * nq):
            raise ValueError('Invalid native thin-gap layout.')
    s_out = np.empty((count, degree + 1), dtype=np.complex128) if want_s else None
    k_out = np.empty((count, degree + 1), dtype=np.complex128) if want_k else None
    kk = complex(kk)
    series = [_f64(s_series), _complex_pairs(v_series)]
    status = _dll().ghost_thin_remainder(
        count, degree, len(s_series), nq, *(a.ctypes.data for a in columns),
        *(_pointer(v) for v in kernels), kk.real, kk.imag, *(a.ctypes.data for a in series),
        _pointer(s_out), _pointer(k_out), _mode())
    _check(status, 'thin-gap remainder')
    return s_out, k_out


def thin_contract(outer, values):
    """einsum('nqa,nqb->nab', outer, values) of real outer weights and complex values, or None."""
    if not available():
        return None
    outer = _f64(outer)
    values = np.ascontiguousarray(values, dtype=np.complex128)
    n, q, size = outer.shape
    if values.shape != (n, q, size):
        raise ValueError('Invalid native thin-gap layout.')
    out = np.empty((n, size, size), dtype=np.complex128)
    _check(_dll().ghost_thin_contract(n, q, size, outer.ctypes.data, values.ctypes.data, out.ctypes.data),
           'thin-gap contraction')
    return out
