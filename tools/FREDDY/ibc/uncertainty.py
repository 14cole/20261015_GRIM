"""Bounded adaptive impedance-envelope estimates, never certified bounds."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .compute import (ETA0, build_uncertainty_scales, compute_stack_impedance_many,
                      describe_uncertainty_sampling)

# Per frequency, across all axes. Only scalar-frequency responses are retained
# during refinement, so storage does not grow as cases * sweep length.
MAX_REFINEMENT_EVALUATIONS = 1024


def impedance_envelope(frequencies, layers, backing, config, nominal=None):
    """Seed cases plus axis extrema/resonance refinement at each frequency.

    Susceptance zero crossings locate narrow impedance poles even when none
    of the seed samples hits their peak. Real/imaginary impedance and
    reflection extrema are then searched on each side. Ordinary sampled
    extrema are refined too. Other axes stay nominal: coupled interior
    extrema and unbracketed oscillations remain outside this estimate.
    """
    frequencies = list(frequencies)
    seeds = build_uncertainty_scales(config)
    nominal = np.asarray(compute_stack_impedance_many(frequencies, layers, backing)
                         if nominal is None else nominal, dtype=complex)
    if nominal.shape != (len(frequencies),):
        raise ValueError('Nominal impedance must match the frequency sweep.')

    def metrics(z):
        z = np.asarray(z, dtype=complex)
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite impedance within the tolerance range; a finite envelope is unavailable.')
        reflection = 20 * np.log10(np.maximum(np.abs((z - ETA0) / (z + ETA0)), 1e-15))
        if not np.isfinite(reflection).all():
            raise ValueError('Nonfinite reflection within the tolerance range.')
        return np.stack((z.real, z.imag, reflection), axis=-1)

    low = metrics(nominal).copy()
    high = low.copy()
    # Retain only the three axis lines used for bracketing; corners feed the
    # envelope directly and do not multiply the adaptive search workspace.
    axes = [{} for _ in range(3)]
    for point in seeds:
        z = nominal if point == (1., 1., 1.) else np.asarray(compute_stack_impedance_many(
            frequencies, layers, backing, thickness_scale=point[0],
            eps_scale=point[1], mu_scale=point[2]), dtype=complex)
        value = metrics(z)
        np.minimum(low, value, out=low)
        np.maximum(high, value, out=high)
        for axis in range(3):
            if all(point[j] == 1. for j in range(3) if j != axis):
                axes[axis][point[axis]] = z

    extra = exhausted = unfinished = 0

    class BudgetReached(Exception):
        pass

    for fi, frequency in enumerate(frequencies):
        calls = 0
        cache = {}

        def evaluate(axis, x):
            nonlocal calls
            key = (axis, float(x))
            if key in cache:
                return cache[key]
            if calls >= MAX_REFINEMENT_EVALUATIONS:
                raise BudgetReached
            point = [1., 1., 1.]
            point[axis] = float(x)
            z = complex(compute_stack_impedance_many([frequency], layers, backing,
                thickness_scale=point[0], eps_scale=point[1], mu_scale=point[2])[0])
            values = metrics(z)
            low[fi] = np.minimum(low[fi], values)
            high[fi] = np.maximum(high[fi], values)
            cache[key] = (z, values)
            calls += 1
            return cache[key]

        def refine(axis, a, b, component, sign):
            nonlocal unfinished
            if b - a <= 1e-12:
                return
            result = minimize_scalar(lambda x: sign * evaluate(axis, x)[1][component],
                bounds=(a, b), method='bounded', options={'xatol': 1e-12, 'maxiter': 64})
            if not result.success:
                unfinished += 1

        try:
            for axis, line in enumerate(axes):
                xs = sorted(line)
                if len(xs) < 3:
                    continue
                zs = [complex(line[x][fi]) for x in xs]
                values = [metrics(z) for z in zs]
                for x, z, value in zip(xs, zs, values):
                    cache[(axis, x)] = (z, value)
                # Search local extrema in all three displayed quantities.
                brackets = set()
                for i in range(1, len(xs) - 1):
                    for component in range(3):
                        left, mid, right = (values[j][component] for j in (i-1, i, i+1))
                        if mid >= max(left, right) and mid > min(left, right):
                            brackets.add((xs[i-1], xs[i+1], component, -1))
                        if mid <= min(left, right) and mid < max(left, right):
                            brackets.add((xs[i-1], xs[i+1], component, 1))
                # Im(1/Z) crosses zero at a parallel resonance; unlike Im(Z)
                # it remains well conditioned near a high, narrow peak.
                for i in range(len(xs) - 1):
                    if zs[i] == 0 or zs[i+1] == 0:
                        continue
                    ya, yb = (1 / zs[i]).imag, (1 / zs[i+1]).imag
                    if ya * yb > 0 or ya == yb:
                        continue
                    def susceptance(x):
                        z = evaluate(axis, x)[0]
                        return (1 / z).imag if z else math.nan
                    center = brentq(susceptance, xs[i], xs[i+1], xtol=1e-13)
                    evaluate(axis, center)
                    for a, b in ((xs[i], center), (center, xs[i+1])):
                        for component in range(3):
                            for sign in (-1, 1):
                                brackets.add((a, b, component, sign))
                for a, b, component, sign in sorted(brackets):
                    refine(axis, a, b, component, sign)
        except BudgetReached:
            exhausted += 1
        extra += calls

    note = (describe_uncertainty_sampling(config) +
            f' Impedance refinement added {extra} scalar-frequency evaluations around '
            'axis extrema and susceptance zero crossings; other scales remain nominal. '
            'Narrow unbracketed resonances and coupled interior extrema may still be missed.')
    if exhausted or unfinished:
        note += (f' Refinement incomplete: {exhausted} frequency points reached the '
                 f'{MAX_REFINEMENT_EVALUATIONS}-evaluation limit; {unfinished} extrema searches did not converge.')
    return dict(impedance_bounds=dict(real_min=low[:, 0], real_max=high[:, 0],
                    imag_min=low[:, 1], imag_max=high[:, 1]),
                reflection_bounds=(low[:, 2], high[:, 2]), note=note,
                refinement_evaluations=extra, budget_exhausted_frequencies=exhausted,
                unconverged_searches=unfinished)
