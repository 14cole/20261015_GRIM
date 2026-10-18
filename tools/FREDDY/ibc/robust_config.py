"""Validated, serializable robust-design controls (no numerical dependencies)."""
import math

ROBUST_DEFAULTS = dict(samples=64, seed=12345, repeats=4, levels=2)


def validate_robust_setup(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError('Robust design settings must be an object.')
    values = {}
    for key, default in ROBUST_DEFAULTS.items():
        value = raw.get(key, default)
        try:
            integer = int(value)
            if isinstance(value, bool) or float(value) != integer:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f'{key}: enter an integer.') from None
        values[key] = integer
    n = values['samples']
    if not 16 <= n <= 2048 or n & (n - 1):
        raise ValueError('Robust samples must be a power of two from 16 to 2048.')
    if not 0 <= values['seed'] < 2**32:
        raise ValueError('Robust seed must be between 0 and 2³²−1.')
    if not 2 <= values['repeats'] <= 16 or not 1 <= values['levels'] <= 4:
        raise ValueError('Use 2–16 independent yield repeats and 1–4 grid refinements.')
    return values
