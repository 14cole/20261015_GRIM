"""Consistent dB requirement comparisons without changing computed responses."""

# Far below meaningful engineering precision; absorb solver roundoff at an
# exact requirement boundary, including a lossless PEC stack at 0 dB.
REQUIREMENT_TOL_DB = 1e-10


def requirement_margin(target, worst):
    margin = float(target) - float(worst)
    return 0.0 if abs(margin) <= REQUIREMENT_TOL_DB else margin
