"""Shared labels and stable option values for FREDDY forms/projects."""
from __future__ import annotations
import math

from .compute import MIX_RULE_LABELS, MIX_RULES


INVERSE_SCORE_WHOLE_BAND = "Whole-band PEC reflection requirement (worst point)"
INVERSE_SCORE_ROBUST = "Per-layer tolerances: worst sampled reflection gap"
INVERSE_SCORE_YIELD = "Per-layer tolerances: modeled failure fraction (%)"
MANUFACTURING_OBJECTIVES = (INVERSE_SCORE_ROBUST, INVERSE_SCORE_YIELD)
INVERSE_SCORE_MODE_OPTIONS = (
    "Worst-corner mean PEC reflection |Γ| (dB)",
    "Average-corner mean PEC reflection |Γ| (dB)",
    INVERSE_SCORE_WHOLE_BAND,
    *MANUFACTURING_OBJECTIVES,
)


def inverse_requirement_target(score_mode, value):
    """Only requirement scoring uses a target; legacy mean objectives ignore it."""
    if score_mode not in (INVERSE_SCORE_WHOLE_BAND, *MANUFACTURING_OBJECTIVES):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('Whole-band reflection target must be numeric.') from None
    if not math.isfinite(value) or not -200 <= value <= 0:
        raise ValueError('Whole-band reflection target must be finite and between -200 and 0 dB.')
    return value


def inverse_polarizations(value):
    from .compute import normalize_wave_polarization
    return ('te', 'tm') if str(value).strip().lower() == 'both' else (normalize_wave_polarization(value),)

MIX_RULE_LABEL_OPTIONS = tuple(MIX_RULE_LABELS[key] for key in MIX_RULES)

MIX_OBJECTIVE_FORWARD = "Predict properties from a known recipe"

MIX_OBJECTIVE_PROPERTY = "Find a recipe for target properties"

MIX_OBJECTIVE_PERFORMANCE = "Find a recipe for target stack performance"

MIX_OBJECTIVE_OPTIONS = (
    MIX_OBJECTIVE_FORWARD,
    MIX_OBJECTIVE_PROPERTY,
    MIX_OBJECTIVE_PERFORMANCE,
)

MIX_PROP_SOURCE_OPTIONS = ("Constant values", "Material file")

MIX_PERFORMANCE_METRIC_OPTIONS = (
    ("PEC-backed reflection |Γ| (dB)", "metal_loss_db", "at_most", "dB", -10.0),
    ("PEC-backed absorption (%)", "metal_absorption_db", "at_least", "%", 90.0),
    ("Air-backed reflection |Γ| (dB)", "air_loss_db", "at_most", "dB", -10.0),
    ("Air-backed absorption (%)", "air_absorption_db", "at_least", "%", 50.0),
    ("Air-backed transmission |S21| (dB)", "insertion_loss_db", "at_most", "dB", -10.0),
)

MIX_PERFORMANCE_SPEC_BY_LABEL = {
    label: {
        "label": label,
        "metric_key": metric_key,
        "direction": direction,
        "unit": unit,
        "default_target": default_target,
    }
    for label, metric_key, direction, unit, default_target in MIX_PERFORMANCE_METRIC_OPTIONS
}

MIX_SCORE_MODE_OPTIONS = (
    "Worst-case across uncertainty corners (robust)",
    "Average across uncertainty corners",
)
