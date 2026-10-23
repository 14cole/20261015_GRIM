"""Shared, explicit phase declarations for imaging and reference transforms."""
from __future__ import annotations

import re
import numpy as np


RANGE_PHASE_METADATA_KEYS = (
    "range_phase_convention", "phase_law", "amplitude_convention", "phase_reference",
)


def metadata_is_unspecified(value) -> bool:
    """Producer placeholders are absence of evidence, not incompatible physics."""
    text = re.sub(r"[^a-z0-9]+", " ", str("" if value is None else value).casefold()).strip()
    placeholders = {"", "none", "null", "nan", "na", "n a", "unknown", "unspecified",
                    "undefined", "unverified", "not applicable", "not specified",
                    "not provided", "not known", "not declared", "arbitrary"}
    if text in placeholders:
        return True
    for subject in ("phase reference", "phase center", "phase centre", "geometry",
                    "motion compensation", "time convention", "elevation convention"):
        if text in {f"{subject} {word}" for word in placeholders if word} or text in {
            f"{word} {subject}" for word in placeholders if word
        }:
            return True
    return False


def _declarations(dataset, keys):
    for container in (dataset.units or {}, dataset.extra or {}):
        for key in keys:
            value = container.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            if array.size != 1:
                raise ValueError(f"metadata {key!r} must be scalar")
            yield str(array.reshape(-1)[0])


def declared_time_sign(dataset) -> str | None:
    """Resolve all supported time declarations, including conflicts in one value."""
    signs = set()
    for value in _declarations(dataset, ("time_convention", "phase_reference", "amplitude_convention")):
        compact = value.casefold().replace("ω", "omega").replace("*", "")
        compact = re.sub(r"\s+", "", compact).replace("−", "-").replace("–", "-")
        if compact in {"+jwt", "-jwt"}:
            signs.add(compact)
        for sign in re.findall(r"exp\(([+-]?)j(?:omega|w)t\)", compact):
            signs.add("-jwt" if sign == "-" else "+jwt")
    if len(signs) > 1:
        raise ValueError("dataset declares contradictory time conventions")
    return next(iter(signs), None)


def declared_range_phase_sign(dataset) -> int | None:
    """Resolve explicit two-way exp(±j2kR) laws; one-way laws are insufficient."""
    signs = set()
    for value in _declarations(dataset, RANGE_PHASE_METADATA_KEYS):
        compact = value.casefold().replace("−", "-").replace("–", "-")
        compact = re.sub(r"[\s*·^{}()\[\]_=~]+", "", compact)
        matches = re.findall(r"(?:exp|e)([+-])j2(?:\.0+)?kr", compact)
        if "negativetwowayrangephase" in compact:
            matches.append("-")
        if "positivetwowayrangephase" in compact:
            matches.append("+")
        signs.update(-1 if match == "-" else 1 for match in matches)
    if len(signs) > 1:
        raise ValueError("dataset contains contradictory two-way range-phase declarations")
    return next(iter(signs), None)


def plot_phase_conventions(dataset, *, notes: list[str] | None = None):
    """Resolve usable declarations without vetoing a read-only plot.

    Ambiguous time metadata leaves samples unconjugated; ambiguous range
    metadata leaves the plot's default range convention in effect. Strict
    parsers remain available for operations that modify or combine datasets.
    """
    resolved = []
    for parser, fallback in (
        (declared_time_sign, "samples are used without time-convention conjugation"),
        (declared_range_phase_sign, "the default range-phase convention is used"),
    ):
        try:
            resolved.append(parser(dataset))
        except (TypeError, ValueError) as exc:
            resolved.append(None)
            if notes is not None:
                notes.append(f"{exc}; {fallback}")
    return tuple(resolved)
