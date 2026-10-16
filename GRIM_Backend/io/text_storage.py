"""Managed text import storage and immutable, source-checked discovery plans."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import os
import sys
from types import MappingProxyType

from GRIM_Backend.datasets.result_storage import operation_result_storage

_DISCOVERY = ContextVar("grim_text_discovery", default=None)


def source_signature(path):
    stat = os.stat(path)
    return (os.path.normcase(os.path.abspath(os.fspath(path))), stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, set):
        return frozenset(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return set(value)
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class TextDiscoveryPlan:
    """Coordinate sets and schema facts, never sample rows or mutable arrays."""
    signature: tuple
    stages: object


@contextmanager
def discovery_session(path, plan=None):
    signature = source_signature(path)
    # A completed file may legitimately be replaced between planning and load.
    # Ignore stale discovery and repeat it against the new source.
    stages = (dict(plan.stages) if isinstance(plan, TextDiscoveryPlan)
              and plan.signature == signature else {})
    state = (signature, stages)
    token = _DISCOVERY.set(state)
    try:
        yield state
        if source_signature(path) != signature:
            raise ValueError("table changed during import; reload the completed file")
    finally:
        _DISCOVERY.reset(token)


def discover_coordinates(path, key, discover):
    state = _DISCOVERY.get()
    if state is None or state[0] != source_signature(path):
        return discover()
    stages = state[1]
    if key not in stages:
        stages[key] = _freeze(discover())
    return _thaw(stages[key])


def text_reader(function):
    """Keep direct reader APIs compatible while accepting managed-load options."""
    @wraps(function)
    def wrapped(*args, storage=None, cache_dir=None, discovery_plan=None, **kwargs):
        path = kwargs.get("path") if not args else (args[1] if isinstance(args[0], type) and len(args) > 1 else kwargs.get("path") if isinstance(args[0], type) else args[0])
        if storage is None and discovery_plan is None:
            return function(*args, **kwargs)
        with discovery_session(path, discovery_plan):
            with operation_result_storage(storage or "memory", cache_dir=cache_dir):
                return function(*args, **kwargs)
    return wrapped


def estimate_text_import(path, *, storage="memory", discovery_plan=None):
    """Plan recognized flat/vendor tables once, retaining only coordinate facts."""
    from GRIM_Backend.io.csv import has_flat_csv_signature, load_flat_csv
    from GRIM_Backend.io.cst import estimate_vendor_text_import
    with discovery_session(path, discovery_plan) as (signature, stages):
        with operation_result_storage(storage):
            if has_flat_csv_signature(path):
                report = load_flat_csv(path, None, None, _estimate_only=True)
            else:
                report = estimate_vendor_text_import(path)
        if report is None:
            return None
        return dict(report, discovery_plan=TextDiscoveryPlan(
            signature, MappingProxyType(dict(stages))))
