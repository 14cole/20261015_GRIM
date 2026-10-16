"""Cooperative cancellation and throttled progress for bounded work loops."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time


class WorkCancelled(RuntimeError):
    """The caller canceled work before its result was published."""


@dataclass
class _Control:
    cancelled: object = None
    progress: object = None
    last_report: float = 0.0


_CONTROL = ContextVar("grim_work_control", default=None)


@contextmanager
def work_control(cancelled=None, progress=None):
    """Scope callbacks to this worker; nested scopes inherit omitted callbacks.

    Progress is a stage-local (completed, total, detail) report, not an estimate
    of overall completion. Thread-pool callers must propagate their context.
    """
    parent = _CONTROL.get()
    control = _Control(
        cancelled if cancelled is not None else getattr(parent, "cancelled", None),
        progress if progress is not None else getattr(parent, "progress", None),
    )
    token = _CONTROL.set(control)
    try:
        yield
    finally:
        _CONTROL.reset(token)


def checkpoint(completed=None, total=None, detail=""):
    """Check interruption at a safe boundary and optionally report progress."""
    control = _CONTROL.get()
    if control is None:
        return
    if control.cancelled is not None and control.cancelled():
        raise WorkCancelled("Canceled")
    if control.progress is not None and completed is not None and total is not None:
        now = time.monotonic()
        if completed == total or now - control.last_report >= 0.1:
            control.last_report = now
            control.progress(int(completed), int(total), str(detail))
