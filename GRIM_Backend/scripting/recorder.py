"""Record dataset and plotting actions as executable Python scripts."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import math

import numpy as np


@dataclass(frozen=True)
class DatasetReference:
    """Stable recorder identity plus the best durable source for one GUI row."""

    dataset_id: str
    name: str
    path: str = ""


@dataclass(frozen=True)
class _Code:
    text: str


def _safe_comment(value) -> str:
    """Collapse untrusted display text to one inert Python comment line."""

    text = " ".join(str(value).splitlines())
    return " ".join(text.replace("\x00", " ").split()).strip()


def _compact_numeric_sequence(value) -> str | None:
    """Represent long uniform numeric axes with three hard-coded numbers.

    The emitted expression must replay to exactly the recorded value: a list
    stays a list (``.tolist()``) and integer sequences stay integers
    (``np.arange``), so operation arguments and the provenance derived from
    them do not change between the GUI and the replayed script.
    """

    as_list = isinstance(value, list)
    if as_list:
        if not value or any(isinstance(item, (bool, np.bool_)) for item in value):
            return None
        if all(isinstance(item, (int, np.integer)) for item in value):
            kind = "i"
        elif all(isinstance(item, (float, np.floating)) for item in value):
            kind = "f"
        else:
            return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size < 16 or array.dtype.kind not in "iuf":
        return None
    if not as_list:
        kind = "f" if array.dtype.kind == "f" else "i"
        if kind == "f" and array.dtype != np.float64:
            return None
    suffix = ".tolist()" if as_list else ""
    if kind == "i":
        integers = [int(item) for item in array.tolist()]
        step = integers[1] - integers[0]
        if step <= 0:
            return None
        if any(right - left != step for left, right in zip(integers, integers[1:])):
            return None
        stop = integers[-1] + step
        dtype_text = ""
        replayed = np.arange(integers[0], stop, step)
        if not as_list and replayed.dtype != array.dtype:
            dtype_text = f", dtype=np.{array.dtype.name}"
            replayed = np.arange(integers[0], stop, step, dtype=array.dtype)
        if not np.array_equal(replayed, array) or (
            not as_list and replayed.dtype != array.dtype
        ):
            return None
        return f"np.arange({integers[0]}, {stop}, {step}{dtype_text}){suffix}"
    numeric = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        return None
    steps = np.diff(numeric)
    if steps.size == 0 or not np.all(steps > 0.0):
        return None
    expected = np.linspace(numeric[0], numeric[-1], numeric.size, dtype=np.float64)
    if not np.array_equal(numeric, expected):
        return None
    return (
        f"np.linspace({_literal(float(numeric[0]))}, "
        f"{_literal(float(numeric[-1]))}, {int(numeric.size)}){suffix}"
    )


def _literal(value) -> str:
    """Render deterministic Python literals for resolved GUI parameters."""

    if isinstance(value, _Code):
        return value.text
    if isinstance(value, Path):
        return f"Path({_literal(str(value))})"
    if isinstance(value, np.generic):
        return _literal(value.item())
    if isinstance(value, np.ndarray):
        compact = _compact_numeric_sequence(value)
        if compact is not None:
            return compact
        return _literal(value.tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0 else "float('-inf')"
        return repr(value)
    if isinstance(value, tuple):
        body = ", ".join(_literal(item) for item in value)
        if len(value) == 1:
            body += ","
        return f"({body})"
    if isinstance(value, list):
        compact = _compact_numeric_sequence(value)
        if compact is not None:
            return compact
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = [f"{_literal(key)}: {_literal(item)}" for key, item in value.items()]
        return "{" + ", ".join(pairs) + "}"
    return repr(value)


def _call_text(callable_text: str, args=(), kwargs=None, *, indent: str = "") -> str:
    kwargs = kwargs or {}
    rendered = [_literal(value) for value in args]
    rendered.extend(f"{key}={_literal(value)}" for key, value in kwargs.items())
    if not rendered:
        return f"{callable_text}()"
    if len(rendered) == 1 and len(rendered[0]) < 72:
        return f"{callable_text}({rendered[0]})"
    continuation = indent + "    "
    return (
        f"{callable_text}(\n"
        + "".join(f"{continuation}{value},\n" for value in rendered)
        + f"{indent})"
    )


class PythonScriptRecorder:
    """Record dataset and plotting operations as executable Python.

    Bindings use the dataset UUID. Source files are loaded when first used by a
    recorded operation.
    """

    _HEADER = f'''\
"""Generated by GRIM.  Dataset operations and plot output only."""

import os
from pathlib import Path
import sys

# Set GRIM_MODULE_DIR to the GRIM_Backend folder when replaying
# outside the source/install tree used to create this script.
_grim_module_override = os.environ.get("GRIM_MODULE_DIR")
if _grim_module_override:
    _grim_module_candidates = [Path(_grim_module_override).expanduser()]
else:
    _script_dir = Path(__file__).resolve().parent
    _grim_module_candidates = [
        Path({str(Path(__file__).resolve().parents[1])!r}),
        _script_dir,
        _script_dir / "GRIM_Backend",
    ]

GRIM_MODULE_DIR = None
for _candidate in _grim_module_candidates:
    _candidate = _candidate.resolve()
    if (_candidate / "datasets/api.py").is_file():
        GRIM_MODULE_DIR = _candidate
        if str(_candidate.parent) not in sys.path:
            sys.path.insert(0, str(_candidate.parent))
        break

try:
    import numpy as np

    from GRIM_Backend.datasets.api import RcsGrid
    from GRIM_Backend.scripting.api import load_dataset, operation_result_storage
    from GRIM_Backend.scripting.workspace import (
        band_statistics_dataset,
        crop_dataset,
        coherent_divide,
        convert_extrusion,
        duplicate_dataset,
        flip_phase,
        decimate_axis,
        join_datasets,
        medianize_azimuth,
        offset_db,
        plot_datasets,
        regrid_axis,
        save_dataset_batch,
        shift_dataset,
        stitch_datasets,
        time_gate,
        translate_phase_center,
        wedge_to_conic,
        wrap_phase,
    )
except ModuleNotFoundError as exc:
    if exc.name == "GRIM_Backend" or (exc.name or "").startswith("GRIM_Backend."):
        raise ModuleNotFoundError(
            "GRIM could not be imported. Install the GRIM package, place this "
            "script beside GRIM_Backend, or set GRIM_MODULE_DIR to the "
            "GRIM_Backend folder."
        ) from exc
    raise
'''

    def __init__(self, on_change: Callable[[str], None] | None = None) -> None:
        self._on_change = on_change
        self.action_listener = None
        self._lines: list[str] = []
        self._dataset_vars: dict[str, str] = {}
        self._result_storage: dict[str, dict] = {}
        self._next_dataset = 1
        self._next_plot = 1
        self._current_plot_var: str | None = None
        self._current_plot_signature: str | None = None
        self._notify()

    @property
    def script(self) -> str:
        body = "\n".join(self._lines)
        if body:
            body = "\n" + body.rstrip() + "\n"
        return self._HEADER + body

    def set_on_change(self, callback: Callable[[str], None] | None) -> None:
        self._on_change = callback
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self.script)

    def clear(self) -> None:
        self._lines.clear()
        self._dataset_vars.clear()
        self._result_storage.clear()
        self._next_dataset = 1
        self._next_plot = 1
        self._current_plot_var = None
        self._current_plot_signature = None
        self._notify()

    def _new_dataset_var(self, dataset_id: str) -> str:
        variable = f"dataset_{self._next_dataset}"
        self._next_dataset += 1
        self._dataset_vars[str(dataset_id)] = variable
        return variable

    def bind_loaded(self, reference: DatasetReference, *, load_options=None) -> str | None:
        existing = self._dataset_vars.get(reference.dataset_id)
        if existing is not None:
            return existing
        if not reference.path:
            return None
        variable = self._new_dataset_var(reference.dataset_id)
        resolved = str(Path(reference.path).expanduser().resolve())
        options = {key: value for key, value in (load_options or {}).items()
                   if key in {'storage', 'selection', 'max_output_bytes', 'cache_dir', 'declared_units'}
                   and value is not None
                   and not (key == 'storage' and value == 'memory')}
        if (load_options or {}).get('allow_exceed'):
            import sys
            options['max_output_bytes'] = sys.maxsize
        suffix = ''.join(f", {key}={_literal(value)}" for key, value in sorted(options.items()))
        self._lines.extend(
            [
                f"# Load {_safe_comment(reference.name)}",
                f"{variable} = load_dataset(Path({_literal(resolved)}){suffix})",
                "",
            ]
        )
        self._notify()
        return variable

    def _ensure(self, reference: DatasetReference) -> str | None:
        return self._dataset_vars.get(reference.dataset_id) or self.bind_loaded(reference)

    def _skip_unbound(self, operation: str, references: Sequence[DatasetReference]) -> None:
        names = ", ".join(
            _safe_comment(ref.name) for ref in references if not ref.path
        )
        self._lines.extend(
            [
                f"# {_safe_comment(operation)} was not recorded: "
                f"{names or 'an input dataset'} has no replayable source.",
                "# Save that dataset, clear this script, and repeat the operation to capture it.",
                "",
            ]
        )
        self._notify()

    def bind_result_storage(self, dataset_id, options):
        """Keep each output's captured allocation policy for script replay."""
        self._result_storage[str(dataset_id)] = {
            key: value for key, value in dict(options).items()
            if key in {'mode', 'ram_budget_bytes', 'cache_dir', 'allow_exceed'} and value is not None
        }

    def _append_result_statement(self, outputs, statement):
        options = next((self._result_storage[output.dataset_id] for output in outputs
                        if output.dataset_id in self._result_storage), None)
        if options:
            context = _call_text('operation_result_storage', kwargs=options)
            self._lines.append(f'with {context}:')
            self._lines.extend('    ' + line if line else '' for line in statement.splitlines())
        else:
            self._lines.append(statement)
        self._lines.append('')

    def record_method(
        self,
        output: DatasetReference,
        source: DatasetReference,
        method: str,
        *,
        args=(),
        kwargs=None,
        comment: str = "",
    ) -> str | None:
        self._record_action([output], [source], [dict(
            kind='method', name=method, args=args, kwargs=kwargs or {},
        )], comment or method)
        source_var = self._ensure(source)
        if source_var is None:
            self._skip_unbound(comment or method, [source])
            return None
        output_var = self._new_dataset_var(output.dataset_id)
        if comment:
            self._lines.append(f"# {_safe_comment(comment)}")
        call = _call_text(f"{source_var}.{method}", args, kwargs)
        self._append_result_statement([output], f"{output_var} = {call}")
        self._notify()
        return output_var

    def record_function(
        self,
        output: DatasetReference,
        function: str,
        sources: Sequence[DatasetReference],
        *,
        args=(),
        kwargs=None,
        comment: str = "",
    ) -> str | None:
        self._record_action([output], sources, [dict(
            kind='function', name=function, args=args, kwargs=kwargs or {},
        )], comment or function)
        variables = [self._ensure(reference) for reference in sources]
        if any(variable is None for variable in variables):
            self._skip_unbound(comment or function, sources)
            return None
        output_var = self._new_dataset_var(output.dataset_id)
        if comment:
            self._lines.append(f"# {_safe_comment(comment)}")
        call_args = tuple(_Code(str(variable)) for variable in variables) + tuple(args)
        call = _call_text(function, call_args, kwargs)
        self._append_result_statement([output], f"{output_var} = {call}")
        self._notify()
        return output_var

    def record_multi_function(
        self,
        outputs: Sequence[DatasetReference],
        function: str,
        sources: Sequence[DatasetReference],
        *,
        args=(),
        kwargs=None,
        comment: str = "",
    ) -> tuple[str, ...]:
        self._record_action(outputs, sources, None, comment or function)
        variables = [self._ensure(reference) for reference in sources]
        if any(variable is None for variable in variables):
            self._skip_unbound(comment or function, sources)
            return ()
        output_vars = tuple(self._new_dataset_var(output.dataset_id) for output in outputs)
        if comment:
            self._lines.append(f"# {_safe_comment(comment)}")
        call_args = tuple(_Code(str(variable)) for variable in variables) + tuple(args)
        call = _call_text(function, call_args, kwargs)
        self._append_result_statement(outputs, f"{', '.join(output_vars)} = {call}")
        self._notify()
        return output_vars

    def record_expression(
        self,
        output: DatasetReference,
        sources: Sequence[DatasetReference],
        expression: Callable[[tuple[str, ...]], str],
        *,
        comment: str,
        action_steps=None,
    ) -> str | None:
        self._record_action([output], sources, action_steps, comment)
        variables = tuple(self._ensure(reference) for reference in sources)
        if any(variable is None for variable in variables):
            self._skip_unbound(comment, sources)
            return None
        output_var = self._new_dataset_var(output.dataset_id)
        self._lines.append(f"# {_safe_comment(comment)}")
        self._append_result_statement([output], f"{output_var} = {expression(tuple(str(v) for v in variables))}")
        self._notify()
        return output_var

    def _record_action(self, outputs, sources, steps, label):
        # Notify even for unsaved inputs: action recipes substitute a selected
        # dataset and have no dependency on the Python recorder's file bindings.
        if self.action_listener is not None:
            self.action_listener(outputs, sources, steps, label)

    def record_save(self, reference: DatasetReference, path: str) -> bool:
        variable = self._ensure(reference)
        if variable is None:
            self._skip_unbound("Save dataset", [reference])
            return False
        resolved = str(Path(path).expanduser().resolve())
        self._lines.extend(
            [
                f"# Save {_safe_comment(reference.name)}",
                f"{variable}.save(Path({_literal(resolved)}), compressed=\"auto\")",
                "",
            ]
        )
        self._notify()
        return True

    def record_save_batch(
        self,
        entries: Sequence[tuple[DatasetReference, str]],
    ) -> bool:
        """Emit one transactional save call while preserving GUI plan order."""

        if not entries:
            return False
        variables = [self._ensure(reference) for reference, _path in entries]
        if any(variable is None for variable in variables):
            self._skip_unbound(
                "Save dataset batch", [reference for reference, _path in entries]
            )
            return False
        self._lines.extend(
            ["# Transactionally save the dataset batch", "save_dataset_batch(["]
        )
        for variable, (_reference, path) in zip(variables, entries):
            resolved = str(Path(path).expanduser().resolve())
            self._lines.append(
                f"    ({variable}, Path({_literal(resolved)})),"
            )
        self._lines.extend(["])", ""])
        self._notify()
        return True

    def record_plot(
        self,
        datasets: Sequence[DatasetReference],
        *,
        names: Sequence[str],
        mode: str,
        parameters: dict[str, object],
    ) -> str | None:
        # One source of truth: every mode the headless plotter can draw is
        # recordable (a private list here once silently dropped CDF and waterfall).
        from GRIM_Backend.plotting.registry import HEADLESS_PLOT_MODES, PLOT_MODES
        mode_key = str(mode).strip().lower()
        if mode_key not in HEADLESS_PLOT_MODES:
            labels = ", ".join(PLOT_MODES[key].label for key in sorted(HEADLESS_PLOT_MODES))
            self.record_unsupported_plot(
                mode_key,
                f"the headless plotter supports these plots only: {labels}",
            )
            return None
        variables = [self._ensure(reference) for reference in datasets]
        if any(variable is None for variable in variables):
            self._skip_unbound(f"Create {mode} plot", datasets)
            self._current_plot_var = None
            self._current_plot_signature = None
            return None
        signature = _literal(
            (
                tuple(
                    (reference.dataset_id, str(name))
                    for reference, name in zip(datasets, names)
                ),
                mode_key,
                parameters,
            )
        )
        if (
            self._current_plot_var is not None
            and signature == self._current_plot_signature
        ):
            return self._current_plot_var
        variable = f"plot_{self._next_plot}"
        self._next_plot += 1
        named = [
            (str(name), _Code(str(dataset_var)))
            for name, dataset_var in zip(names, variables)
        ]
        kwargs = {"mode": mode_key, **parameters}
        self._lines.append(f"# Create {mode_key.replace('_', ' ')} plot")
        call = _call_text("plot_datasets", (named,), kwargs)
        self._lines.extend([f"{variable} = {call}", ""])
        self._current_plot_var = variable
        self._current_plot_signature = signature
        self._notify()
        return variable

    def record_unsupported_plot(self, mode: str, reason: str) -> None:
        """Explain an omitted plot and prevent export from targeting stale code."""

        signature = _literal(("unsupported", str(mode), str(reason)))
        if self._current_plot_var is None and self._current_plot_signature == signature:
            return
        self._current_plot_var = None
        self._current_plot_signature = signature
        self._lines.extend(
            [
                f"# {_safe_comment(str(mode).replace('_', ' '))} plot was not "
                f"recorded: {_safe_comment(reason)}.",
                "# The plot shown in GRIM can still be exported manually.",
                "",
            ]
        )
        self._notify()

    def invalidate_current_plot(self) -> None:
        """Forget the export target without recording a presentation-only action."""

        self._current_plot_var = None
        self._current_plot_signature = None

    def record_plot_save(self, path: str, *, dpi: int = 200) -> bool:
        if self._current_plot_var is None:
            self._lines.extend(
                [
                    "# Plot export was not recorded: create a recorded plot first.",
                    "",
                ]
            )
            self._notify()
            return False
        resolved = str(Path(path).expanduser().resolve())
        self._lines.extend(
            [
                f"{self._current_plot_var}.savefig(",
                f"    Path({_literal(resolved)}),",
                f"    dpi={int(dpi)},",
                "    bbox_inches='tight',",
                ")",
                "",
            ]
        )
        self._notify()
        return True
