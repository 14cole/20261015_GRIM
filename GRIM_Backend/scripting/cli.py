"""Command-line dataset loading, operations, audit, and export."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from GRIM_Backend.datasets.combine import combine_datasets
from GRIM_Backend.datasets.grid import RcsGrid
from GRIM_Backend.io.loaders import _matching_dataset_paths, load_dataset, load_folder


def audit_dataset(dataset: RcsGrid, **kwargs):
    """Return a non-mutating, JSON-serializable dataset health report.

    The report contains status, errors, warnings, info, and metrics. Samples are
    checked in bounded blocks; malformed data is reported without repair.
    """

    return dataset.audit(**kwargs)


def _audit_json_default(value):
    """Serialize the numeric/container values used by audit reports."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=str)
    for method_name in ("to_dict", "as_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return method()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


STITCH_POLICIES = ("priority-first", "priority-last", "power-mean", "coherent-mean")


def _stitch_policy(value: str) -> str:
    # Accept the spellings RcsGrid.stitch() normalizes (case, underscores).
    return str(value).strip().lower().replace("_", "-")


def _parser():
    parser = argparse.ArgumentParser(description="Headless GRIM dataset operations")
    parser.add_argument("inputs", nargs="*", help="input files")
    parser.add_argument(
        "-o",
        "--output",
        help="output .grim path, or optional JSON report path with --audit",
    )
    parser.add_argument("--folder", help="load a folder instead of explicit inputs")
    parser.add_argument("--pattern", default="*", help="folder glob pattern")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--operation",
        choices=("join", "stitch", "coherent-add", "incoherent-add"),
        default="join",
        help=(
            "combination operation; join rejects conflicting finite overlaps, "
            "while stitch resolves them with --stitch-policy"
        ),
    )
    parser.add_argument("--overlap", choices=("error", "first", "last"), default="error")
    parser.add_argument(
        "--max-gib",
        type=float,
        default=None,
        help="maximum estimated dense output and working allocation",
    )
    parser.add_argument(
        "--stitch-policy",
        type=_stitch_policy,
        choices=STITCH_POLICIES,
        default="priority-first",
        help=(
            "stitch overlap policy: priority-first, priority-last, power-mean, "
            "or coherent-mean"
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-6,
        help="numeric coordinate matching tolerance for stitch",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "audit each input independently and emit JSON instead of creating "
            "a derived dataset; --output is optional"
        ),
    )
    parser.add_argument(
        "--attest-coherent-metadata",
        action="store_true",
        help=(
            "legacy compatibility option: record an explicit user attestation "
            "for missing coherent metadata; operations no longer require it"
        ),
    )
    return parser


def _resolve_input_paths(parser, args) -> list[str]:
    """Validate the input selection before any dataset is loaded."""

    if args.folder and args.inputs:
        parser.error("give either input files or --folder, not both")
    if args.folder:
        if not os.path.isdir(args.folder):
            parser.error(f"--folder does not exist or is not a folder: {args.folder}")
        try:
            return _matching_dataset_paths(
                args.folder, pattern=args.pattern, recursive=args.recursive
            )
        except ValueError:
            parser.error(
                f"no supported datasets in {args.folder} matched --pattern "
                f"{args.pattern!r}" + (" (recursive)" if args.recursive else "")
            )
    if not args.inputs:
        parser.error("provide input files or --folder")
    return list(args.inputs)


def _load_input(parser, path: str) -> RcsGrid:
    try:
        return load_dataset(path)
    except FileNotFoundError:
        parser.error(f"input file not found: {path}")
    except IsADirectoryError:
        parser.error(f"input is a folder, not a dataset file (use --folder): {path}")


def _validate_dataset_output(parser, output: str | None, inputs: list[str]) -> None:
    """Reject an unusable --output before an expensive combination runs."""

    if not output:
        parser.error("--output is required")
    resolved = os.path.abspath(output)
    if os.path.isdir(resolved):
        parser.error(f"--output is a folder, not a file path: {output}")
    parent = os.path.dirname(resolved)
    if parent and not os.path.isdir(parent):
        parser.error(f"--output folder does not exist: {parent}")
    identity = os.path.normcase(os.path.realpath(resolved))
    for path in inputs:
        same = identity == os.path.normcase(os.path.realpath(os.path.abspath(path)))
        if not same and os.path.exists(resolved):
            try:
                same = os.path.samefile(resolved, path)
            except OSError:
                same = False
        if same:
            parser.error("--output must not overwrite an input dataset")


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    limit = None if args.max_gib is None else int(args.max_gib * 1024**3)
    paths = _resolve_input_paths(parser, args)
    if args.audit:
        reports = [
            {
                "input": os.path.abspath(path),
                "report": audit_dataset(_load_input(parser, path)),
            }
            for path in paths
        ]
        payload = {"operation": "audit", "datasets": reports}
        rendered = json.dumps(
            payload,
            default=_audit_json_default,
            indent=2,
            sort_keys=True,
        )
        if args.output:
            output = os.path.abspath(args.output)
            output_identity = os.path.normcase(os.path.realpath(output))
            input_paths = {
                os.path.normcase(os.path.realpath(os.path.abspath(path)))
                for path in paths
            }
            same_existing_file = False
            if os.path.exists(output):
                for path in paths:
                    try:
                        if os.path.samefile(output, path):
                            same_existing_file = True
                            break
                    except OSError:
                        continue
            if output_identity in input_paths or same_existing_file:
                raise SystemExit(
                    "audit --output must not overwrite an audited input dataset"
                )
            with open(output, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
                stream.write("\n")
            print(output)
        else:
            print(rendered)
        return 0
    _validate_dataset_output(parser, args.output, paths)
    if args.folder:
        result = load_folder(
            args.folder, pattern=args.pattern, recursive=args.recursive,
            operation=args.operation, workers=args.workers, overlap=args.overlap,
            max_output_bytes=limit,
            coherent_metadata_attested=args.attest_coherent_metadata,
            stitch_policy=args.stitch_policy,
            tol=args.tol,
        )
    else:
        result = combine_datasets(
            [_load_input(parser, path) for path in paths], args.operation,
            overlap=args.overlap,
            max_output_bytes=limit,
            coherent_metadata_attested=args.attest_coherent_metadata,
            stitch_policy=args.stitch_policy,
            tol=args.tol,
        )
    output = result.save(args.output, compressed="auto")
    print(output)
    return 0
