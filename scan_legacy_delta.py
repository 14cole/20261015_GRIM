#!/usr/bin/env python3
"""Find (and optionally convert) .grim files written with the retired GHOST
"delta" tags.

Scanning is read-only: it opens each .grim as a NumPy zip and writes nothing.

    python scan_legacy_delta.py <folder> [<folder> ...]
    python scan_legacy_delta.py <folder> [<folder> ...] --convert OUT_DIR

Categories
----------
AT-RISK SOURCE  storage_domain=raw_field with power_domain=delta_amp_sq and
                rcs_domain=delta. The samples on disk are correct, but GRIM
                builds released before commit 3231933 (complex I/Q storage,
                before the delta fixes) displayed, exported and computed from
                them with the wrong scale: sigma_3d about +10.99 dB too high,
                sigma_2d low by 10*log10(4k) (about -19 dB at 1 GHz, -29 dB at
                10 GHz).
DERIVED         A file whose recorded provenance names an AT-RISK SOURCE as an
                input. If it was produced by the older GRIM, its samples were
                written with the wrong scale and stay wrong in every version.
SAFE LEGACY     A legacy delta stored as power/phase (no raw solver field):
                read correctly by every GRIM version.
REFUSED         A legacy delta stored as power/phase plus a raw amplitude
                pair: the older GRIM refused to load it, so it produced no
                bad data in GRIM.
DELTA TAG ONLY  rcs_domain=delta with the standard power_domain: ordinary
                samples carrying the retired semantic tag.

Current GRIM and GHOST convert every one of these files when they load it, so
conversion is optional. ``--convert OUT_DIR`` writes standard v2 copies of every
non-OTHER file (loaded and saved through GRIM's RcsGrid) under OUT_DIR, keeping
each file's path relative to the scanned folder, plus
``legacy_delta_conversion_report.json``. Originals are never modified, and an
existing output file is never overwritten. DERIVED files are listed in the
report but not converted: their samples cannot be repaired automatically.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

CONVERTIBLE = ("AT-RISK SOURCE", "SAFE LEGACY", "REFUSED", "DELTA TAG ONLY")
REPORT_NAME = "legacy_delta_conversion_report.json"


def _scalar(z, name):
    if name not in z.files:
        return ""
    value = np.asarray(z[name]).reshape(-1)
    if value.size != 1:
        return ""
    value = value[0]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def classify(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            power_domain, rcs_domain = _scalar(z, "power_domain"), _scalar(z, "rcs_domain")
            domain = _scalar(z, "storage_domain") or "physical"
            legacy = power_domain == "delta_amp_sq" and rcs_domain == "delta"
            inputs = []
            text = _scalar(z, "operation_provenance_json")
            if text:
                try:
                    for event in json.loads(text).get("events", []):
                        for item in event.get("inputs", []):
                            if item.get("source_path"):
                                inputs.append(str(item["source_path"]))
                except (ValueError, AttributeError):
                    pass
            history = _scalar(z, "history")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return "UNREADABLE", str(exc), []
    if legacy and domain == "raw_field":
        kind = "AT-RISK SOURCE"
    elif legacy and {"rcs_amp_real", "rcs_amp_imag"} <= files:
        kind = "REFUSED"
    elif legacy:
        kind = "SAFE LEGACY"
    elif rcs_domain == "delta":
        kind = "DELTA TAG ONLY"
    else:
        kind = "OTHER"
    return kind, history, inputs


def scan(folders):
    """Return {resolved path: (kind, history, inputs, scanned root)}."""
    records = {}
    for folder in folders:
        root = Path(folder).resolve()
        for path in root.rglob("*.grim"):
            records[path.resolve()] = classify(path) + (root,)
    return records


def _derived(records):
    at_risk = {p for p, record in records.items() if record[0] == "AT-RISK SOURCE"}
    at_risk_names = {p.name for p in at_risk}
    derived = {}
    for path, (kind, history, inputs, _root) in records.items():
        if kind == "OTHER":
            hit = [i for i in inputs if Path(i).resolve() in at_risk or Path(i).name in at_risk_names]
            if hit:
                derived[path] = hit
    return at_risk, derived


def report(records):
    at_risk, derived = _derived(records)
    for path, (kind, history, inputs, _root) in sorted(records.items()):
        if kind == "OTHER":
            if path in derived:
                print(f"DERIVED         {path}\n                inputs: {', '.join(derived[path])}\n                history: {history[:160]}")
            continue
        print(f"{kind:15s} {path}")
    print(f"\n{len(records)} .grim files scanned; {len(at_risk)} at-risk source(s).")
    print("Files derived from an at-risk source with no recorded source path cannot be found\n"
          "automatically; check results made from those sources by hand.")


def _source_physical_power(path):
    """Physical power implied by a legacy file's own samples (|A|**2)."""
    with np.load(path, allow_pickle=False) as z:
        if "rcs_complex" in z.files:
            return np.abs(np.asarray(z["rcs_complex"])) ** 2
        if {"rcs_amp_real", "rcs_amp_imag"} <= set(z.files):
            real = np.asarray(z["rcs_amp_real"], dtype=float)
            imag = np.asarray(z["rcs_amp_imag"], dtype=float)
            return real * real + imag * imag
        return np.asarray(z["rcs_power"], dtype=float)


def convert(records, out_dir):
    """Write converted copies of every convertible file; return the report."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from GRIM_Backend.datasets.grid import RcsGrid

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _at_risk, derived = _derived(records)
    entries = []
    for path, (kind, _history, _inputs, root) in sorted(records.items()):
        if path in derived:
            entries.append({"source": str(path), "category": "DERIVED", "status": "not converted",
                            "inputs": derived[path]})
            continue
        if kind not in CONVERTIBLE:
            continue
        target = (out_dir / root.name / path.relative_to(root)).resolve()
        entry = {"source": str(path), "category": kind, "output": str(target)}
        if target == path:
            entry["status"] = "skipped: output would overwrite the original"
        elif target.exists():
            entry["status"] = "skipped: output already exists"
        else:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                RcsGrid.load(path).save(target)
                converted = RcsGrid.load(target)
                expected = _source_physical_power(path)
                power = np.asarray(converted.power, dtype=float)
                finite = np.isfinite(expected) & (expected > 0)
                relative = (float(np.max(np.abs(power[finite] - expected[finite]) / expected[finite]))
                            if np.any(finite) else 0.0)
                with np.load(target, allow_pickle=False) as z:
                    tags = {key: _scalar(z, key) for key in
                            ("grim_format_version", "storage_kind", "storage_domain",
                             "power_domain", "rcs_domain")}
                entry.update(status="converted", output_tags=tags,
                             max_relative_power_error=relative)
            except Exception as exc:  # report and continue with the next file
                entry["status"] = f"failed: {exc}"
        entries.append(entry)
    result = {"schema": "grim.legacy-delta-conversion.v1", "out_dir": str(out_dir), "files": entries}
    report_path = out_dir / REPORT_NAME
    if report_path.exists():
        stem = 1
        while (out_dir / f"legacy_delta_conversion_report_{stem}.json").exists():
            stem += 1
        report_path = out_dir / f"legacy_delta_conversion_report_{stem}.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report_path"] = str(report_path)
    for entry in entries:
        print(f"{entry['status']:40.40s} {entry['source']}")
    print(f"\nConversion report: {report_path}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="folders to scan recursively for .grim files")
    parser.add_argument("--convert", metavar="OUT_DIR",
                        help="write converted standard v2 copies and a report under OUT_DIR")
    args = parser.parse_args(argv)
    records = scan(args.folders)
    report(records)
    if args.convert:
        return convert(records, args.convert)
    return records


if __name__ == "__main__":
    main()
