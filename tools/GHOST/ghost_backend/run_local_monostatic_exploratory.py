#!/usr/bin/env python3
"""Simple, uncertified 2D monostatic runner; no backend edits required.

Edit CONFIG and run this file; no command-line arguments are needed.
Relative paths are relative to your current working directory.

Skips condition estimation, strict quality-gate rejection, mesh certification,
and adaptive mesh comparisons. Input validation, residual calculations, and
numerical solver errors remain. This runner does not support BoR.
"""
from datetime import datetime
import math
from pathlib import Path
import sys
from uuid import uuid4

# CONFIG -- use an absolute path if running from different working directories.
GEOMETRY_FILE = "your_geometry.geo"
FREQUENCIES_GHZ = [1.0]
AZIMUTHS_DEG = [float(a) for a in range(0, 360, 5)]
GEOMETRY_UNITS = "inches"  # "inches" or "meters"
OUTPUT_DIR = "ghost_exploratory_results"
# Run only the initial base (P2 when eligible) or fine/check (P3 when eligible)
# candidate used by certification. No comparison or adaptive refinement runs.
# "geometry" retains the original global linear mesh from the geometry's N.
MESH_MODE = "certification_base"  # "certification_base", "certification_fine", "geometry"
FACTORIZATION = "adaptive"  # automatic RAM-aware solver choice; "dense" forces dense
RAM_BUDGET_GIB = None  # None uses available memory; optionally supply a positive GiB limit


def main():
    geometry = Path(GEOMETRY_FILE).expanduser().resolve()
    if not geometry.is_file():
        raise ValueError(f"Geometry does not exist: {geometry}. Edit GEOMETRY_FILE in CONFIG.")
    if not FREQUENCIES_GHZ or any(not math.isfinite(f) or f <= 0 for f in FREQUENCIES_GHZ):
        raise ValueError("FREQUENCIES_GHZ must contain finite values greater than zero.")
    if not AZIMUTHS_DEG or any(not math.isfinite(a) for a in AZIMUTHS_DEG):
        raise ValueError("AZIMUTHS_DEG must contain finite values.")
    if GEOMETRY_UNITS not in ("inches", "meters"):
        raise ValueError('GEOMETRY_UNITS must be "inches" or "meters".')

    # Supports running this script directly from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ghost_backend.runs.inputs import load_geometry_snapshot
    from ghost_backend.runs.exploratory import MESH_MODES, select_exploratory_mesh
    from ghost_backend.linalg.refined_lu import linear_precision
    from ghost_backend.twod.solver import solve_monostatic_rcs_2d
    from ghost_backend.io.grim import export_result_to_grim

    if MESH_MODE not in MESH_MODES:
        raise ValueError('MESH_MODE must be one of: ' + ', '.join(MESH_MODES))
    snapshot, material_base = load_geometry_snapshot(str(geometry), {})
    run_dir = Path(OUTPUT_DIR).expanduser().resolve() / (
        f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Exploratory 2D solve (VV and HH). Output: {run_dir}", flush=True)

    def progress(done, total, message):
        print(f"[{done}/{total}] {message}", flush=True)

    # Export immediately so an error at a later frequency keeps earlier results.
    for index, frequency in enumerate(FREQUENCIES_GHZ, start=1):
        print(f"Frequency {index}/{len(FREQUENCIES_GHZ)}: {frequency:g} GHz", flush=True)
        mesh, options, mesh_record = select_exploratory_mesh(
            snapshot, frequency, mesh_mode=MESH_MODE, geometry_units=GEOMETRY_UNITS,
            material_base_dir=material_base, factorization=FACTORIZATION,
            ram_budget_gib=RAM_BUDGET_GIB,
        )
        print(f"Single mesh: {mesh_record['candidate']}, P{mesh_record['basis_order']}, "
              f"{mesh_record['mesh_strategy']} sizing, "
              f"coarsening factor {mesh_record['coarsening_factor']:g}. "
              "No convergence comparison or certification.", flush=True)
        if mesh_record.get('selection_reason'):
            print(mesh_record['selection_reason'], flush=True)
        with linear_precision('double'):
            result = solve_monostatic_rcs_2d(
                geometry_snapshot=mesh,
                frequencies_ghz=[frequency],
                # The 2D backend calls its in-plane angle argument elevations_deg.
                elevations_deg=AZIMUTHS_DEG,
                geometry_units=GEOMETRY_UNITS,
                material_base_dir=material_base,
                progress_callback=progress,
                strict_quality_gate=False,
                compute_condition_number=False,
                solver_method="auto",
                execution_options=options,
            )
        metadata = result.setdefault("metadata", {})
        metadata.update(
            run_profile="exploratory",
            mesh_convergence_certified=False,
            strict_quality_gate=False,
            compute_condition_number=False,
            exploratory_mesh=mesh_record,
        )
        # Keep the solver's actual diagnostics, including failed quality gates.
        gate = metadata.get("quality_gate", {})
        if gate and not gate.get("passed", False):
            print(f"Quality warning (export continues): {gate.get('reason', gate)}", flush=True)
        destination = run_dir / f"{geometry.stem}_{index:04d}_{frequency:g}GHz.grim"
        written = export_result_to_grim(
            result, str(destination), source_path=str(geometry),
            history=(f"Exploratory 2D monostatic: {MESH_MODE}, "
                     f"P{mesh_record['basis_order']}, {mesh_record['mesh_strategy']} mesh; "
                     "no mesh certification, convergence comparison, or condition "
                     "estimation; strict quality gate disabled."),
        )
        for path in written:
            print(f"Saved: {path}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
