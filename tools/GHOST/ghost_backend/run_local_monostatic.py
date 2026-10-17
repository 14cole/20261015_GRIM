#!/usr/bin/env python3
"""Local monostatic 2-D RCS sweep (no SLURM).

Edit the parameters below and run:

    python run_local_monostatic.py

The workflow, scheduling and worker machinery live in
ghost_backend/runs/local_twod_driver.py.
"""
import sys

# ===============================================================================
# CONFIG
# ===============================================================================

# Geometry folders. Every *.geo file under these (recursively) is included.
FRD_DIR = "ghost_backend/geometry/geometries/FRD"
OPN_DIR = "ghost_backend/geometry/geometries/OPN"

# Sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "ghost_backend/results/rcs_runs"

GEOMETRY_UNITS = "inches"         # "inches" | "meters"
MESH_CERTIFICATION = True        # True: compare base/fine meshes. False: one uncertified mesh.
ACCURACY_TARGET = "standard"     # "standard" | "tight"

# Optional resource caps. The backend, mesh, threads and memory admission are
# chosen automatically for each solve.
WORKERS = None                    # max concurrent solves; None = cpu_count() - 1
PLANNING_WORKERS = None           # processes for the mesh/storage planning; None = min(8, CPUs), 1 = serial
MAX_SOLVE_GB = None               # per-solve RAM ceiling in GiB; None = from available RAM

# ===============================================================================


if __name__ == "__main__":
    from pathlib import Path
    if sys.version_info < (3, 10):
        raise SystemExit("GHOST requires Python 3.10 or newer; see ghost_backend/hpc/check_environment.py.")
    _here = Path(__file__).resolve().parent
    if _here.name == "ghost_backend":
        sys.path.insert(0, str(_here.parent))
    from ghost_backend.runs.local_twod_driver import run
    run(globals(), __file__)
