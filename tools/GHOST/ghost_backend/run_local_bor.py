#!/usr/bin/env python3
"""Local BoR monostatic RCS sweep (no SLURM).

Edit the parameters below and run:

    python run_local_bor.py

The workflow, scheduling and worker machinery live in
ghost_backend/runs/local_bor_driver.py.
"""
import sys

# ===============================================================================
# CONFIG
# ===============================================================================

GEOMETRY_DIRS = ["ghost_backend/geometry/geometries/BOR"]      # every *.geo under these, recursively

FREQUENCIES_GHZ = [1.0, 2.0, 4.0]
AZIMUTHS_DEG    = [float(a) for a in range(0, 360, 5)]
ELEVATIONS_DEG  = [float(e) for e in range(-60, 61, 5)]

# Body-axis attitude in the radar/earth frame.  The default is horizontal,
# nose toward azimuth 0.  Roll changes feature orientation about the BoR axis.
BODY_AXIS_AZ_DEG = 0.0
BODY_AXIS_EL_DEG = 0.0
BODY_ROLL_DEG    = 0.0

OUTPUT_DIR = "ghost_backend/results/rcs_runs_bor"

# Hard ceiling on units in flight. None -> cores // WORKERS_PER_UNIT, since
# each unit already runs WORKERS_PER_UNIT threads for its mode sweep and
# streaming tiles. The memory budget below is usually the binding constraint.
WORKERS = None

# --- Solver knobs (mirror bor_dispatch.solve_monostatic_rcs_bor) -----------
GEOMETRY_UNITS   = "inches"       # "inches" or "meters"
CFIE_ALPHA       = 0.5            # closed PEC bodies -> CFIE
N_MODES          = None           # None = auto (adaptive truncation)
MODE_TOL         = 1e-6
MAX_ELEMENTS     = 50_000
ASSEMBLY         = "auto"         # "auto" | "tables" | "streaming"
TABLE_PRECISION  = "auto"         # "auto" | "single" | "double"
ACCURACY_TARGET  = "standard"     # "standard" | "tight"; mesh comparison limits
STREAM_BUDGET_GB = 8.0            # maximum held streaming-block budget; the
                                  # scheduler predicts total peak separately
MESH_CERTIFICATION = True         # recommended: compare base/fine meshes;
                                  # False solves the base mesh only
WORKERS_PER_UNIT = 4              # threads inside one BoR solve
BLAS_THREADS_PER_WORKER = 1
BOR_EXECUTION_OPTIONS = {}        # BoR execution profile (factorization, angle
                                  # batching, ...); {} = defaults

# --- Memory admission ------------------------------------------------------
# A local machine has far less RAM than a compute node and is usually running
# other things, so the same guard the cluster path uses matters more here, not
# less. One unit is always admitted, so a solve larger than the whole budget
# still runs (and fails loudly) instead of hanging.
MEMORY_HEADROOM = 0.75            # fraction of detected RAM the scheduler may
                                  # reserve for concurrent solves. Lower than
                                  # the cluster default of 0.85: a workstation
                                  # has a desktop and a page cache to leave
                                  # room for.

# Pool worker lifetime, in units. Lower than the 2-D default because a BoR
# unit's streaming blocks are large and worth returning to the OS promptly
# (a respawn costs about half a second: 130 ms per pair at this interval).
TASKS_PER_CHILD = 4

# ===============================================================================


if __name__ == "__main__":
    from pathlib import Path
    if sys.version_info < (3, 10):
        raise SystemExit("GHOST requires Python 3.10 or newer; see ghost_backend/hpc/check_environment.py.")
    _here = Path(__file__).resolve().parent
    if _here.name == "ghost_backend":
        sys.path.insert(0, str(_here.parent))
    from ghost_backend.runs.local_bor_driver import run
    run(globals(), __file__)
