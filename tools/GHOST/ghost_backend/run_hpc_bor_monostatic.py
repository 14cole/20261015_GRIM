#!/usr/bin/env python3
"""HPC monostatic BoR RCS sweep (SLURM).

Edit the parameters below and run:

    python run_hpc_bor_monostatic.py

The workflow, scheduling and worker machinery live in
ghost_backend/hpc/bor_driver.py.
"""
import sys

# ===============================================================================
# CONFIG -- the only section most users need to edit
# ===============================================================================

GEOMETRY_DIRS = ["ghost_backend/geometry/geometries/BOR"]

# Requested monostatic grid.
FREQUENCIES_GHZ = [1.0, 2.0, 3.0]
AZIMUTHS_DEG    = [float(a) for a in range(0, 360, 5)]
ELEVATIONS_DEG  = [float(e) for e in range(-60, 61, 5)]

# Body-axis attitude in the radar/earth frame. Default: horizontal with the
# nose at azimuth 0. Roll affects later feature orientation, not the bare BoR.
BODY_AXIS_AZ_DEG = 0.0
BODY_AXIS_EL_DEG = 0.0
BODY_ROLL_DEG    = 0.0

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "ghost_backend/results/rcs_runs_bor"

# --- Multi-node / multi-submission parallelism -----------------------------
N_NODES = 1
N_JOBS  = 1

# ===============================================================================
# ADVANCED -- fine tuning (SLURM resources, solver knobs, monostatic grid)
# ===============================================================================

SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None
SLURM_QOS       = None
SLURM_TIME      = None
CORES_PER_NODE  = None            # None = whole node via --exclusive
MEM_PER_NODE    = "0"             # "0" = all node memory (see 2D driver notes)
MAX_WORKERS_PER_NODE = None       # cap concurrent UNITS per node (memory-heavy
                                  # units: peak ~ streaming blocks + mode LU)
SLURM_MAIL_TYPE = None
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]
JOB_PROLOGUE = []  # type: List[str]

# --- Solver knobs (mirror bor_dispatch.solve_monostatic_rcs_bor) -----------
GEOMETRY_UNITS          = "inches"       # "inches" or "meters"
CFIE_ALPHA              = 0.5            # closed PEC bodies -> CFIE
N_MODES                 = None           # None = auto (adaptive truncation)
MODE_TOL                = 1e-6
MAX_ELEMENTS            = 50_000
ASSEMBLY                = "auto"         # "auto" | "tables" | "streaming"
TABLE_PRECISION         = "auto"         # "auto" | "single" | "double"
ACCURACY_TARGET         = "standard"     # "standard" | "tight"; mesh comparison limits
STREAM_BUDGET_GB        = 8.0            # held streaming-block budget per unit
MESH_CERTIFICATION      = True           # recommended base/fine comparison;
                                         # False = base mesh only
WORKERS_PER_UNIT        = 4              # threads inside one BoR solve (modes
                                         # + streaming tiles); pool size =
                                         # cores // WORKERS_PER_UNIT
BLAS_THREADS_PER_WORKER = 1
BOR_EXECUTION_OPTIONS   = {}           # BoR execution profile (factorization,
                                         # angle batching, ...); {} = defaults

# --- Scheduling ------------------------------------------------------------
# Units are costed at submit time and assigned longest-first, then claimed
# at run time through atomic files in <run_dir>/claims/.
MEMORY_HEADROOM     = 0.85   # fraction of node memory the scheduler may reserve
CLAIM_STALE_SECONDS = 7200   # a quiet claim older than this is stealable
TASKS_PER_CHILD     = 2      # pool worker lifetime, in units

GEOMETRY_EXTS = (".geo",)
PYTHON_EXE    = sys.executable
SUBMIT        = True

# ===============================================================================


if __name__ == "__main__":
    from pathlib import Path
    if sys.version_info < (3, 10):
        raise SystemExit("GHOST requires Python 3.10 or newer; see ghost_backend/hpc/check_environment.py.")
    _here = Path(__file__).resolve().parent
    _runtime = _here / "runtime"
    if "--worker" in sys.argv[:-1]:
        _runtime = Path(sys.argv[sys.argv.index("--worker") + 1]).resolve() / "runtime"
    if (_runtime / "ghost_backend" / "execution" / "paths.py").is_file():
        sys.path.insert(0, str(_runtime))
    elif _here.name == "ghost_backend":
        sys.path.insert(0, str(_here.parent))
    from ghost_backend.hpc.bor_driver import run
    run(globals(), __file__)
