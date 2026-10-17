#!/usr/bin/env python3
"""HPC monostatic 2-D RCS sweep (SLURM).

Edit the parameters below and run:

    python run_hpc_monostatic.py

The workflow, scheduling and worker machinery live in
ghost_backend/hpc/twod_driver.py.
"""
import sys

# ===============================================================================
# CONFIG -- the only section most users need to edit
# ===============================================================================

# Input geometry folders. Every *.geo file found under these paths
# (recursively) is added to the sweep. Source folder is NOT injected into
# output filenames -- the geometry filename is preserved verbatim.
FRD_DIR = "ghost_backend/geometry/geometries/FRD"
OPN_DIR = "ghost_backend/geometry/geometries/OPN"

# Requested sweep.
FREQUENCIES_GHZ = [2.0, 4.0, 6.0, 8.0, 10.0]
AZIMUTHS_DEG    = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

GEOMETRY_UNITS = "inches"         # "inches" | "meters"
MESH_CERTIFICATION = True        # True: compare base/fine meshes. False: one uncertified mesh.
ACCURACY_TARGET = "standard"     # "standard" | "tight"

# Output root. A new run_YYYYMMDD_HHMMSS/ subfolder is created inside.
OUTPUT_DIR = "ghost_backend/results/rcs_runs"

# --- How much of the cluster to use ----------------------------------------
# N_NODES is the size of each sbatch job array, N_JOBS the number of separate
# submissions (use more than one to spread across partitions or accounts).
# Total parallelism is N_NODES x N_JOBS nodes.
#
# Unlike the previous round-robin scheme these are pure throughput knobs. Array
# tasks are interchangeable and pull work from a shared claim directory, so
# nothing is stranded if a task never starts, is canceled, or is preempted,
# and raising N_NODES on a later submission simply adds capacity to a run that
# is already going.
N_NODES = 1
N_JOBS  = 1

# Cap on array tasks running at once (SLURM's `--array=...%N`). None = no cap.
ARRAY_THROTTLE = None

# Worker processes for the submit-time mesh/storage planning (one geometry per
# task; the plan is identical whatever the count). None = GHOST_PLANNING_WORKERS
# or min(8, usable CPUs); 1 = serial. Sweeps below four geometries plan serially.
PLANNING_WORKERS = None

# ===============================================================================
# SLURM allocation and job environment. The backend, mesh, threads and memory
# admission are chosen automatically for each solve.
# ===============================================================================

# --- SLURM resources (per array task = one node) ---------------------------
SLURM_PARTITION = "compute"
SLURM_ACCOUNT   = None            # e.g. "my_project"; None to omit
SLURM_QOS       = None
SLURM_TIME      = None            # None = no walltime limit; or "HH:MM:SS"
CORES_PER_NODE  = None            # None = request whole node via --exclusive
                                  # (pool size auto-detected from SLURM env).
                                  # Or set an integer, e.g. 32.
MEM_PER_NODE    = "0"             # "0" = ALL memory of the node (SLURM idiom;
                                  # recommended with --exclusive). None = omit
                                  # the directive -> cluster default applies
                                  # (often DefMemPerCPU ~3.5G x CPUs, which can
                                  # be far less than node RAM). Or e.g. "64G".
MAX_WORKERS_PER_NODE = None       # Hard ceiling on concurrent solves per node.
                                  # None = one per allocated core. The memory
                                  # budget below is usually the binding
                                  # constraint, so this rarely needs setting.
MAX_SOLVE_GB    = None            # Hard ceiling on ONE solve's estimated
                                  # footprint, exported to the job as
                                  # GHOST_MAX_SOLVE_GB. None = derive it from
                                  # current available RAM (0.9 x available,
                                  # with no minimum floor).
                                  #
                                  # Set it when you intend to run something
                                  # very large. Detection needs a number it
                                  # can trust: with MEM_PER_NODE = "0" SLURM
                                  # reports 0, so the limit falls back to the
                                  # cgroup or /proc/meminfo, and if neither is
                                  # meaningful the ceiling collapses to 32 GB
                                  # and a big solve is refused on a big node.
                                  # Either set this, or give MEM_PER_NODE an
                                  # explicit size like "750G".
SLURM_MAIL_TYPE = None            # e.g. "END,FAIL"
SLURM_MAIL_USER = None
SLURM_EXTRA_SBATCH = []  # type: List[str]  # raw extra lines, e.g. "--constraint=intel"

JOB_PROLOGUE = []  # type: List[str]

PYTHON_EXE    = sys.executable           # interpreter used inside the job
SUBMIT        = True                     # False -> write .slurm files but don't sbatch

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
    from ghost_backend.hpc.twod_driver import run
    run(globals(), __file__)
