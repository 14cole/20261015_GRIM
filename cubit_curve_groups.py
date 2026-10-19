"""Plot local elevation and visibility for groups of already-projected Cubit curves.

Each CURVE_GROUPS entry is a list of existing curve IDs and makes one plot.
Run in Cubit's Python Journal Editor. Existing geometry is only queried.
All model surfaces are searched at each curve sample. Ambiguous normals are
flagged and excluded from angle calculations, not silently assigned a face.
Default frame: +Y nose, +X right wing, +Z up. Azimuth 0 faces the nose;
positive azimuth rotates toward the right wing (+90 degrees).
Requires Cubit, NumPy and Matplotlib. See curve_groups_quick_start.txt.
"""

from pathlib import Path
from datetime import datetime
import csv
import json
import math
import copy

import numpy as np


# --------------------------- USER SETTINGS ---------------------------
# One list = one plot. Only curve IDs are needed; surfaces are found automatically.
CURVE_GROUPS = [
    [6],                 # Your current curve
    # [12, 13, 14],      # Another plot combining several curves
]
# Shadowing normally tests all model surfaces. Optional overrides:
OCCLUDING_SURFACE_IDS = None
IGNORED_OCCLUDING_SURFACE_IDS = [2]  # Your construction sheet at Y=10; clear for other models
NORMAL_AMBIGUITY_ANGLE_DEG = 0.1     # Flag matching surfaces whose normals differ more than this

SAMPLES_PER_CURVE = 40                 # Equal arc-length midpoint samples
# Stop is included if the step lands on it; otherwise stop before passing it.
AZIMUTH_START_DEG = -90
AZIMUTH_STOP_DEG = 90
AZIMUTH_STEP_DEG = 1
ELEVATION_START_DEG = -45
ELEVATION_STOP_DEG = 45
ELEVATION_STEP_DEG = 1
# Default aircraft frame. Positive azimuth turns from the nose toward the right.
VIEW_FORWARD_AXIS = (0.0, 1.0, 0.0)    # Azimuth 0: viewer on +Y, looking at nose
VIEW_RIGHT_AXIS = (1.0, 0.0, 0.0)      # Azimuth +90: viewer on +X, right wing side
VIEW_UP_AXIS = (0.0, 0.0, 1.0)         # Elevation +90: viewer above, on +Z
NORMAL_SIGN_BY_SURFACE = {}            # Example: {7: -1} for a reversed normal
FRONT_FACES_ONLY = True                # Appropriate for an aircraft exterior
GRAZING_TOLERANCE_DEG = 0.001           # Exactly grazing points are excluded
RAY_OFFSET = None                     # Model units; None = max(1e-6, D * 1e-6)
SURFACE_DISTANCE_TOLERANCE = None      # None = max(1e-6, D * 1e-5)
AGGREGATION = "mean"                  # "mean", "min" or "max" visible elevation
OUTPUT_ROOT = None                    # None = script folder, or Cubit working folder, / curve_group_results
# ---------------------------------------------------------------------


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Expected a finite, nonzero vector.")
    return v / length


def validate_ids(cubit, kind, ids):
    missing = [int(i) for i in ids if not cubit.entity_exists(kind, int(i))]
    if missing:
        raise ValueError("Missing %s IDs: %s" % (kind, missing))


def validate_groups(requested):
    """One list of existing curve IDs per output plot."""
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError("Set CURVE_GROUPS, for example [[6], [12, 13, 14]].")
    groups = []
    for index, ids in enumerate(requested, 1):
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("Group %d needs a nonempty list of curve IDs." % index)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0 for v in ids):
            raise ValueError("Group %d must contain positive integer curve IDs." % index)
        groups.append(list(dict.fromkeys(int(v) for v in ids)))
    return groups


def sample_curves(cubit, groups, count, distance_tolerance):
    """Find containing trimmed surfaces and flag samples without a unique normal."""
    surfaces = []
    for sid in cubit.get_entities("surface"):
        box = np.asarray(cubit.get_bounding_box("surface", sid), dtype=float)
        surfaces.append((sid, cubit.surface(sid), box[[0, 3, 6]], box[[1, 4, 7]]))
    if not surfaces:
        raise ValueError("The model has no surfaces from which to obtain normals.")
    points, normals, weights, curve_ids, surface_ids, group_ids = [], [], [], [], [], []
    statuses, candidate_ids, spreads, flags = [], [], [], []
    for group_id, group in enumerate(groups, 1):
        for cid in group:
            curve = cubit.curve(cid)
            length, start = float(curve.length()), curve.start_param()
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Curve %d has invalid length." % cid)
            for j in range(count):
                fraction = (j + 0.5) / count
                p = np.asarray(curve.point_from_arc_length(start, length * fraction))
                matches = []
                for sid, surface, lower, upper in surfaces:
                    if np.any(p < lower - distance_tolerance) or np.any(p > upper + distance_tolerance):
                        continue
                    q = np.asarray(surface.closest_point_trimmed(p.tolist()))
                    distance = float(np.linalg.norm(p - q))
                    if distance <= distance_tolerance:
                        sign = NORMAL_SIGN_BY_SURFACE.get(sid, 1)
                        if sign not in (-1, 1):
                            raise ValueError("Surface normal signs must be +1 or -1.")
                        normal = sign * unit(surface.normal_at(q.tolist()))
                        matches.append((distance, int(sid), q, normal))
                matches.sort(key=lambda item: (item[0], item[1]))
                spread = float("nan")
                normal = np.full(3, np.nan)
                sid, q = 0, p
                status = "no_matching_surface"
                if matches:
                    ns = np.asarray([item[3] for item in matches])
                    spread = float(np.degrees(np.arccos(np.clip(np.min(ns @ ns.T), -1, 1))))
                    if spread > NORMAL_AMBIGUITY_ANGLE_DEG:
                        status = "ambiguous_normal"
                    else:
                        _, sid, q, normal = matches[0]
                        status = "ok"
                ids_string = ";".join(str(item[1]) for item in matches)
                if status != "ok":
                    flags.append({"point_index": len(points), "group": "group_%02d" % group_id,
                                  "curve_id": cid, "sample_index": j, "arc_fraction": fraction,
                                  "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                                  "flag": status, "candidate_surface_ids": ids_string,
                                  "normal_spread_deg": spread,
                                  "candidates_json": json.dumps([
                                      {"surface_id": item[1], "distance": item[0], "normal": item[3].tolist()}
                                      for item in matches])})
                points.append(q)
                normals.append(normal)
                weights.append(length / count)
                curve_ids.append(cid)
                surface_ids.append(sid)
                group_ids.append(group_id)
                statuses.append(status)
                candidate_ids.append(ids_string)
                spreads.append(spread)
    samples = {"points": np.asarray(points), "normals": np.asarray(normals),
               "weights": np.asarray(weights), "curve_ids": np.asarray(curve_ids),
               "surface_ids": np.asarray(surface_ids), "group_ids": np.asarray(group_ids),
               "normal_status": np.asarray(statuses), "candidate_surface_ids": np.asarray(candidate_ids),
               "normal_spread_deg": np.asarray(spreads), "valid": np.asarray(statuses) == "ok"}
    return samples, flags


def write_point_flags(out, flags):
    fields = ["point_index", "group", "curve_id", "sample_index", "arc_fraction", "x", "y", "z",
              "flag", "candidate_surface_ids", "normal_spread_deg", "candidates_json"]
    with (out / "point_flags.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flags)


def angle_range(start, stop, step, name):
    """Fixed steps toward stop, including stop only when it lies on the grid."""
    start, stop, step = float(start), float(stop), float(step)
    if not all(math.isfinite(v) for v in (start, stop, step)) or step == 0:
        raise ValueError(name + ": start/stop/step must be finite; step cannot be zero.")
    intervals = (stop - start) / step
    if intervals < 0:
        raise ValueError(name + ": step must point from start toward stop.")
    count = int(math.floor(intervals + 1e-10)) + 1
    values = start + step * np.arange(count, dtype=float)
    if np.isclose(values[-1], stop, rtol=0, atol=1e-10):
        values[-1] = stop
    return values


def view_grid():
    if not all(-90 <= v <= 90 for v in (ELEVATION_START_DEG, ELEVATION_STOP_DEG)):
        raise ValueError("Elevation start and stop must be between -90 and +90 degrees.")
    if abs(AZIMUTH_STOP_DEG - AZIMUTH_START_DEG) > 360:
        raise ValueError("Use an azimuth span of at most 360 degrees.")
    az = angle_range(AZIMUTH_START_DEG, AZIMUTH_STOP_DEG, AZIMUTH_STEP_DEG, "Azimuth")
    el = angle_range(ELEVATION_START_DEG, ELEVATION_STOP_DEG, ELEVATION_STEP_DEG, "Elevation")
    forward, right, up = unit(VIEW_FORWARD_AXIS), unit(VIEW_RIGHT_AXIS), unit(VIEW_UP_AXIS)
    if max(abs(float(forward @ right)), abs(float(forward @ up)), abs(float(right @ up))) > 1e-8:
        raise ValueError("VIEW_FORWARD_AXIS, VIEW_RIGHT_AXIS and VIEW_UP_AXIS must be mutually perpendicular.")
    a, e = np.meshgrid(np.radians(az), np.radians(el))
    directions = (np.cos(e)[..., None] * np.cos(a)[..., None] * forward +
                  np.cos(e)[..., None] * np.sin(a)[..., None] * right +
                  np.sin(e)[..., None] * up)
    return az, el, directions


def first_blocker(cubit, point, normal, direction, surface_ids, offset):
    """Return nearest blocking surface ID; zero means clear, -1 means unknown ID."""
    origin = point + offset * normal
    locations, ids = cubit.fire_ray(origin.tolist(), direction.tolist(), "surface", surface_ids, 0, 0.0)
    hits = []
    for i, hit in enumerate(locations):
        distance = float((np.asarray(hit) - origin) @ direction)
        if distance > offset:
            sid = int(ids[i]) if len(ids) == len(locations) else -1
            hits.append((distance, sid))
    return min(hits)[1] if hits else 0


def shadow_backend(cubit):
    if callable(getattr(cubit, "fire_ray", None)):
        return "fire_ray"
    if callable(getattr(cubit, "parse_locations", None)):
        return "parse_locations"
    raise RuntimeError("This Cubit needs either fire_ray or parse_locations for shadowing.")


def make_shadow_query(cubit, occluders, offset):
    """Use the native API when available, otherwise the older location parser."""
    if shadow_backend(cubit) == "fire_ray":
        return lambda p, n, d: first_blocker(cubit, p, n, d, occluders, offset)

    box = np.asarray(cubit.get_total_bounding_box("surface", occluders), dtype=float)
    center = (box[[0, 3, 6]] + box[[1, 4, 7]]) / 2
    diagonal = float(np.linalg.norm(box[[1, 4, 7]] - box[[0, 3, 6]]))
    targets = " ".join(str(int(sid)) for sid in occluders)
    surfaces = [(int(sid), cubit.surface(int(sid))) for sid in occluders]
    # Some builds expose a message handler; restore it after each query.
    can_capture = all(hasattr(cubit, name) for name in (
        "CubitStringMessageHandler", "get_cubit_message_handler", "set_cubit_message_handler"))
    capture = cubit.CubitStringMessageHandler() if can_capture else None

    def query(point, normal, direction):
        # A forward ray that misses emits parser errors in older Cubit. Reverse
        # the ray from outside the model toward the known surface point instead.
        travel = np.linalg.norm(point - center) + diagonal + 10 * offset
        origin = point + travel * direction
        xyz = " ".join("%.17g" % v for v in origin)
        vec = " ".join("%.17g" % v for v in -direction)
        expression = "location fire ray location %s direction %s at surface %s ray radius 0" % (xyz, vec, targets)
        previous = cubit.get_cubit_message_handler() if can_capture else None
        if can_capture:
            capture.mMessage = ""
            cubit.set_cubit_message_handler(capture)
        try:
            errors = cubit.get_error_count()
            hits = np.asarray(cubit.parse_locations(expression), dtype=float)
            if cubit.get_error_count() != errors or hits.size == 0:
                detail = str(capture.mMessage) if can_capture else "See Cubit's console."
                raise RuntimeError("Legacy ray query failed; visibility was not assumed. " + detail)
        finally:
            if can_capture:
                cubit.set_cubit_message_handler(previous)
        if hits.ndim != 2 or hits.shape[1] != 3 or not np.all(np.isfinite(hits)):
            raise RuntimeError("Unexpected coordinates returned by the legacy ray query.")
        relative = hits - point
        distances = relative @ direction
        perpendicular = relative - distances[:, None] * direction
        if np.any(np.linalg.norm(perpendicular, axis=1) > offset):
            raise RuntimeError("Legacy ray query returned points off the requested ray.")
        # A normal-distance tolerance expands along the ray at grazing angles.
        # This also handles the kernel coalescing two nearly tangent hits.
        endpoint_tolerance = offset / max(abs(float(normal @ direction)), 1e-12)
        if not np.any(np.linalg.norm(relative, axis=1) <= endpoint_tolerance):
            raise RuntimeError("Legacy ray missed its known surface point (nearest hit distance %.6g, tolerance %.6g). "
                               "Check curve/surface tolerance and RAY_OFFSET." % (np.min(np.linalg.norm(relative, axis=1)), endpoint_tolerance))
        candidates = np.flatnonzero((distances > endpoint_tolerance) & (distances < travel))
        if not len(candidates):
            return 0
        hit = hits[candidates[np.argmin(distances[candidates])]]
        # The parser returns positions without IDs. Recover the blocker from
        # trimmed-surface distances; retain -1 if its identity is unresolved.
        matches = [(float(np.linalg.norm(hit - np.asarray(s.closest_point_trimmed(hit.tolist())))), sid)
                   for sid, s in surfaces]
        distance, sid = min(matches)
        return sid if distance <= offset else -1

    return query


def compute_visibility(cubit, samples, directions, occluders, offset, progress=True):
    # Status codes: 0 unresolved normal, 1 back-facing, 2 grazing, 3 blocked, 4 visible.
    shape = directions.shape[:2] + (len(samples["points"]),)
    values = np.full(shape, np.nan, dtype=np.float32)
    status = np.zeros(shape, dtype=np.uint8)
    blockers = np.zeros(shape, dtype=np.int32)
    query = make_shadow_query(cubit, occluders, offset)
    threshold = math.sin(math.radians(GRAZING_TOLERANCE_DEG))
    for row in range(shape[0]):
        for col in range(shape[1]):
            direction = directions[row, col]
            if col > 0 and np.allclose(direction, directions[row, 0], rtol=0, atol=1e-12):
                values[row, col] = values[row, 0]
                status[row, col] = status[row, 0]
                blockers[row, col] = blockers[row, 0]
                continue
            dots = samples["normals"] @ direction
            valid = samples["valid"]
            grazing = valid & (np.abs(dots) <= threshold)
            status[row, col, grazing] = 2
            if FRONT_FACES_ONLY:
                status[row, col, valid & (dots < -threshold)] = 1
                candidate = valid & (dots > threshold)
            else:
                candidate = valid & (np.abs(dots) > threshold)
            for index in np.flatnonzero(candidate):
                normal = samples["normals"][index]
                if not FRONT_FACES_ONLY and dots[index] < 0:
                    normal = -normal
                blocker = query(samples["points"][index], normal, direction)
                if blocker:
                    status[row, col, index] = 3
                    blockers[row, col, index] = blocker
                else:
                    status[row, col, index] = 4
                    value = dots[index] if FRONT_FACES_ONLY else abs(dots[index])
                    values[row, col, index] = np.degrees(np.arcsin(np.clip(value, -1, 1)))
        if progress:
            print("Visibility: elevation row %d/%d" % (row + 1, shape[0]), flush=True)
    return values, status == 4, status, blockers


def aggregate(values, visible, weights, method):
    visible_length = np.sum(visible * weights, axis=-1)
    fraction = visible_length / float(np.sum(weights))
    result = np.full(visible_length.shape, np.nan)
    if method == "mean":
        numerator = np.sum(np.where(visible, values, 0.0) * weights, axis=-1)
        np.divide(numerator, visible_length, out=result, where=visible_length > 0)
    elif method in ("min", "max"):
        fill = np.inf if method == "min" else -np.inf
        reducer = np.min if method == "min" else np.max
        result = reducer(np.where(visible, values, fill), axis=-1)
        result[visible_length == 0] = np.nan
    else:
        raise ValueError("AGGREGATION must be mean, min, or max.")
    return result, fraction


def cell_edges(centers, step):
    """Give single-angle sweeps a nonzero display width too."""
    if len(centers) == 1:
        return np.array([centers[0] - abs(step) / 2, centers[0] + abs(step) / 2])
    middle = (centers[:-1] + centers[1:]) / 2
    return np.r_[2 * centers[0] - middle[0], middle, 2 * centers[-1] - middle[-1]]


def plot_maps(path, az, el, angle_map, fraction, title, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("#e4e7ec")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.20, top=0.80, wspace=0.35)
    fig.suptitle(title, fontsize=14)
    az_edges = cell_edges(az, AZIMUTH_STEP_DEG)
    el_edges = cell_edges(el, ELEVATION_STEP_DEG)
    first = axes[0].pcolormesh(az_edges, el_edges, np.ma.masked_invalid(angle_map),
                             shading="flat", cmap=cmap, vmin=0, vmax=90)
    second = axes[1].pcolormesh(az_edges, el_edges, 100 * fraction, shading="flat",
                              cmap="cividis", vmin=0, vmax=100)
    axes[0].set_title("%s local elevation of visible portions" % method.capitalize(), fontsize=10)
    axes[1].set_title("Confirmed visible fraction of total curve length", fontsize=10)
    fig.colorbar(first, ax=axes[0], label="Local elevation (degrees)")
    fig.colorbar(second, ax=axes[1], label="Visible length (%)")
    for ax in axes:
        ax.set(xlabel="Viewing azimuth (degrees)", ylabel="Viewing elevation (degrees)",
               xlim=(az[0], az[-1]) if len(az) > 1 else tuple(az_edges),
               ylim=(el[0], el[-1]) if len(el) > 1 else tuple(el_edges))
        if len(az) == 1:
            ax.set_xticks(az)
        if len(el) == 1:
            ax.set_yticks(el)
    fig.text(0.075, 0.035, "Gray = no evaluated visible samples (hidden or unresolved). See point_flags.csv. Local angle: 0° grazing, 90° normal.", fontsize=9)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(path, samples):
    """Show the sampled existing curves in model coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Registers 3D axes in older Matplotlib.
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cid in np.unique(samples["curve_ids"]):
        mask = samples["curve_ids"] == cid
        first_group = samples["group_ids"][mask].min()
        p = samples["points"][mask & (samples["group_ids"] == first_group)]
        ax.plot(*p.T, linewidth=2, label="Curve %d" % cid)
    bad = ~samples["valid"]
    if np.any(bad):
        ax.scatter(*samples["points"][bad].T, color="red", marker="x", s=35, label="Flagged normal")
    ax.set(xlabel="Model X", ylabel="Model Y", zlabel="Model Z")
    span = np.ptp(samples["points"], axis=0)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(np.maximum(span, max(float(span.max()) * 0.1, 1e-6)))
    ax.view_init(elev=25, azim=55)
    ax.set_title("Already-projected curve samples")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_outputs(out, az, el, directions, values, visible, status, blockers, samples, records, metadata, curve_groups):
    np.savez_compressed(out / "point_results.npz", azimuth_deg=az, elevation_deg=el,
                        local_elevation_deg=values, visible=visible, visibility_status=status, blocking_surface_ids=blockers, **samples)
    with (out / "curve_points.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["point_index", "group", "curve_id", "surface_id",
                         "x", "y", "z", "nx", "ny", "nz", "represented_arc_length", "normal_status", "candidate_surface_ids", "normal_spread_deg"])
        for i, p in enumerate(samples["points"]):
            writer.writerow([i, "group_%02d" % samples["group_ids"][i], samples["curve_ids"][i],
                             samples["surface_ids"][i], *p, *samples["normals"][i], samples["weights"][i], samples["normal_status"][i], samples["candidate_surface_ids"][i], samples["normal_spread_deg"][i]])
    groups = [("group_%02d" % i, group, samples["group_ids"] == i)
              for i, group in enumerate(curve_groups, 1)]
    with (out / "view_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "azimuth_deg", "elevation_deg", AGGREGATION + "_local_elevation_deg",
                         "visible_length_fraction", "back_facing_fraction", "grazing_fraction", "occluded_fraction", "unresolved_normal_fraction", "blocking_surface_ids", "view_dx", "view_dy", "view_dz"])
        for name, ids, mask in groups:
            angles, fraction = aggregate(values[..., mask], visible[..., mask], samples["weights"][mask], AGGREGATION)
            group_weights = samples["weights"][mask]
            total_length = group_weights.sum()
            fractions = {code: np.sum((status[..., mask] == code) * group_weights, axis=-1) / total_length
                         for code in (0, 1, 2, 3)}
            for i, e in enumerate(el):
                for j, a in enumerate(az):
                    writer.writerow([name, a, e, "" if np.isnan(angles[i, j]) else float(angles[i, j]), float(fraction[i, j]),
                                     fractions[1][i, j], fractions[2][i, j], fractions[3][i, j], fractions[0][i, j],
                                     ";".join(map(str, np.unique(blockers[i, j, mask][blockers[i, j, mask] != 0]))),
                                     *directions[i, j]])
            label = ", ".join(str(cid) for cid in ids[:8])
            if len(ids) > 8:
                label += ", ... (%d curves)" % len(ids)
            plot_maps(out / (name + ".png"), az, el, angles, fraction,
                      name.replace("_", " ").title() + " | Curves: " + label +
                      " | Unresolved length: %.1f%%" % (100 * fractions[0][0, 0]), AGGREGATION)
    plot_geometry(out / "geometry.png", samples)
    metadata["curve_mapping"] = records
    with (out / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(curve_groups=None):
    """Run settings above; optionally call main([[6], [12, 13, 14]])."""
    import cubit
    # No cubit.init() here: use the model in the calling Cubit session.
    backend = shadow_backend(cubit)
    print("Shadowing interface: " + backend, flush=True)
    if not isinstance(SAMPLES_PER_CURVE, int) or SAMPLES_PER_CURVE < 2:
        raise ValueError("SAMPLES_PER_CURVE must be an integer >= 2.")
    if AGGREGATION not in ("mean", "min", "max"):
        raise ValueError("AGGREGATION must be mean, min, or max.")
    if not 0 <= GRAZING_TOLERANCE_DEG < 90:
        raise ValueError("GRAZING_TOLERANCE_DEG must be in [0, 90).")
    az, el, directions = view_grid()
    if not 0 < NORMAL_AMBIGUITY_ANGLE_DEG < 90:
        raise ValueError("NORMAL_AMBIGUITY_ANGLE_DEG must be between 0 and 90.")
    # Check plotting dependencies before doing the analysis.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot
    # Cubit's Journal Editor does not supply __file__ in version 2026.8.
    script_file = globals().get("__file__")
    script_dir = Path(script_file).resolve().parent if script_file else Path.cwd()
    root = Path(OUTPUT_ROOT) if OUTPUT_ROOT else script_dir / "curve_group_results"
    out = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    groups = validate_groups(CURVE_GROUPS if curve_groups is None else curve_groups)
    for index, group in enumerate(groups, 1):
        print("group_%02d.png: curves %s (automatic surface lookup)" % (index, group), flush=True)
    validate_ids(cubit, "curve", sorted({cid for group in groups for cid in group}))
    default_occluders = list(cubit.get_entities("surface"))
    occluders = sorted(set(default_occluders if OCCLUDING_SURFACE_IDS is None else OCCLUDING_SURFACE_IDS) - set(IGNORED_OCCLUDING_SURFACE_IDS))
    if not occluders:
        raise ValueError("Provide the surfaces that can block visibility.")
    validate_ids(cubit, "surface", occluders)
    diagonal = float(cubit.get_total_bounding_box("surface", occluders)[9])
    offset = max(1e-6, diagonal * 1e-6) if RAY_OFFSET is None else float(RAY_OFFSET)
    tolerance = max(1e-6, diagonal * 1e-5) if SURFACE_DISTANCE_TOLERANCE is None else float(SURFACE_DISTANCE_TOLERANCE)
    if not math.isfinite(offset) or not math.isfinite(tolerance) or min(offset, tolerance) <= 0:
        raise ValueError("Distance tolerances must be finite and positive.")
    out.mkdir(parents=True, exist_ok=False)
    samples, flags = sample_curves(cubit, groups, SAMPLES_PER_CURVE, tolerance)
    write_point_flags(out, flags)
    used_surfaces = set(samples["surface_ids"][samples["valid"]].tolist())
    if not used_surfaces.issubset(set(occluders)):
        raise ValueError("The detected supporting surfaces %s must be included in occluders for self-shadowing."
                         % sorted(used_surfaces - set(occluders)))
    print("Normal flags: %d of %d samples; details in %s" % (len(flags), len(samples["points"]), out / "point_flags.csv"), flush=True)
    records = []
    for group_id, group in enumerate(groups, 1):
        for cid in group:
            mask = (samples["group_ids"] == group_id) & (samples["curve_ids"] == cid)
            records.append({"group": "group_%02d" % group_id, "curve_id": cid,
                            "surface_ids_used": np.unique(samples["surface_ids"][mask & samples["valid"]]).tolist()})
    print("Computing %d points across %d viewing directions." % (len(samples["points"]), len(az) * len(el)), flush=True)
    values, visible, status, blockers = compute_visibility(cubit, samples, directions, occluders, offset)
    metadata = {"cubit_version": cubit.get_version(), "geometry_mode": "already projected; query only",
                "shadowing_backend": backend,
                "occluding_surface_ids": occluders, "samples_per_curve": SAMPLES_PER_CURVE,
                "curve_groups": {"group_%02d" % i: group for i, group in enumerate(groups, 1)},
                "group_id_meaning": "existing projected curve IDs",
                "surface_selection": "search all model surfaces at each sample; flag conflicting normals or no matches",
                "normal_ambiguity_angle_deg": NORMAL_AMBIGUITY_ANGLE_DEG,
                "ignored_occluding_surface_ids": list(IGNORED_OCCLUDING_SURFACE_IDS),
                "flagged_sample_count": len(flags),
                "visibility_status_codes": {"0": "unresolved_normal", "1": "back_facing", "2": "grazing", "3": "occluded", "4": "visible"},
                "azimuth_start_deg": AZIMUTH_START_DEG, "azimuth_stop_deg": AZIMUTH_STOP_DEG,
                "elevation_start_deg": ELEVATION_START_DEG, "elevation_stop_deg": ELEVATION_STOP_DEG,
                "actual_azimuth_deg": az.tolist(), "actual_elevation_deg": el.tolist(),
                "azimuth_step_deg": AZIMUTH_STEP_DEG, "elevation_step_deg": ELEVATION_STEP_DEG,
                "ray_offset_model_units": offset, "surface_distance_tolerance": tolerance,
                "normal_sign_by_surface": NORMAL_SIGN_BY_SURFACE, "front_faces_only": FRONT_FACES_ONLY,
                "grazing_tolerance_deg": GRAZING_TOLERANCE_DEG,
                "view_forward_axis": list(VIEW_FORWARD_AXIS), "view_right_axis": list(VIEW_RIGHT_AXIS),
                "view_up_axis": list(VIEW_UP_AXIS), "azimuth_convention": "0 forward, +90 right, -90 left",
                "aggregation": AGGREGATION, "local_elevation": "asin(normal dot direction_to_viewer), degrees",
                "mean_weighting": "represented physical curve arc length, restricted to visible samples",
                "notes": "Far-field visibility. NaN means hidden or unresolved normal; consult status codes. Geometry ray tests; no diffraction or lighting model."}
    (out / "curve_surface_mapping.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_outputs(out, az, el, directions, values, visible, status, blockers, samples, records, metadata, groups)
    print("Results saved to: " + str(out), flush=True)
    return out


if __name__ in ("__main__", "__coreformcubit__"):
    main()
