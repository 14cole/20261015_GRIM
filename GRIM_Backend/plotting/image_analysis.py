"""Physical image records, native cuts, and explicit A/B comparison policies.

Arrays use Matplotlib's row/column order (y, x). Source arrays are never edited.
An image record can reference an artist's native snapshot; dialogs freeze their
own temporary-file snapshots so later plots or dataset edits cannot change A/B.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import threading

import numpy as np

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.isar.comparison import comparison_issues
from GRIM_Backend.datasets.result_storage import allocate_result, operation_result_storage, result_blocks
from GRIM_Backend.datasets.constants import C0

AXES = ("azimuth", "elevation", "frequency")
AXIS_ATTRIBUTES = {"azimuth": "azimuths", "elevation": "elevations", "frequency": "frequencies"}
LENGTH_SCALE = {"m": 1., "cm": .01, "mm": .001, "in": .0254, "ft": .3048}


def compact_metadata(mapping):
    """Copy only small comparison declarations, never solver arrays/meshes."""
    keys = ("phase_reference", "polarization_basis", "amplitude_convention", "complex_field_domain",
            "phase_center_xyz_m", "time_convention", "source_file", "filename", "source", "name",
            "azimuth", "elevation", "frequency", "angular_coordinate_system", "rcs_linear_quantity", "rcs_log_unit")
    result = {}
    for key in keys:
        value = (mapping or {}).get(key)
        if value is None or isinstance(value, (str, bool, int, float)):
            if key in (mapping or {}):
                result[key] = value
        elif key == "phase_center_xyz_m" and isinstance(value, (tuple, list, np.ndarray)) and np.shape(value) == (3,):
            result[key] = np.asarray(value, float).tolist()
    return result


@dataclass
class ImageRecord:
    name: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    x_axis: str
    y_axis: str
    x_unit: str
    y_unit: str
    quantity: str
    value_unit: str
    logarithmic: bool = False
    phase: bool = False
    notes: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    storage: object = None
    _lease_count: int = field(default=0, init=False, repr=False, compare=False)
    _close_requested: bool = field(default=False, init=False, repr=False, compare=False)
    _lease_lock: object = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def frozen(self, *, directory=None, cancel=None):
        """Copy the image into private snapshot storage in bounded blocks.

        ``cancel`` may abandon a long copy (the partial file is closed);
        ``directory`` overrides the configured snapshot working folder.
        """
        storage = common.NativeSamples(np.shape(self.values), directory=directory)
        try:
            for selection in result_blocks(storage.values.shape, maximum_cells=65_536):
                _cancel(cancel)
                storage.values[selection] = np.ma.filled(self.values[selection], np.nan)
            storage.values.flush()
            return replace(self, x=np.array(self.x, copy=True), y=np.array(self.y, copy=True),
                           values=storage.values, metadata=deepcopy(self.metadata), storage=storage)
        except BaseException:
            storage.close()
            raise

    def lease(self):
        """Keep a capture alive through cancellation or replacement by the UI."""
        with self._lease_lock:
            if self._close_requested:
                raise ValueError("Image capture has closed")
            self._lease_count += 1
        return _ImageLease(self)

    def close(self):
        with self._lease_lock:
            self._close_requested = True
            if self._lease_count == 0 and self.storage is not None:
                self.values = np.empty((0, 0))
                self.storage.close()
                self.storage = None


class _ImageLease:
    def __init__(self, record):
        self.record = record

    def close(self):
        record, self.record = self.record, None
        if record is not None:
            with record._lease_lock:
                record._lease_count -= 1
                if record._close_requested:
                    record.close()

    def __del__(self):
        self.close()


def heatmap_plane(dataset, indices, x_axis, y_axis, *, remaining_index=None, phase=False):
    """Selected X-by-Y plane, or mean-power/circular-mean over the third axis.

    ``indices`` is in azimuth/elevation/frequency/polarization order. The
    polarization must be a singleton. Phase averaging uses unit phasors and
    returns a complex unit value; magnitude averaging is in linear power.
    """
    if x_axis not in AXES or y_axis not in AXES or x_axis == y_axis:
        raise ValueError("Choose two different waterfall axes")
    remaining = next(a for a in AXES if a not in (x_axis, y_axis))
    selected = [np.asarray(ids, dtype=int) for ids in indices]
    axis_index = AXES.index(remaining)
    if remaining_index is not None:
        selected[axis_index] = np.asarray([remaining_index], dtype=int)
    selection = np.ix_(*selected)
    data = dataset.rcs_slice(selection)[..., 0] if phase else dataset.power[selection][..., 0]
    if phase:
        finite = np.isfinite(data) & (np.abs(data)>0)
        data = np.where(finite, np.exp(1j * np.angle(data)), np.nan + 1j * np.nan)
    else:
        finite = np.isfinite(data) & (data >= 0)
    count = finite.sum(axis=axis_index)
    total = np.where(finite, data, 0).sum(axis=axis_index)
    with np.errstate(invalid="ignore", divide="ignore"):
        plane = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    if phase:
        plane = np.where(np.abs(plane) > 1.e-12, plane, np.nan + 1j * np.nan)
    remaining_order = [a for a in AXES if a != remaining]
    if remaining_order[0] != x_axis:
        plane = plane.T
    frequencies = np.asarray(dataset.frequencies[selected[2]], float)
    if "frequency" == x_axis:
        frequencies = frequencies[:, None]
    elif "frequency" == y_axis:
        frequencies = frequencies[None, :]
    else:
        # For frequency aggregation the renderer converts every frequency to
        # the requested physical ordinate before averaging (see Heatmap).
        frequencies = frequencies[0] if len(frequencies) == 1 else frequencies
    return plane, frequencies


def heatmap_display_plane(dataset, indices, x_axis, y_axis, *, remaining_index=None, phase=False, linear=False):
    """Shared GUI/headless X-by-Y display plane with physical aggregation."""
    raw, frequency = heatmap_plane(dataset,indices,x_axis,y_axis,remaining_index=remaining_index,phase=phase)
    if phase:
        degrees = np.rad2deg(np.angle(raw))
        return np.mod(degrees,360.) if (dataset.units or {}).get("phase_wrap") == "0_360" else (degrees+180.)%360.-180.
    if linear:
        return raw
    if remaining_index is None and "frequency" not in (x_axis,y_axis):
        total, count = np.zeros(raw.shape), np.zeros(raw.shape,dtype=np.int64)
        for fi in indices[2]:
            plane, f = heatmap_plane(dataset,indices,x_axis,y_axis,remaining_index=fi)
            valid = np.isfinite(plane) & (plane >= 0)
            # Convert a unit power to the requested log reference to retain
            # legitimate zero power in the mean without inventing a dB floor.
            factor = 10.**(dataset.linear_to_default_db(1.,frequency_value=f,eps=0.)/10.)
            total += np.where(valid,plane*factor,0.)
            count += valid
        with np.errstate(invalid="ignore",divide="ignore"):
            mean = np.where(count,total/np.maximum(count,1),np.nan)
            return np.where(mean>0,10*np.log10(mean),np.nan)
    return dataset.linear_to_default_db(np.where(np.isfinite(raw)&(raw>0),raw,np.nan),frequency_value=frequency,eps=0.)


def attach_isar_record(owner, mesh, band, params):
    """Attach frame/recipe provenance to a rendered ISAR image for A/B capture."""
    dataset = params["dataset"]
    reduced = bool(band.get("display_decimated"))
    if reduced:
        x = (np.asarray(band["display_x_edges"][:-1]) + band["display_x_edges"][1:]) / 2
        y = (np.asarray(band["display_y_edges"][:-1]) + band["display_y_edges"][1:]) / 2
    else:
        x, y = band["x_range"], band["y_range"]
    source = {"units": compact_metadata(dataset.units), "metadata": compact_metadata(dataset.extra),
              "selected_polarization": str(dataset.polarizations[params["pol_idx"]]),
              "selected_elevation_degrees": float(params["elevation_deg"]),
              "selected_frequency_values_hz": np.asarray(dataset._frequency_value_to_hz(
                  dataset.frequencies[np.asarray(params.get("freq_indices_sorted", []),dtype=int)])).tolist(),
              "selected_azimuth_values_native": common.convert_axis_values(
                  band["az_values"], "azimuth", "deg", common.axis_unit(dataset, "azimuth")).tolist()}
    manifest = {"source": source, "formation": {"window": params.get("window_name"),
                "reconstruction": params.get("recon")}}
    contract = {key: deepcopy(band.get(key)) for key in
                ("image_contract", "resolved_reconstruction", "az_values")}
    notes = ("Overview block maxima: cuts, ROI and differences use displayed blocks, not native pixels.",) if reduced else ()
    if not owner._plot_scale_is_linear():
        notes += ("ISAR display intensity uses the formation display floor of −120 dB.",)
    names = getattr(owner,"_selected_datasets",lambda:[])()
    source_name = next((name for name,candidate in names if candidate is dataset),"ISAR")
    mesh._grim_image_record = ImageRecord(
        f"{source_name} | ISAR | {source['selected_polarization']} | {band['az_values'][0]:g}–{band['az_values'][-1]:g}°",
        np.asarray(x), np.asarray(y), np.asanyarray(mesh.get_array()).reshape(len(y),len(x)), "cross_range", "range",
        params["unit_name"], params["unit_name"], "image_intensity", "linear" if owner._plot_scale_is_linear() else "dB",
        logarithmic=not owner._plot_scale_is_linear(), notes=notes,
        metadata={"kind": "isar", "source_name":source_name, "manifest": manifest, "band": contract})


def records_from_figure(owner):
    return [artist._grim_image_record for ax in owner.plot_figure.axes
            for artist in (*ax.images, *ax.collections)
            if getattr(artist, "_grim_image_record", None) is not None]


COMPARISON_BLOCK_CELLS = 65_536


def _cancel(cancel):
    if cancel is not None and cancel():
        from GRIM_Backend.plotting.requests import PlotCancelled
        raise PlotCancelled()


class _OrderedImage:
    """Read sorted image tiles without sorting/copying the full native grid."""
    def __init__(self, record):
        self.record = record
        x, y = np.asarray(record.x, float), np.asarray(record.y, float)
        if x.ndim != 1 or y.ndim != 1 or np.shape(record.values) != (len(y), len(x)):
            raise ValueError("Image dimensions do not match its coordinate axes")
        if not len(x) or not len(y) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Image axes must contain finite coordinates")
        self.xi, self.yi = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
        self.x, self.y = x[self.xi], y[self.yi]
        if np.any(np.diff(self.x) <= 0) or np.any(np.diff(self.y) <= 0):
            raise ValueError("Image coordinates must be unique for linked inspection")

    def read(self, rows, cols):
        return np.asarray(np.ma.filled(self.record.values[np.ix_(self.yi[rows], self.xi[cols])], np.nan))


def ordered(record, *, cancel=None, cache_dir=None):
    """Sorted image, retaining a view when axes are already increasing."""
    source = _OrderedImage(record)
    if (np.array_equal(source.xi, np.arange(len(source.x))) and
            np.array_equal(source.yi, np.arange(len(source.y))) and not np.ma.isMaskedArray(record.values)):
        return source.x, source.y, np.asarray(record.values)
    with operation_result_storage("auto", ram_budget_bytes=32 * 1024**2,
                                  cache_dir=cache_dir or common.snapshot_directory()):
        out = allocate_result(np.shape(record.values), np.float64)
        for rows, cols in result_blocks(out.shape, maximum_cells=COMPARISON_BLOCK_CELLS):
            _cancel(cancel)
            out[rows, cols] = source.read(np.arange(len(source.y))[rows], np.arange(len(source.x))[cols])
    return source.x, source.y, out


def cuts(record, x, y):
    """Nearest native row/column and actual selected center coordinates."""
    xs, ys = np.asarray(record.x), np.asarray(record.y)
    ix, iy = int(np.argmin(abs(xs - x))), int(np.argmin(abs(ys - y)))
    return xs, np.asarray(record.values[iy, :]), ys, np.asarray(record.values[:, ix]), float(xs[ix]), float(ys[iy])


def _convert_axis(values, axis, source_unit, target_unit):
    if source_unit == target_unit:
        return np.asarray(values)
    if axis in AXES:
        return common.convert_axis_values(values, axis, source_unit, target_unit)
    if source_unit in LENGTH_SCALE and target_unit in LENGTH_SCALE:
        return np.asarray(values) * (LENGTH_SCALE[source_unit] / LENGTH_SCALE[target_unit])
    raise ValueError(f"Cannot convert {source_unit} to {target_unit}")


def comparison_advisories(a, b):
    notes = list(a.notes) + list(b.notes)
    if (a.x_axis, a.y_axis) != (b.x_axis, b.y_axis):
        notes.append("Axis meanings differ; compare as supplied uses numeric coordinates only.")
    if a.quantity != b.quantity or a.value_unit != b.value_unit:
        notes.append("Image ordinate quantities or units differ; select relative-peak or explicit numeric comparison.")
    if a.metadata.get("kind") == b.metadata.get("kind") == "isar":
        notes.extend(comparison_issues(a.metadata["manifest"], a.metadata["band"], b.metadata["manifest"], b.metadata["band"]))
    for key in ("polarization", "remaining_selection", "coordinate_system", "frequency_values_hz", "peak_normalized"):
        if a.metadata.get(key) != b.metadata.get(key):
            notes.append(f"Different {key.replace('_', ' ')}; images are compared as supplied.")
    return tuple(dict.fromkeys(notes))


def _log_reference_factor(record, x=None, y=None):
    if record.value_unit.lower() != "dbke" or record.metadata.get("frequency_aggregation"):
        return 1.
    if record.x_axis == "frequency":
        frequency = common.convert_axis_values(record.x if x is None else x,"frequency",record.x_unit,"Hz")[None,:]
    elif record.y_axis == "frequency":
        frequency = common.convert_axis_values(record.y if y is None else y,"frequency",record.y_unit,"Hz")[:,None]
    else:
        selection = record.metadata.get("remaining_selection")
        if not selection or selection[0] != "frequency":
            raise ValueError("dBke comparison needs the source frequency reference; choose displayed numeric comparison")
        frequency = common.convert_axis_values([selection[1]],"frequency",selection[2],"Hz")[0]
    return C0/(2*np.pi*frequency)


def _to_linear(values, record, *, physical=False, x=None, y=None):
    if not record.logarithmic:
        return np.asarray(values)
    result = np.power(10.,np.asarray(values)/10.)
    return result * _log_reference_factor(record,x,y) if physical else result


def _sample_grid(xb, yb, values, x, y, policy):
    """Sample only inside B support; interpolate linear values, not dB.

    An exact center is copied independently of missing neighbors. Bilinear
    samples require only corners with nonzero interpolation weights.
    """
    def positions(source, target):
        pos = np.searchsorted(source, target)
        lo, hi = np.clip(pos - 1, 0, len(source)-1), np.clip(pos, 0, len(source)-1)
        span = source[hi] - source[lo]
        weight = np.divide(target-source[lo], span, out=np.zeros_like(target), where=span != 0)
        near_lo = np.isclose(target, source[lo], rtol=0, atol=8*np.finfo(float).eps*max(1., np.max(abs(source))))
        near_hi = np.isclose(target, source[hi], rtol=0, atol=8*np.finfo(float).eps*max(1., np.max(abs(source))))
        weight[near_lo], weight[near_hi] = 0., 1.
        return lo, hi, weight
    xl, xh, wx = positions(xb, x)
    yl, yh, wy = positions(yb, y)
    if policy == "nearest":
        return values[np.ix_(np.where(wy <= .5, yl, yh), np.where(wx <= .5, xl, xh))]
    out = np.zeros((len(y), len(x)), dtype=np.result_type(values.dtype, float))
    for rows, rowweight in ((yl, 1-wy), (yh, wy)):
        for cols, colweight in ((xl, 1-wx), (xh, wx)):
            weight = rowweight[:, None] * colweight[None, :]
            sample = values[np.ix_(rows, cols)]
            out += np.where(weight > 0, sample * weight, 0)
    return out


def _positions(source, target):
    pos = np.searchsorted(source, target)
    lo, hi = np.clip(pos - 1, 0, len(source)-1), np.clip(pos, 0, len(source)-1)
    span = source[hi] - source[lo]
    weight = np.divide(target-source[lo], span, out=np.zeros_like(target), where=span != 0)
    tolerance = 8*np.finfo(float).eps*max(1., np.max(abs(source)))
    weight[np.isclose(target, source[lo], rtol=0, atol=tolerance)] = 0.
    weight[np.isclose(target, source[hi], rtol=0, atol=tolerance)] = 1.
    return lo, hi, weight


def _converted_tile(source, rows, cols, mode, phase):
    values = source.read(rows, cols)
    if mode == "displayed":
        return values
    if phase:
        return np.exp(1j * np.deg2rad(values))
    return _to_linear(values, source.record, physical=mode == "physical",
                      x=source.x[cols], y=source.y[rows])


def _linear_peak(source, cancel):
    peak = np.nan
    rows, cols = np.arange(len(source.y)), np.arange(len(source.x))
    for row, col in result_blocks((len(rows), len(cols)), maximum_cells=COMPARISON_BLOCK_CELLS):
        _cancel(cancel)
        values = _to_linear(source.read(rows[row], cols[col]), source.record)
        peak = np.fmax(peak, np.fmax.reduce(values.reshape(-1), initial=np.nan))
    return peak


def compare_records(a, b, *, policy="exact", mode="physical", as_supplied=False, maximum_cells=None,
                    cancel=None, storage="auto", ram_budget_bytes=32 * 1024**2, cache_dir=None):
    """Exact native A-minus-B values using bounded tiles and managed outputs.

    Logarithmic differences are physical ratios in dB, phase differences are
    wrapped to [-180, 180), and relative peaks use all source samples. Only the
    displayed overview may be reduced; this result always retains native cells.
    """
    if policy not in ("exact", "nearest", "linear") or mode not in ("physical", "relative", "displayed"):
        raise ValueError("Unknown image comparison policy")
    if mode == "relative" and (a.phase or b.phase):
        raise ValueError("Relative-peak normalization needs two level images; phase has no intensity peak")
    source_a, source_b = _OrderedImage(a), _OrderedImage(b)
    xa, ya, xb, yb = source_a.x, source_a.y, source_b.x, source_b.y
    coordinate_notes = ()
    axis_match = (a.x_axis, a.y_axis) == (b.x_axis, b.y_axis)
    if not axis_match and not as_supplied:
        raise ValueError("Axis meanings differ. Enable compare as supplied to use numeric coordinates.")
    if axis_match:
        try:
            xb = _convert_axis(source_b.x, a.x_axis, b.x_unit, a.x_unit)
            yb = _convert_axis(source_b.y, a.y_axis, b.y_unit, a.y_unit)
        except ValueError:
            if not as_supplied:
                raise
            xb, yb = source_b.x, source_b.y
            coordinate_notes = ("Axis units could not be converted; explicit compare-as-supplied uses numeric coordinates without a physical unit conversion.",)
    if mode == "physical" and (a.quantity != b.quantity or a.phase != b.phase):
        raise ValueError("Physical quantities differ. Choose relative-peak or explicit displayed numeric comparison.")
    if mode == "physical" and a.value_unit != b.value_unit and a.logarithmic == b.logarithmic and not (a.quantity == b.quantity and a.quantity in ("image_intensity", "range_profile_intensity")):
        raise ValueError("Ordinate units differ. Choose relative-peak or explicit displayed numeric comparison.")
    xtol = 8*np.finfo(float).eps*max(1., np.max(abs(xa)), np.max(abs(xb)))
    ytol = 8*np.finfo(float).eps*max(1., np.max(abs(ya)), np.max(abs(yb)))
    ix = np.flatnonzero((xa >= xb[0]-xtol) & (xa <= xb[-1]+xtol))
    iy = np.flatnonzero((ya >= yb[0]-ytol) & (ya <= yb[-1]+ytol))
    if not len(ix) or not len(iy):
        raise ValueError("Images have no overlapping sample centers")
    shape = (len(iy), len(ix))
    if maximum_cells is not None and shape[0]*shape[1] > maximum_cells:
        raise ValueError("The overlap exceeds the requested cell budget; select smaller input images or raise maximum_cells")
    x, y = xa[ix], ya[iy]
    if policy == "exact":
        def match(source, target):
            tolerance = 8*np.finfo(float).eps*max(1., np.max(abs(source)), np.max(abs(target)))
            ids = common.unique_axis_selection(source, target, tolerance)
            if ids is None:
                raise ValueError("Grids differ. Select nearest-neighbor or linear interpolation on the overlapping grid.")
            return ids
        bx, by = match(xb, x), match(yb, y)
    else:
        xl, xh, wx = _positions(xb, x)
        yl, yh, wy = _positions(yb, y)
    pa = pb = 1.
    if mode == "relative":
        pa, pb = _linear_peak(source_a, cancel), _linear_peak(source_b, cancel)
        if not (np.isfinite(pa) and np.isfinite(pb) and pa > 0 and pb > 0):
            raise ValueError("Relative-peak comparison needs a positive finite peak in each image")
    phase = a.phase and mode != "displayed"
    logarithmic = mode != "displayed" and not phase and (mode == "relative" or a.logarithmic or b.logarithmic)
    if mode == "displayed":
        unit = f"numeric A ({a.value_unit}) − B ({b.value_unit})"
    elif phase:
        unit = "wrapped phase difference (deg)"
    else:
        unit = "relative-peak difference (dB)" if mode == "relative" else ("difference (dB)" if logarithmic else a.value_unit)
    _cancel(cancel)
    with operation_result_storage(storage, ram_budget_bytes=ram_budget_bytes,
                                  cache_dir=cache_dir or common.snapshot_directory()):
        outputs = {key: allocate_result(shape, np.float64) for key in ("a", "b", "difference")}
        for rows, cols in result_blocks(shape, maximum_cells=COMPARISON_BLOCK_CELLS):
            _cancel(cancel)
            left = source_a.read(iy[rows], ix[cols]) if phase else _converted_tile(source_a, iy[rows], ix[cols], mode, False) / pa
            if policy == "exact":
                right = _converted_tile(source_b, by[rows], bx[cols], mode, phase) / pb
            elif policy == "nearest":
                right = _converted_tile(source_b, np.where(wy[rows] <= .5, yl[rows], yh[rows]),
                                        np.where(wx[cols] <= .5, xl[cols], xh[cols]), mode, phase) / pb
            else:
                right = np.zeros(left.shape, dtype=complex if phase else float)
                for indices_y, weights_y in ((yl, 1-wy), (yh, wy)):
                    for indices_x, weights_x in ((xl, 1-wx), (xh, wx)):
                        weight = weights_y[rows, None] * weights_x[None, cols]
                        sample = _converted_tile(source_b, indices_y[rows], indices_x[cols], mode, phase) / pb
                        right += np.where(weight > 0, sample * weight, 0.)
            with np.errstate(invalid="ignore", divide="ignore"):
                if phase:
                    right = np.where(abs(right) > 1.e-12, np.rad2deg(np.angle(right)), np.nan)
                    delta = (left - right + 180.) % 360. - 180.
                elif logarithmic:
                    reference = _log_reference_factor(a, x[cols], y[rows]) if a.logarithmic and mode == "physical" else 1.
                    left = np.where(left > 0, 10*np.log10(left/reference), np.nan)
                    right = np.where(right > 0, 10*np.log10(right/reference), np.nan)
                    delta = left - right
                else:
                    delta = left - right
            outputs["a"][rows, cols], outputs["b"][rows, cols], outputs["difference"][rows, cols] = left, right, delta
    level_unit = (a.value_unit if a.logarithmic else ("dB relative to one " + a.value_unit)) if a.logarithmic or b.logarithmic else a.value_unit
    return {"x": x, "y": y, **outputs, "unit": unit, "level_unit": level_unit,
            "phase": bool(phase), "notes": comparison_advisories(a, b)+coordinate_notes, "policy": policy}


def summarize_roi(result, bounds=None, *, cancel=None):
    """Exact native-cell statistics; scratch space does not grow with ROI size."""
    x, y = result["x"], result["y"]
    if bounds is None:
        bounds = (min(x), max(x), min(y), max(y))
    x0, x1, y0, y1 = bounds
    xi = np.flatnonzero((x >= min(x0, x1)) & (x <= max(x0, x1)))
    yi = np.flatnonzero((y >= min(y0, y1)) & (y <= max(y0, y1)))
    selected = len(xi)*len(yi)
    count, total, squared, circular = 0, np.longdouble(0), np.longdouble(0), 0j
    minimum, maximum = np.inf, -np.inf
    for rows, cols in result_blocks((len(yi), len(xi)), maximum_cells=COMPARISON_BLOCK_CELLS):
        _cancel(cancel)
        values = np.asarray(result["difference"][np.ix_(yi[rows], xi[cols])])
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        count += finite.size
        total += np.sum(finite, dtype=np.longdouble)
        squared += np.sum(finite.astype(np.longdouble)**2, dtype=np.longdouble)
        minimum, maximum = min(minimum, float(finite.min())), max(maximum, float(finite.max()))
        if result.get("phase"):
            circular += np.exp(1j*np.deg2rad(finite)).sum()
    if not count:
        raise ValueError("ROI contains no finite image samples")
    mean = float(total/count)
    if result.get("phase"):
        mean = float(np.rad2deg(np.angle(circular))) if abs(circular/count) > 1.e-12 else float("nan")
    return {"pixels": int(count), "mean_delta": mean, "rms_delta": float(np.sqrt(squared/count)),
            "minimum_delta": minimum, "maximum_delta": maximum, "selected_cells": int(selected),
            "excluded_cells": int(selected-count), "unit": result["unit"],
            "mean_definition": "circular" if result.get("phase") else "arithmetic"}


def _block_extreme(values, row_starts, col_starts, *, minimum=False, cancel=None):
    """Reduce either signed extreme with bounded native reads."""
    rows, columns = values.shape
    output = np.full((len(row_starts), len(col_starts)), np.nan)
    width = min(columns, max(1, int(np.sqrt(COMPARISON_BLOCK_CELLS))))
    height = max(1, COMPARISON_BLOCK_CELLS // width)
    reduce = np.fmin if minimum else np.fmax
    for first_row in range(0, rows, height):
        stop_row = min(rows, first_row + height)
        ri = max(0, int(np.searchsorted(row_starts, first_row, side="right")) - 1)
        rj = int(np.searchsorted(row_starts, stop_row, side="left"))
        local_rows = np.r_[0, row_starts[ri+1:rj] - first_row]
        for first_col in range(0, columns, width):
            _cancel(cancel)
            stop_col = min(columns, first_col + width)
            ci = max(0, int(np.searchsorted(col_starts, first_col, side="right")) - 1)
            cj = int(np.searchsorted(col_starts, stop_col, side="left"))
            local_cols = np.r_[0, col_starts[ci+1:cj] - first_col]
            tile = values[first_row:stop_row, first_col:stop_col]
            tile = np.where(np.isfinite(tile), tile, np.nan)
            reduced = reduce.reduceat(reduce.reduceat(tile, local_rows, axis=0), local_cols, axis=1)
            reduce(output[ri:rj, ci:cj], reduced, out=output[ri:rj, ci:cj])
    return output


def comparison_display(result, *, max_cells=250_000, max_side=1024, cancel=None):
    """Bound artist memory while keeping exact comparison arrays for probes.

    A/B level overviews preserve block maxima. Signed differences preserve the
    largest absolute difference (with its sign); phase overviews select native
    block centers because an angle has no useful maximum-intensity meaning.
    """
    x, y = result["x"], result["y"]
    shape = (len(y), len(x))
    ny, nx = common._bounded_image_shape(*shape, max_side=max_side, max_cells=max_cells)
    row_starts = np.arange(ny, dtype=np.int64)*len(y)//ny
    col_starts = np.arange(nx, dtype=np.int64)*len(x)//nx
    row_stops, col_stops = np.r_[row_starts[1:], len(y)], np.r_[col_starts[1:], len(x)]
    rows, cols = (row_starts+row_stops-1)//2, (col_starts+col_stops-1)//2
    reduced = (ny, nx) != shape
    output = {"x": x[cols], "y": y[rows],
              "x_edges": common.coordinate_edges(x)[np.r_[col_starts, len(x)]],
              "y_edges": common.coordinate_edges(y)[np.r_[row_starts, len(y)]], "reduced": reduced}
    limits, scale = [np.inf, -np.inf], 0.
    for key in ("a", "b", "difference"):
        _cancel(cancel)
        values = result[key]
        for selection in result_blocks(shape, maximum_cells=COMPARISON_BLOCK_CELLS):
            _cancel(cancel)
            block = np.asarray(values[selection])
            finite = block[np.isfinite(block)]
            if len(finite):
                if key == "difference":
                    scale = max(scale, float(np.max(np.abs(finite))))
                else:
                    limits[0], limits[1] = min(limits[0], float(finite.min())), max(limits[1], float(finite.max()))
        if not reduced:
            output[key] = values
        elif result.get("phase"):
            output[key] = values[np.ix_(rows, cols)]
        elif key == "difference":
            positive = _block_extreme(values, row_starts, col_starts, cancel=cancel)
            negative = _block_extreme(values, row_starts, col_starts, minimum=True, cancel=cancel)
            output[key] = np.where(np.abs(negative) > np.abs(positive), negative, positive)
        else:
            output[key] = _block_extreme(values, row_starts, col_starts, cancel=cancel)
    output["limits"] = tuple(limits) if np.isfinite(limits).all() else (0., 1.)
    output["difference_scale"] = max(scale, 1.e-12)
    output["note"] = ("Overview uses native block centers for phase; cuts and ROI use all captured image cells." if result.get("phase") else
                      "Overview shows A/B block maxima and the signed largest absolute difference per block; cuts and ROI use all captured image cells.") if reduced else ""
    return output


def image_display(record, *, cancel=None, cache_dir=None):
    """Bound a single captured image for the incomplete/unsupported A/B view."""
    x, y, values = ordered(record, cancel=cancel, cache_dir=cache_dir)
    return comparison_display({"x": x, "y": y, "a": values, "b": values,
                               "difference": values, "phase": record.phase}, cancel=cancel)
