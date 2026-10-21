"""Bounded statistical workspaces and exact, private disk-backed selection."""
from __future__ import annotations

import math
import warnings

import numpy as np

from .memory import _bounded_grid_selections
from .result_storage import allocate_result, operation_result_storage, working_folder


BLOCK_CELLS = 65_536
RAM_WORKSPACE_BYTES = 8 * 1024**2


def workspace_array(shape, dtype, *, ram_limit_bytes=RAM_WORKSPACE_BYTES):
    """Private scratch; large buffers map to disk regardless of output mode."""
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    dtype = np.dtype(dtype)
    if math.prod(shape) * dtype.itemsize <= ram_limit_bytes:
        return np.empty(shape, dtype=dtype)
    # Follow the operation's folder, else the user's working folder.
    with operation_result_storage("disk", cache_dir=working_folder()):
        return allocate_result(shape, dtype)


def domain_values(dataset, selection, domain, field_reader=None):
    if domain == "complex":
        # A reader from _bounded_complex_slice_reader validates legacy arrays
        # once instead of hashing the whole grid for every block.
        return field_reader(selection) if field_reader is not None else dataset.rcs_slice(selection)
    values = dataset.power[selection]
    if domain in {"db", "dbsm"}:
        return dataset.linear_to_dbsm(values)
    if domain == "dbke":
        frequency = dataset.frequencies[selection[2]]
        return dataset.linear_to_dbke(values, frequency.reshape(1, 1, -1, 1))
    return values


def _population_blocks(dataset, axes, output_index, domain, field_reader=None):
    shape = tuple(size if axis in axes else 1 for axis, size in enumerate(dataset.shape))
    for block in _bounded_grid_selections(shape, BLOCK_CELLS):
        selection = tuple(part if axis in axes else slice(output_index[axis], output_index[axis] + 1)
                          for axis, part in enumerate(block))
        yield np.asarray(domain_values(dataset, selection, domain, field_reader)).reshape(-1)


def _exact_rank(blocks, population_size, dtype, statistic, percentile, *, interpolate=False):
    values = workspace_array(population_size, dtype)
    count = 0
    for block in blocks:
        valid = block[~np.isnan(block)]
        values[count:count + valid.size] = valid
        count += valid.size
    if not count:
        return np.nan
    values = values[:count]
    if statistic == "median":
        ranks = ((count - 1) // 2, count // 2)
    else:
        q = float(percentile) / 100.
        if not 0. <= q <= 1.:
            raise ValueError("Percentiles must be in the range [0, 100]")
        if not interpolate:
            # Empirical inverse CDF: select the first observed value whose
            # cumulative fraction reaches q, with q=0 selecting the minimum.
            rank = max(int(np.ceil(q * count)) - 1, 0)
            values.partition(rank)
            return values[rank]
        position = (count - 1) * q
        lower = int(np.floor(position))
        # NumPy's linear method retains the following neighbor even when the
        # interpolation fraction is exactly zero (relevant beside infinity).
        ranks = (lower, min(lower + 1, count - 1))
    # ndarray.partition works in-place on the private mapping and needs only
    # bounded introspective selection scratch, unlike nanpercentile's copy.
    values.partition(ranks)
    pair = np.asarray([values[ranks[0]], values[ranks[1]]], dtype=dtype)
    if statistic == "median":
        return np.mean(pair) if count % 2 == 0 else pair[0]
    fraction = position - ranks[0]
    # Delegate the two bracketing values to NumPy for its exact linear
    # interpolation/dtype/infinity semantics, including endpoint percentiles.
    return np.percentile(pair, fraction * 100.)


def _stream_scalar(blocks, statistic, dtype):
    count = 0
    mean = np.complex128(0.) if np.dtype(dtype).kind == "c" else np.float64(0.)
    m2 = np.float64(0.)
    extreme = None
    infinite = []
    for block in blocks:
        valid = block[~np.isnan(block)]
        if not valid.size:
            continue
        if statistic in {"min", "max"}:
            value = np.min(valid) if statistic == "min" else np.max(valid)
            extreme = value if extreme is None else (np.minimum(extreme, value) if statistic == "min" else np.maximum(extreme, value))
            continue
        if np.any(np.isinf(valid)):
            # Only sign/component representatives are required to retain
            # NumPy's infinity behavior without retaining every such sample.
            exceptional = valid[np.isinf(valid)]
            infinite.extend((np.min(exceptional), np.max(exceptional)))
            if len(infinite) > 4:
                infinite = [np.sum(infinite)]
            valid = valid[np.isfinite(valid)]
        if not valid.size:
            continue
        promoted = np.asarray(valid, dtype=np.complex128 if np.iscomplexobj(valid) else np.float64)
        n = promoted.size
        block_mean = np.mean(promoted)
        total = count + n
        delta = block_mean - mean
        if statistic == "std":
            centered = promoted - block_mean
            block_m2 = np.sum(centered.real * centered.real + centered.imag * centered.imag) if np.iscomplexobj(centered) else np.sum(centered * centered)
            m2 += block_m2 + np.abs(delta) ** 2 * (count * n / total)
        mean = block_mean if not count else mean + delta * (n / total)
        count = total
    if statistic in {"min", "max"}:
        return np.nan if extreme is None else extreme
    if infinite:
        if statistic == "std":
            return np.nan
        with np.errstate(invalid="ignore", over="ignore"):
            return np.mean(np.asarray(infinite, dtype=dtype))
    if not count:
        return np.nan
    return np.sqrt(m2 / count) if statistic == "std" else mean


_VECTORIZED_RANK = {"median": np.median, "percentile": np.percentile}


def _vectorized_nan_rank(values, axes, rank, args, kwargs=None):
    """``np.nanmedian``/``np.nanpercentile`` over ``axes`` without a lane loop.

    Without NaN the nan-function equals the plain function. Otherwise lanes
    are grouped by their non-NaN count n and each group is reduced at once
    after the exact compaction numpy's ``_remove_nan_1d`` applies per lane
    (NaN holes among the first n samples are filled, in order, by the
    trailing non-NaN samples), so ties such as -0.0/+0.0 select identically.
    Percentiles default to the empirical inverse CDF unless ``kwargs``
    explicitly selects another method; median retains its usual definition.
    """
    kwargs = {"method": "inverted_cdf"} if kwargs is None and rank is np.percentile else (kwargs or {})
    nan = np.isnan(values)
    if not nan.any():
        return rank(values, *args, axis=axes, keepdims=True, **kwargs)
    keep = [axis for axis in range(values.ndim) if axis not in axes]
    order = keep + list(axes)
    lanes = np.transpose(values, order).reshape(
        math.prod(values.shape[axis] for axis in keep), -1)
    lane_nan = np.transpose(nan, order).reshape(lanes.shape)
    population = lanes.shape[1]
    counts = population - np.count_nonzero(lane_nan, axis=1)
    reduced = np.empty(lanes.shape[0], dtype=np.float64)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        if count == 0:
            reduced[rows] = np.nan
            warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=3)
            continue
        group = lanes[rows]
        if count < population:
            head = np.array(group[:, :count], copy=True)
            tail = group[:, count:]
            head[lane_nan[rows, :count]] = tail[~lane_nan[rows, count:]]
            group = head
        reduced[rows] = rank(group, *args, axis=1, **kwargs)
    shape = [1] * values.ndim
    for position, axis in enumerate(keep):
        shape[axis] = values.shape[axis]
    return reduced.reshape(shape)


def reduce_dataset(dataset, domain, axes, statistic, percentile=50., *, interpolate=False):
    """Reduce complete populations; percentiles use nearest rank by default."""
    reduced_shape = tuple(1 if axis in axes else count for axis, count in enumerate(dataset.shape))
    probe = tuple(slice(0, 1) for _axis in dataset.shape)
    field_reader = dataset._bounded_complex_slice_reader()[0] if domain == "complex" else None
    dtype = np.asarray(domain_values(dataset, probe, domain, field_reader)).dtype
    reducer = {"mean": np.nanmean, "std": np.nanstd, "median": np.nanmedian,
               "min": np.nanmin, "max": np.nanmax, "percentile": np.nanpercentile}[statistic]
    args = (float(percentile),) if statistic == "percentile" else ()
    kwargs = {"method": "linear" if interpolate else "inverted_cdf"} if statistic == "percentile" else {}
    result_dtype = np.asarray(reducer(np.ones(1, dtype=dtype), *args, **kwargs)).dtype
    result = allocate_result(reduced_shape, result_dtype)
    population_size = math.prod(dataset.shape[axis] for axis in axes)
    # Small populations stay vectorized across independent outputs. Larger
    # ones stream even when the complete reduction produces only one scalar.
    if population_size <= BLOCK_CELLS:
        output_cells = max(1, BLOCK_CELLS // population_size)
        for selection in _bounded_grid_selections(reduced_shape, output_cells):
            source = tuple(slice(None) if axis in axes else part for axis, part in enumerate(selection))
            values = np.asarray(domain_values(dataset, source, domain, field_reader))
            # Reduce in float64/complex128 like the streamed path; float32
            # cancellation otherwise costs precision in mean and variance.
            values = values.astype(np.complex128 if values.dtype.kind == "c" else np.float64, copy=False)
            vectorized = _VECTORIZED_RANK.get(statistic)
            if vectorized is not None and values.dtype.kind == "f":
                # nanmedian/nanpercentile with an axis fall back to a Python
                # loop over every output lane (apply_along_axis); the
                # vectorized form below returns the same bits.
                result[selection] = _vectorized_nan_rank(values, axes, vectorized, args, kwargs)
            else:
                result[selection] = reducer(values, *args, axis=axes, keepdims=True, **kwargs)
        return result
    for index in np.ndindex(reduced_shape):
        blocks = _population_blocks(dataset, axes, index, domain, field_reader)
        if statistic in {"median", "percentile"}:
            value = _exact_rank(blocks, population_size, dtype, statistic, percentile, interpolate=interpolate)
        else:
            value = _stream_scalar(blocks, statistic, dtype)
        result[index] = value
    return result
