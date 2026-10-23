"""Spatial image cuts and approximate Fourier phase-history reconstruction.

Arrays use (cross-range, down-range) order, independent of display resolution.
The inverse retains aperture taper/coverage; it does not divide window zeros
or pretend that polar regridding and a scene crop can be losslessly undone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
import os
import tempfile

import numpy as np
from matplotlib.path import Path

from GRIM_Backend.datasets.result_storage import allocate_result, operation_result_storage, result_blocks
from GRIM_Backend.plotting.requests import PlotCancelled


RECONSTRUCTION_NOTE = (
    "Approximate reconstructed radar data: aperture taper and coverage are retained; "
    "polar regridding is interpolated back to the selected native samples. "
    "Unsupported or originally missing samples remain unknown. This is not the original measurement."
)


@dataclass
class MaskImage:
    name: str
    x: np.ndarray
    y: np.ndarray
    magnitude: np.ndarray
    complex_image: np.ndarray | None = None
    inverse: dict | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def reconstruction_reason(self):
        if self.inverse is not None and self.complex_image is not None:
            return RECONSTRUCTION_NOTE
        if self.metadata.get("kind") == "composite":
            return "Image dataset only: Composite combines magnitudes and has no recoverable complex phase."
        return "Image dataset only: radar reconstruction requires Fourier ISAR (Fast/Accurate)."


def reconstruction_extra(dataset):
    """Retain coordinate/field lineage without claiming solver certification."""
    extra = dataset._safe_derived_scalar_extra(include_field_conventions=True)
    for key in ("phase_center_xyz_m", "phase_reference", "time_convention", "polarization_basis",
                "range_phase_convention", "measurement_geometry", "measurement_domain",
                "motion_compensation", "phase_center_motion"):
        if key in (dataset.extra or {}):
            extra[key] = deepcopy(dataset.extra[key])
    dataset._invalidate_assembly_sampling_hash(extra, "ISAR Mask Cut reconstruction")
    return extra


def image_from_band(band, params):
    kind = "composite" if band.get("composite") else "isar"
    az = band["az_values"]
    inverse = band.get("mask_inverse") or {}
    dataset = params.get("dataset")
    polarization = (str(dataset.polarizations[params["pol_idx"]])
                    if dataset is not None and "pol_idx" in params else None)
    name = f"{'Composite' if kind == 'composite' else 'ISAR'} {az[0]:g}°–{az[-1]:g}°"
    return MaskImage(name, np.asarray(band["x_range"]).copy(), np.asarray(band["y_range"]).copy(),
                     band.get("mask_magnitude", band["magnitude"]), band.get("complex_image"),
                     band.get("mask_inverse"),
                     {"kind": kind, "unit": params.get("unit_name", "m"),
                      "contract": deepcopy(band.get("image_contract", {})),
                      "source_name": str(getattr(dataset, "source_path", "") or ""),
                      "source_selection_digest": band.get("source_selection_digest"),
                      "frequency_hz": np.asarray(params.get("freq_hz", inverse.get("native_frequency_hz", []))).tolist(),
                      "azimuth_degrees": np.asarray(az).tolist(),
                      "elevation_degrees": params.get("elevation_deg"), "polarization": polarization,
                      "window": params.get("window_name", inverse.get("window")),
                      "reconstruction": band.get("resolved_reconstruction"),
                      "notes": list(band.get("isar_metadata_notes", ())), "cuts": []})


def _check(cancel):
    if cancel is not None and cancel():
        raise PlotCancelled()


def cut_image(source, vertices, *, include, name, cancel=None):
    """Zero the discarded pixels on the original coordinate grid, preserving NaNs."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("Draw a rectangle or polygon with at least three finite vertices.")
    area = abs(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                      - vertices[:, 1] * np.roll(vertices[:, 0], -1)))
    if area <= 0:
        raise ValueError("The selected area is empty.")
    shape = (len(source.x), len(source.y))
    if source.magnitude.shape != shape:
        raise ValueError("This plot has no native mask source. Replot it before using Mask Cut.")
    # Positive radius includes the drawn boundary for counter-clockwise paths.
    signed = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                    - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    if signed < 0:
        vertices = vertices[::-1]
    polygon = Path(np.vstack((vertices, vertices[0])))
    tolerance = max(float(np.ptp(source.x)), float(np.ptp(source.y)), 1.) * 1e-10
    with operation_result_storage("auto"):
        magnitude = allocate_result(shape, source.magnitude.dtype)
        complex_image = (allocate_result(shape, source.complex_image.dtype)
                         if source.complex_image is not None else None)
        selected = 0
        for block in result_blocks(shape):
            _check(cancel)
            x, y = np.meshgrid(source.x[block[0]], source.y[block[1]], indexing="ij")
            inside = polygon.contains_points(np.column_stack((x.ravel(), y.ravel())),
                                             radius=tolerance).reshape(x.shape)
            selected += int(inside.sum())
            keep = inside if include else ~inside
            tile = source.magnitude[block]
            magnitude[block] = np.where(keep | ~np.isfinite(tile), tile, 0.)
            if complex_image is not None:
                tile = source.complex_image[block]
                complex_image[block] = np.where(keep | ~np.isfinite(tile), tile, 0.)
        if not selected:
            raise ValueError("The selected area contains no image pixel centers. Draw a larger area.")
    metadata = deepcopy(source.metadata)
    metadata.setdefault("cuts", []).append({"operation": "include" if include else "exclude",
                                            "vertices": vertices.tolist(), "source": source.name})
    return MaskImage(name.strip() or "Mask cut", source.x.copy(), source.y.copy(), magnitude,
                     complex_image, source.inverse, metadata)


def _unflipped_image(image, cancel=None):
    """Restore a cropped/mirrored image to its full shifted Fourier grid."""
    context = image.inverse
    full = allocate_result((len(context["full_x"]), len(context["full_y"])), np.complex64, fill=0)
    signs = image.metadata.get("contract", {}).get("coordinate_signs", [1, 1])
    x, y = signs[0] * image.x, signs[1] * image.y
    ix = np.rint((x - context["full_x"][0]) / np.diff(context["full_x"])[0]).astype(int)
    iy = np.rint((y - context["full_y"][0]) / np.diff(context["full_y"])[0]).astype(int)
    if np.any(ix < 0) or np.any(ix >= full.shape[0]) or np.any(iy < 0) or np.any(iy >= full.shape[1]):
        raise ValueError("Image coordinates do not match the retained reconstruction grid.")
    for block in result_blocks(image.complex_image.shape):
        _check(cancel)
        full[np.ix_(ix[block[0]], iy[block[1]])] = image.complex_image[block]
    return full


def inverse_spectrum(image, cancel=None):
    """Invert the normalized, padded IFFT in bounded batches of 1-D FFTs."""
    if image.inverse is None or image.complex_image is None:
        raise ValueError(image.reconstruction_reason)
    try:
        from scipy.fft import fft
    except ImportError:  # NumPy may use complex128 scratch; batches stay bounded.
        from numpy.fft import fft
    context = image.inverse
    full = _unflipped_image(image, cancel)
    na, nf = full.shape
    nv, nu = len(context["v"]), len(context["u"])
    stage = allocate_result((na, nv), np.complex64)
    rows = max(1, 8 * 1024**2 // (32 * nf))
    for start in range(0, na, rows):
        _check(cancel)
        tile = full[start:start + rows]
        stage[start:start + rows] = fft(np.fft.ifftshift(tile, axes=1), axis=1)[:, :nv]
    del full
    result = allocate_result((nu, nv), np.complex64)
    cols = max(1, 8 * 1024**2 // (32 * na))
    scale = context["gain"] / (na * nf)
    for start in range(0, nv, cols):
        _check(cancel)
        tile = stage[:, start:start + cols]
        result[:, start:start + cols] = fft(np.fft.ifftshift(tile, axes=0), axis=0)[:nu] * scale
    return result


def reconstruct_radar(image, cancel=None):
    """Return an RcsGrid on the selected original azimuth/frequency axes."""
    from GRIM_Backend.datasets.grid import RcsGrid
    with operation_result_storage("auto"):
        spectrum = inverse_spectrum(image, cancel)
        context = image.inverse
        az = np.asarray(context["native_azimuth_degrees"])
        frequencies = np.asarray(context["native_frequency_hz"])
        u, v = np.asarray(context["u"]), np.asarray(context["v"])
        output = allocate_result((len(az), len(frequencies)), np.complex64)
        for block in result_blocks(output.shape):
            _check(cancel)
            psi = np.deg2rad(az[block[0]] - context["look_center_degrees"])[:, None]
            f = frequencies[block[1]][None, :]
            qu = f * np.sin(psi)
            qv = f * (np.cos(psi) if context["accurate"] else np.ones_like(psi))
            a = (qu - u[0]) / (u[-1] - u[0]) * (len(u) - 1)
            b = (qv - v[0]) / (v[-1] - v[0]) * (len(v) - 1)
            valid = ((a >= -1e-7) & (a <= len(u)-1+1e-7)
                     & (b >= -1e-7) & (b <= len(v)-1+1e-7)
                     & context["native_valid"][block])
            a, b = np.clip(a, 0, len(u)-1), np.clip(b, 0, len(v)-1)
            ia = np.minimum(a.astype(int), len(u)-2)
            ib = np.minimum(b.astype(int), len(v)-2)
            da, db = a - ia, b - ib
            tile = ((1-da) * ((1-db)*spectrum[ia, ib] + db*spectrum[ia, ib+1])
                    + da * ((1-db)*spectrum[ia+1, ib] + db*spectrum[ia+1, ib+1]))
            if context["conjugate_time"]:
                tile = tile.conj()
            output[block] = np.where(valid, tile, complex(np.nan, np.nan))
        extra = dict(context["extra"])
        extra["mask_cut_json"] = json.dumps({"version": 1, "image": image.metadata,
                                              "reconstruction": RECONSTRUCTION_NOTE,
                                              "window": context["window"]}, default=_json_value)
        extra["data_origin"] = "ISAR mask reconstructed phase history"
        extra.pop("source_path", None)
        history = context["history"] + "\nMask Cut: " + image.name + ". " + RECONSTRUCTION_NOTE
        return RcsGrid(context["azimuths"], context["elevations"], context["frequencies"],
                       context["polarizations"], rcs=output[:, None, :, None],
                       units=dict(context["units"]), extra=extra, history=history)


def finish_cut(source, vertices, *, include, name, cancel=None):
    result = cut_image(source, vertices, include=include, name=name, cancel=cancel)
    radar = reconstruct_radar(result, cancel) if result.inverse is not None else None
    return result, radar


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported image metadata: {type(value).__name__}")


def save_image(path, image):
    """Atomic NPZ artifact; numeric arrays and JSON only, no pickle."""
    arrays = {"x": image.x, "y": image.y, "magnitude": image.magnitude}
    if image.complex_image is not None:
        arrays["complex_image"] = image.complex_image
    context = None
    if image.inverse is not None:
        context = {}
        for key, value in image.inverse.items():
            if isinstance(value, np.ndarray):
                arrays["inverse_" + key] = value
            else:
                context[key] = value
    arrays["manifest"] = np.asarray(json.dumps({"format": "GRIM-mask-image", "version": 1,
        "name": image.name, "metadata": image.metadata, "inverse": context}, default=_json_value))
    folder = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".grim-mask-", suffix=".npz", dir=folder)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_image(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest"]))
        if manifest.get("format") != "GRIM-mask-image" or manifest.get("version") != 1:
            raise ValueError("This is not a supported GRIM Mask Cut image dataset.")
        x, y, magnitude = (archive[key] for key in ("x", "y", "magnitude"))
        if (x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)
                or magnitude.shape != (len(x), len(y)) or np.iscomplexobj(magnitude)):
            raise ValueError("Invalid image axes or magnitude shape.")
        complex_image = archive["complex_image"] if "complex_image" in archive else None
        if complex_image is not None and (complex_image.shape != magnitude.shape or not np.iscomplexobj(complex_image)):
            raise ValueError("Invalid complex image shape or type.")
        inverse = manifest["inverse"]
        if inverse is not None:
            inverse.update({key[len("inverse_"):]: archive[key] for key in archive.files if key.startswith("inverse_")})
        return MaskImage(manifest["name"], x, y, magnitude, complex_image, inverse, manifest["metadata"])
