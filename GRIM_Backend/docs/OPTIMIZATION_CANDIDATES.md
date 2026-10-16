# Optimization candidates (survey of 2026-10-15)

Scope: new performance, memory and responsiveness work not already done or tabled.
Excluded as done: column-mapped import storage/cancel, model-backed coordinate lists,
Capture A/B off the UI thread, OUT managed storage, PIO/SS cancellation, SS plan reuse,
statistics estimate, PTM caps, SS descending sweeps, legacy CSV unit confirmation.
Excluded as tabled: per-format import prompts, shared immutable sample storage,
indexed native format.

How this list was produced: timing and profiling runs on this machine (system Python
3.11, numpy 2.4.2, matplotlib 3.10.8) with synthetic grids of 289k, 2.9M and 2.2M cells
(the last with 100,000 azimuths), plus six read-only code audits by area. Every item
cites the lines it rests on. "Measured" means a number from this machine; everything
else is a cost derived from the code. Effort: S under a day, M a few days, L longer.

## 1. Measured hotspots (highest confidence)

### 1.1 Flat CSV import and export run one Python step per cell
- Where: `io/csv.py` `_write_flat_csv_direct` (443-494) and `_finite_csv_number`
  (325-333); reader `_FlatCsvSource.rows` (636-649), `_discover` (756-792), samples
  pass (939-1019), `_parse_number` (542-555), `_power_candidates` (558-615).
- Measured: 289,440 rows (38 MB) take 5.8 s to write and 5.3 s to read (two passes).
  The writer calls `_finite_csv_number` 2,026,080 times, each with three numpy scalar
  predicates; the reader calls `cell()` 7.2M times and numpy scalar predicates 1.7M
  times. A 2.9M-cell grid is therefore about a minute each way.
- Prototype on the same rows: block formatting (one `np.isfinite` per 65,536-row tile,
  `.tolist()`, `format(v, ".17g")`) writes identical text in 0.70 s (7.4x);
  `np.loadtxt(usecols=...)` parses the numeric columns in 0.24 s versus 0.78 s for
  `csv.DictReader` plus per-cell `float()`, before any of GRIM's per-row validation.
- Fix: per tile, vectorize dB and phase conversions, format from `.tolist()`, write
  with `writer.writerows`, report `checkpoint(done, total)`; read with `csv.reader`
  (or `np.loadtxt(max_rows=...)` for numeric columns) into typed blocks, validate per
  block with numpy, map coordinates with `searchsorted`, and recover the offending row
  index for the existing "line N:" messages. Column-mapped tables share the reader.
- Effort M. Risk: exact `.17g` text, blank/`-inf` cells, line-numbered error messages,
  legacy modes; covered by `test_flat_csv_schema.py`, `test_csv_export.py`,
  `test_table_import.py`, `test_format_dispatch.py`.

### 1.2 Selecting a dataset with a large axis stalls the GUI in selection queries
- Where: `ui/coordinate_list.py` `selected_rows` (222-223) and `selectedItems`
  (225-226) use `selectionModel().selectedIndexes()`, which makes Qt call the model's
  `flags()` once per selected row; `ui/dataset_actions.py` `_apply_default_param_selection`
  (1788-1835) calls `selected_rows()` twelve times and `widget.selectedItems()` as an
  emptiness test (1828).
- Measured: selecting a 100,000-azimuth dataset costs 1.63 s on the GUI thread
  (1.16 s in `selected_rows`, 1.11 s in `selectedItems`, 107,135 `flags()` calls);
  switching back to a normal dataset afterwards costs 1.70 s because the previous
  selection is captured the same way. Every plot also pays 0.1 s here.
- Fix: derive rows from `selectionModel().selection()` ranges (one `np.arange` per
  range), add `has_selection()` backed by `hasSelection()`, and cache `count()` in
  `select_rows` (`rowCount` was called 507k times).
- Effort S. Risk: none functional; `test_coordinate_list_view.py`,
  `test_parameter_selection_persistence.py`.

### 1.3 Restoring a non-contiguous axis selection is quadratic
- Where: `ui/dataset_actions.py` `_restore_param_selection` (1733-1738): for each
  previously selected value, `np.isclose(target, value)` scans the whole new axis.
- Measured: 5,000 scattered selections restored onto a 100,000-sample axis take
  1.52 s; 25,000 take 3.74 s. A sorted `searchsorted` window gives the same rows in
  under 1 ms.
- Fix: sort the target once, `searchsorted(value - tol)` / `searchsorted(value + tol)`
  per wanted value (vectorized), expand with `np.repeat`.
- Effort S. Risk: tolerance semantics unchanged (same `atol`).

### 1.4 The Reports workspace counts distinct azimuths with a quadratic loop
- Where: `reports/workspace.py` `_physical_angle_sample_count` (2722-2746): each value
  is compared against every value already kept. Called from `_sync_azimuth_band_axis`
  (1690) on every catalog change and dataset toggle via `_refresh_availability`.
- Measured: 1,800 azimuths 0.056 s, 3,600 0.226 s, 7,200 0.903 s (4x per doubling),
  so a 36,000-sample axis is about 20 s and 100,000 about three minutes, on the GUI
  thread. `np.unique` on the rounded modulo values takes 0.2 ms.
- Fix: `np.unique(np.round((values % period) / tol))` with the seam alias folded.
- Effort S. Risk: none; `test_ppt_workspace.py`.

### 1.5 Startup imports scipy through the ISAR plot mode
- Where: `plotting/modes/isar_mode.py:28-30` imports `scipy.fft` at module import;
  `ui/analysis_controls.py` imports `plotting.modes`, so the app always pays it. No
  other scipy import is on the startup path (`engineering_tools` loads on demand).
- Measured: `import GRIM_Backend.ui.app` takes 1,016 ms; 303 ms of that is the
  `isar_mode` subtree (scipy.fft pulls scipy.special, numpy.f2py and charset_normalizer).
  Window construction adds 0.38 s, of which the eager GHOST workflow import in 7.3 is
  the largest single piece.
- Fix: a cached lazy accessor for `scipy.fft` used by the three FFT helpers and the
  dtype decision at line 580.
- Effort S. Risk: ISAR pays the import on first use instead.

### 1.6 Native .grim save spends its time in deflate for a few percent of size
- Where: `io/native.py` `save` (170-171) uses `np.savez_compressed`, which fixes zlib
  level 6; `_native_archive_payload` (96-101) validates without `_validated_complex`,
  so a complex store is re-derived to power and phase just to scan for infinities.
- Measured: 2.9M cells of noisy float64 samples: compressed write 1.17-1.36 s for a
  43.8 MB file; level 1 0.97 s; uncompressed 0.02 s for 46.2 MB (5% larger). Reads are
  0.13-0.15 s either way.
- Fix: decide per save like `isar/artifact.py` already does (compress a 1-2 MiB sample
  at level 1 and store uncompressed when the gain is small), write members through
  `zipfile` with `compresslevel=1` when compressing, pass `_validated_complex=True`
  for complex storage, and report progress per member.
- Effort S. Risk: tests that assert compression decisions or sizes
  (`test_complex_storage_efficiency.py`, `test_native_cache_footprint.py`).

### 1.7 Phase heatmaps build a complex field to get unit phasors
- Where: `plotting/image_analysis.py` `heatmap_plane` (127-129): `rcs_slice` derives
  the full complex plane, then `exp(1j * angle(data))`.
- Measured: azimuth by frequency phase plane on 2.9M cells takes 0.17 s with a 107 MB
  peak versus 0.025 s and 61 MB for the power plane.
- Fix: build the phasor from `phase_slice` (degrees to radians) and the finite mask from
  `power_slice`, without the complex field.
- Effort S. Risk: phase convention for zero-power cells; `test_image_analysis.py`.

## 2. Plotting (verified by reading; auditor timings on this machine)

### 2.1 Rectilinear images are drawn as QuadMesh, not images
- Where: `plotting/modes/waterfall_mode.py:185`, `range_freq_mode.py:301-305`,
  `delta_map_mode.py:190-193`, `ui/image_comparison.py:196-198`,
  `az_vs_range_mode.py:383-387`; `isar/geometry.py:228-236` already switches to
  `imshow` for uniform edges.
- Cost: one 2M-cell QuadMesh draw is about 270 ms against 45 ms for `imshow` and
  20 ms for `NonUniformImage`; panels are budgeted at 2M cells and 8M per figure, and
  every pan, zoom, cut click, z-limit keystroke and colormap toggle redraws.
- Fix: use the `draw_image` rule everywhere (imshow for uniform edges, otherwise
  `NonUniformImage`), keep explicit edges on the artist, and include `ax.images` in
  `_waterfall_coordinate_limits`.
- Effort M (many tests read `ax.collections[0]`). Risk: artist type changes in tests.

### 2.2 Comparison dialog recomputes linked cuts on every mouse move
- Where: `ui/image_comparison.py:116, 400-421`: no coalescing timer, `relim` and
  `autoscale_view` on both cut axes, full `draw_idle` of four meshes; column cuts read a
  strided column from disk-mapped results above 32 MiB.
- Fix: the main window's 30 ms single-shot timer pattern, then blit the cut and
  cursor lines. Effort S for the timer, M for blitting.

### 2.3 ISAR rebuilds the whole figure per result, including every Play frame
- Where: `plotting/modes/isar_render.py:357-367, 373-394, 423-428, 486-507`.
- Fix: keep a figure signature and reuse `AxesImage.set_data` plus `set_clim` when band
  count, shapes and extents match; move the magnitude-to-dB conversion into the worker.
  Effort M. Risk: A/B capture reads the record on the artist, so re-attach it.

### 2.4 Polarization availability derives power and phase on the GUI thread
- Where: `plotting/actions.py:207-216` on every polarization click and phase toggle.
- Fix: for complex storage take `np.isfinite` of the field block directly.
  Effort S (touches `datasets/storage.py`).

### 2.5 Marker drag, pan and dataset highlight redraw the full figure per event
- Where: `plotting/markers.py:196-204`, `actions.py:1983-1999`,
  `dataset_style.py:127-137`. Fix: blit markers from a cached background; restyle only
  the lines whose highlight changed. Effort M.

### 2.6 Each slider step re-runs the complete renderer
- Where: `plotting/slider.py:64-107`, `actions.py:1582-1651, 2168-2177, 2629-2704`:
  128 temporary files, 128 artists and a rebuilt legend per step, plus two fit passes
  each calling `_apply_plot_limits`.
- Fix: update existing lines and refill their memmaps when the trace keys match; fold
  `_fit_both` into one `relim`/`autoscale_view`. Effort M-L (S for the fit change).

### 2.7 Waterfall preparation reduces the native plane twice
- Where: `plotting/requests.py:216-228` and `246-248`. Fix: compute the display shape
  with `_bounded_image_shape` and call `image_grid` once. Effort S-M.

### 2.8 Circular median is a per-column Python loop
- Where: `plotting/modes/common.py:594-626`, used by phase sweeps. Fix: block the
  columns and evaluate candidate costs with broadcast rank counts. Effort M.

### 2.9 PBP heatmap fill loops per sample and hovers by polygon hit test
- Where: `plotting/actions.py:764-808, 2830-2838`. Fix: broadcast the edge and value
  grids, attach a `searchsorted` probe. Effort S.

### 2.10 Figure export renders twice at 200 dpi
- Where: `ui/dataset_actions.py:3581` (`bbox_inches="tight"`). Fix: measure the box
  from the on-screen renderer and pass a `Bbox`. Effort S.

## 3. Datasets

### 3.1 Wrap azimuth derives power and phase for every cell and runs seam math on every row
- Where: `datasets/axes.py:1394-1486`, merge helper `1061-1119` (`phase_delta` from
  `exp` and `angle` is computed even when no cell overlaps, 1095-1097).
- Cost: about 35-40 ms per million cells plus 8-32 bytes per cell of scratch, for an
  operation that is a copy except at the seam. Fix: copy single-member groups directly,
  merge only the seam, return a field for raw storage. Effort M.

### 3.2 Derived grids that are already clean skip the adoption token and pay full copies
- Where: `datasets/transforms.py` `_copy_grid` (28-54) never passes
  `_adopt_clean_arrays`; used by `offset_db`, `shift_dataset`, `coherent_divide`,
  `decimate_axis`, `translate_phase_center`, `medianize_azimuth`; also
  `arithmetic.py` `incoherent_subtract` (579-589), `arithmetic_db_subtract` (626-637),
  `calibration.py` `range_calibrate` (487-498).
- Cost: one or two extra grid-sized arrays and an `isfinite` pass per operation
  (`grid.py:141, 186-190, 331-348`). Fix: clean inside the block loop and pass the
  token. Effort S per site.

### 3.3 Join and stitch do seam trigonometry on every block and fetch power and phase separately
- Where: `datasets/combine.py:1112-1134` (join), `421-422` and `257-271` (stitch
  validation pass over canonical storage that cannot contain what it checks).
- Fix: `power_phase_slice` for both components, gate the phase delta on `np.any(both)`,
  skip the validation pass for canonical storage. Effort S.

### 3.4 Derived results are allocated in float64 regardless of source precision
- Where: `axes.py:30, 585, 635-637, 504-506`; `transforms.py:782-785, 924-925,
  1360, 1551-1553, 654, 479-480, 415`; `arithmetic.py:552-553, 610-613`;
  `combine.py:315-319`; `calibration.py:296`.
- Cost: float32 datasets double permanently after regrid, decimate, gate, translate or
  dB subtract. Fix: allocate retained results in the source precision while keeping
  float64 block arithmetic. Effort M; needs a per-operation precision decision and
  test tolerance review.

### 3.5 Legacy array access makes every later block read re-hash the whole grid
- Where: `datasets/storage.py` `_sample_fingerprint` (445-457),
  `_ensure_legacy_source_current` (459-471); `grid.py:525-534, 578`.
- Cost: scripting users only (no application module reads `rcs_power`); after one
  access, block readers hash the full arrays once per block (SHA-256 at about 1.9 GB/s
  measured, so tens of seconds on a 50M-cell grid). Fix: validate once per reader.
  Effort S-M.

### 3.6 Audit derives the field twice and reads blocks through advanced indexing
- Where: `datasets/audit.py:248-276, 463-502, 573-586`. Fix: basic-slice blocks and a
  single accumulating pass. Effort S.

### 3.7 Overlap copies every input three or four times
- Where: `datasets/combine.py:1386-1403, 1427-1466`. Fix: compute the keep mask
  block-wise from the sources and gather once. Effort M.

### 3.8 Slice reads copy an array that already owns its data
- Where: `datasets/storage.py:369-381` (`np.array(..., copy=True)` on a fancy-index
  result). Fix: the `owndata` check `_complex_samples` already uses. Effort S.

### 3.9 Gathers use 4,096-cell blocks
- Where: `datasets/result_storage.py:357-371`. Auditor measured a 1.16M-cell crop at
  12.0 ms with 4,096-cell blocks versus 7.1 ms with 65,536. Fix: use `BLOCK_CELLS`.
  Effort S.

### 3.10 In-place sample edits rewrite the whole field (scripting API only)
- Where: `datasets/storage.py:557-644`. Fix: copy the stored field once, derive only
  the selection, pass `_result=True`. Effort M.

## 4. I/O (beyond 1.1 and 1.6)

### 4.1 Unrecognized tables get three unbounded header searches with regexes per cell
- Where: `io/loaders.py:146-165`; `io/cst.py:282-309, 733-767, 1059-1071`; the
  estimate path already bounds its probe to 256 rows (`cst.py:109, 127`).
- Fix: `itertools.islice(rows, 256)` for the header searches. Effort S.

### 4.2 Vendor text readers use `np.errstate` and numpy scalar predicates per sample
- Where: `io/cst.py:192-211, 167-171, 351-377, 783-826, 1115-1141, 1172-1229`;
  `io/sentri.py:200-217, 225-281`; `io/out.py:195`.
- Fix: `math` equivalents and `10.0 ** (v / 10.0)` inside `try/except OverflowError`.
  Effort S.

### 4.3 SS record scan parses the full header per record; sample decode loops per signal
- Where: `io/xpatch.py` `read_ss` (364-438, `_parse_table` of HDRC per record at
  400-402), `_load_ss_bounded` (755-784).
- Cost: about 100 numpy calls per record, then about 110 per signal; 36k signals is
  seconds, 360k is minutes. Fix: validate record 0, then read all records through
  strided views and decode in signal blocks; fall back to the loop at the first
  mismatch so error text is unchanged. Effort M.

### 4.4 PTM reader and writer iterate per aspect; `save_ptm` derives each row twice
- Where: `io/ptm.py:340-351, 589-602, 774-792`. Fix: process
  `BLOCK_CELLS // num_frequencies` aspects per iteration. Effort S-M.

### 4.5 PIO export validates the whole slice up front and has no checkpoints
- Where: `io/pioneer.py:787-795, 896-923`. Fix: test phase-missing per block, add
  progress. Effort S.

### 4.6 Native estimate opens and decodes the archive three times per load
- Where: `io/loaders.py:283-298`, `datasets/memory.py:364-472`,
  `io/native_cache.py:78-120`. Fix: carry the estimate report into `load` like
  `discovery_plan`. Effort S-M.

### 4.7 The raw-pair consistency check loops in Python over every frequency
- Where: `io/native.py` `_raw_complex_consistency_report`, loop at 341-366: one iteration per
  frequency, each slicing four arrays and running a dozen numpy calls on a one-frequency
  plane. Added 2026-10-16 from the project audit (`PROJECT_AUDIT.md`, F4).
- Measured: the slowest test in the suite (16.6 s) is a raw field with 512,000 frequencies;
  the same cell count with 512 frequencies is fast.
- Fix: process blocks that span many frequencies and broadcast the per-frequency power scale
  (`4 pi` for sigma_3d, `1/(4 k0)` for sigma_2d) across the frequency axis. Effort S.

## 5. Assembly

### 5.1 .asy files are JSON with base64 of uncompressed NPZ, loaded on the GUI thread
- Where: `assembly/tree.py:1083-1100, 1181-1220, 1245-1268, 1351-1386, 2819-2867`.
- Cost: files at least 1.33x the raw samples, load peak near 2.7x in text plus arrays,
  and the decode freezes the GUI (save already runs in a worker). Fix: a v6 ZIP
  container with a manifest and one NPZ member per leaf, decoded in a worker, with the
  v1-v5 readers kept. Effort M-L.

### 5.2 Build Platform preflight overestimates by about 15x and refuses feasible builds
- Where: `assembly/tree.py:3667-3671` (`cells() * 256` summed at every level),
  `2766-2771`. A 9.9M-cell grid under root, branch and two leaves is estimated at
  10 GB against a real peak near 0.8 GB. Fix: model the actual peak per branch.
  Effort S.

### 5.3 Dragging a branch onto the Datasets table builds synchronously
- Where: `ui/app.py:2365-2367` calls `build_assembly_grid` on the GUI thread; the
  Build button path has a worker and progress dialog (`tree.py:2735-2792`).
  Fix: route the drop through the same worker. Effort M.

### 5.4 Combined results allocate three to five extra grid-sized arrays
- Where: `assembly/tree.py:3619-3644, 3072-3079`; `grid.py:127-134, 185-190`.
  Fix: build the coherent result as raw-field storage and adopt owned fields.
  Effort M.

### 5.5 Axes-only grids carry full-size sample stubs
- Where: `assembly/tree.py:2972-2990`. Fix: power-only zeros with the adoption token.
  Effort S.

### 5.6 Physics SHA-256 builds 48-64 bytes per cell of temporaries every build
- Where: `assembly/tree.py:548-577`. Fix: hash in azimuth slabs (identical digest),
  memoize per grid object. Effort S-M.

### 5.7 Every .asy save re-validates and re-derives every embedded dataset
- Where: `assembly/tree.py:1086-1091`; `io/native.py:576-619`. Fix: the same
  `_validated_complex` shortcut as 1.6 and a per-grid payload cache. Effort S-M.

### 5.8 Spatial feature tree rebuilt wholesale; readiness computed twice per edit
- Where: `assembly/panel.py:970-1063, 3806-3827, 3682-3695, 4137-4145`;
  `assembly/model.py:563-589`. Fix: update only the changed column, coalesce readiness
  behind a zero-delay timer, cache the sidecar read. Effort S-M.

### 5.9 3-D preview draws an uncapped quiver arrow per point
- Where: `assembly/workspace.py:408-523, 1369-1396, 1621-1657`. Fix: apply the
  existing 250-arrow cap and hide arrows during interaction. Effort S.

### 5.10 Provenance JSON is re-parsed ten or more times per branch
- Where: `assembly/tree.py:594-601, 3187, 3284, 3311-3321, 3537-3553, 3762-3769`.
  Fix: decode once per build. Effort M.

## 6. Reports, ISAR, scripting

### 6.1 Report series are validated sample by sample in Python, 8-10 passes per point, on the GUI thread
- Where: `reports/report.py:236-262, 827`; `reports/plot_data.py:677, 882,
  1012-1023`; `reports/workspace.py:110-125, 1469-1500, 2115-2170, 2185-2213`.
- Cost: 4 datasets, 60 frequencies and a 0.1 degree sweep is about 8M Python
  iterations (3-5 s frozen) per preview build. Fix: validate with numpy once and skip
  re-validation when restyling. Effort M. Risk: `PlotSeries.x/y` stay tuples.

### 6.2 Plot availability recomputed 2-4 times per build with a Python intersection loop
- Where: `reports/plot_data.py:333-370, 424-460, 531, 741, 1055, 1252`;
  `reports/workspace.py:1560, 2224`. Fix: vectorize `_numeric_intersection` and pass
  the availability down. Effort S-M.

### 6.3 The report plan is rebuilt for data bounds, recipe apply and save, then again for preview
- Where: `reports/workspace.py:1908-1914, 2380-2390`; `reports/workflow.py`.
  Fix: memoize the last plan on the setup signature. Effort S-M.

### 6.4 Preview and export rasterize every plot twice
- Where: `reports/workspace.py:360-414`; `reports/report.py:906-955, 1519-1526,
  1616-1620, 757-769`. Fix: render previews at export dpi and reuse them. Effort M.

### 6.5 The Python recorder re-joins and re-renders the whole script per recorded action
- Where: `scripting/recorder.py:195-207`; `ui/app.py:1252-1266`. Fix: cache the joined
  text, append deltas, track dirtiness by counter. Effort S.

### 6.6 Accurate PFA interpolation runs its 32-tap loop in float64 with per-tap coverage gathers
- Where: `isar/interpolation.py:93-110, 184-215`; `plotting/modes/isar_mode.py:766-785`.
  Fix: float32 mix, hoist the index arithmetic, compute coverage once. Effort M.

### 6.7 The interpolation plan cache hashes every block but never hits on realistic grids
- Where: `isar/interpolation.py:24-65, 184-215`. Fix: skip caching when the plan would
  not fit. Effort S.

### 6.8 Test-only numerics (low priority)
- `isar/repeats.py` widens stacks to complex128; `isar/bpde.py` re-checks finiteness
  per block and applies the operator twice per power iteration. Neither has a GUI
  caller yet.

## 7. User interface (auditor findings, spot-checked)

### 7.1 Every published row rebuilds the Reports and Assembly catalogs
- Where: `ui/dataset_actions.py` `_add_dataset_row` defaults to `notify=True`
  (1418-1421) and the multi-result publish loops call it once per row (decimate 4182,
  statistics 3151, align 3843, regrid 4090, mirror 4273, wrap 4374, shift 4464,
  offset 4886, time gate 4993, duplicate 5519, among others); `_set_dataset_row_saved`
  (1494-1496) notifies once per saved row. Each notify runs `ui/app.py`
  `_notify_dataset_catalog_changed` (2326-2339), then `reports/workspace.py`
  `set_dataset_catalog` (606-663) and `_refresh_availability` (1538-1613): a list item
  per row, plot availability recomputed over every checked grid, one combo item per
  common azimuth and elevation, one list item per common frequency with O(F x checked)
  float compares (1732-1755), plus the quadratic count of 1.4.
- Measured: one notify with two datasets of 3,600 azimuths costs 0.076 s; an operation
  over K rows pays it K times, after its progress bar reaches 100%.
- Fix: `notify=False` in the publish loops with one notify at the end (the batch-load
  path already does this at 906-914), a `notify` flag on the saved-row helper, and a
  zero-delay coalescing timer that only marks hidden Reports/Assembly tabs stale (the
  Reports tab already re-syncs on show at `app.py:2255-2258`; `test_gui_shell.py` 789
  and 814 query the catalog synchronously, so keep a synchronous resync when queried).
- Effort S for the loops, M for the coalescing.

### 7.2 Multi-row delete, drag reorder and recolor run the handlers once per row
- Where: `ui/dataset_actions.py:3234-3235` removes rows with table signals live;
  `ui/dataset_sidebar.py:121-151` `_reorder_to_drop`; `ui/dataset_actions.py:3752-3774`
  color actions; handlers connected at `ui/app.py:928-929`. The undo path already
  blocks signals (3256-3267).
- Cost: each removed row runs `_on_dataset_selection_changed`; when the current row
  lands on another selected grid it repopulates the parameter lists and scans
  availability for a dataset about to be deleted. Each reordered row triggers a full
  7.1 cascade.
- Fix: block table signals and updates around the loops and keep the explicit cleanup.
  Effort S.

### 7.3 The main window imports GHOST's feature workflow at construction
- Where: `ui/app.py:843-851` `load_ghost_module("feature_workflow", ...)`;
  `integrations/ghost.py:124-181` walks the GHOST tree in `ghost_module_paths`
  (`execution/diagnostics.py:524-542`, glob plus rglob) for the lookup and again in
  `_assert_loaded_module_origins`.
- Cost: the auditor timed 0.113 s of the 0.26 s constructor in this import (a
  4,694-line module with its geometry imports). The GHOST and FREDDY widgets already
  load lazily on first show.
- Fix: resolve the feature service on the Assembly tab's first show or from its build
  entry points; cache the module path map per backend. Effort S-M.

### 7.4 Polarization availability is rescanned on every dataset switch with no memo
- The scan of 2.4 also runs from `_populate_params` through
  `_apply_default_param_selection` (1812) on every selection: linear in cells (auditor:
  8 ms power-only, 15 ms with phase on a 230 MB grid; a full file read for disk-backed
  grids). Fix: memoize the masks per (grid object, polarizations, phase flag); edits
  replace grids rather than mutating them. Effort S.

### 7.5 Slider labels are rebuilt for every row on every selection change
- Where: `plotting/slider.py:42-62` builds `item(row).text()` for all rows;
  `ui/analysis_controls.py:417-425` copies the list but reads one label at a time.
  Auditor measured 34 ms per 100k rows plus 100k proxy objects per gesture. Fix: pass
  a row count plus a label callable. Effort S.

### 7.6 Per-row Python loops in the coordinate list helpers
- Where: `ui/coordinate_list.py:68-73, 253-254` (`.item()` per element);
  `_displayed_indices` turns the index array into a Python set
  (`dataset_actions.py:2017-2019`). Auditor measured 18 ms per 100k rows against
  0.9 ms with `tolist()`, 8-10 calls per dataset switch. Fix together with 1.2.
  Effort S.

### 7.7 Minor
- `ui/storage_usage.py:48-76` scans every cache folder synchronously in the dialog
  constructor and on Refresh; `ui/dataset_actions.py:1634-1648, 2067-2075` use list
  membership inside loops (matters past a few hundred rows). Effort S each.

## 8. Suggested order
1. Quick GUI-thread wins (all S): 1.2, 1.3, 1.4, 1.5, 7.1 (loops), 7.2, 7.4, 7.6,
   2.2 (timer), 5.2, 5.9.
2. Import and export throughput: 1.1, 1.6, 4.1, 4.2, 4.3.
3. Redraw cost: 2.1, 2.3, 2.5, 2.6.
4. Memory of derived datasets: 3.1, 3.2, 3.3, 3.4, 5.4, 5.5.
5. Startup: 7.3 after 1.5.
6. Assembly archive format and reports pipeline: 5.1, 5.3, 6.1-6.4.

## Appendix: benchmark numbers (this machine, synthetic data)

| Operation (2.9M cells unless noted) | Time | Peak RAM |
|---|---|---|
| write_flat_csv, 289k rows | 5.84 s | 1.7 MB |
| load_dataset flat CSV, 289k rows | 5.32 s | 11.5 MB |
| save_pio | 0.08 s | 42 MB |
| load_dataset PIO | 0.02 s | 14 MB |
| native save compressed | 1.36 s | 64 MB |
| native load | 0.15 s | 93 MB |
| native load, disk mode | 0.19 s | 4 MB |
| native load, azimuth 0-90 selection | 0.18 s | 14 MB |
| convert_axis_units | 0.004 s | 46 MB |
| axis_crop half azimuth | 0.03 s | 23 MB |
| coherent_add | 0.22 s | 62 MB |
| statistics median over azimuth | 0.03 s | 0.6 MB |
| heatmap azimuth by frequency, dB | 0.03 s | 61 MB |
| heatmap azimuth by frequency, phase | 0.17 s | 107 MB |
| first legacy rcs_power access | 0.03 s | n/a |

| GUI thread | Time |
|---|---|
| import GRIM_Backend.ui.app | 1.02 s |
| construct main window | 0.38 s |
| add row, 2.9M cells | 0.08 s |
| select row, 2.9M cells | 0.08 s |
| select row, 100k azimuths | 1.63 s |
| select previous row again | 1.70 s |
| azimuth rect plot, 128 curves | 0.83-1.25 s wall, 0.43 s on the GUI thread |
| waterfall, 2.9M cells | 0.29-0.56 s |
| azimuth rect plot, 100k azimuths | 2.18 s |
