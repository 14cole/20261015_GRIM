# FREDDY

FREDDY evaluates one-dimensional, infinite-planar material stacks. It computes
TE/TM reflection, transmission, absorption, and front-face input impedance for
frequency-dependent dielectric/magnetic layers and zero-thickness resistive
sheets. Layers are ordered from the incident side to the backing.

FREDDY does **not** calculate finite-object radar cross section or dBsm. Its
metal-backed impedance export can be used as a planar equivalent boundary
condition by the companion GHOST RCS solvers when that physical approximation
is appropriate.

## File Converter

Choose **File → File Converter…** in FREDDY (standalone or embedded in GRIM).
Open or drop a CSV, TXT, DAT, ASC, ASCII, or TSV table. The preview supports
comma, tab, semicolon, and whitespace delimiters, optional headers, skipped
preamble lines, UTF-8/BOM and Windows-1252 text, and Fortran `D` exponents.
Blank lines and full-line `#`, `%`, `!`, and `//` comments are ignored.

Each output row lets you label a variable, choose its source column or a
constant for every row, and select its input/output units. Uncheck unwanted
columns. Frequency conversion supports Hz, kHz, MHz, GHz, and THz in both
directions; angles support degrees/radians, with length, time and impedance
units also available. **As written** preserves a column without conversion.
No frequency unit is inferred from the size of a number.

Use **Preview conversion**, then **Convert and save…** to write a new table.
Converted headers include units, such as `frequency_ghz`. All rows are checked
before the output is published; a failure leaves an existing output untouched.
The source cannot be used as the destination. For a FREDDY material file,
label columns `frequency`, `eps_real`, `eps_imag`, `mu_real`, `mu_imag`, choose
Hz output for frequency, and select comma output. Missing relative permeability
columns can be constants `1` and `0`. The resulting header is
`frequency_hz,eps_real,eps_imag,mu_real,mu_imag`.

## Launch

FREDDY is available as the **FREDDY** tab in the main GRIM window. From the
repository root, install the integrated application and run it with:

```text
py -3 -m pip install -e .
grim
```

The embedded tab and standalone window use the same authoritative code in
this directory. Inverse Design supports **Stop and keep best** and resume;
Material Mix supports stopping with incomplete results discarded.
**Thickness** supports stopping while retaining the previous completed study. Ordinary
analysis/export jobs run to completion. GRIM prevents closing while a job is running.

**User-defined constant layers:** choose **Add Layer → Material source →
Constant εr / μr (all frequencies)**. Enter the four relative-property components
(εr real/imaginary and μr real/imaginary) and thickness. For example, εr = 6 − 0.8j
and μr = 1 − 0.1j means inputs `6`, `-0.8`, `1`, `-0.1`. Negative imaginary values
represent passive loss. Values must be finite and the complex ε/μ magnitudes
must be nonsingular, using the same validation as material CSVs. These isotropic
values are applied directly at every frequency and saved in the project; no
temporary material CSV is created. Constant and measured layers can share a
stack. Impedance, Off Angle, Thickness, IBC Batch, inverse design, and coating
checks support both. Material Explorer continues to inspect measured CSVs.

**About & Guide** is the single home for application help, angle and polarization
conventions, material definitions, and result interpretation. Search its topic
list for a plot or task. Eight mode-specific workflows explain setup, the
expected response, how to make a decision, and the next step. Plot catalogs
cover sweep/IBC views, inverse candidates, blend targets, and Material Explorer.
**F1** opens the active mode's workflow; **Explain view** in sweep and inverse
Results opens its plot catalog. Workflow links open Setup while preserving
inputs and completed results. The guide fills the workspace without layer
controls or plots. File and View actions remain in the menu bar.

Analysis Results also offer **Export comparison CSV…** in the plot toolbar.
The report includes every sampled thickness/angle, reflection basis and bound,
target/band, passing bandwidth and coverage, worst reflection, passing margin,
full-sweep sampled null frequency, output path, and captured run/stack context.
The sortable **Margin (dB)** column is target minus worst reflection: a
nonnegative value passes the entire selected band. Span displays compare nominal
reflection. These reports are analysis records, not nominal IBC inputs. Figure
display reduction and table sorting do not change exported samples or row identity.

Dense Off Angle and Thickness sweeps retain NumPy grids and update tolerance
bounds in place instead of repeatedly boxing arrays into Python lists. Shared
material paths are read once per stack load and refreshed on each later run.
Unchanged comparison tables and nominal band metrics are reused during plot
changes. The public computation API still returns lists unless array output is
explicitly requested.

For a standalone window on Windows, double-click `Launch_FREDDY_GUI.bat` or
run the following from this directory:

```text
py -3 -m pip install -r requirements.txt
py -3 impedance_gui.py
```

`FREDDY_ROOT_PATH` is an optional GRIM development override that can point the
embedded tab at another FREDDY root. The Windows launcher first uses the
repository-root `.venv`, then an active `VIRTUAL_ENV`, then a system Python;
FREDDY does not require a separate environment under `tools/FREDDY`.

FREDDY output is CSV, not `.grim`: save a three-column IBC impedance file or a
five-column material file for GHOST. In integrated GRIM, **Export and attach to
current GHOST geometry** accepts only a nominal PEC-backed IBC or nominal
material export. GHOST must already have an active saved/loaded `.geo`; the
handoff validates and copies the file beside that geometry, then you press
**Save Geometry** to persist its reference. Off-angle, thickness, uncertainty,
and other analysis CSVs cannot be attached through this action. GRIM
deliberately does not load any FREDDY CSV into its RCS dataset table.

## Demo material presets

**Add Layer → Preset material → Use** and Material Mix's component dialog
include four synthetic examples alongside the Air reference:

| Preset | Useful for testing | εr model (high, strength, relaxation GHz) | μr model |
| --- | --- | --- | --- |
| Demo: Low-loss dielectric | Dielectric spacing and thickness resonances | (2.1, 0.1, 20) | 1 |
| Demo: High permittivity | Stronger impedance mismatch and thinner resonances | (8, 2, 30) | 1 |
| Demo: Lossy dielectric | Absorption and multilayer matching | (3, 7, 8) | 1 |
| Demo: Magnetic loss | Combined electric and magnetic loss | (2.5, 1.5, 12) | (1, 2, 6) |

These are illustrative passive Debye models, not measured commercial materials.
For each parameter tuple, the response is `high + strength / (1 + j*f/relaxation)`
with frequency in GHz, using FREDDY's `exp(+jωt)` convention (negative imaginary
parts indicate loss). CSVs cover **1 MHz–1 THz**, with 80 intervals per decade;
FREDDY interpolates their stored samples. Regenerate them with
`python tools/generate_demo_materials.py` from the FREDDY folder.

For a quick multilayer trial, add a low-loss layer and a lossy or magnetic layer,
then set each layer's thickness minimum, maximum and step in **Thickness**.
Import that stack into **Material Explorer** to compare its material properties.

## Material Explorer

Use **Material Explorer** to compare any number of validated five-column
material CSVs without changing the FREDDY layer stack or running a solver. Add
files directly, drag them onto the explorer, bring in the current stack or
Material Mix inputs, or add the bundled Air reference. The workspace plots the
stored real and imaginary relative permittivity and permeability on each
file's native frequency grid. It also provides a compact range/coverage table
and a lazy raw-value table that remains responsive with large files. A
same-frequency view compares every material in one row-oriented table;
between stored samples it uses component-wise linear interpolation and marks
out-of-range files instead of extrapolating.

Explorer sources are session-only: adding, removing, or reloading them does not
dirty a FREDDY project and cannot emit a GHOST attachment. Changed or missing
source files are marked; **Reload selected** rereads them explicitly and keeps
the last valid cached data if a reload fails. The two loss tangents shown in
the tables are clearly labeled derived values and use `-imaginary/real`; no
other material properties are inferred. Density and conductivity are not in
the FREDDY material schema and therefore are not guessed by the explorer.
Curves above 5,000 stored rows use display-only extrema-preserving decimation;
the complete values remain available in the raw table.

## Portable project files

FREDDY JSON projects store layer materials, Material Mix files, and output
locations relative to the project JSON when they are in its directory tree or
one neighboring directory. Moving that folder structure to another machine
therefore keeps those links intact. More distant or cross-volume absolute paths
remain absolute, are listed in the JSON's `path_portability` metadata, and
produce a warning when loaded because the external file must be copied or
reselected separately. Project and nominal solver-facing CSV writes are
same-directory atomic, so a failed write does not truncate an existing file.

## Material CSV format

Material inputs and mixed-material exports use exactly five comma-separated
columns with a required header:

```csv
frequency_hz,eps_real,eps_imag,mu_real,mu_imag
1000000000,3.2,-0.15,1.0,0.0
```

Use a `.csv` extension. Both tools accept UTF-8 with or without a BOM, blank
lines, and full-line `#` comments. Column order, capitalization, spaces and
brackets may vary; for example `Frequency (GHz),EPS REAL,EPS IMAG,MU REAL,MU IMAG`
is accepted. Each required physical quantity must occur exactly once, and the
frequency header must specify Hz, kHz, MHz or GHz. Space/tab-separated and headerless
files are rejected. See the [shared file format](../GHOST/MATERIAL_CSV_FORMAT.md).

Exports use Hz; input frequency units are converted explicitly. FREDDY and GHOST use the `e^(+j omega t)` convention, so a
passive lossy material has negative imaginary permittivity and permeability.
Positive imaginary values are rejected as active/gain media. Frequencies must
be positive and unique; interpolation is linear in the real and imaginary
property components and extrapolation is not performed.

## Impedance CSV format

Nominal impedance exports use the following canonical header, comma separator,
Hz frequency and impedance in ohms. GHOST also accepts equivalent presentation,
column order and explicitly declared frequency units as described above:

```csv
frequency_hz,resistance_ohm,reactance_ohm
1000000000,120,15
```

Uncertainty envelope estimates are written to a separate `_uncertainty.csv` analysis file
so the nominal file remains directly readable by GHOST. The seed envelope is the
min/max over sampled tolerance cases: the corners of the ±δ thickness/ε/μ box
plus 9 evenly spaced points along each scale axis across `[-δ, +δ]` (the other
two scales nominal). Impedance exports and plots additionally refine sampled
axis extrema and susceptance zero crossings at each frequency. This catches
narrow impedance resonances between seed points; resistance, reactance and
reflection extrema are searched on both sides. Refinement is capped at 1,024
additional evaluations per frequency; the report identifies incomplete searches.
One scale per axis applies to all layers together (fully correlated).
Off-axis interior extrema and unbracketed resonances can still be missed:
these are sampled estimates, not certified worst-case bounds. Leading `#`
lines in the sidecar state the seed count, refinement work and limitations.
The fixed seed case set (without impedance-specific adaptive refinement) drives
Off Angle/Thickness tolerance envelopes and the Inverse Design and Material
Mix worst/average tolerance scores.

Free-space (air) backed exports are planar analysis results, not one-sided
IBCs. They are written with a leading `#` note and the distinct header
`frequency_hz,free_space_backed_resistance_ohm,free_space_backed_reactance_ohm`,
which GHOST's IBC loader refuses; PEC-backed output keeps the header above.

Phase uncertainty
bounds are unwrapped about the nominal phase and can therefore lie outside
`[-180, 180]`; this avoids false 360-degree spans at the phase branch cut.

## Exhaustive individual and combined thickness studies

**Thickness → Setup** is the single home for thickness sweeps and tolerance
analysis. Check **Vary** on each selected bulk layer and enter **Minimum,
Maximum, and Step**, in inches. Ranges are shared with Inverse Design and can
also be changed in the layer editor. Unchecked layers stay nominal. Set the
frequency and incidence-angle sweeps, polarization, and PEC reflection target.
Angles are measured from the surface normal. The former standalone Sensitivity
only, Sensitivity + statistical trials, and thickness-envelope modes have been
consolidated; there are no sampling-count controls in this workflow.

Every minimum + k × step value not exceeding maximum is evaluated. An off-step
maximum is excluded: 0.090–0.110 in with step 0.006 gives 0.090, 0.096, 0.102,
and 0.108 in. The table shows the count and actual last value. Equal limits
define one value. Three layers with 3, 4, and 5 values give 60 joint stacks.
The calculation covers every layer individually (others nominal), plus every
joint combination, reusing shared evaluations. Material properties, sheet
resistances, and layer order stay nominal. Manufacturing distributions do not
restrict this grid. Nominal is a separate reference and is included in bounds
only when it belongs to the defined grid; it can lie outside the bounds.

Results include:

- **Thickness envelopes:** pointwise minimum/maximum PEC reflection versus
  frequency. More negative means less reflection. Choose a polarization and
  one angle, or pool all selected angles. Click a point to inspect the actual
  thickness combinations producing its bounds.
- **Whole-band best/worst stacks:** realizable stacks ranked by their largest
  reflection across all selected frequencies, angles, and polarizations. The
  best minimizes that peak; the worst maximizes it. The plot displays those
  fixed stacks at the chosen angle/polarization, or their worst-angle curves.
  An envelope can use a different stack at every point; these curves cannot.
- **Layer sensitivity:** largest whole-band margin loss from varying each
  layer individually relative to nominal. This is not a variance decomposition.
- **Layer details:** the existing single-layer heatmap, frequency curves,
  phase, power balance, bandwidth and slice tools, using captured results at
  the chosen angle. Other layers remain nominal.

A passing grid case meets the limit at every selected operating point. Counts
are not manufacturing yield. Requirement comparisons allow 1e-10 dB for
floating-point roundoff; the calculated reflection values remain unchanged.
Every discrete grid case is evaluated, but extrema between thickness,
frequency, or angle steps are not certified. Refine the steps to check narrow
resonances. Preflight reports the work and rejects more than 250,000 joint
combinations, 50 million response points, or excessive retained array storage;
it never silently samples or skips cases. Individual metric grids are retained
for the detailed viewer, while joint responses are reduced as they are computed.

**Export envelope CSV** includes every panel, polarization, angle, and frequency,
with the actual thickness combinations attaining its lower and upper bounds.
When both polarizations were analyzed, it writes separate `<name>_TE.csv` and
`<name>_TM.csv` files. Each contains only its named polarization, including all
individual/combined panels and bound witnesses. A single-polarization run keeps
the chosen filename. JSON remains one complete report containing both.
**Export study JSON** includes captured settings, layers, individual metrics,
case indices, bounds, best/worst stacks, method version and a fingerprint of
the material values actually loaded. Ties use the lowest captured case index.
Editing inputs does not relabel results. Cancellation or error keeps the previous
completed study. Projects save inputs, and loading a project clears results.

## Inverse manufacturing model

Inverse Design shows its frequency, incidence-angle, objective and tolerance
settings directly. The combination count updates as inputs change and shows
the total response points to evaluate. **Keep best for comparison** controls
storage only; every defined design is still evaluated. Layer Edit, Thickness
and inverse layer selection share the same **Vary / Minimum / Maximum / Step**
controls. Unchecked Vary means Fixed; inactive range fields are ignored.

IBC Batch's **Use current layer range** copies the selected layer's range in
the chosen inch/mm units.

Results offer upper/lower bounds only when those bounds were captured for the
selected metric and polarization. Tolerance span/envelope requires both bounds;
nominal-only results do not imply zero uncertainty. The editable **Comparison
limit** affects display comparisons, not a captured inverse ranking or study.

The separate distribution-based inverse objectives and candidate verification
remain available under **Inverse Design → Edit per-layer tolerance bounds**.
They support thickness, sheet resistance, and signed ε/μ component bounds,
uniform or truncated-normal distributions, and shared Gaussian factors.
These assumptions estimate modeled yield; they do not change the exhaustive
Thickness grid. Old project layer tolerance metadata is preserved. Retired
standalone sample-count and seed settings do not affect Thickness runs.

## Thickness-batch IBC export

Use **IBC Batch** to write one nominal solver-compatible IBC CSV for each
requested thickness of one material layer. Choose the layer, thickness
start/stop/step, and `in` or `mm`; the default 0.015-to-0.030 inch sweep
uses a 0.001 inch step and writes `ibc_0p015in.csv` through `ibc_0p03in.csv`.
Saved projects retain inches or millimeters. The Impedance frequency sweep is shared
with this mode. Every batch output is broadside and PEC-backed, and all other
layers, material tables, and stack ordering remain unchanged.

The preflight line shows the exact file count and endpoint names before the
run. Existing destinations require one confirmation for the complete set.
FREDDY computes and stages the complete set before publishing it, and restores
prior files if publication fails, so a partial batch is not left behind. A
multi-file batch is deliberately not auto-selected for GHOST; attach the CSV
for the desired thickness explicitly.

IBC Batch now opens **Results** after export. Compare reflection curves,
frequency/thickness maps, passing bandwidth, coverage, resistance/reactance,
and the sampled frequency of minimum reflection. The optional table links each
thickness to its output file. **Use selected IBC for GHOST** verifies the chosen
file against the exported result before enabling the existing GHOST handoff.
Results are published only for a completed batch; changed output files cannot
silently replace the data shown in its plots.

## Analysis results workspaces

**Impedance, IBC Batch, Thickness, and Off Angle** have separate Setup/Results
tabs, large plots, optional comparison tables, and run-context image exports.
Each mode retains its own last successful run. Editing setup does not relabel
the cached results; loading a project clears the session's result caches.

Thickness and Off Angle add reflection threshold contours, multiple frequency
overlays, nominal/worst-case passing bandwidth and coverage, tolerance envelopes,
and selected-frequency slices. Thickness also tracks the sampled null location.
A user-selected comparison band reports passing sampled thickness/angle ranges.
Bandwidth interpolates frequency samples without extrapolation, and grouped
passing samples do not guarantee untested intermediate conditions.

Hovering a sweep, TE/TM, coating-error, inverse angle, or sensitivity failure
map shows `(x, y, z)` in the plot toolbar, where z is the sampled value of the
cell under the cursor. Clicking a sweep map snaps **Selected** and **Slice GHz** to
the nearest computed samples, marks them with dashed crosshairs, and the toolbar
keeps reporting that `Selected (x, y, z)` point beside the hover readout.

Maps have a **Color scale** row that is automatic by default. Clearing **Auto**
fixes the displayed range. Enter round **Min**/**Max** values: later runs keep
the scale, so maps of two stacks match in a presentation. Each metric and its
tolerance span keep separate limits, so a dB scale is never applied to phase.
Fixed scales are saved with the project; a project saved without one keeps the
current session's scales. Colorbar arrows mark values clipped to fixed limits.
The inverse angle map has the same row. The Material Mix performance map uses
the **Auto color scale**, **Min** and **Max** fields above it; press Enter in
Min or Max to apply typed limits.

Off Angle uses one **Polarization: TE / TM / Both** selector. Both computes two
responses and writes `<name>_TE.csv` and `<name>_TM.csv`, with the same metric
columns and full sample grids. Results compare them on a shared map scale.
Comparison CSV exports also include every captured polarization in separate
named files, regardless of which polarization is displayed. Existing destinations
are checked before writing. Unsupported oblique TM for directional materials is
rejected explicitly; select TE for the supported directional calculation.
Broadside scalar IBC exports remain solver-compatible three-column files: those
workflows compute one normal-incidence impedance, not two oblique polarizations.

Impedance displays resistance/reactance, reflection, tolerance bounds, and
nominal reflected/absorbed/transmitted power for its selected backing. The
GHOST coating check supplies TE/TM approximation-error charts and maps, with
the original report under Run details. It remains a planar scalar-IBC check,
not finite-body RCS certification.

See [Workflow shortcuts](../../WORKFLOW_GUIDE.md) for view definitions and the
file-selection workflow. Frequency overlays above 5,000 points retain extrema
while reducing display points only; exported values and band metrics are intact.

## Inverse design: analyze all combinations

**Fixed / variable layers…** defines fixed values or finite minimum/maximum/step
ranges. **Analyze all combinations** evaluates their complete Cartesian product
in a stable order. The setup displays the exact combination count, per-layer
value count and sampled limits, and frequency/angle/tolerance workload. Values
start at Minimum and advance by Step without exceeding Maximum; an off-step
Maximum is excluded. A varying range needs an explicit step.

The design grid is always exhaustive. Joint tolerance sample counts and seeds
control the uncertainty analysis, not how many designs are searched. Legacy
design-search budget/refinement settings remain ignored. **Keep best for
comparison** limits displayed candidates only, not evaluations. **Stop and keep
best** retains complete scores and marks an interrupted analysis incomplete.
**Resume remaining** finishes the same grid without rescoring completed designs,
or finishes interrupted plots. It is disabled once the run is complete. Inputs
and material contents must still match. For recovery after closing the
application, choose an optional **Recovery file** before starting. Completed
scores are saved about every 30 seconds at combination boundaries and when
scoring stops or completes. **Choose / save...** can also save an idle search.
Reopen the matching saved project, choose **Load checkpoint...**, then
**Resume remaining**. Clearing the recovery field disables disk saves.
Starting a fresh analysis with an existing recovery path selects a new
`-fresh-...fsearch` file automatically and leaves the existing file intact.
The Recovery file field shows the new destination. A loaded checkpoint can be
copied with **Choose / save...** before Resume rebuilds its plots.
Recovery archives contain scores and identity checks, not a project or material
copies; changed code, changed inputs, and corrupted scores are rejected. Plots
are rebuilt after loading without rescoring completed combinations. Recovery
files are limited to 512 MiB of scores; larger grids can still run with the
optional recovery field cleared.
Saved candidates receive separate project and output destinations. See the
[workflow guide](../../WORKFLOW_GUIDE.md) for examples. Material Mix retains its
separate recipe-search behavior.

Inverse Design now separates **Setup** and **Results**. Results opens after a
run and provides a full-width plot with three views: reflection overlays,
null depth versus passing bandwidth, and analysis history. The
candidate table can be shown for sorting and choosing overlays, or hidden for
more plot height. The selected candidate can be applied or saved from Results;
sorting does not change its identity. The plot toolbar exports the current view.

An adjustable reflection target (initially −10 dB) supports comparison of deep
narrow nulls against broader responses. The default curve is the pointwise
worst analyzed angle/tolerance case; analyzed-point percentiles remain available.
Band coverage and the widest contiguous passing interval are estimated by
linear interpolation in dB on the sampled sweep, without extrapolation.
Discrete targets report point coverage only, with no inferred bandwidth.
These are display metrics for retained candidates; the search continues to
rank its captured objective. Increase **Keep best** to review more
alternatives and use a finer frequency sweep to verify narrow features.

Analysis history shows every completed combination and the running best score,
including work preserved by Resume. A complete run finds the best selected
objective on the specified finite grid, not between grid values or under
untested conditions. A partial run is explicitly incomplete.

New workspaces default to **Whole-band PEC reflection requirement (worst point)**.
For a requirement over the full requested region, select this Score and enter **Reflection Target
(dB)**. The objective minimizes `max(reflection_dB) − target_dB` across every
requested frequency, incidence angle, and nominal/tolerance case. Lower is
better; **Gap ≤ 0 passes**, and **Req. margin = −Gap**. A flat −11 dB design
therefore beats a −40/−1 dB narrow-null design against a −10 dB limit, even though
the narrow-null design has a better mean. The original mean objectives remain
available; existing projects preserve their saved objective.

Choose **Both** to rank each design against TE and TM together. Whole-band and
worst-gap objectives take the worst of all requested polarizations; legacy mean
objectives average both. Comparison curves/maps use the worse polarization at
each angle/case, including their nominal curves and point percentiles.

Two **Per-layer tolerances** objectives use the bounds entered through **Inverse Design → Edit per-layer tolerance bounds** directly during exhaustive inverse ranking. Selecting either automatically disables shared stack variation, since these are alternative uncertainty models.

- **Worst sampled reflection gap** checks nominal, individual parameter ends,
  all box corners for up to eight parameters (the two all-low/all-high corners
  otherwise), and joint Sobol interior samples. It minimizes the worst sampled
  reflection minus the limit. These rectangular support probes ignore group and
  distribution restrictions; they are conservative scenarios, not certified bounds.
- **Modeled failure fraction (%)** minimizes the percentage of statistical
  trials missing the limit at any sampled frequency, angle or polarization.
  It uses the entered bounded distributions and groups. All designs share the
  same normalized trials for a fair comparison. Nominal is shown separately
  and does not enter the failure percentage. A zero score is not a guarantee.

Bounds, distributions, groups, sample count and seed are captured in search
identity; changes invalidate resume. Percentage bounds follow each candidate's
nominal thickness/resistance. Absolute bounds must remain physically valid at
the minimum allowed design value. New numerical implementation versions also
invalidate older checkpoints; saved projects remain loadable.

Use **Results → Verify candidates…** to check the selected candidate or all
retained candidates independently. Set verification samples/seed, independent
yield repeats and frequency/angle refinements in that dialog; only applicable
settings are shown. Ranking shows samples/seed only for manufacturing objectives.
Unused sampling inputs do not block a nominal search. Verification bisects every frequency/angle
interval at each configured level and doubles joint interior tolerance samples,
while retaining previous samples, nominal and boundary probes. Discrete targets
remain discrete in frequency; angle intervals are still refined. It uses either
per-layer bounds or shared systematic scales and refuses ambiguous simultaneous
models. For per-layer bounds it also estimates yield using fresh independent
scrambles on the finest grid. Verification seeds differ from the ranking seed.

The comparison table sorts by verified worst margin or modeled yield. It shows
the final refinement change, repeat range and sampling standard error. A
"sampled stable" result requires two successive changes of at most 0.05 dB;
this is evidence of numerical stability, never a continuous guarantee. The
details identify the worst sampled frequency, angle and polarization.
**Export verification JSON** saves the designs, bounds, seeds, refinement history,
worst condition and provenance. Stop discards incomplete verification and keeps
the previous captured report. Settings persist in projects; verification results
are exported separately. Verification compares retained candidates only and
does not claim to find an optimum among discarded designs.

The captured search requirement appears in Results. Its gap and margin remain
fixed when the display target or percentile changes. Discrete targets apply the
limit at each requested point, without claiming an intervening band. Refine the
frequency/angle grid to verify between samples. Search recovery and candidate
application/save validate the objective, active limit, and constant properties;
changed inputs require a fresh search. Recovery files from older solver code
are rejected by the existing implementation fingerprint check; saved projects
remain loadable.

The selected candidate also has an angle/frequency map and a tolerance envelope
at a selected angle. The run retains explicit angle/tolerance labels for these
views; later setup changes cannot change their interpretation.

Combinations are generated by index without allocating all proposals. Five
scores per evaluated design are retained in a compact numeric array; full
response curves are computed only for the retained candidates. Material and
wave-term preparation is reused across the grid. Equal scores retain stable
grid order, making identical runs reproducible without a random seed.

## Numerical scope

- Incidence angle is measured from the surface normal and must satisfy
  `0 <= angle < 90 degrees`.
- Use TE/TM labels. HH=TE and VV=TM are retained only as legacy aliases for a
  vertical plane of incidence.
- Directional materials are supported only on their measured 0- or 90-degree
  principal axes. Arbitrary tensor rotation and cross-polarization require a
  full anisotropic field solver.
- Air-backed impedance is a planar analysis result, not generally a valid
  one-sided boundary condition for a closed transmitting body.
- Effective-medium rules are morphology-dependent approximations. The GUI
  reports their assumptions and should not be treated as a substitute for
  measured mixture properties.

## Material Mix workflows

The Material Mix tab supports three related jobs:

- Predict effective frequency-dependent ε and μ from a known volume recipe.
- Search bounded volume fractions for a target ε/μ curve or constant.
- Search bounded volume fractions and a specified layer thickness for a
  reflection, absorption, or transmission requirement across a frequency and
  incidence-angle grid.

Recipe samples are evaluated as they are generated; at most 100 candidates are
retained for comparison. Fixed bounds defining one recipe evaluate it once.
The setup shows the sample budget plus up to 300 additional refinement
evaluations per retained candidate. **Stop search** cancels sampling, refinement,
or result preparation without publishing an incomplete result. Changing inputs
while a search runs invalidates its eventual result. A blank **Seed** draws a
random seed for that run; the result message reports the seed actually used,
so entering it reproduces the search.

**Export selected CSV** and **Add selected as layer** use the properties,
frequency grid, and thickness captured in the displayed result. They do not
recalculate from material files that may have changed on disk. Recalculate
explicitly to analyze updated material data. Project loading rejects malformed
recipe entries before changing the open study.

Performance targets can use PEC or air backing and TE or TM polarization. A
candidate's requirement gap is evaluated at every requested frequency/angle
point: for an upper limit the gap is `max(value) - target`, and for a lower
limit it is `target - min(value)`. A gap at or below zero passes. When material
or thickness uncertainty is enabled, the results separately report the worst
uncertainty-case gap (box corners plus per-axis samples, as described under
the impedance uncertainty report); this is the pass/fail value even if
average-case scoring was selected for ranking.

The predicted or optimized effective properties still depend on the selected
mixing law and its morphology assumptions. Performance optimization does not
remove that limitation; validate a promising recipe with measured mixture
data before treating the result as a manufactured-material specification.

### Public-data validation pack

`materials/validation/nist_bam_pdms` contains unmodified, checksum-verified
NIST broadband BaM/PDMS source tables, deterministic conversion to FREDDY's Hz
and negative-loss-imaginary convention, and solver-level forward/inverse
Maxwell-Garnett regressions. Rebuild and validate it from the repository root:

```text
py -3 tools/FREDDY/tools/convert_nist_bam_pdms.py
py -3 tools/FREDDY/tools/validate_material_mix.py
```

The converted files can also be selected directly in the Material Mix tab as
normal FREDDY material inputs or targets. See
`materials/validation/nist_bam_pdms/README.md` for provenance, exclusions, and
the numerical acceptance limits.

## Tests

From the `tools/FREDDY` directory:

```text
py -3 -m unittest discover -s tests -v
```

The regression suite covers CSV sign/units, reference slab and sheet cases,
TE/TM power conservation, causal negative-index branches, layer ordering,
scalar/vector equivalence, stable thick-loss transmission, phase wrapping,
effective-medium formulas, deterministic public-data conversion, and measured
forward/inverse material-mix validation.
