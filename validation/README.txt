GRIM VALIDATION EXAMPLES

Open GRIM_validation_simple.xlsx. There is one sheet per operation: 46 sheets
with 353 selected comparisons, plus an index.
The previous detailed workbook, if present locally, is excluded from version
control. The simplified workbook is the current deliverable.

Yellow cells hold raw data and settings. Blue cells contain live Excel formulas,
beside the saved GRIM result and percent error. Each sheet identifies the formula
cells. Editing inputs changes Excel calculations, not the saved GRIM exports.

Percent error is |GRIM - Excel| / |Excel|. A zero reference is n/a. RCS dB results
are compared in equivalent linear power. Phase uses a separate wrapped error in
degrees; an angle has no percent error. Missing data remains distinct from zero.

REPEAT AN EXAMPLE
Load the native .grim inputs from datasets/<operation>/, follow its GUIDE.txt,
then compare the output with the worksheet. Use .grim inputs rather than CSV
copies so declarations such as range-phase and SENTRi conventions are preserved.
The folders retain 165 native files and 194 CSV exports, including the full
input/output evidence. GUIDE.txt maps each displayed comparison to its export.

The compact workbook shows selected examples. The full specs retain 2,197
recorded comparisons; they are not all recalculated by this workbook. Scope
limits are in each GUIDE.txt. BETA labels remain unchanged. GUI interaction and
full GHOST, FREDDY, and Assembly physics/geometry are outside this workbook.

REGENERATE (from the repository root)
  python validation/scripts/generate_all.py
  node validation/scripts/build_workbook.mjs --render
  powershell -File validation/scripts/validate_excel.ps1
  python validation/scripts/package_report.py
Use the GRIM Python environment, Microsoft Excel, and the bundled Node runtime
with @oai/artifact-tool. If the excluded validation/.work/node_modules link is
absent, this workstation can locate the library with:
  $env:NODE_PATH = 'C:/Users/14col/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
  $env:Path = 'C:/Users/14col/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin;' + $env:Path

See validation_report.txt for the current Excel result and manifest.json for
file checksums. All fixture values are synthetic.
