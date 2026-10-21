"""Inventory the compact workbook after its native Excel recalculation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'validation'
specs = sorted((json.loads(p.read_text(encoding='utf-8')) for p in (V / 'specs').glob('*.json')),
               key=lambda x: x['title'])
compact = {p.stem: json.loads(p.read_text(encoding='utf-8'))
           for p in (V / 'compact_specs').glob('*.json')}
layouts = {item['slug']: item for item in json.loads(
    (V / '.work/workbook_layout.json').read_text(encoding='utf-8'))}
results = json.loads((V / 'excel_validation_results.json').read_text(encoding='utf-8-sig'))
bytitle = {s['sheet']: s for s in results['operations']}
assert set(bytitle) == {s['title'] for s in specs}
assert set(compact) == set(layouts) == {s['slug'] for s in specs}
for s in specs:
    slug = s['slug']
    chosen = compact[slug]['selectedCheckIndices']
    assert [c['originalIndex'] for c in layouts[slug]['checks']] == chosen, slug
    assert bytitle[s['title']]['checks'] == len(chosen), ('stale Excel report', slug)
checks = sum(len(c['selectedCheckIndices']) for c in compact.values())
full_checks = sum(len(s['checks']) for s in specs)
failures = sum(len(s['failures']) for s in results['operations'])
assert results['recalculated']
assert not results['formula_errors'], results['formula_errors']
assert all(x['passed'] for x in results['dynamic_checks'])

for s in specs:
    slug = s['slug']
    c = compact[slug]
    layout = layouts[slug]
    for f in s['files']:
        for key in ('native', 'csv'):
            if f.get(key):
                assert (V / f[key]).is_file(), f[key]
    guide = [s['title'], '-' * len(s['title']), '',
             'WORKBOOK EXAMPLE', c['settings'],
             f"The worksheet shows {len(c['selectedCheckIndices'])} selected comparisons.",
             'Yellow cells contain raw inputs. Blue cells calculate the Excel result beside the saved GRIM result.',
             'Changing the inputs recalculates Excel only; rerun GRIM to create new outputs.', '',
             'REPEAT IN GRIM',
             'Load the .grim input files, not their CSV copies. Native files retain conventions that CSV may omit.',
             s['settings'],
             'Result files are reference outputs, not additional operands. Match coordinates and polarization when comparing exports.',
             'Plot CSV files record the numerical plot values.', '',
             'WORKSHEET EVIDENCE',
             'Original check indices are zero-based positions in specs/' + slug + '.json.']
    for check in layout['checks']:
        guide.append(f"{check['label']}: Excel {check['expectedCell']}; GRIM {check['actualCell']}; "
                     f"percent error {check['errorCell']}; original check {check['originalIndex']}; "
                     f"source {check['source']}")
    guide += ['', 'FILES']
    guide.extend(f"{f['name']}: " + (f.get('native', '') + '; ' if f.get('native') else '') + f['csv']
                 for f in s['files'])
    guide += ['', 'SCOPE', c['scope'],
              f"The full specification retains {len(s['checks'])} recorded checks. Only the selected comparisons are recalculated in this workbook.",
              '', 'RECORDED NOTES', *(s['notes'] or ['None.'])]
    (V / 'datasets' / slug / 'GUIDE.txt').write_text('\n'.join(guide) + '\n', encoding='utf-8')

native = list((V / 'datasets').rglob('*.grim'))
exports = list((V / 'datasets').rglob('*.csv'))
readme = f'''GRIM VALIDATION EXAMPLES

Open GRIM_validation_simple.xlsx. There is one sheet per operation: {len(specs)} sheets
with {checks} selected comparisons, plus an index.
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
The folders retain {len(native)} native files and {len(exports)} CSV exports, including the full
input/output evidence. GUIDE.txt maps each displayed comparison to its export.

The compact workbook shows selected examples. The full specs retain {full_checks:,}
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
'''
(V / 'README.txt').write_text(readme, encoding='utf-8')
report = [f'GRIM compact validation: {len(specs)} operations; {checks} selected comparisons',
          'Workbook: GRIM_validation_simple.xlsx.',
          f'Engine: {results["engine"]}; current workbook fully recalculated.',
          f'Failed comparisons: {failures}; formula errors: {len(results["formula_errors"])}.',
          f'Full evidence: {full_checks} recorded checks retained in specs; not all are displayed or recalculated here.',
          'No production code or BETA labels changed. See README and operation GUIDE files.', '',
          'Excel behavior checks']
report.extend(f"{item['test']}: {'passed' if item['passed'] else 'failed'}" for item in results['dynamic_checks'])
report += ['', 'Operation | selected comparisons | failures | full recorded checks']
for s in specs:
    report.append(f"{s['title']} | {len(compact[s['slug']]['selectedCheckIndices'])} | "
                  f"{len(bytitle[s['title']]['failures'])} | {len(s['checks'])}")
(V / 'validation_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
with (V / 'operations.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Operation', 'Category', 'Selected comparisons', 'Failures',
                     'Full recorded checks', 'Folder', 'Example settings', 'Scope'])
    for s in specs:
        c = compact[s['slug']]
        writer.writerow([s['title'], s['category'], len(c['selectedCheckIndices']),
                         len(bytitle[s['title']]['failures']), len(s['checks']),
                         'datasets/' + s['slug'], c['settings'], c['scope']])
files = []
for directory, children, names in os.walk(V):
    children[:] = sorted(name for name in children if name not in ('.work', '__pycache__'))
    for name in sorted(names):
        if name in ('manifest.json', 'GRIM_button_validation.xlsx') or name.startswith('~$') or name.endswith('.inspect.ndjson'):
            continue
        p = Path(directory) / name
        files.append(dict(path=p.relative_to(V).as_posix(), bytes=p.stat().st_size,
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                generated_utc=datetime.now(timezone.utc).isoformat(), operation_count=len(specs),
                current_workbook='GRIM_validation_simple.xlsx',
                comparison_count=checks, full_evidence_comparison_count=full_checks,
                native_dataset_count=len(native), csv_count=len(exports), failed_comparisons=failures, files=files)
(V / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))
