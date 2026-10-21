"""Small, visible worked examples drawn from the full arithmetic fixtures.

This selects presentation examples only. Original datasets, exports, formula
specifications, and their broader edge-case evidence remain unchanged.
"""
from pathlib import Path
import csv
import json

VALIDATION = Path(__file__).resolve().parents[1]


def load(slug):
    return json.loads((VALIDATION / 'specs' / (slug + '.json')).read_text(encoding='utf-8'))


def cell(spec, key):
    item = next(x for x in spec['inputs'] if x['id'] == key)
    return dict(id=key, value=item['value'])


def rows(slug, name):
    with (VALIDATION / 'datasets' / slug / (name + '.csv')).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value == '':
        return 'missing'
    try:
        return float(value)
    except ValueError:
        return value


def context(row):
    return [number(row['azimuth']), number(row['elevation']), number(row['frequency']), row['polarization']]


def base(spec, headers, raw, selected, labels=None, formulas=None, settings=None):
    result = {key: spec[key] for key in ('slug', 'title', 'description', 'settings', 'scope')}
    result.update(rawHeaders=headers, rawRows=raw, selectedCheckIndices=selected)
    if labels:
        result['labelOverrides'] = {str(k): v for k, v in labels.items()}
    if formulas:
        result['formulaOverrides'] = {str(k): v for k, v in formulas.items()}
    if settings:
        result['settings'] = settings
    return result


def coherent(slug):
    spec = load(slug)
    raw = []
    for operand in ('a', 'b'):
        for n, row in enumerate(rows(slug, 'input_' + operand)):
            raw.append([operand.upper(), *context(row), cell(spec, f'{operand}_i{n}'),
                        cell(spec, f'{operand}_q{n}'),
                        dict(formula=f'=IF(OR(@{operand}_i{n}="missing",@{operand}_q{n}="missing"),"missing",@{operand}_i{n}^2+@{operand}_q{n}^2)')])
    selected = [n * 3 for n in range(7)] + [1, 2]
    formulas = {}
    sign = '+' if slug == 'coherent_add' else '-'
    for n in range(7):
        guard = f'OR(@a_i{n}="missing",@a_q{n}="missing",@b_i{n}="missing",@b_q{n}="missing")'
        if slug == 'coherent_divide':
            expr = f'IF(@b_i{n}^2+@b_q{n}^2=0,"missing",(@a_i{n}^2+@a_q{n}^2)/(@b_i{n}^2+@b_q{n}^2))'
        else:
            expr = f'(@a_i{n}{sign}@b_i{n})^2+(@a_q{n}{sign}@b_q{n})^2'
        formulas[n * 3] = f'=IF({guard},"missing",{expr})'
    labels = {n * 3: f'Az {n}: output linear RCS' for n in range(7)}
    if slug == 'coherent_divide':
        labels = {n * 3: f'Az {n}: output power ratio' for n in range(7)}
    labels.update({1: 'Az 0: output I', 2: 'Az 0: output Q'})
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)',
                       'Pol', 'I (sqrt m²)', 'Q (sqrt m²)', 'RCS (m², calc.)'],
                raw, selected, labels, formulas,
                'Load input_a and input_b. Operand order is A then B. The workbook shows the two-operand result.')


def incoherent(slug):
    spec = load(slug)
    raw = []
    for operand in ('a', 'b'):
        for n, row in enumerate(rows(slug, 'input_' + operand)):
            raw.append([operand.upper(), *context(row), cell(spec, f'{operand}{n}')])
    selected = [n * 2 for n in range(6)]
    labels = {i: f'Az {i // 2}: output linear RCS' for i in selected}
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)'],
                raw, selected, labels)


def delta():
    spec = load('delta_db')
    raw = []
    for operand in ('a', 'b'):
        for n, row in enumerate(rows('delta_db', 'input_' + operand)):
            raw.append([operand.upper(), *context(row), cell(spec, f'{operand}{n}')])
    selected = [n * 2 for n in range(6)] + [1, 7]
    labels = {n * 2: f'Az {n}: A / B power ratio' for n in range(6)}
    labels.update({1: 'Az 0: A minus B level (dB)', 7: 'Az 3: A minus B level (dB)'})
    formulas = {n * 2: next(x['formula'] for x in spec['calculations'] if x['id'] == f'r{n}') for n in range(6)}
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)'],
                raw, selected, labels, formulas)


def stats():
    spec = load('stats')
    selected_rows = [r for r in rows('stats', 'input') if r['elevation'] == '-5' and r['frequency'] == '8' and r['polarization'] == 'VV']
    raw = [['Input', *context(row), cell(spec, f'p000_{n}')] for n, row in enumerate(selected_rows)]
    selected = []
    labels = {}
    formulas = {}
    for stat, function, label in [('mean', 'AVERAGE', 'Mean'), ('median', 'MEDIAN', 'Median'),
                                  ('min', 'MIN', 'Minimum'), ('max', 'MAX', 'Maximum'),
                                  ('std', 'STDEV.P', 'Population standard deviation'),
                                  ('percentile', 'PERCENTILE.INC', '90th percentile')]:
        idx = next(i for i, x in enumerate(spec['checks']) if x['case'].startswith(stat + ':') and '@p000_0' in x['formula'])
        selected.append(idx)
        labels[idx] = label
        formulas[idx] = f'={function}(@p000_0:@p000_4' + (',@q/100)' if stat == 'percentile' else ')')
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)'],
                raw, selected, labels, formulas,
                'Reduce azimuth in linear power. This example uses 8 GHz, elevation -5°, VV. Percentile setting: 90.')


def percentile():
    spec = load('percentile')
    raw = []
    selected = []
    labels = {}
    for operand in ('a', 'b'):
        source = [r for r in rows('percentile', 'input_' + operand) if r['frequency'] == '8' and r['polarization'] == 'VV']
        raw.extend([[operand.upper(), *context(row), cell(spec, f'{operand}00_{n}')] for n, row in enumerate(source)])
        idx = next(i for i, x in enumerate(spec['checks']) if f'@{operand}00_0' in x['formula'])
        selected.append(idx)
        labels[idx] = f'Dataset {operand.upper()}: 90th percentile'
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)'],
                raw, selected, labels,
                settings='Percentile 90 over azimuth, separately for A and B. Shown trace: 8 GHz, VV. The datasets are not pooled.')


def offset():
    spec = load('offset')
    raw = [['Input', *context(row), cell(spec, f'p{n}'), cell(spec, f'ph{n}')]
           for n, row in enumerate(rows('offset', 'input'))]
    selected = [n * 3 for n in range(5)] + [1, 7, 2, 8]
    labels = {n * 3: f'Az {n}: power after +6 dB' for n in range(5)}
    labels.update({1: 'Az 0: preserved phase', 7: 'Az 2: preserved phase',
                   2: 'Az 0: power after reversing offset', 8: 'Az 2: power after reversing offset'})
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)', 'Phase (deg)'],
                raw, selected, labels)


def flip():
    spec = load('flip_phase')
    raw = [['Input', *context(row), cell(spec, f'a_i{n}'), cell(spec, f'a_q{n}'),
            dict(formula=f'=IF(OR(@a_i{n}="missing",@a_q{n}="missing"),"missing",@a_i{n}^2+@a_q{n}^2)')]
           for n, row in enumerate(rows('flip_phase', 'input'))]
    selected = [n * 6 + 2 for n in range(6)] + [0, 18, 1]
    labels = {n * 6 + 2: f'Az {n}: Q after phase flip' for n in range(6)}
    labels.update({0: 'Az 0: preserved linear RCS', 18: 'Az 3: zero linear RCS', 1: 'Az 0: preserved I'})
    formulas = {n * 6 + 2: f'=IF(@a_q{n}="missing","missing",-@a_q{n})' for n in range(6)}
    formulas[0] = '=@a_i0^2+@a_q0^2'
    formulas[18] = '=@a_i3^2+@a_q3^2'
    formulas[1] = '=@a_i0'
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol',
                       'I (sqrt m²)', 'Q (sqrt m²)', 'RCS (m², calc.)'], raw, selected, labels, formulas,
                'Apply Flip Phase once. Complex conjugation preserves I and negates Q.')


def medianize():
    spec = load('medianize')
    raw = []
    for name, prefix in [('periodic', 'p'), ('gapped', 'g')]:
        raw.extend([[name.title(), *context(row), cell(spec, f'{prefix}{n}')]
                    for n, row in enumerate(rows('medianize', 'input_' + name))])
    selected = [i for i, x in enumerate(spec['checks']) if x['metric'] == 'Median power']
    labels = {i: spec['checks'][i]['case'] for i in selected}
    return base(spec, ['Dataset', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m²)'],
                raw, selected, labels)


def extrusion():
    spec = load('dbke_dbsm')
    raw = []
    for n, row in enumerate(rows('dbke_dbsm', 'input_3d')):
        raw.append([n, number(row['azimuth']), number(row['elevation']), cell(spec, f'f{n}'),
                    row['polarization'], cell(spec, f'p3_{n}'), cell(spec, f'p2_{n}'),
                    number(row['phase_deg'])])
    selected = [n * 4 + k for n in (0, 1, 2, 4, 5) for k in (0, 2)] + [1, 3]
    labels = {}
    formulas = {}
    for n in (0, 1, 2, 4, 5):
        labels[n * 4] = f'3-D sample {n}: scattering width (m)'
        labels[n * 4 + 2] = f'2-D sample {n}: RCS area (m²)'
        for offset, key in ((0, f'w{n}'), (2, f's{n}')):
            formulas[n * 4 + offset] = next(x['formula'] for x in spec['calculations'] if x['id'] == key)
    labels.update({1: '3-D sample 0: converted dBke', 3: '2-D sample 0: converted dBsm'})
    return base(spec, ['Sample', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', '3-D RCS (m²)', '2-D RCS (m)', 'Phase (deg)'],
                raw, selected, labels, formulas)


def generate():
    specs = [coherent(slug) for slug in ('coherent_add', 'coherent_subtract', 'coherent_divide')]
    specs += [incoherent(slug) for slug in ('incoherent_add', 'incoherent_subtract')]
    specs += [delta(), stats(), percentile(), offset(), flip(), medianize(), extrusion()]
    output = VALIDATION / 'compact_specs'
    output.mkdir(exist_ok=True)
    for result in specs:
        assert len(result['rawHeaders']) <= 8
        assert all(len(row) == len(result['rawHeaders']) for row in result['rawRows'])
        ids = [v['id'] for row in result['rawRows'] for v in row if isinstance(v, dict) and 'id' in v]
        assert len(ids) == len(set(ids)), result['slug']
        original = load(result['slug'])
        for idx in result['selectedCheckIndices']:
            assert original['checks'][idx]['metric'] != 'Invalid input handling'
        (output / (result['slug'] + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f"{result['slug']}: {len(result['rawRows'])} input rows, {len(result['selectedCheckIndices'])} comparisons")
    return specs


if __name__ == '__main__':
    generate()
