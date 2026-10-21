"""Choose small, readable worked examples from the coordinate validation evidence.

This changes presentation only. Expected formulas and GRIM export snapshots stay
owned by the full specifications. Raw cells retain their semantic input IDs.
"""
from pathlib import Path
import json
import re

V = Path(__file__).resolve().parents[1]
FIELDS = ('az', 'el', 'f', 'pol', 'p', 'ph')
RECORD = re.compile(r'^(.*)_(\d+)_(az|el|f|pol|p|ph)$')
REF = re.compile(r'@([A-Za-z0-9_]+)')


def pick(spec, case, metrics):
    indices = []
    for metric in metrics:
        matches = [i for i, item in enumerate(spec['checks'])
                   if item['case'] == case and item['metric'] == metric]
        assert len(matches) == 1, (spec['slug'], case, metric, matches)
        indices.extend(matches)
    return indices


def build(slug, selections, records, settings, *, overrides=None, labels=None,
          headers=None, description=None):
    spec = json.loads((V / 'specs' / (slug + '.json')).read_text())
    selected = [i for case, metrics in selections for i in pick(spec, case, metrics)]
    formulas = overrides or {}
    inputs = {item['id']: item for item in spec['inputs']}
    calculations = {item['id']: item for item in spec['calculations']}
    used_inputs = set()

    def visit(formula):
        for key in REF.findall(formula):
            if key in calculations:
                visit(calculations[key]['formula'])
            else:
                assert key in inputs, (slug, key)
                used_inputs.add(key)

    for index in selected:
        visit(formulas.get(str(index), spec['checks'][index]['formula']))
    chosen = {(name, number) for name, numbers in records for number in numbers}
    # Every formula operand is displayed. Non-grid controls are shown by the
    # workbook builder as a short parameter list below the raw table.
    for key in used_inputs:
        if match := RECORD.fullmatch(key):
            chosen.add((match[1], int(match[2])))
    file_order = {item['name']: i for i, item in enumerate(spec['files'])}
    rows = []
    for name, number in sorted(chosen, key=lambda x: (file_order[x[0]], x[1])):
        raw = [name]
        for field in FIELDS:
            key = f'{name}_{number}_{field}'
            raw.append(dict(id=key, value=inputs[key]['value']))
        rows.append(raw)
    assert len(rows) <= 12, (slug, len(rows))
    known = {cell['id'] for row in rows for cell in row if isinstance(cell, dict)}
    assert all(not RECORD.fullmatch(key) or key in known for key in used_inputs)
    result = dict(slug=slug, title=spec['title'],
                  description=description or spec['description'], settings=settings,
                  scope='Worked examples from the supplied input and output files. ' + spec['scope'],
                  rawHeaders=headers or ['Dataset', 'Azimuth (deg)', 'Elevation (deg)',
                                         'Frequency (GHz)', 'Polarization', 'RCS (m2)', 'Phase (deg)'],
                  rawRows=rows, selectedCheckIndices=selected,
                  labelOverrides=labels or {}, formulaOverrides=formulas)
    folder = V / 'compact_specs'
    folder.mkdir(exist_ok=True)
    (folder / (slug + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(f'{slug}: {len(rows)} raw rows, {len(selected)} comparisons')
    return result


def generate():
    results = []
    add = lambda *a, **k: results.append(build(*a, **k))
    add('crop_slice', [('result sample 1', ['azimuth', 'elevation', 'frequency', 'magnitude_power_linear']),
                       ('result sample 2', ['azimuth', 'magnitude_power_linear'])],
        [('input', [7, 15, 23, 31])],
        'Crop azimuth 10 to 30 deg, stride 2. Shown slice: elevation 5 deg, 10 GHz, HH.')
    add('align', [('result sample 1', ['azimuth', 'magnitude_power_linear']),
                  ('result sample 2', ['azimuth', 'magnitude_power_linear'])],
        [('reference', range(3)), ('input', range(4))],
        'Reference first; intersect mode. Snap the two matching input azimuths to the reference.')
    add('regrid', [('Sample 2', ['Azimuth', 'Power']), ('Sample 3', ['Power']),
                   ('Sample 4', ['Azimuth', 'Power'])],
        [('input', range(3))],
        'Interpolate I and Q to 5 and 15 deg. The 10 deg result is an unchanged source node.')
    add('mirror', [('result sample 1', ['azimuth', 'magnitude_power_linear']),
                   ('result sample 3', ['azimuth', 'magnitude_power_linear']),
                   ('result sample 5', ['azimuth', 'magnitude_power_linear'])],
        [('input', range(6))],
        'Mirror about 15 deg: new azimuth = 2 x 15 - old azimuth; keep each RCS sample.')
    add('wrap', [('result sample 1', ['azimuth', 'magnitude_power_linear']),
                 ('result sample 2', ['azimuth', 'magnitude_power_linear', 'phase']),
                 ('result sample 3', ['azimuth', 'magnitude_power_linear', 'phase']),
                 ('result sample 4', ['azimuth', 'magnitude_power_linear'])],
        [('input', range(5))],
        'Wrap azimuth to 0..360 and phase to -180..180. At the repeated seam, keep the -180 deg input.')
    add('shift', [('result sample 1', ['azimuth', 'elevation', 'magnitude_power_linear', 'phase']),
                  ('result sample 3', ['azimuth', 'elevation', 'magnitude_power_linear', 'phase'])],
        [('input', range(4))],
        'Add 15 deg azimuth, subtract 7 deg elevation, and add 30 deg phase.')
    add('round', [('result sample 1', ['azimuth', 'elevation', 'frequency', 'magnitude_power_linear']),
                  ('result sample 2', ['frequency']), ('result sample 3', ['elevation']),
                  ('result sample 5', ['azimuth']), ('result sample 9', ['azimuth'])],
        [('input', [0, 1, 2, 4, 8])],
        'Round to one decimal, using ties-to-even. Exact halfway examples: -1.25 and +0.25 deg.')
    add('duplicate', [(f'result sample {i}', ['magnitude_power_linear']) for i in range(1, 9)] +
                      [('result sample 8', ['azimuth', 'frequency'])],
        [('input', range(8))], 'Duplicate the dataset. Compare copied RCS, zero, missing, and coordinates.')
    add('axis_units', [('result sample 1', ['elevation', 'frequency']),
                       ('result sample 5', ['azimuth', 'magnitude_power_linear']),
                       ('result sample 8', ['elevation', 'frequency']),
                       ('result_inverse sample 8', ['elevation', 'frequency'])],
        [('input', range(8))],
        'Convert deg to rad and GHz to Hz. The final two comparisons convert back.',
        labels={'2': 'Elevation -30 deg to rad', '3': 'Frequency 8 GHz to Hz',
                '25': 'Azimuth 90 deg to rad', '29': 'RCS retained (m2)',
                '44': 'Elevation 30 deg to rad', '45': 'Frequency 12 GHz to Hz',
                '96': 'Inverse elevation (deg)', '97': 'Inverse frequency (GHz)'})
    add('swap_el_az', [(f'result sample {i}', ['azimuth', 'elevation', 'magnitude_power_linear'])
                       for i in [1, 3, 11]],
        [('input', range(12))],
        'Swap angular axes and reorder samples. Input elevation is rad; input azimuth is deg.',
        headers=['Dataset', 'Azimuth (deg)', 'Elevation (rad)', 'Frequency (GHz)',
                 'Polarization', 'RCS (m2)', 'Phase (deg)'])
    add('join_merge', [(policy + ' sample 2', ['Power']) for policy in
                       ['priority-first', 'priority-last', 'power-mean', 'coherent-mean']] +
                      [('result_union sample 4', ['azimuth', 'magnitude_power_linear']),
                       ('result_union sample 5', ['azimuth', 'magnitude_power_linear'])],
        [('input_a', range(3)), ('input_b', range(3)), ('input_c_no_overlap', range(2))],
        'Merge A+B: compare the four policies at 10 deg. Join A+C: retain the new 40 and 50 deg samples.',
        labels={'4': '10 deg: first input RCS', '13': '10 deg: last input RCS',
                '22': '10 deg: mean power', '31': '10 deg: power of mean field',
                '55': 'Joined azimuth 40 deg', '59': 'Joined RCS at 40 deg',
                '61': 'Joined azimuth 50 deg', '65': 'Joined RCS at 50 deg'})
    add('overlap', [(f'Dataset {dataset} sample {i}', ['Common finite power'])
                    for dataset in (1, 2) for i in range(1, 5)],
        [('input_a', range(6)), ('input_b', range(6))],
        'Keep common azimuths 10 and 20 deg. A missing HH value at 10 deg masks that cell in both results.',
        overrides={'2': '=IF(AND(ISNUMBER(@input_a_3_p),ISNUMBER(@input_b_1_p)),@input_a_3_p,"missing")',
                   '14': '=IF(AND(ISNUMBER(@input_a_3_p),ISNUMBER(@input_b_1_p)),@input_b_1_p,"missing")'},
        labels={str(index): f'{dataset}, {az} deg {pol}: RCS' for index, dataset, az, pol in
                [(2, 'A', 10, 'HH'), (5, 'A', 10, 'VV'), (8, 'A', 20, 'HH'), (11, 'A', 20, 'VV'),
                 (14, 'B', 10, 'HH'), (17, 'B', 10, 'VV'), (20, 'B', 20, 'HH'), (23, 'B', 20, 'VV')]})
    add('sentri_el_grim', [(f'result sample {i}', ['azimuth', 'elevation', 'magnitude_power_linear'])
                           for i in [1, 3, 5, 13]],
        [('input', range(0, 18, 2))],
        'Shown 9 GHz slice. GRIM elevation = 90 - native theta; azimuth = phi modulo 360.',
        headers=['Dataset', 'Phi (deg)', 'Theta (deg)', 'Frequency (GHz)',
                 'Polarization', 'RCS (m2)', 'Phase (deg)'])
    add('el_az360', [(f'result sample {i}', ['azimuth', 'magnitude_power_linear'])
                     for i in range(1, 5)] + [('result sample 1', ['elevation'])],
        [('input', range(4))],
        'Attested opposite elevation pair. Shift +10 deg cut by 180 deg azimuth; output elevation stays -10 deg.')
    add('decimate', [(f'keep bin {i}', ['Coordinate center', 'Power mean']) for i in range(1, 4)] +
                    [(f'Coherent bin {i}', ['Power of mean field']) for i in range(1, 4)],
        [('input_power', range(5)), ('input_coherent', range(5))],
        'Factor 2, keep final partial bin. Power mode averages finite powers; coherent mode averages I/Q.')
    assert len(results) == 15
    return results


if __name__ == '__main__':
    generate()
