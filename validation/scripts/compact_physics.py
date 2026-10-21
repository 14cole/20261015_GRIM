"""Compact worked examples for the seven physical-operation worksheets.

Original fixtures and complete comparison specifications are kept unchanged.
The range examples use the closed-form rectangular-window point response,
which is algebraically equivalent to the original direct DFT expectation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def original(slug):
    return json.loads((ROOT / 'specs' / (slug + '.json')).read_text(encoding='utf-8'))


def mapped(spec, key):
    item = next(x for x in spec['inputs'] if x['id'] == key)
    return {'id': key, 'value': item['value']}


def raw_csv(slug, name):
    with (ROOT / 'datasets' / slug / (name + '.csv')).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def extra(spec, key, label, value, unit=''):
    spec.setdefault('extraInputs', []).append(dict(id=key, label=label, value=value, unit=unit))
    return dict(id=key, value=value)


def calculation(spec, key, label, formula, unit=''):
    spec.setdefault('extraCalculations', []).append(dict(id=key, label=label, formula=formula, unit=unit))
    return '@' + key


def base(source, description, settings, scope):
    return dict(slug=source['slug'], title=source['title'], description=description,
                settings=settings, scope=scope, rawHeaders=[], rawRows=[],
                selectedCheckIndices=[], labelOverrides={}, formulaOverrides={})


def pick(source, case, metric):
    matches = [i for i, check in enumerate(source['checks']) if check['case'] == case and check['metric'] == metric]
    assert len(matches) == 1, (source['slug'], case, metric, matches)
    return matches[0]


def phase_center():
    s = original('phase_center')
    out = base(s, 'Translate one angled sample using the two-way phase ramp.',
               'x = 0.031 m, y = -0.017 m, z = 0.009 m; both declared time signs.',
               'One sample for each time convention. The full input files contain additional angles and frequencies.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'RCS (m2)', 'Phase (deg)']
    for label, text in [('positive_time', '+ time'), ('negative_time', '- time')]:
        tag = label + '_0'
        out['rawRows'].append([text, mapped(s, tag+'_az'), mapped(s, tag+'_el'), mapped(s, tag+'_f'), 'VV', mapped(s, tag+'_p'), mapped(s, tag+'_phase')])
        for metric in ['Field I', 'Field Q', 'Power preserved']:
            i = pick(s, label+' sample 1', metric)
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = text + ': ' + ('RCS (m2)' if metric == 'Power preserved' else metric+' (m)')
    return out


def range_cal():
    s = original('range_cal')
    out = base(s, 'Multiply the DUT field by exact/measured reference, then correct signed delay.',
               '8 GHz example; signed reference offsets +0.0125 m and -0.0125 m.',
               'Synthetic complex-gain correction. Each sign uses DUT, measured standard and exact standard as separate inputs.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'Field I (m)', 'Field Q (m)']
    for tag in ['plus', 'minus']:
        for label, prefix in [('DUT', 'd'), ('measured', 'm'), ('reference', 'e')]:
            out['rawRows'].append([('+' if tag == 'plus' else '-')+' '+label, 0, 0,
                                   mapped(s, tag+'_0_f') if prefix == 'd' else 8e9,
                                   'VV', mapped(s, tag+'_0_'+prefix+'i'), mapped(s, tag+'_0_'+prefix+'q')])
        for metric in ['Field I', 'Field Q']:
            i = pick(s, tag+' f1 formula', metric)
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = ('+ offset' if tag == 'plus' else '- offset') + ': ' + metric+' (m)'
    return out


def time_gate():
    s = original('time_gate')
    out = base(s, 'A compensated point at the gate center retains its known complex field.',
               'Keep gate -0.03 to +0.04 m; taper 0.4; compensation on; point range 0.005 m.',
               'This sheet checks the analytic compensated center-point control. Full gated-DFT cases remain in the dataset folder and original specification.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'RCS (m2)', 'Phase (deg)']
    rows = raw_csv('time_gate', 'input_center_point')
    center = calculation(out, 'compact_center', 'Gate center', '=(@start+@stop)/2', 'm')
    for n, row in enumerate(rows):
        frequency = extra(out, f'compact_f{n}', f'Frequency {n+1}', (n+1)*1e9, 'Hz')
        power = extra(out, f'compact_p{n}', f'Input RCS at {n+1} GHz', 1., 'm2')
        out['rawRows'].append(['Center point', 0, 0, frequency, 'VV', power, float(row['phase_deg'])])
        angle = calculation(out, f'compact_angle{n}', f'Phase at {n+1} GHz', f'=-4*PI()*@compact_f{n}*{center}/@c', 'rad')
        for metric, trig in [('Field I', 'COS'), ('Field Q', 'SIN')]:
            i = pick(s, f'Compensated center frequency {n+1}', metric)
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = f'{n+1} GHz: {metric} (m)'
            out['formulaOverrides'][str(i)] = f'=SQRT(@compact_p{n})*{trig}({angle})'
    return out


def wedge_conic():
    s = original('wedge_conic')
    out = base(s, 'Waterline wedge angles map to signed conic azimuth; RCS is preserved.',
               'Waterline elevation 0 deg; 10 GHz; HH polarization.',
               'Waterline coordinate example only. Tilted polarization mixing and sparse-angle interpolation are outside this compact sheet.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (GHz)', 'Pol', 'RCS (m2)']
    for n in range(4):
        out['rawRows'].append(['Waterline', mapped(s, f'water_a{n}'), 0, 10, 'HH', mapped(s, f'water_p{n}')])
    for n in range(4):
        for metric in ['Conic azimuth', 'Power']:
            if metric == 'Power' and n == 3:
                continue
            i = pick(s, f'Waterline row {n+1}', metric)
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = f'Output row {n+1}: ' + ('RCS (m2)' if metric == 'Power' else 'azimuth (deg)')
    return out


def response_formula(power, phase_step):
    # Integer N has the same power limit at every multiple of 2*pi. The
    # inner guard also keeps the inactive division defined in formula engines
    # that evaluate both IF branches. No comparison tolerance is changed.
    denominator = f'SIN({phase_step}/2)'
    limit = f'ABS({denominator})<=1E-12'
    safe_denominator = f'IF({limit},1,{denominator})'
    return (f'=IF({limit},{power},'
            f'{power}*(SIN(@n*{phase_step}/2)/(@n*{safe_denominator}))^2)')


def range_frequency():
    s = original('range_frequency')
    out = base(s, 'Rectangular-window point response: P [sin(N theta/2)/(N sin(theta/2))]^2.',
               'Eight frequencies, 1-8 GHz; 64 range bins; rectangular window; declared negative range law.',
               'A single synthetic point and three adjacent range bins. At multiples of 2 pi the limiting response is the input RCS. Other conventions and full profiles remain in the folder.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'RCS (m2)', 'Phase (deg)', 'Target range (m)']
    rows = raw_csv('range_frequency', 'input_negative_range_law')
    for n, row in enumerate(rows):
        power = extra(out, 'compact_power', 'Point RCS', 4., 'm2') if n == 0 else 4.
        out['rawRows'].append(['Point', 0, 0, mapped(s, f'f{n}'), 'VV', power, float(row['phase_deg']), mapped(s, 'target') if n == 0 else next(x['value'] for x in s['inputs'] if x['id'] == 'target')])
    for k in [36, 37, 38]:
        radius = calculation(out, f'compact_range{k}', f'Bin {k} range', f'=({k}-@size/2)*@c/(2*(@f1-@f0)*@size)', 'm')
        step = calculation(out, f'compact_step{k}', f'Bin {k} phase step', f'=4*PI()*(@f1-@f0)*({radius}-@target)/@c', 'rad')
        i = pick(s, f'negative_range_law bin {k}', 'Power')
        out['selectedCheckIndices'].append(i)
        out['labelOverrides'][str(i)] = f'Bin {k}: RCS (m2)'
        out['formulaOverrides'][str(i)] = response_formula('@compact_power', step)
        i = pick(s, f'Range bin {k}', 'Range coordinate')
        out['selectedCheckIndices'].append(i)
        out['labelOverrides'][str(i)] = f'Bin {k}: range (m)'
        out['formulaOverrides'][str(i)] = '=' + radius
    return out


def azimuth_range():
    s = original('azimuth_range')
    out = base(s, 'Two single-point range profiles use the analytic rectangular-window response.',
               'Azimuths 0 and 30 deg; eight frequencies, 1-8 GHz; 32 range bins; no peak normalization.',
               'Peak and adjacent-bin examples for two synthetic profiles. The plot uses float32 FFT work; the full profiles remain in the folder.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'RCS (m2)', 'Phase (deg)', 'Target range (m)']
    rows = raw_csv('azimuth_range', 'input')
    for a in range(2):
        for j in range(8):
            row = rows[a*8+j]
            frequency = extra(out, f'compact_f{j}', f'Frequency {j+1}', (j+1)*1e9, 'Hz') if a == 0 and j < 2 else (j+1)*1e9
            power = extra(out, f'compact_power{a}', f'Profile {a+1} RCS', [4., 1.][a], 'm2') if j == 0 else [4., 1.][a]
            out['rawRows'].append([f'Profile {a+1}', mapped(s, f'az{a}') if j == 0 else a*30, 0, frequency, 'VV', power, float(row['phase_deg']), mapped(s, f'r{a}') if j == 0 else next(x['value'] for x in s['inputs'] if x['id'] == f'r{a}')])
    for a, bins in [(0, [15, 16]), (1, [20, 21])]:
        for k in bins:
            radius = calculation(out, f'compact_range{a}_{k}', f'Az {a*30}, bin {k}: range', f'=({k}-@size/2)*@c/(2*(@compact_f1-@compact_f0)*@size)', 'm')
            step = calculation(out, f'compact_step{a}_{k}', f'Az {a*30}, bin {k}: phase step', f'=4*PI()*(@compact_f1-@compact_f0)*({radius}-@r{a})/@c', 'rad')
            i = pick(s, f'Az {a*30} bin {k}', 'Intensity')
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = f'Az {a*30}, bin {k}: intensity'
            out['formulaOverrides'][str(i)] = response_formula(f'@compact_power{a}', step)
            if a == 0:
                i = pick(s, f'Range bin {k}', 'Coordinate')
                out['selectedCheckIndices'].append(i)
                out['labelOverrides'][str(i)] = f'Bin {k}: range (m)'
                out['formulaOverrides'][str(i)] = '=' + radius
    return out


def isar_image():
    s = original('isar_image')
    out = base(s, 'Origin-point peak intensity and nominal image sampling from the aperture limits.',
               'Full input: 17 azimuths, -4 to 4 deg; 33 frequencies, 9-11 GHz; Fast and Accurate PFA.',
               'Three representative raw samples are shown; the native file contains all 561. Checks cover an origin point and sampling formulas, not arbitrary images.')
    out['rawHeaders'] = ['Input', 'Azimuth (deg)', 'Elevation (deg)', 'Frequency (Hz)', 'Pol', 'RCS (m2)', 'Phase (deg)']
    out['rawRows'] = [
        ['Origin 1', mapped(s, 'az0'), 0, mapped(s, 'f0'), 'VV', mapped(s, 'sigma'), 0],
        ['Origin middle', 0, 0, 10e9, 'VV', 4., 0],
        ['Origin 561', mapped(s, 'az1'), 0, mapped(s, 'f1'), 'VV', 4., 0],
    ]
    for mode in ['fast', 'accurate']:
        for metric in ['Origin peak intensity', 'Nominal range resolution', 'Nominal cross resolution', 'Range half extent']:
            i = pick(s, 'origin '+mode, metric)
            out['selectedCheckIndices'].append(i)
            out['labelOverrides'][str(i)] = mode.capitalize()+': '+metric.replace('Nominal ', '').replace('Origin ', '')+(' (m)' if metric != 'Origin peak intensity' else '')
    return out


def validate_and_save(out):
    s = original(out['slug'])
    ids = {x['id'] for x in s['inputs']} | {x['id'] for x in out.get('extraInputs', [])}
    mapped_ids = [x['id'] for row in out['rawRows'] for x in row if isinstance(x, dict)]
    assert len(mapped_ids) == len(set(mapped_ids)), out['slug']
    assert all(key in ids for key in mapped_ids), (out['slug'], set(mapped_ids)-ids)
    assert all(len(row) == len(out['rawHeaders']) for row in out['rawRows'])
    assert len(out['rawHeaders']) <= 8
    assert len(set(out['selectedCheckIndices'])) == len(out['selectedCheckIndices'])
    assert all(0 <= i < len(s['checks']) for i in out['selectedCheckIndices'])
    target = ROOT / 'compact_specs'
    target.mkdir(exist_ok=True)
    (target / (out['slug']+'.json')).write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    print(f"{out['slug']}: {len(out['selectedCheckIndices'])} comparisons, {len(out['rawRows'])} raw rows")


def generate():
    for maker in (phase_center, range_cal, time_gate, wedge_conic, range_frequency, azimuth_range, isar_image):
        validate_and_save(maker())


if __name__ == '__main__':
    generate()
