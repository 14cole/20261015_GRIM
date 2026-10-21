"""Independent Excel formula cases for arithmetic and statistical buttons.

Run from the repository root with Python. GRIM is used only for actual results;
Excel expectations refer to the literal source inputs below.
"""
from __future__ import annotations
import math
import numpy as np
from common import Spec, grid, number
from GRIM_Backend.datasets.transforms import (
    coherent_divide, convert_extrusion, flip_phase, medianize_azimuth, offset_db,
)


def field_inputs(spec, prefix, values):
    for n, value in enumerate(values):
        spec.input(f'{prefix}_i{n}', f'{prefix} sample {n}: I', value.real, 'sqrt(power)')
        spec.input(f'{prefix}_q{n}', f'{prefix} sample {n}: Q', value.imag, 'sqrt(power)')


def csv_component(row, imaginary=False):
    power, phase = number(row, 'magnitude_power_linear'), number(row, 'phase_deg')
    if power == 'missing':
        return 'missing'
    if power == 0:
        return 0.
    if phase == 'missing':
        return 'missing'
    return math.sqrt(power) * (math.sin if imaginary else math.cos)(math.radians(phase))


def field_checks(spec, rows, n, i_formula, q_formula, *, prefix='r', file='result', case=None, unit='m2'):
    case = case or f'Azimuth {n} deg'
    ii = spec.calc(f'{prefix}_i{n}', f'{case}: expected I', i_formula, 'sqrt(power)')
    qq = spec.calc(f'{prefix}_q{n}', f'{case}: expected Q', q_formula, 'sqrt(power)')
    power = f'=IF(OR({ii}="missing",{qq}="missing"),"missing",{ii}^2+{qq}^2)'
    spec.exported(case, 'Linear power', power, rows, n, 'magnitude_power_linear', unit, file)
    spec.check(case, 'I from exported power and phase', '=' + ii,
               csv_component(rows[n]), 'sqrt(power)', f'{file}.csv row {n+2}: sqrt(power)*COS(phase)')
    spec.check(case, 'Q from exported power and phase', '=' + qq,
               csv_component(rows[n], True), 'sqrt(power)', f'{file}.csv row {n+2}: sqrt(power)*SIN(phase)')


def coherent_cases():
    a_values = [1+2j, 1+0j, 0+0j, 3-4j, complex(np.nan, np.nan), -2+2j, 1e-5+2e-5j]
    b_values = [3-1j, -1+0j, 2j, 0j, 1+1j, 2-2j, 1e-5-2e-5j]
    c_values = [1j, 2j, 1+0j, 2-1j, 1+0j, -1+0j, 2e-5+0j]
    specs = []
    for slug, title, method, sign in (
        ('coherent_add', 'Coherent +', 'coherent_add', '+'),
        ('coherent_subtract', 'Coherent -', 'coherent_subtract', '-'),
        ('coherent_divide', 'Coherent divide', '', '/'),
    ):
        spec = Spec(slug, title,
                    'Independent Cartesian I/Q arithmetic, including cancellation, zero and missing samples.',
                    settings=('Select input_a first, input_b second. A is the numerator/minuend.' +
                              ('' if slug == 'coherent_divide' else ' For result_three also select input_c last.')),
                    scope='Common synthetic phase reference and exact matching grids; no claim that unrelated acquisitions are coherent.')
        field_inputs(spec, 'a', a_values)
        field_inputs(spec, 'b', b_values)
        a, b = grid(field=a_values), grid(field=b_values)
        spec.save('input_a', a)
        spec.save('input_b', b)
        result = coherent_divide(a, b) if sign == '/' else getattr(a, method)(b)
        rows = spec.save('result', result)
        for n in range(len(a_values)):
            guard = f'OR(@a_i{n}="missing",@a_q{n}="missing",@b_i{n}="missing",@b_q{n}="missing")'
            if sign == '/':
                denom = spec.calc(f'd{n}', f'Sample {n}: denominator power',
                                  f'=IF(OR(@b_i{n}="missing",@b_q{n}="missing"),"missing",@b_i{n}^2+@b_q{n}^2)', 'power')
                ii = f'=IF({guard},"missing",IF({denom}=0,"missing",(@a_i{n}*@b_i{n}+@a_q{n}*@b_q{n})/{denom}))'
                qq = f'=IF({guard},"missing",IF({denom}=0,"missing",(@a_q{n}*@b_i{n}-@a_i{n}*@b_q{n})/{denom}))'
            else:
                ii = f'=IF({guard},"missing",@a_i{n}{sign}@b_i{n})'
                qq = f'=IF({guard},"missing",@a_q{n}{sign}@b_q{n})'
            field_checks(spec, rows, n, ii, qq, unit='ratio' if sign == '/' else 'm2')
        if sign != '/':
            field_inputs(spec, 'c', c_values)
            c = grid(field=c_values)
            spec.save('input_c', c)
            three = a.coherent_add_many(b, c) if sign == '+' else a.coherent_subtract(b).coherent_subtract(c)
            rows3 = spec.save('result_three', three)
            for n in range(len(a_values)):
                guard = f'OR(@a_i{n}="missing",@a_q{n}="missing",@b_i{n}="missing",@b_q{n}="missing")'
                field_checks(spec, rows3, n,
                    f'=IF({guard},"missing",@a_i{n}{sign}@b_i{n}{sign}@c_i{n})',
                    f'=IF({guard},"missing",@a_q{n}{sign}@b_q{n}{sign}@c_q{n})',
                    prefix='three', file='result_three', case=f'Three operands, azimuth {n}')
        opposite = grid(field=b_values, units={'time_convention': 'exp(-jwt)'})
        spec.save('input_opposite_time', opposite)
        spec.reject('Opposite declared time conventions',
                    lambda: coherent_divide(a, opposite) if sign == '/' else getattr(a, method)(opposite),
                    contains='time')
        specs.append(spec.finish())
    return specs


def incoherent_cases():
    a_values = [1., 10., 100., 0., np.nan, 4.]
    b_values = [.25, 2., 100., 0., 3., np.nan]
    specs = []
    for slug, title, method, sign in (
        ('incoherent_add', 'Incoherent +', 'incoherent_add', '+'),
        ('incoherent_subtract', 'Incoherent -', 'incoherent_subtract', '-'),
    ):
        spec = Spec(slug, title, 'Linear power arithmetic; zero and missing samples remain distinct.',
                    settings='Select input_a first and input_b second. Result has no coherent phase.')
        a, b = grid(a_values), grid(b_values)
        spec.save('input_a', a); spec.save('input_b', b)
        rows = spec.save('result', getattr(a, method)(b))
        for n, (av, bv) in enumerate(zip(a_values, b_values)):
            spec.input(f'a{n}', f'A power at azimuth {n}', av, 'm2')
            spec.input(f'b{n}', f'B power at azimuth {n}', bv, 'm2')
            formula = f'=IF(OR(@a{n}="missing",@b{n}="missing"),"missing",@a{n}{sign}@b{n})'
            spec.exported(f'Azimuth {n}', 'Linear power', formula, rows, n, 'magnitude_power_linear', 'm2')
            spec.exported(f'Azimuth {n}', 'Unknown output phase', '="missing"', rows, n, 'phase_deg', 'deg')
        if sign == '-':
            invalid = grid([2., 20., 200., 1., 3., 5.])
            spec.save('input_negative_result', invalid)
            spec.reject('Materially negative power subtraction', lambda: a.incoherent_subtract(invalid), contains='negative')
        specs.append(spec.finish())
    spec = Spec('delta_db', 'Delta dB', 'First-minus-second dB levels are stored as a dimensionless linear power ratio.',
                settings='Select input_a first and input_b second; use Delta dB. Zero denominator is missing.')
    aa = [1., 10., 0., 4., np.nan, 9.]
    bb = [.5, 0., 2., 4., 5., np.nan]
    a, b = grid(aa), grid(bb)
    spec.save('input_a', a); spec.save('input_b', b)
    rows = spec.save('result', a.arithmetic_db_subtract(b))
    for n, (av, bv) in enumerate(zip(aa, bb)):
        spec.input(f'a{n}', f'A power at azimuth {n}', av, 'm2')
        spec.input(f'b{n}', f'B power at azimuth {n}', bv, 'm2')
        ratio = spec.calc(f'r{n}', f'Azimuth {n}: expected ratio',
            f'=IF(OR(@a{n}="missing",@b{n}="missing"),"missing",IF(@b{n}=0,"missing",@a{n}/@b{n}))', 'ratio')
        spec.exported(f'Azimuth {n}', 'Power ratio', '=' + ratio, rows, n, 'magnitude_power_linear', 'ratio')
        spec.exported(f'Azimuth {n}', 'Log ratio',
            f'=IF({ratio}="missing","missing",IF({ratio}=0,"-inf",10*LOG10({ratio})))', rows, n, 'magnitude_db', 'dB')
    specs.append(spec.finish())
    return specs


def statistical_cases():
    specs = []
    az, el, frequencies, pol = [-10., 0., 10., 20., 30.], [-5., 5.], [8., 12.], ['VV', 'HH']
    powers = np.empty((5, 2, 2, 2))
    for e in range(2):
        for f in range(2):
            for p in range(2):
                powers[:, e, f, p] = np.asarray([0., 1., 4., 16., 100.]) + e*3 + f*10 + p*2
    powers[2, 0, 0, 1] = np.nan
    powers[:, 1, 1, 1] = np.nan
    source = grid(powers, az=az, el=el, freq=frequencies, pol=pol)
    spec = Spec('stats', 'Stats', 'Independent linear-power statistics over azimuth, with frequency, elevation and polarization kept separate.',
                settings='Azimuth reduction; broadcast enabled. Result files name each statistic. A second mean case reduces elevation and frequency without broadcasting.')
    spec.save('input', source)
    for e in range(2):
        for f in range(2):
            for p in range(2):
                for a in range(5):
                    spec.input(f'p{e}{f}{p}_{a}', f'el {el[e]}, f {frequencies[f]}, {pol[p]}, az {az[a]}', powers[a,e,f,p], 'm2')
    spec.input('q', 'Percentile setting', 90., '%')
    for stat, func in [('mean','AVERAGE'),('median','MEDIAN'),('min','MIN'),('max','MAX'),('std','STDEV.P'),('percentile','PERCENTILE.INC')]:
        rows = spec.save('result_' + stat, source.statistics_dataset(statistic=stat, axes=('azimuth',), domain='magnitude', percentile=90, broadcast_reduced=True))
        for idx, row in enumerate(rows):
            e = el.index(float(row['elevation'])); f = frequencies.index(float(row['frequency'])); p = pol.index(row['polarization'])
            refs = f'@p{e}{f}{p}_0:@p{e}{f}{p}_4'
            expr = f'{func}({refs}' + (',@q/100)' if stat == 'percentile' else ')')
            formula = f'=IF(COUNT({refs})=0,"missing",{expr})'
            case = f'{stat}: az {row["azimuth"]}, el {row["elevation"]}, f {row["frequency"]}, {row["polarization"]}'
            spec.exported(case, 'Reduced linear power', formula, rows, idx, 'magnitude_power_linear', 'm2', 'result_' + stat)
    compact = source.statistics_dataset(statistic='mean', axes=('elevation','frequency'), domain='magnitude', broadcast_reduced=False)
    rows = spec.save('result_mean_el_freq_compact', compact)
    for n, row in enumerate(rows):
        a = az.index(float(row['azimuth'])); p = pol.index(row['polarization'])
        refs = ','.join(f'@p{e}{f}{p}_{a}' for e in range(2) for f in range(2))
        spec.exported(f'Compact az {az[a]}, {pol[p]}', 'Mean over elevation and frequency', f'=AVERAGE({refs})', rows,n,'magnitude_power_linear','m2','result_mean_el_freq_compact')
    specs.append(spec.finish())

    spec = Spec('percentile', 'Percentile', 'Azimuth percentile calculated separately inside each selected dataset, for each frequency and polarization.',
                settings='Percentile 90. Select input_a and input_b together. Each result repeats its own percentile across the original azimuths.')
    spec.input('q', 'Percentile setting', 90., '%')
    for dataset_name, multiplier in [('a',1.),('b',3.)]:
        pp = np.empty((5,1,2,2))
        for f in range(2):
            for p in range(2):
                pp[:,0,f,p] = (np.asarray([0.,1.,4.,16.,100.]) + f*10+p*2)*multiplier
        pp[2,0,0,1]=np.nan
        for f in range(2):
            for p in range(2):
                for a in range(5):
                    spec.input(f'{dataset_name}{f}{p}_{a}', f'{dataset_name.upper()}, f {frequencies[f]}, {pol[p]}, az {az[a]}', pp[a,0,f,p], 'm2')
        ds = grid(pp, az=az, freq=frequencies, pol=pol)
        spec.save('input_' + dataset_name, ds)
        rows = spec.save('result_' + dataset_name, ds.statistics_dataset(statistic='percentile', axes=('azimuth',), percentile=90, domain='magnitude', broadcast_reduced=True))
        for n,row in enumerate(rows):
            f=frequencies.index(float(row['frequency'])); p=pol.index(row['polarization'])
            refs=f'@{dataset_name}{f}{p}_0:@{dataset_name}{f}{p}_4'
            formula=f'=PERCENTILE.INC({refs},@q/100)'
            spec.exported(f'{dataset_name.upper()}, az {row["azimuth"]}, f {row["frequency"]}, {row["polarization"]}', '90th percentile power', formula, rows,n,'magnitude_power_linear','m2','result_'+dataset_name)
    spec.data['notes'].append('Do not pool input_a and input_b. The GUI applies the reduction to each dataset independently.')
    specs.append(spec.finish())
    return specs


def offset_and_flip():
    specs=[]
    powers=[1.,0.,100.,np.nan,4.]
    phases=[179.,np.nan,-179.,45.,np.nan]
    spec=Spec('offset','Offset','Constant dB offset scales linear power while preserving known phase and missing data.',settings='Offset +6 dB. A second output applies -6 dB to the first output.')
    spec.input('offset','Level offset',6.,'dB')
    source=grid(powers,phases)
    spec.save('input',source)
    result=offset_db(source,6.)
    rows=spec.save('result',result)
    returned=spec.save('result_reverse',offset_db(result,-6.))
    for n,(power,phase) in enumerate(zip(powers,phases)):
        pp=spec.input(f'p{n}',f'Azimuth {n}: original power',power,'m2')
        ph=spec.input(f'ph{n}',f'Azimuth {n}: original phase',phase,'deg')
        formula=f'=IF({pp}="missing","missing",{pp}*10^(@offset/10))'
        spec.exported(f'Azimuth {n}','Offset power',formula,rows,n,'magnitude_power_linear','m2')
        spec.exported(f'Azimuth {n}','Preserved phase',f'=IF(OR({pp}="missing",{pp}=0,{ph}="missing"),"missing",{ph})',rows,n,'phase_deg','deg')
        spec.exported(f'Azimuth {n}','Reverse offset power','='+pp,returned,n,'magnitude_power_linear','m2','result_reverse')
    specs.append(spec.finish())
    spec=Spec('flip_phase','Flip Phase','Complex conjugation keeps I and negates Q; power and coordinates are unchanged.',settings='Apply Flip Phase once for result, twice for result_twice.')
    values=[1+2j,-1+1e-8j,-1-1e-8j,0j,complex(np.nan,np.nan),3-4j]
    field_inputs(spec,'a',values)
    source=grid(field=values)
    spec.save('input',source)
    once=flip_phase(source)
    rows=spec.save('result',once)
    twice=spec.save('result_twice',flip_phase(once))
    for n in range(len(values)):
        field_checks(spec,rows,n,f'=@a_i{n}',f'=IF(@a_q{n}="missing","missing",-@a_q{n})')
        field_checks(spec,twice,n,f'=@a_i{n}',f'=@a_q{n}',prefix='twice',file='result_twice',case=f'Twice, azimuth {n}')
    specs.append(spec.finish())
    return specs


def medianize_case():
    spec=Spec('medianize','Medianize','Sliding medians of linear power, with periodic seam handling and empty-window preservation.',
              settings='Periodic: window 200 deg, slide 90 deg. Open/gapped: window 20 deg, slide 20 deg, nonperiodic.')
    az=[0.,90.,180.,270.,360.]; power=[1.,4.,9.,16.,1.]
    source=grid(power,az=az)
    spec.save('input_periodic',source)
    rows=spec.save('result_periodic',medianize_azimuth(source,window_degrees=200.,slide_degrees=90.,periodic=True))
    for n,value in enumerate(power): spec.input(f'p{n}',f'Periodic source at {az[n]} deg',value,'m2')
    for n,neighbors in enumerate([(3,0,1),(0,1,2),(1,2,3),(2,3,0)]):
        refs=','.join(f'@p{i}' for i in neighbors)
        spec.exported(f'Periodic center {n*90} deg','Median power',f'=MEDIAN({refs})',rows,n,'magnitude_power_linear','m2','result_periodic')
        spec.exported(f'Periodic center {n*90} deg','Center',f'={n}*90',rows,n,'azimuth','deg','result_periodic')
    az2=[0.,20.,80.,100.]; power2=[1.,np.nan,9.,16.]
    source2=grid(power2,az=az2)
    spec.save('input_gapped',source2)
    rows2=spec.save('result_gapped',medianize_azimuth(source2,window_degrees=20.,slide_degrees=20.,periodic=False))
    for n,value in enumerate(power2): spec.input(f'g{n}',f'Gapped source at {az2[n]} deg',value,'m2')
    groups=[(0,1),(1,),(),(2,),(2,3)]
    for n,neighbors in enumerate(groups):
        refs=','.join(f'@g{i}' for i in neighbors)
        formula=f'=IF(COUNT({refs})=0,"missing",MEDIAN({refs}))' if refs else '="missing"'
        spec.exported(f'Gapped center {10+n*20} deg','Median power',formula,rows2,n,'magnitude_power_linear','m2','result_gapped')
        spec.exported(f'Gapped center {10+n*20} deg','Center',f'=10+{n}*20',rows2,n,'azimuth','deg','result_gapped')
    spec.reject('Nonpositive window',lambda:medianize_azimuth(source,window_degrees=0,slide_degrees=90),contains='positive')
    spec.data['notes'].append('The 360-degree closing sample repeats the 0-degree direction and is counted once.')
    return spec.finish()


def extrusion_case():
    spec=Spec('dbke_dbsm','dBke and dBsm','Both conversion directions checked with independent linear-width and area inputs, including frequency dependence.',
              settings='Extrusion length 2.5 m. Convert input_3d to dBke and input_2d to dBsm.',
              scope='Validates implemented broadside uniform-extrusion arithmetic. Does not establish the approximation for arbitrary finite 3-D geometry.')
    spec.input('length','Extrusion length',2.5,'m')
    spec.input('c0','Speed of light',299792458.,'m/s')
    frequencies=[8.,12.]
    values3=[1.,4.,0.,0.,np.nan,16.]
    values2=[.01,.02,0.,0.,np.nan,.04]
    phase=[20.,-40.,np.nan,np.nan,np.nan,170.]
    source3=grid(values3,phase,az=[0.,30.,60.],freq=frequencies)
    source2=grid(values2,phase,az=[0.,30.,60.],freq=frequencies,units={'rcs_linear_quantity':'sigma_2d','rcs_log_unit':'dBke'})
    spec.save('input_3d',source3); spec.save('input_2d',source2)
    result2=spec.save('result_2d',convert_extrusion(source3,to='dbke',length_m=2.5))
    result3=spec.save('result_3d',convert_extrusion(source2,to='dbsm',length_m=2.5))
    for n in range(6):
        p3=spec.input(f'p3_{n}',f'3-D source sample {n}',values3[n],'m2')
        p2=spec.input(f'p2_{n}',f'2-D source sample {n}',values2[n],'m')
        ff=spec.input(f'f{n}',f'Sample {n} frequency',frequencies[n%2]*1e9,'Hz')
        width=spec.calc(f'w{n}',f'Sample {n}: expected width',f'=IF({p3}="missing","missing",{p3}*@c0/(2*@length^2*{ff}))','m')
        area=spec.calc(f's{n}',f'Sample {n}: expected area',f'=IF({p2}="missing","missing",{p2}*2*@length^2*{ff}/@c0)','m2')
        spec.exported(f'3-D to 2-D, sample {n}','Scattering width','='+width,result2,n,'magnitude_power_linear','m','result_2d')
        spec.exported(f'3-D to 2-D, sample {n}','Frequency-normalized level',f'=IF({width}="missing","missing",IF({width}=0,"-inf",10*LOG10(2*PI()*{ff}*{width}/@c0)))',result2,n,'magnitude_dbke','dBke','result_2d')
        spec.exported(f'2-D to 3-D, sample {n}','RCS area','='+area,result3,n,'magnitude_power_linear','m2','result_3d')
        spec.exported(f'2-D to 3-D, sample {n}','RCS level',f'=IF({area}="missing","missing",IF({area}=0,"-inf",10*LOG10({area})))',result3,n,'magnitude_dbsm','dBsm','result_3d')
    spec.reject('Nonpositive length',lambda:convert_extrusion(source3,to='dbke',length_m=0),contains='positive')
    spec.data['notes'].append('dBke = 10 log10(k * width), k = 2*pi*f/c. The exported linear column contains width, not k*width.')
    return spec.finish()


def generate():
    specs=coherent_cases()+incoherent_cases()+statistical_cases()+offset_and_flip()
    specs.extend([medianize_case(),extrusion_case()])
    return specs


if __name__ == '__main__':
    results=generate()
    print(f'Generated {len(results)} arithmetic/statistical operation sheets with {sum(len(s["checks"]) for s in results)} comparisons.')
