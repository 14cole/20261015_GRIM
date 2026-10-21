"""Independent Excel comparisons for coordinate and data-management buttons."""
from __future__ import annotations
import itertools
import math
import os
from pathlib import Path
import numpy as np

from common import Spec, grid, number, VALIDATION, RcsGrid
from GRIM_Backend.datasets import transforms as T


def phase_from_iq(real, imag):
    """Signed angle with explicit quadrants, avoiding ATAN2 order ambiguity."""
    return (f'=IF(AND({real}=0,{imag}=0),"missing",'
            f'IF({real}=0,IF({imag}>0,90,-90),'
            f'MOD(DEGREES(ATAN({imag}/{real}))+IF({real}<0,180,0)+180,360)-180))')


def source(s, name, *, az, el=(0.,), freq=(10.,), pol=('VV',), power=None, phase=None, units=None, extra=None):
    shape = (len(az), len(el), len(freq), len(pol))
    count = math.prod(shape)
    p = np.array(power if power is not None else np.arange(1., count + 1.), float).reshape(shape)
    ph = np.array(phase if phase is not None else np.arange(count) * 13. - 150., float).reshape(shape)
    g = grid(p, ph, az=az, el=el, freq=freq, pol=pol, units=units, extra=extra)
    s.save(name, g)
    records = []
    for k, (ia, ie, iff, ip) in enumerate(np.ndindex(shape)):
        prefix = f'{name}_{k}'
        fields = {}
        for key, value, unit in [('az', az[ia], (units or {}).get('azimuth', 'deg')),
                                 ('el', el[ie], (units or {}).get('elevation', 'deg')),
                                 ('f', freq[iff], (units or {}).get('frequency', 'GHz')),
                                 ('pol', pol[ip], ''), ('p', p[ia,ie,iff,ip], 'linear power'),
                                 ('ph', ph[ia,ie,iff,ip], 'deg')]:
            fields[key] = s.input(f'{prefix}_{key}', f'{name} sample {k+1}: {key}', value, unit)
        fields['index'] = (ia, ie, iff, ip)
        fields['power'] = p[ia,ie,iff,ip]
        fields['phase'] = ph[ia,ie,iff,ip]
        records.append(fields)
    return g, records


def mapped(s, name, result, records, *, az=None, el=None, f=None, phase=None, power=None, check_phase=True):
    """Check a known independent permutation; never derive mapping from output."""
    rows = s.save(name, result)
    s.check(name, 'Export row count', '=' + str(len(records)), len(rows), 'rows', name + '.csv')
    for i, record in enumerate(records):
        case = f'{name} sample {i+1}'
        for key, col, unit, transform in [('az', 'azimuth', '', az), ('el', 'elevation', '', el),
                                           ('f', 'frequency', '', f), ('pol', 'polarization', '', None),
                                           ('p', 'magnitude_power_linear', 'linear power', power)]:
            formula = transform(record) if transform else '=' + record[key]
            s.exported(case, col, formula, rows, i, col, unit, name)
        # Phase at exactly zero power is arbitrary; compare the zero power,
        # not the renderer/storage convention used to represent its angle.
        if check_phase and record['power'] != 0:
            if phase:
                formula = phase(record)
            else:
                formula = f'=IF(OR({record["p"]}="missing",{record["ph"]}="missing"),"missing",MOD({record["ph"]}+180,360)-180)'
            s.exported(case, 'phase', formula, rows, i, 'phase_deg', 'deg', name)
    return rows


def crop():
    s=Spec('crop_slice','Crop Slice','Selected coordinates retain the corresponding original samples.', settings='Azimuth range 10..30 deg, stride 2; elevation +5 deg; frequency 10 GHz; polarization HH.')
    g,r=source(s,'input',az=[0.,10.,20.,30.],el=[-5.,5.],freq=[9.,10.],pol=['VV','HH'])
    out=T.crop_dataset(g,azimuth_range=(10.,30.),azimuth_stride=2,elevations=[5.],frequencies=[10.],polarizations=['HH'])
    selected=[x for x in r if x['index'][0] in [1,3] and x['index'][1:]==(1,1,1)]
    mapped(s,'result',out,selected)
    s.reject('Zero stride',lambda:T.crop_dataset(g,azimuth_stride=0),contains='positive integer')
    return s.finish()


def align():
    s=Spec('align','Align','One-to-one intersection matching to the reference axes.', settings='Reference selected first; Align mode intersect. Near-matching azimuths differ by 0.0000002 deg.')
    ref,rr=source(s,'reference',az=[10.,20.,30.],power=[1.,2.,3.])
    g,r=source(s,'input',az=[0.,10.0000002,20.0000002,40.],power=[7.,8.,0.,11.],phase=[20.,30.,40.,50.])
    out=g.align_to(ref,mode='intersect')
    mapping=[r[1],r[2]]
    rows=mapped(s,'result',out,mapping,az=lambda x:'='+rr[x['index'][0]-1]['az'])
    s.reject('Exact mode mismatched grid',lambda:g.align_to(ref,mode='exact'))
    return s.finish()


def regrid():
    s=Spec('regrid','Regrid','Linear interpolation of I and Q, including exact source nodes.', settings='Azimuth target [0,5,10,15,20] deg; no extrapolation.', scope='Checks linear interpolation arithmetic and missing support, not adequacy of source angular sampling.')
    g,r=source(s,'input',az=[0.,10.,20.],power=[1.,5.,9.],phase=[0.,math.degrees(math.atan2(1,2)),0.])
    rows=s.save('result',T.regrid_axis(g,'azimuth',values=[0.,5.,10.,15.,20.]))
    for i,angle in enumerate([0.,5.,10.,15.,20.]):
        lo=min(i//2,1); hi=lo+1
        target=s.input(f'target_{i}',f'Target azimuth {i+1}',angle,'deg')
        w=s.calc(f'w{i}',f'Fraction between source samples {lo+1} and {hi+1}',f'=({target}-{r[lo]["az"]})/({r[hi]["az"]}-{r[lo]["az"]})')
        re=s.calc(f're{i}',f'Interpolated I at {angle:g} deg',f'=(1-{w})*SQRT({r[lo]["p"]})*COS(RADIANS({r[lo]["ph"]}))+{w}*SQRT({r[hi]["p"]})*COS(RADIANS({r[hi]["ph"]}))')
        im=s.calc(f'im{i}',f'Interpolated Q at {angle:g} deg',f'=(1-{w})*SQRT({r[lo]["p"]})*SIN(RADIANS({r[lo]["ph"]}))+{w}*SQRT({r[hi]["p"]})*SIN(RADIANS({r[hi]["ph"]}))')
        s.exported(f'Sample {i+1}','Azimuth','='+target,rows,i,'azimuth','deg')
        s.exported(f'Sample {i+1}','Power',f'={re}^2+{im}^2',rows,i,'magnitude_power_linear','linear power')
        s.exported(f'Sample {i+1}','Phase',phase_from_iq(re,im),rows,i,'phase_deg','deg')
    s.reject('Extrapolation',lambda:T.regrid_axis(g,'azimuth',values=[-1.,0.,10.]),contains='extrapolat')
    s.reject('Repeated targets',lambda:T.regrid_axis(g,'azimuth',values=[0.,0.,10.]),contains='strictly increasing')
    return s.finish()


def mirror():
    s=Spec('mirror','Mirror','Azimuth reflection and sample permutation.',settings='Mirror about 15 deg; output sorted ascending.')
    center=s.input('center','Mirror reference',15.,'deg')
    g,r=source(s,'input',az=[-10.,5.,40.],el=[-5.,5.],power=[1.,2.,0.,4.,float('nan'),6.])
    mapped(s,'result',g.mirror_about_azimuth(15.),r[4:6]+r[2:4]+r[0:2],az=lambda x:f'=2*{center}-{x["az"]}')
    return s.finish()


def wrap():
    s=Spec('wrap','Wrap','Wrap azimuth and phase; keep opening sample at a conflicting closed seam.',settings='0..360 azimuth; signed phase. Source -180 and +180 share the seam; default keeps -180.')
    g,r=source(s,'input',az=[-180.,-90.,0.,90.,180.],power=[1.,2.,0.,4.,99.],phase=[-181.,-90.,0.,181.,179.])
    wrapped=g.wrap_azimuth('0_360').wrap_phase('-180_180')
    mapped(s,'result',wrapped,[r[2],r[3],r[0],r[1]],az=lambda x:f'=MOD({x["az"]},360)')
    s.reject('Strict conflicting seam',lambda:g.wrap_azimuth('0_360',seam_conflict='error'),contains='conflict')
    # A second interval exercises both azimuth and phase wrapping directions.
    g2,r2=source(s,'input_unsigned',az=[0.,90.,180.,270.],power=[1.,2.,3.,4.],phase=[-10.,10.,179.,-179.])
    mapped(s,'result_signed',g2.wrap_azimuth('-180_180').wrap_phase('0_360'),[r2[2],r2[3],r2[0],r2[1]],az=lambda x:f'=MOD({x["az"]}+180,360)-180',phase=lambda x:f'=MOD({x["ph"]},360)')
    return s.finish()


def shift():
    s=Spec('shift','Shift','Coordinate translation and constant phase rotation.',settings='Azimuth +15 deg; elevation -7 deg; phase +30 deg.')
    s.data['notes'].append('Phase is not compared at exactly zero power because a zero complex field has no physical phase.')
    da=s.input('da','Azimuth shift',15.,'deg');de=s.input('de','Elevation shift',-7.,'deg');dp=s.input('dp','Phase shift',30.,'deg')
    g,r=source(s,'input',az=[0.,90.],el=[-5.,5.],power=[1.,0.,3.,float('nan')],phase=[170.,0.,-175.,float('nan')])
    mapped(s,'result',T.shift_dataset(g,azimuth_degrees=15.,elevation_degrees=-7.,phase_degrees=30.),r,az=lambda x:f'={x["az"]}+{da}',el=lambda x:f'={x["el"]}+{de}',phase=lambda x:f'=IF(OR({x["p"]}="missing",{x["ph"]}="missing"),"missing",MOD({x["ph"]}+{dp}+180,360)-180)')
    return s.finish()


def rounded():
    s=Spec('round','Round','Coordinate rounding preserves the response and rejects coordinate collisions.',settings='Round azimuth/elevation/frequency to 1 decimal. Includes exact halfway values using ties-to-even rounding.')
    d=s.input('digits','Decimals',1)
    g,r=source(s,'input',az=[-1.25,0.25,2.34],el=[-2.36,3.24],freq=[9.14,10.26])
    # Excel ROUND uses half-away-from-zero. The explicit even-tie expression
    # independently expresses GRIM/NumPy's intended ties-to-even policy.
    def formula(ref):
        x=f'({ref}*10^{d})'
        return f'=IF(MOD(ABS({x}),1)=0.5,2*ROUND({x}/2,0),ROUND({x},0))/10^{d}'
    mapped(s,'result',g.round_axes(1),r,az=lambda x:formula(x['az']),el=lambda x:formula(x['el']),f=lambda x:formula(x['f']))
    coll,r2=source(s,'input_collision',az=[0.01,0.02],power=[1.,2.])
    s.reject('Coordinates collapse',lambda:coll.round_axes(1),contains='duplicate')
    s.data['notes'].append('Excel ROUND alone uses a different tie rule. The worksheet explicitly applies ties-to-even for halfway coordinates.')
    return s.finish()


def duplicate():
    s=Spec('duplicate','Duplicate','A copy retains all coordinates and samples.',settings='Duplicate dataset; compare every sample and metadata units.')
    g,r=source(s,'input',az=[0.,45.],el=[-5.,5.],freq=[9.,10.],power=[1.,0.,float('nan'),4.,5.,6.,7.,8.])
    copied=T.duplicate_dataset(g)
    mapped(s,'result',copied,r)
    copied.set_power_phase((0,0,0,0),power=12345.)
    s.check('Independent edit','Original first power','='+r[0]['p'],float(g.power[0,0,0,0]),'linear power','Original memory after editing duplicate')
    return s.finish()


def units():
    s=Spec('axis_units','Axis Units','Equivalent unit conversions preserve response samples.',settings='Degrees to radians, GHz to Hz; then inverse conversion.')
    g,r=source(s,'input',az=[0.,90.],el=[-30.,30.],freq=[8.,12.])
    result=g.convert_axis_units(azimuth='rad',elevation='rad',frequency='Hz')
    rows=mapped(s,'result',result,r,az=lambda x:f'=RADIANS({x["az"]})',el=lambda x:f'=RADIANS({x["el"]})',f=lambda x:f'={x["f"]}*10^9')
    for col,value in [('azimuth_unit','rad'),('elevation_unit','rad'),('frequency_unit','Hz')]:
        s.exported('Converted units',col,f'="{value}"',rows,0,col)
    mapped(s,'result_inverse',result.convert_axis_units(azimuth='deg',elevation='deg',frequency='GHz'),r)
    return s.finish()


def swap():
    s=Spec('swap_el_az','Swap El Az','Transpose both angular coordinates and every response sample.',settings='Swap elevation and azimuth; mixed angle units test metadata exchange.')
    g,r=source(s,'input',az=[0.,90.],el=[-0.2,0.,0.2],freq=[9.,10.],units={'elevation':'rad'})
    rec=sorted(r,key=lambda x:(x['index'][1],x['index'][0],x['index'][2],x['index'][3]))
    rows=mapped(s,'result',g.swap_elevation_azimuth(),rec,az=lambda x:'='+x['el'],el=lambda x:'='+x['az'])
    s.exported('Unit exchange','Azimuth unit','="rad"',rows,0,'azimuth_unit')
    s.exported('Unit exchange','Elevation unit','="deg"',rows,0,'elevation_unit')
    return s.finish()


def join():
    s=Spec('join_merge','Join Merge','Union existing samples and resolve overlaps by the selected policy.',settings='Join default rejects conflicts. Merge tests priority-first/last, power-mean, coherent-mean.',scope='Checks a one-axis union and all overlap policies with missing and zero samples; higher-dimensional union resource limits are separate.')
    a,ra=source(s,'input_a',az=[0.,10.,20.],power=[1.,4.,0.],phase=[0.,0.,0.])
    b,rb=source(s,'input_b',az=[10.,20.,30.],power=[16.,float('nan'),9.],phase=[180.,float('nan'),90.])
    s.reject('Conflicting join overlap',lambda:T.join_datasets(a,b),contains='conflict')
    for policy in ['priority-first','priority-last','power-mean','coherent-mean']:
        name='result_'+policy.replace('-','_')
        rows=s.save(name,T.stitch_datasets(a,b,policy=policy,metadata_attested=True))
        records=[ra[0],ra[1],ra[2],rb[2]]
        for i,r in enumerate(records):
            formula='='+r['p']
            if i==1:
                if policy=='priority-last':formula='='+rb[0]['p']
                elif policy=='power-mean':formula=f'=AVERAGE({ra[1]["p"]},{rb[0]["p"]})'
                elif policy=='coherent-mean':
                    re=s.calc('merge_i','Overlap mean I',f'=(SQRT({ra[1]["p"]})*COS(RADIANS({ra[1]["ph"]}))+SQRT({rb[0]["p"]})*COS(RADIANS({rb[0]["ph"]})))/2')
                    im=s.calc('merge_q','Overlap mean Q',f'=(SQRT({ra[1]["p"]})*SIN(RADIANS({ra[1]["ph"]}))+SQRT({rb[0]["p"]})*SIN(RADIANS({rb[0]["ph"]})))/2')
                    formula=f'={re}^2+{im}^2'
            s.exported(f'{policy} sample {i+1}','Azimuth','='+r['az'],rows,i,'azimuth','deg',name)
            s.exported(f'{policy} sample {i+1}','Power',formula,rows,i,'magnitude_power_linear','linear power',name)
            if i==1:
                ph = ('="missing"' if policy=='power-mean' else
                      f'=MOD({rb[0]["ph"]}+180,360)-180' if policy=='priority-last' else
                      phase_from_iq(re,im) if policy=='coherent-mean' else
                      f'=MOD({ra[1]["ph"]}+180,360)-180')
                s.exported(f'{policy} overlap','Phase',ph,rows,i,'phase_deg','deg',name)
    c,rc=source(s,'input_c_no_overlap',az=[40.,50.],power=[7.,8.])
    mapped(s,'result_union',T.join_datasets(a,c),ra+rc)
    return s.finish()


def overlap():
    s=Spec('overlap','Overlap','All selected datasets share common coordinates and finite-cell support.',settings='Inputs overlap at azimuths 10 and 20 deg; VV/HH. A missing HH cell masks HH in both outputs.')
    a,ra=source(s,'input_a',az=[0.,10.,20.],pol=['VV','HH'],power=[1.,2.,3.,4.,0.,6.])
    b,rb=source(s,'input_b',az=[10.,20.,30.],pol=['VV','HH'],power=[7.,float('nan'),9.,10.,11.,12.])
    out=RcsGrid.overlap_many(a,b)
    # Common polarization labels are lexically sorted (HH before VV), unlike
    # these intentionally reversed source channels.
    for k,(g,recs) in enumerate(zip(out,[[ra[3],ra[2],ra[5],ra[4]],[rb[1],rb[0],rb[3],rb[2]]])):
        rows=s.save(f'result_{k+1}',g)
        for i,r in enumerate(recs):
            s.exported(f'Dataset {k+1} sample {i+1}','Azimuth','='+r['az'],rows,i,'azimuth','deg',f'result_{k+1}')
            s.exported(f'Dataset {k+1} sample {i+1}','Polarization','='+r['pol'],rows,i,'polarization','',f'result_{k+1}')
            formula='="missing"' if i==0 else '='+r['p']
            s.exported(f'Dataset {k+1} sample {i+1}','Common finite power',formula,rows,i,'magnitude_power_linear','linear power',f'result_{k+1}')
    return s.finish()


def sentri():
    s=Spec('sentri_el_grim','SENTRi El to GRIM','Native polar theta converts to signed elevation without changing response samples.',settings='Theta [0,90,180] deg; phi [-90,0,90] deg. Elevation = 90 - theta; azimuth modulo 360.',scope='Exact relabel and permutation for explicitly native SENTRi coordinates; correct source-convention identification remains required.',category='BETA dataset operation')
    g,r=source(s,'input',az=[-90.,0.,90.],el=[0.,90.,180.],freq=[9.,10.],units={'elevation_coordinate_convention':'sentri_theta_top_zero'})
    records=sorted(r,key=lambda x:([2,0,1][x['index'][0]],-x['index'][1],x['index'][2]))
    out=g.convert_sentri_elevation_to_grim()
    mapped(s,'result',out,records,az=lambda x:f'=MOD({x["az"]},360)',el=lambda x:f'=90-{x["el"]}')
    s.reject('Already converted',out.convert_sentri_elevation_to_grim,contains='already')
    bad,rb=source(s,'input_bad_theta',az=[0.],el=[-1.,90.],power=[1.,2.])
    s.reject('Theta outside 0..180',bad.convert_sentri_elevation_to_grim,contains='0..180')
    return s.finish()


def elaz():
    s=Spec('el_az360','El to Az360','Stitch an explicitly attested pair of opposite elevation acquisitions.',settings='Elevations -10/+10 deg; base azimuth [0,90] deg. Higher cut shifts +180 deg; output elevation -10 deg.',scope='Acquisition-specific relabel only; does not prove equivalence of arbitrary elevation and azimuth sweeps.',category='BETA dataset operation')
    g,r=source(s,'input',az=[0.,90.],el=[-10.,10.],power=[1.,2.,3.,4.],phase=[0.,30.,60.,90.])
    out=g.combine_elevation_pair_to_azimuth_360(assumptions_attested=True)
    mapped(s,'result',out,[r[0],r[2],r[1],r[3]],az=lambda x:f'={x["az"]}+IF({x["el"]}>0,180,0)',el=lambda x:'='+r[0]['el'])
    s.reject('No acquisition attestation',g.combine_elevation_pair_to_azimuth_360,contains='acquisition-specific')
    bad,rr=source(s,'input_asymmetric',az=[0.,90.],el=[-5.,10.],power=[1.,2.,3.,4.])
    s.reject('Asymmetric elevation pair',lambda:bad.combine_elevation_pair_to_azimuth_360(assumptions_attested=True),contains='equal-and-opposite')
    seam,rs=source(s,'input_seam_conflict',az=[0.,180.],el=[-10.,10.],power=[1.,2.,3.,4.])
    s.reject('Conflicting 180-degree seam',lambda:seam.combine_elevation_pair_to_azimuth_360(assumptions_attested=True),contains='conflict')
    return s.finish()


def decimate():
    s=Spec('decimate','Decimate','Finite boxcar means followed by integer-factor downsampling.',settings='Factor 2, azimuth; power and coherent modes. Test keeping/dropping final partial bin.',scope='Validates block averaging and support handling. A boxcar is not a universal guarantee of adequate anti-alias attenuation.',category='BETA dataset operation')
    g,r=source(s,'input_power',az=[0.,1.,2.,3.,4.],power=[0.,2.,float('nan'),6.,10.],phase=[0.,0.,float('nan'),0.,0.])
    for partial, groups in [('keep',[(0,1),(2,3),(4,)]),('drop',[(0,1),(2,3)])]:
        name='result_power_'+partial
        rows=s.save(name,T.decimate_axis(g,axis='azimuth',factor=2,mode='power',partial=partial))
        s.check(partial,'Output count','='+str(len(groups)),len(rows),'rows',name+'.csv')
        for i,idx in enumerate(groups):
            az=','.join(r[k]['az'] for k in idx);p=','.join(r[k]['p'] for k in idx)
            s.exported(f'{partial} bin {i+1}','Coordinate center',f'=AVERAGE({az})',rows,i,'azimuth','deg',name)
            s.exported(f'{partial} bin {i+1}','Power mean',f'=AVERAGE({p})',rows,i,'magnitude_power_linear','linear power',name)
            s.exported(f'{partial} bin {i+1}','Unknown aggregate phase','="missing"',rows,i,'phase_deg','deg',name)
    c,rc=source(s,'input_coherent',az=[0.,1.,2.,3.,4.],power=[1.,1.,4.,4.,9.],phase=[0.,180.,90.,90.,45.])
    rows=s.save('result_coherent',T.decimate_axis(c,axis='azimuth',factor=2,mode='coherent',metadata_attested=True))
    for i,idx in enumerate([(0,1),(2,3),(4,)]):
        re=s.calc(f'di{i}',f'Coherent bin {i+1} mean I','=('+'+'.join(f'SQRT({rc[k]["p"]})*COS(RADIANS({rc[k]["ph"]}))' for k in idx)+f')/{len(idx)}')
        im=s.calc(f'dq{i}',f'Coherent bin {i+1} mean Q','=('+'+'.join(f'SQRT({rc[k]["p"]})*SIN(RADIANS({rc[k]["ph"]}))' for k in idx)+f')/{len(idx)}')
        s.exported(f'Coherent bin {i+1}','Power of mean field',f'={re}^2+{im}^2',rows,i,'magnitude_power_linear','linear power','result_coherent')
    irregular,ri=source(s,'input_irregular',az=[0.,1.,3.,4.],power=[1.,2.,3.,4.])
    s.reject('Irregular source sampling',lambda:T.decimate_axis(irregular,axis='azimuth',factor=2),contains='uniformly')
    return s.finish()


def generate():
    results=[]
    for fn in [crop,align,regrid,mirror,wrap,shift,rounded,duplicate,units,swap,join,overlap,sentri,elaz,decimate]:
        value=fn(); results.append(value)
        print(f'{value["slug"]}: {len(value["checks"])} comparisons',flush=True)
    return results


if __name__=='__main__':
    tmp=VALIDATION/'.work'/'tmp';tmp.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(tmp);os.environ['TMP']=str(tmp)
    generate()
