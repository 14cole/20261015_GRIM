"""Small independently specified physical-operation qualification fixtures.

Expected formulas are plain Excel arithmetic and direct DFT sums. GRIM is used
only for the actual branch. Actual fields are reconstructed from CSV columns.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from common import Spec, VALIDATION, grid, number
from GRIM_Backend.datasets.constants import C0
from GRIM_Backend.datasets.transforms import translate_phase_center, time_gate, wedge_to_conic


def csv_field(row):
    power, phase = number(row, 'magnitude_power_linear'), number(row, 'phase_deg')
    if power == 'missing':
        return 'missing', 'missing'
    if power == 0:
        return 0., 0.
    if phase == 'missing':
        return 'missing', 'missing'
    return math.sqrt(power) * math.cos(math.radians(phase)), math.sqrt(power) * math.sin(math.radians(phase))


def check_field(spec, case, rows, row, real_formula, imag_formula, file):
    real, imag = csv_field(rows[row])
    origin = f'{file}.csv row {row + 2}: sqrt(magnitude_power_linear) and phase_deg'
    spec.check(case, 'Field I', real_formula, real, 'sqrt(m2)', origin)
    spec.check(case, 'Field Q', imag_formula, imag, 'sqrt(m2)', origin)


def phase_center_case():
    s = Spec('phase_center', 'Phase Center',
             'Known body-axis phase ramp; positive and negative time conventions.',
             settings='Translation x=0.031 m, y=-0.017 m, z=0.009 m; angles in degrees.',
             scope='Checks phase sign, two-way factor, direction cosines, magnitude preservation, and point recentering. Does not establish unknown acquisition frames.',
             category='BETA dataset operation')
    x, y, z = .031, -.017, .009
    for key, value in [('x',x),('y',y),('z',z),('c',C0)]:
        s.input(key, key if key != 'c' else 'Speed of light', value, 'm/s' if key == 'c' else 'm')
    az, el, freq = [-75.,0.,45.], [-20.,30.], [8.,11.]
    shape = (3,2,2,1)
    powers = np.array([1.,4.,9.,.25,2.,7.,3.,8.,5.,6.,10.,11.]).reshape(shape)
    phases = np.array([10.,-35.,179.,-179.,45.,90.,-90.,25.,120.,-150.,70.,-10.]).reshape(shape)
    for sign, label in [(1,'positive_time'),(-1,'negative_time')]:
        source = grid(powers, sign*phases, az=az, el=el, freq=freq,
                      units={'time_convention': 'exp(+jwt)' if sign == 1 else 'exp(-jwt)'})
        s.save('input_' + label, source)
        rows = s.save('result_' + label, translate_phase_center(source,x_m=x,y_m=y,z_m=z))
        for n, row in enumerate(rows):
            ai,ei,fi,_ = np.unravel_index(n,shape)
            tag=f'{label}_{n}'
            for name,value,unit in [('az',az[ai],'deg'),('el',el[ei],'deg'),('f',freq[fi]*1e9,'Hz'),
                                    ('p',powers.ravel()[n],'m2'),('phase',sign*phases.ravel()[n],'deg')]:
                s.input(tag+'_'+name,f'{label} sample {n+1}: {name}',value,unit)
            projection=s.calc(tag+'_projection',f'{label} sample {n+1}: body projection',
                f'=COS(RADIANS(@{tag}_el))*COS(RADIANS(@{tag}_az))*@x+COS(RADIANS(@{tag}_el))*SIN(RADIANS(@{tag}_az))*@y+SIN(RADIANS(@{tag}_el))*@z','m')
            angle=s.calc(tag+'_angle',f'{label} sample {n+1}: output angle',
                f'=RADIANS(@{tag}_phase)-{sign}*4*PI()*@{tag}_f*{projection}/@c','rad')
            check_field(s,f'{label} sample {n+1}',rows,n,
                        f'=SQRT(@{tag}_p)*COS({angle})',f'=SQRT(@{tag}_p)*SIN({angle})','result_'+label)
            s.exported(f'{label} sample {n+1}','Power preserved',f'=@{tag}_p',rows,n,'magnitude_power_linear','m2','result_'+label)
    # A target at the translated point must have zero phase after recentering.
    projection=np.cos(np.deg2rad(el))[None,:]*np.cos(np.deg2rad(az))[:,None]*x+np.cos(np.deg2rad(el))[None,:]*np.sin(np.deg2rad(az))[:,None]*y+np.sin(np.deg2rad(el))[None,:]*z
    point=2*np.exp(4j*np.pi*projection[:,:,None,None]*np.asarray(freq)[None,None,:,None]*1e9/C0)
    point_grid=grid(field=point,az=az,el=el,freq=freq)
    s.save('input_point_at_translation',point_grid)
    rows=s.save('result_point_at_origin',translate_phase_center(point_grid,x_m=x,y_m=y,z_m=z))
    s.input('point_amplitude','Point amplitude',2.,'sqrt(m2)')
    for n in range(len(rows)):
        check_field(s,f'Recenter point {n+1}',rows,n,'=@point_amplitude','=0','result_point_at_origin')
    s.reject('Zero translation',lambda:translate_phase_center(point_grid,x_m=0,y_m=0,z_m=0),contains='zero')
    no_phase=grid([1.,2.],az=[0.,30.])
    s.save('input_missing_phase',no_phase)
    s.reject('No phase available',lambda:translate_phase_center(no_phase,x_m=x,y_m=y,z_m=z),contains='phase')
    return s.finish()


def range_cal_case():
    s=Spec('range_cal','Range Cal','Complex substitution calibration with independent signed-range equations.',
           settings='Measured and exact references use matching grids; correction limit 60 dB; signed offset +/-0.0125 m.',
           scope='Synthetic complex-gain and delay recovery. Does not establish measured-standard quality or acquisition comparability.',category='BETA dataset operation')
    s.input('c','Speed of light',C0,'m/s')
    frequencies=np.array([8.,10.,12.])
    truth=np.array([1.+2j, .5-.4j, 2.-.2j])
    exact=np.array([2.+.5j, 1.5-.2j, .8+.4j])
    gain=np.array([.5+.3j, .8-.4j, .7+.2j])
    for sign in [1,-1]:
        tag='plus' if sign==1 else 'minus'
        offset=sign*.0125
        s.input(tag+'_offset',f'{tag} range offset',offset,'m')
        measured=exact*gain*np.exp(-4j*np.pi*frequencies*1e9*offset/C0)
        dut=truth*gain
        dg=grid(field=dut,az=[0.],freq=frequencies)
        mg=grid(field=measured,az=[0.],freq=frequencies)
        eg=grid(field=exact,az=[0.],freq=frequencies)
        s.save('input_dut_'+tag,dg);s.save('input_measured_cal_'+tag,mg);s.save('input_exact_'+tag,eg)
        rows=s.save('result_'+tag,dg.range_calibrate(mg,eg,offset,convention_attested=True))
        for j in range(3):
            t=f'{tag}_{j}'
            for name,value,unit in [('f',frequencies[j]*1e9,'Hz'),('di',dut[j].real,'sqrt(m2)'),('dq',dut[j].imag,'sqrt(m2)'),
                                    ('ei',exact[j].real,'sqrt(m2)'),('eq',exact[j].imag,'sqrt(m2)'),
                                    ('mi',measured[j].real,'sqrt(m2)'),('mq',measured[j].imag,'sqrt(m2)'),
                                    ('ti',truth[j].real,'sqrt(m2)'),('tq',truth[j].imag,'sqrt(m2)')]:
                s.input(t+'_'+name,f'{tag} f{j+1}: {name}',value,unit)
            den=s.calc(t+'_den',f'{tag} f{j+1}: measured power',f'=@{t}_mi^2+@{t}_mq^2','m2')
            qi=s.calc(t+'_qi',f'{tag} f{j+1}: exact/measured I',f'=(@{t}_ei*@{t}_mi+@{t}_eq*@{t}_mq)/{den}')
            qq=s.calc(t+'_qq',f'{tag} f{j+1}: exact/measured Q',f'=(@{t}_eq*@{t}_mi-@{t}_ei*@{t}_mq)/{den}')
            pi=s.calc(t+'_pi',f'{tag} f{j+1}: DUT times quotient I',f'=@{t}_di*{qi}-@{t}_dq*{qq}')
            pq=s.calc(t+'_pq',f'{tag} f{j+1}: DUT times quotient Q',f'=@{t}_di*{qq}+@{t}_dq*{qi}')
            angle=s.calc(t+'_angle',f'{tag} f{j+1}: signed range angle',f'=-4*PI()*@{t}_f*@{tag}_offset/@c','rad')
            check_field(s,f'{tag} f{j+1} formula',rows,j,f'={pi}*COS({angle})-{pq}*SIN({angle})',f'={pi}*SIN({angle})+{pq}*COS({angle})','result_'+tag)
            check_field(s,f'{tag} f{j+1} known truth',rows,j,f'=@{t}_ti',f'=@{t}_tq','result_'+tag)
    zero_exact=grid(field=[0j,1+0j,2+0j],az=[0.],freq=frequencies)
    s.save('input_zero_exact_standard',zero_exact)
    zero_rows=s.save('result_zero_exact_standard',dg.range_calibrate(mg,zero_exact,offset,convention_attested=True))
    s.exported('Zero exact standard','Uninformative bin remains missing','="missing"',zero_rows,0,'magnitude_power_linear','m2','result_zero_exact_standard')
    wrong=grid(field=measured,az=[0.],freq=frequencies,units={'time_convention':'exp(-jwt)'})
    s.save('input_wrong_time',wrong)
    s.reject('Opposite declared time law',lambda:dg.range_calibrate(wrong,eg,offset),contains='declared metadata')
    return s.finish()


def time_gate_case():
    s=Spec('time_gate','Time Gate','Explicit inverse and forward DFT sums independently reproduce the gated field.',
           settings='4 frequencies from 1 to 4 GHz; 16-point padding; gate -0.03 to 0.04 m; taper 0.4; compensation off except center-point control.',
           scope='Checks direct DFT, taper, keep/remove/full gates, time sign and center-point compensation. Does not prove scatterer separation in arbitrary measured data.',category='BETA dataset operation')
    count,size=4,16
    field=np.array([1+2j,2-.5j,-.5+.25j,3+1j])
    frequencies=np.arange(1.,5.)
    start,stop,taper=-.03,.04,.4
    for name,value,unit in [('c',C0,'m/s'),('df',1e9,'Hz'),('n',count,''),('size',size,''),('start',start,'m'),('stop',stop,'m'),('taper',taper,'')]:
        s.input(name,name,value,unit)
    for n,value in enumerate(field):
        s.input(f'i{n}',f'Frequency {n+1} input I',value.real,'sqrt(m2)')
        s.input(f'q{n}',f'Frequency {n+1} input Q',value.imag,'sqrt(m2)')
    edge=s.calc('edge','Raised cosine edge width','=@taper*(@stop-@start)/2','m')
    for k in range(size):
        r=s.calc(f'r{k}',f'Range cell {k}',f'={k if k<size//2 else k-size}*@c/(2*@df*@size)','m')
        mask=s.calc(f'm{k}',f'Range cell {k} gate',f'=IF(AND({r}>=@start,{r}<=@stop),IF({r}<@start+{edge},0.5-0.5*COS(PI()*({r}-@start)/{edge}),IF({r}>@stop-{edge},0.5-0.5*COS(PI()*(@stop-{r})/{edge}),1)),0)')
        re_terms=[f'(@i{n}*COS(2*PI()*{k}*{n}/@size)-@q{n}*SIN(2*PI()*{k}*{n}/@size))' for n in range(count)]
        im_terms=[f'(@i{n}*SIN(2*PI()*{k}*{n}/@size)+@q{n}*COS(2*PI()*{k}*{n}/@size))' for n in range(count)]
        s.calc(f'ri{k}',f'Gated range cell {k} I',f'=({"+".join(re_terms)})/@size*{mask}')
        s.calc(f'rq{k}',f'Gated range cell {k} Q',f'=({"+".join(im_terms)})/@size*{mask}')
    for n in range(count):
        re_terms=[f'(@ri{k}*COS(2*PI()*{k}*{n}/@size)+@rq{k}*SIN(2*PI()*{k}*{n}/@size))' for k in range(size)]
        im_terms=[f'(@rq{k}*COS(2*PI()*{k}*{n}/@size)-@ri{k}*SIN(2*PI()*{k}*{n}/@size))' for k in range(size)]
        s.calc(f'out_i{n}',f'Forward DFT frequency {n+1} I','='+'+'.join(re_terms),'sqrt(m2)')
        s.calc(f'out_q{n}',f'Forward DFT frequency {n+1} Q','='+'+'.join(im_terms),'sqrt(m2)')
    source=grid(field=field,az=[0.],freq=frequencies)
    s.save('input',source)
    for mode in ['keep','remove']:
        rows=s.save('result_'+mode,time_gate(source,start_m=start,stop_m=stop,taper=taper,mode=mode,compensate=False))
        for n in range(count):
            i=f'=@out_i{n}' if mode=='keep' else f'=@i{n}-@out_i{n}'
            q=f'=@out_q{n}' if mode=='keep' else f'=@q{n}-@out_q{n}'
            check_field(s,f'{mode} frequency {n+1}',rows,n,i,q,'result_'+mode)
    negative=grid(field=np.conj(field),az=[0.],freq=frequencies,units={'time_convention':'exp(-jwt)'})
    s.save('input_negative_time',negative)
    rows=s.save('result_negative_time',time_gate(negative,start_m=start,stop_m=stop,taper=taper,compensate=False))
    for n in range(count):
        check_field(s,f'Negative time frequency {n+1}',rows,n,f'=@out_i{n}',f'=-@out_q{n}','result_negative_time')
    half=C0/(4e9)
    rows=s.save('result_full_gate',time_gate(source,start_m=-half,stop_m=half,taper=0,compensate=False))
    for n in range(count):
        check_field(s,f'Full gate frequency {n+1}',rows,n,f'=@i{n}',f'=@q{n}','result_full_gate')
    center=.5*(start+stop)
    point=np.exp(-4j*np.pi*frequencies*1e9*center/C0)
    point_grid=grid(field=point,az=[0.],freq=frequencies)
    s.save('input_center_point',point_grid)
    rows=s.save('result_compensated_center',time_gate(point_grid,start_m=start,stop_m=stop,taper=taper,compensate=True))
    for n in range(count):
        phase=f'-4*PI()*{n+1}*@df*(@start+@stop)/2/@c'
        check_field(s,f'Compensated center frequency {n+1}',rows,n,f'=COS({phase})',f'=SIN({phase})','result_compensated_center')
    s.reject('Gate narrower than native range cell',lambda:time_gate(source,start_m=0,stop_m=.001),contains='narrower')
    irregular=grid(field=[1+0j,2+0j,3+0j],az=[0.],freq=[1.,2.,4.])
    s.save('input_irregular_frequency',irregular)
    s.reject('Irregular frequencies',lambda:time_gate(irregular,start_m=start,stop_m=stop),contains='uniform')
    return s.finish()


def wedge_case():
    s=Spec('wedge_conic','Wedge Conic','Exact waterline relabel plus independent tilted Jones-matrix rotation.',
           settings='Waterline: 0/90/180/270 degrees; tilted: 30-degree azimuth steps and tilts -30/-15/0/15/30 degrees; VV HH VH.',
           scope='Waterline mapping, a constant reciprocal Jones control, and unsupported side directions. Does not qualify interpolation of arbitrary sparse wedge measurements.',category='BETA dataset operation')
    powers=[3.,9.,18.,np.nan]
    source=grid(power=powers,az=[0.,90.,180.,270.],pol=['HH'])
    s.save('input_waterline',source)
    rows=s.save('result_waterline',wedge_to_conic(source))
    mapping=[2,1,0,3]
    for j,p in enumerate(powers):
        s.input(f'water_p{j}',f'Waterline input {j*90} deg power',p,'m2')
        s.input(f'water_a{j}',f'Waterline input {j*90} deg angle',j*90,'deg')
    for n,index in enumerate(mapping):
        s.exported(f'Waterline row {n+1}','Conic azimuth',f'=MOD(-@water_a{index}+180,360)-180',rows,n,'azimuth','deg','result_waterline')
        s.exported(f'Waterline row {n+1}','Power',f'=@water_p{index}',rows,n,'magnitude_power_linear','m2','result_waterline')
        s.exported(f'Waterline row {n+1}','Unavailable phase','="missing"',rows,n,'phase_deg','deg','result_waterline')
    az=np.arange(-180.,180.,30.);el=np.arange(-30.,31.,15.)
    vv,hh,vh=2.+.4j,.8+.1j,.35-.2j
    field=np.empty((len(az),len(el),1,3),complex)
    field[...,0]=vv;field[...,1]=hh;field[...,2]=vh
    tilted=grid(field=field,az=az,el=el,pol=['VV','HH','VH'],units={'angular_coordinate_system':'wedge_turntable'})
    s.save('input_tilted',tilted)
    converted=wedge_to_conic(tilted)
    rows=s.save('result_tilted',converted)
    for name,value in [('vvi',vv.real),('vvq',vv.imag),('hhi',hh.real),('hhq',hh.imag),('vhi',vh.real),('vhq',vh.imag)]:
        s.input(name,'Constant wedge '+name,value,'sqrt(m2)')
    for c,(lon,lat) in enumerate([(30.,15.),(-30.,-15.)]):
        t=f't{c}'
        s.input(t+'_lon',f'Control {c+1} conic azimuth',lon,'deg')
        s.input(t+'_lat',f'Control {c+1} conic elevation',lat,'deg')
        x=s.calc(t+'_x',f'Control {c+1} direction x',f'=COS(RADIANS(@{t}_lat))*COS(RADIANS(@{t}_lon))')
        y=s.calc(t+'_y',f'Control {c+1} direction y',f'=COS(RADIANS(@{t}_lat))*SIN(RADIANS(@{t}_lon))')
        z=s.calc(t+'_z',f'Control {c+1} direction z',f'=SIN(RADIANS(@{t}_lat))')
        sp=s.calc(t+'_sp',f'Control {c+1} sin mechanical angle',f'=-{y}')
        cp=s.calc(t+'_cp',f'Control {c+1} cos mechanical angle',f'=SQRT(1-{sp}^2)')
        st=s.calc(t+'_st',f'Control {c+1} sin tilt',f'={z}/{cp}')
        ct=s.calc(t+'_ct',f'Control {c+1} cos tilt',f'={x}/{cp}')
        a=s.calc(t+'_a',f'Control {c+1} wedge V dot conic V',f'={st}*SIN(RADIANS(@{t}_lat))*COS(RADIANS(@{t}_lon))+{ct}*COS(RADIANS(@{t}_lat))')
        b=s.calc(t+'_b',f'Control {c+1} wedge V dot conic H',f'={st}*SIN(RADIANS(@{t}_lon))')
        cval=s.calc(t+'_c',f'Control {c+1} wedge H dot conic V',f'=-{sp}*{ct}*SIN(RADIANS(@{t}_lat))*COS(RADIANS(@{t}_lon))-{cp}*SIN(RADIANS(@{t}_lat))*SIN(RADIANS(@{t}_lon))+{sp}*{st}*COS(RADIANS(@{t}_lat))')
        d=s.calc(t+'_d',f'Control {c+1} wedge H dot conic H',f'=-{sp}*{ct}*SIN(RADIANS(@{t}_lon))+{cp}*COS(RADIANS(@{t}_lon))')
        expressions={'VV':f'{a}^2*@vvX+2*{a}*{cval}*@vhX+{cval}^2*@hhX',
                     'HH':f'{b}^2*@vvX+2*{b}*{d}*@vhX+{d}^2*@hhX',
                     'VH':f'{a}*{b}*@vvX+({a}*{d}+{cval}*{b})*@vhX+{cval}*{d}*@hhX'}
        for pol,expression in expressions.items():
            n=next(n for n,row in enumerate(rows) if float(row['azimuth'])==lon and float(row['elevation'])==lat and row['polarization']==pol)
            check_field(s,f'Tilted {lon:g}/{lat:g} {pol}',rows,n,'='+expression.replace('X','i'),'='+expression.replace('X','q'),'result_tilted')
    for pol in ['VV','HH','VH']:
        n=next(n for n,row in enumerate(rows) if float(row['azimuth'])==90 and float(row['elevation'])==15 and row['polarization']==pol)
        s.exported('Unsupported side direction '+pol,'Remains missing','="missing"',rows,n,'magnitude_power_linear','m2','result_tilted')
    single=grid(field=field[:,3:4,:,:],az=az,el=[15.],pol=['VV','HH','VH'])
    s.save('input_single_tilt',single)
    s.reject('One nonzero tilt cannot fill conic grid',lambda:wedge_to_conic(single),contains='fixed wedge tilt')
    return s.finish()


def range_frequency_case():
    from GRIM_Backend.plotting.modes.range_freq_mode import prepare_range_frequency
    from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest
    s=Spec('range_frequency','Range Freq','Direct DFT power profiles and explicit range-phase normalization.',
           settings='8 frequencies, 1 to 8 GHz; full selected band; Rectangular window; linear scale; 64 padded range bins.',
           scope='Numerical plot values and coordinate direction for both declared range signs and missing declarations. Plot CSV is a direct numerical export, not a derived native dataset.',category='Plot',atol=1e-10,rtol=1e-9)
    n,size=8,64
    df=1e9
    target=5*C0/(2*df*size)
    freqs=np.arange(1.,9.)
    for name,value,unit in [('c',C0,'m/s'),('df',df,'Hz'),('size',size,''),('n',n,''),('target',target,'m'),('amplitude',2.,'sqrt(m2)')]:
        s.input(name,name,value,unit)
    for j,f in enumerate(freqs):
        s.input(f'f{j}',f'Frequency sample {j+1}',f*1e9,'Hz')
        s.calc(f'i{j}',f'Canonical point sample {j+1} I',f'=@amplitude*COS(-4*PI()*@f{j}*@target/@c)','sqrt(m2)')
        s.calc(f'q{j}',f'Canonical point sample {j+1} Q',f'=@amplitude*SIN(-4*PI()*@f{j}*@target/@c)','sqrt(m2)')
    for k in range(size):
        signed=k-size//2
        raw=k+size//2 if k<size//2 else k-size//2
        s.calc(f'r{k}',f'Shifted range bin {k}',f'={signed}*@c/(2*@df*@size)','m')
        ri='+'.join([f'(@i{j}*COS(2*PI()*{raw}*{j}/@size)-@q{j}*SIN(2*PI()*{raw}*{j}/@size))' for j in range(n)])
        rq='+'.join([f'(@i{j}*SIN(2*PI()*{raw}*{j}/@size)+@q{j}*COS(2*PI()*{raw}*{j}/@size))' for j in range(n)])
        s.calc(f'pi{k}',f'Profile bin {k} I',f'=({ri})/@n','sqrt(m2)')
        s.calc(f'pq{k}',f'Profile bin {k} Q',f'=({rq})/@n','sqrt(m2)')
        s.calc(f'p{k}',f'Profile bin {k} power',f'=@pi{k}^2+@pq{k}^2','m2')
    for sign,label in [(-1,'negative_range_law'),(1,'positive_range_law'),(-1,'undeclared')]:
        field=2*np.exp(sign*4j*np.pi*freqs*1e9*target/C0)
        metadata={} if label=='undeclared' else {'range_phase_convention':f'exp({"+" if sign>0 else "-"}j2kR)'}
        source=grid(field=field,az=[0.],freq=freqs,extra=metadata)
        if label=='undeclared':
            source.units.pop('time_convention',None)
        s.save('input_'+label,source)
        selection=([0],[0],np.arange(n),[0])
        request=PlotRequest('range_freq',source,(PlotPlan(label,source,selection),),scale='linear',options={'percent':100.,'window':'Rectangular','unit':'m'})
        result=prepare_range_frequency(request)
        if result.skipped or len(result.items)!=1:
            raise AssertionError((label,result.skipped))
        panel=result.items[0]
        path=VALIDATION/'datasets'/'range_frequency'/('plot_'+label+'.csv')
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['range_m','power_linear','frequency_center_GHz'])
            for row,power in zip(panel['y'],panel['image'][0]):
                writer.writerow([format(float(row),'.17g'),format(float(power),'.17g'),format(float(panel['x'][0]),'.17g')])
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        s.data['files'].append(dict(name='plot_'+label,native='',csv=path.relative_to(VALIDATION).as_posix(),rows=len(rows)))
        for k,row in enumerate(rows):
            s.check(f'{label} bin {k}','Power',f'=@p{k}',float(row['power_linear']),'m2',f'{path.name} row {k+2}: power_linear')
            if label=='negative_range_law':
                s.check(f'Range bin {k}','Range coordinate',f'=@r{k}',float(row['range_m']),'m',f'{path.name} row {k+2}: range_m')
        s.check(label,'Peak location','=@target',float(rows[int(np.argmax([float(row['power_linear']) for row in rows]))]['range_m']),'m',path.name+' maximum-power row')
        result.close()
    return s.finish()


def azimuth_range_case():
    from GRIM_Backend.plotting.modes.az_vs_range_mode import prepare_azimuth_range
    from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest
    s=Spec('azimuth_range','Az vs D.R.', 'Independent DFT checks for two known range profiles.',
           settings='8 samples, 1-8 GHz; azimuths 0 and 30 deg; Rectangular window; peak normalization off; linear intensity.',
           scope='Checks native numerical profile values and axes. Uses 2e-6 relative tolerance for this plot path\'s float32 FFT work; does not validate scene interpretation or all rendering options.',
           category='Plot',atol=1e-7,rtol=2e-6)
    n,size=8,32
    freq=np.arange(1.,9.)
    ranges=[0.,5*C0/(2e9*size)]
    amplitudes=[2.,1.]
    field=np.array([a*np.exp(-4j*np.pi*freq*1e9*r/C0) for a,r in zip(amplitudes,ranges)])
    source=grid(field=field,az=[0.,30.],freq=freq,extra={'range_phase_convention':'exp(-j2kR)'})
    s.save('input',source)
    for name,value,unit in [('c',C0,'m/s'),('df',1e9,'Hz'),('size',size,''),('n',n,'')]:
        s.input(name,name,value,unit)
    for a in range(2):
        s.input(f'az{a}',f'Profile {a+1} azimuth',a*30,'deg')
        s.input(f'r{a}',f'Profile {a+1} range',ranges[a],'m')
        s.input(f'amp{a}',f'Profile {a+1} amplitude',amplitudes[a],'sqrt(m2)')
        for j in range(n):
            s.calc(f'i{a}_{j}',f'Profile {a+1} sample {j+1} I',f'=@amp{a}*COS(-4*PI()*{j+1}*@df*@r{a}/@c)')
            s.calc(f'q{a}_{j}',f'Profile {a+1} sample {j+1} Q',f'=@amp{a}*SIN(-4*PI()*{j+1}*@df*@r{a}/@c)')
    plan=PlotPlan('two point profiles',source,([0,1],[0],list(range(n)),[0]))
    request=PlotRequest('az_vs_range',source,(plan,),scale='linear',options={'window':'Rectangular','unit_name':'m','unit_scale':1.,'peak_norm':False,'azimuth_width':30.})
    result=prepare_azimuth_range(request)
    panel=result.items[0]
    path=VALIDATION/'datasets'/'azimuth_range'/'plot.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['azimuth_deg','range_m','intensity_linear'])
        for a in range(2):
            for k in range(size):
                writer.writerow([panel['az_values'][a],format(float(panel['range_axis'][k]),'.17g'),format(float(panel['display'][a,k]),'.17g')])
    with path.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    s.data['files'].append(dict(name='plot',native='',csv=path.relative_to(VALIDATION).as_posix(),rows=len(rows)))
    for a in range(2):
        for k in range(size):
            raw=(k+size//2)%size
            ri='+'.join(f'(@i{a}_{j}*COS(2*PI()*{raw}*{j}/@size)-@q{a}_{j}*SIN(2*PI()*{raw}*{j}/@size))' for j in range(n))
            rq='+'.join(f'(@i{a}_{j}*SIN(2*PI()*{raw}*{j}/@size)+@q{a}_{j}*COS(2*PI()*{raw}*{j}/@size))' for j in range(n))
            ir=s.calc(f'pi{a}_{k}',f'Profile {a+1} bin {k} I',f'=({ri})/@n')
            qr=s.calc(f'pq{a}_{k}',f'Profile {a+1} bin {k} Q',f'=({rq})/@n')
            row=a*size+k
            s.check(f'Az {a*30} bin {k}','Intensity',f'={ir}^2+{qr}^2',float(rows[row]['intensity_linear']),'relative intensity',f'plot.csv row {row+2}')
            if a==0:
                s.check(f'Range bin {k}','Coordinate',f'={k-size//2}*@c/(2*@df*@size)',float(rows[row]['range_m']),'m',f'plot.csv row {row+2}')
    result.close()
    return s.finish()


def isar_case():
    from GRIM_Backend.plotting.modes.isar_mode import form_isar
    s=Spec('isar_image','ISAR Image', 'Analytic point-target peak, direction, resolution, and origin-level controls.',
           settings='17 azimuths -4 to 4 deg; 33 frequencies 9-11 GHz; coherent aperture; Rectangular window; Fast and Accurate PFA.',
           scope='Limited point-target controls. Off-origin peak coordinates allow one image pixel; arbitrary geometry, sparse/composite modes and display controls are not certified.',
           category='BETA plot',atol=1e-7,rtol=2e-6)
    az=np.linspace(-4.,4.,17);freq=np.linspace(9e9,11e9,33)
    for name,value,unit in [('c',C0,'m/s'),('f0',freq[0],'Hz'),('f1',freq[-1],'Hz'),('df',freq[1]-freq[0],'Hz'),
                            ('az0',az[0],'deg'),('az1',az[-1],'deg'),('da',az[1]-az[0],'deg'),('nf',len(freq),''),('sigma',4.,'m2'),('x',.2,'m'),('y',-.3,'m')]:
        s.input(name,name,value,unit)
    s.calc('umax','Inscribed Cartesian maximum U frequency','= @f0*SIN(RADIANS((@az1-@az0)/2))','Hz')
    s.calc('vmax','Inscribed Cartesian maximum V frequency','=SQRT(@f1^2-@umax^2)','Hz')
    s.data['notes'].append('Fast PFA uses the original frequency bandwidth and small-angle aperture. Accurate PFA uses the inscribed Cartesian rectangle: Umax=fmin*sin(half aperture); Vmax=sqrt(fmax^2-Umax^2). Its resolutions and extents therefore differ.')
    for name,x,y in [('origin',0.,0.),('offset',.2,-.3)]:
        projection=x*np.sin(np.deg2rad(az))[:,None]+y*np.cos(np.deg2rad(az))[:,None]
        field=2*np.exp(-4j*np.pi*freq[None,:]*projection/C0)
        source=grid(field=field,az=az,freq=freq,units={'frequency':'Hz','phase_reference':'fixed origin'},
                    extra={'range_phase_convention':'exp(-j2kR)','measurement_geometry':'far-field monostatic','motion_compensation':'stable'})
        # Selected authored input cells show the target phase law independently.
        input_rows=s.save('input_'+name,source)
        for a,f in [(0,0),(8,16),(16,32)]:
            tag=f'{name}_{a}_{f}'
            s.input(tag+'_az','Input azimuth',az[a],'deg');s.input(tag+'_f','Input frequency',freq[f],'Hz')
            s.input(tag+'_x','Target cross range',x,'m');s.input(tag+'_y','Target down range',y,'m')
            phase=s.calc(tag+'_phase','Independent point phase',f'=-4*PI()*@{tag}_f*(@{tag}_x*SIN(RADIANS(@{tag}_az))+@{tag}_y*COS(RADIANS(@{tag}_az)))/@c','rad')
            check_field(s,tag,input_rows,a*len(freq)+f,f'=SQRT(@sigma)*COS({phase})',f'=SQRT(@sigma)*SIN({phase})','input_'+name)
        for reconstruction in ['fast','accurate']:
            band=form_isar(source,retain_complex=True,aperture_mode='coherent',reconstruction=reconstruction,window='Rectangular')[0][0]
            image=np.asarray(band['complex_image'])
            peak=np.unravel_index(np.argmax(np.abs(image)),image.shape)
            path=VALIDATION/'datasets'/'isar_image'/f'image_{name}_{reconstruction}.csv'
            with path.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['cross_range_m','down_range_m','intensity_linear'])
                for a,xx in enumerate(band['x_range']):
                    for f,yy in enumerate(band['y_range']):
                        writer.writerow([format(float(xx),'.17g'),format(float(yy),'.17g'),format(float(abs(image[a,f])**2),'.17g')])
            with path.open(newline='',encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            s.data['files'].append(dict(name=path.stem,native='',csv=path.relative_to(VALIDATION).as_posix(),rows=len(rows)))
            peak_row=peak[0]*len(band['y_range'])+peak[1]
            actual=rows[peak_row]
            label=name+' '+reconstruction
            dx=abs(float(band['x_range'][1]-band['x_range'][0]));dy=abs(float(band['y_range'][1]-band['y_range'][0]))
            s.check(label,'Peak cross range','=0' if name=='origin' else '=@x',float(actual['cross_range_m']),'m',path.name+f' row {peak_row+2}',tolerance=dx)
            s.check(label,'Peak down range','=0' if name=='origin' else '=@y',float(actual['down_range_m']),'m',path.name+f' row {peak_row+2}',tolerance=dy)
            if name=='origin':
                s.check(label,'Origin peak intensity','=@sigma',float(actual['intensity_linear']),'relative intensity',path.name+f' row {peak_row+2}',tolerance=2e-5)
            # Sampling metrics are exported explicitly, not asserted from the display.
            metrics=VALIDATION/'datasets'/'isar_image'/f'sampling_{name}_{reconstruction}.csv'
            sampling=band['sampling']
            with metrics.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['range_resolution_m','cross_resolution_m','range_half_extent_m'])
                writer.writerow([format(float(sampling[key]),'.17g') for key in ['range_resolution','cross_resolution','range_half_extent']])
            with metrics.open(newline='',encoding='utf-8') as stream:
                metric=next(csv.DictReader(stream))
            s.data['files'].append(dict(name=metrics.stem,native='',csv=metrics.relative_to(VALIDATION).as_posix(),rows=1))
            range_formula='=@c/(2*(@f1-@f0))' if reconstruction=='fast' else '=@c/(2*(@vmax-@f0))'
            cross_formula='=@c/((@f0+@f1)/2)/(2*RADIANS(@az1-@az0))' if reconstruction=='fast' else '=@c/(4*@umax)'
            extent_formula='=@c/(4*@df)' if reconstruction=='fast' else '=@c*(@nf-1)/(4*(@vmax-@f0))'
            s.check(label,'Nominal range resolution',range_formula,float(metric['range_resolution_m']),'m',metrics.name+' row 2')
            s.check(label,'Nominal cross resolution',cross_formula,float(metric['cross_resolution_m']),'m',metrics.name+' row 2')
            s.check(label,'Range half extent',extent_formula,float(metric['range_half_extent_m']),'m',metrics.name+' row 2')
    return s.finish()


def generate():
    return [phase_center_case(),range_cal_case(),time_gate_case(),wedge_case(),range_frequency_case(),azimuth_range_case(),isar_case()]


if __name__=='__main__':
    cases=generate()
    print('Physics validation specifications:', ', '.join(item['title'] for item in cases))
    print('Comparisons:',sum(len(item['checks']) for item in cases))
