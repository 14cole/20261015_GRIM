"""Independent scalar Excel references for GRIM's numerical plot products."""
from __future__ import annotations
import csv
import math
import numpy as np
from common import Spec, grid, number, scalar, VALIDATION, RcsGrid
from GRIM_Backend.scripting.plotting import plot_datasets
from GRIM_Backend.plotting.requests import PlotPlan, PlotRequest, prepare_sweep, prepare_image_plane
from GRIM_Backend.plotting import distribution_analysis as dist
from GRIM_Backend.plotting.modes import sector_stats_mode, delta_map_mode, compare_mode, common as plot_common


def exported_table(spec, name, headers, rows):
    path = VALIDATION / 'datasets' / spec.data['slug'] / (name + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows([[scalar(x) for x in row] for row in rows])
    with path.open(newline='', encoding='utf-8') as f:
        result = list(csv.DictReader(f))
    spec.data['files'].append(dict(name=name, csv=path.relative_to(VALIDATION).as_posix(), rows=len(result)))
    return result


def sample_inputs(s, values, prefix='p'):
    return [s.input(f'{prefix}{i}', f'Input power, sample {i + 1}', v, 'm^2') for i, v in enumerate(values)]


def line_plots():
    powers = [1., 4., 9., 16., 0., np.nan]
    phases = [170., -170., 30., -90., np.nan, np.nan]
    az = [-150., -90., -30., 30., 90., 150.]
    for slug, title, mode, phase in [
        ('azimuth_plot', 'Azimuth Rect', 'azimuth_rect', False),
        ('polar_plot', 'Azimuth Polar', 'azimuth_polar', False),
        ('phase_plot', 'Phase', 'azimuth_rect', True)]:
        s = Spec(slug, title, 'Numerical samples read from a GRIM-rendered line.',
                 settings='VV; elevation 0 deg; frequency 10 GHz; logarithmic magnitude or phase as named.',
                 scope='Verifies plotted coordinates and sample values for this cut. Styling and interactive selection require GUI review.', category='Plot')
        d = grid(powers, phases, az=az)
        s.save('input', d)
        refs = sample_inputs(s, powers)
        phi = [s.input(f'phi{i}', f'Input phase, sample {i+1}', v, 'deg') for i,v in enumerate(phases)]
        aa = [s.input(f'az{i}', f'Input azimuth, sample {i+1}', v, 'deg') for i,v in enumerate(az)]
        fig = plot_datasets([('input',d)], mode=mode, azimuths=az, elevations=[0], frequencies=[10], polarization='VV', phase=phase)
        line = next(x for x in fig.axes[0].lines if len(x.get_xdata()) == len(az))
        rows = exported_table(s,'plot_values',['x','y'],zip(line.get_xdata(),line.get_ydata()))
        fig.savefig(VALIDATION/'datasets'/slug/'plot.png',dpi=100)
        s.check('Complete cut', 'Exported sample count', f'=COUNT({aa[0]}:{aa[-1]})', len(rows), 'count', 'plot_values.csv data-row count', tolerance=0)
        for i in range(len(az)):
            s.exported(f'Sample {i+1}', 'Plotted azimuth', f'={aa[i]}' if mode!='azimuth_polar' else f'=RADIANS({aa[i]})', rows,i,'x','rad' if mode=='azimuth_polar' else 'deg',file='plot_values')
            formula = (f'=IF(OR({refs[i]}="missing",{refs[i]}=0,{phi[i]}="missing"),"missing",MOD({phi[i]}+180,360)-180)' if phase else
                       f'=IF({refs[i]}="missing","missing",IF({refs[i]}<=0,"missing",10*LOG10({refs[i]})))')
            s.exported(f'Sample {i+1}','Phase' if phase else 'Level',formula,rows,i,'y','deg' if phase else 'dBsm',file='plot_values')
        s.finish()


def sweep_plots():
    power = np.array([1,2,4,8,3,9,27,81,5,10,20,40,7,14,28,56,0,6,12,24,9,18,36,72.],float).reshape(3,2,4,1)
    power[1,1,2,0]=np.nan
    d = grid(power,az=[-20,0,20],el=[0,30],freq=[8,9,10,11])
    for slug,title,mode in [('frequency_plot','Frequency','frequency'),('elevation_plot','Elevation Sweep','elevation_sweep')]:
        s=Spec(slug,title,'Azimuth reduction in linear power, then plotted in dBsm.',settings='All azimuths; VV; test Mean, P50, P90 and Max.',category='Plot')
        s.save('input',d)
        refs={idx:s.input('p'+'_'.join(map(str,idx)),f'Power az {d.azimuths[idx[0]]}, el {d.elevations[idx[1]]}, f {d.frequencies[idx[2]]}',power[idx],'m^2') for idx in np.ndindex(power.shape)}
        frefs=[s.input(f'f{i}',f'Authored frequency {i+1}',value,'GHz') for i,value in enumerate([8.,9.,10.,11.])]
        erefs=[s.input(f'e{i}',f'Authored elevation {i+1}',value,'deg') for i,value in enumerate([0.,30.])]
        s.input('q90','Percentile setting',90.,'%')
        cut_range=f'{erefs[0]}:{erefs[-1]}' if mode=='frequency' else f'{frefs[0]}:{frefs[-1]}'
        point_range=f'{frefs[0]}:{frefs[-1]}' if mode=='frequency' else f'{erefs[0]}:{erefs[-1]}'
        sel=tuple(np.arange(n) for n in d.shape)
        for stat,q in [('mean',50),('median',50),('percentile',90),('max',50)]:
            result=prepare_sweep(PlotRequest(mode,d,(PlotPlan('input',d,sel),),scale='dbsm',options={'mode':stat,'percentile':q}))
            records=[]
            for k,(_,series) in enumerate(result.items):
                for j,(x,y) in enumerate(zip(series[0],series[1])): records.append([k,j,x,y])
            rows=exported_table(s,f'{stat}_plot',['cut','sample','x','level'],records)
            s.check(stat,'Returned cut count',f'=COUNT({cut_range})',len(result.items),'count','GRIM prepare_sweep returned series count',tolerance=0)
            s.check(stat,'Exported point count',f'=COUNT({cut_range})*COUNT({point_range})',len(rows),'count',f'{stat}_plot.csv data-row count',tolerance=0)
            for r,row in enumerate(rows):
                k,j=int(row['cut']),int(row['sample'])
                e,f=(k,j) if mode=='frequency' else (j,k)
                terms=','.join(refs[(a,e,f,0)] for a in range(3))
                reduction={'mean':f'AVERAGE({terms})','median':f'MEDIAN({terms})','max':f'MAX({terms})','percentile':''}[stat]
                if stat=='percentile':
                    # Put the three independent source powers into adjacent
                    # helper cells; Excel chooses the valid population itself.
                    vals=[]
                    for a in range(3):
                        vals.append(s.calc(f'p90_{k}_{j}_{a}',f'P90 cut {k+1}, sample {j+1}, azimuth member {a+1}',f'={refs[(a,e,f,0)]}','m^2'))
                    reduction=f'PERCENTILE.INC({vals[0]}:{vals[-1]},@q90/100)'
                reduced=s.calc(f'{stat}_{k}_{j}',f'{stat} power, cut {k+1}, sample {j+1}',f'=IF(COUNT({terms})=0,"missing",{reduction})','m^2')
                s.exported(f'{stat}, cut {k+1}, sample {j+1}','Level',f'=IF({reduced}="missing","missing",IF({reduced}<=0,"missing",10*LOG10({reduced})))',rows,r,'level','dBsm',file=f'{stat}_plot')
                axis_ref=frefs[j] if mode=='frequency' else erefs[j]
                s.exported(f'{stat}, cut {k+1}, sample {j+1}','Plotted coordinate',f'={axis_ref}',rows,r,'x','GHz' if mode=='frequency' else 'deg',file=f'{stat}_plot')
            result.close()
        s.finish()


def waterfall():
    p=np.array([1,2,4,3,6,12,5,10,20,7,14,28.]).reshape(2,2,3,1)
    d=grid(p,az=[0,90],el=[0,30],freq=[8,9,10])
    s=Spec('waterfall','Waterfall','Native image values averaged over elevation in linear power.',settings='Axes azimuth/frequency; aggregation Mean power; dBsm.',scope='Native pixels and axis association; shared cut axes and display decimation need GUI checks.',category='Plot')
    s.save('input',d)
    refs={idx:s.input('p'+'_'.join(map(str,idx)),f'Power at az/el/f index {idx[:3]}',p[idx],'m^2') for idx in np.ndindex(p.shape)}
    plan=PlotPlan('input',d,tuple(np.arange(n) for n in d.shape))
    out=prepare_image_plane(PlotRequest('waterfall',d,(plan,),options={'axes':('azimuth','frequency')}),plan)
    rows=exported_table(s,'image_values',['azimuth_deg','frequency_GHz','level_dbsm'],[(d.azimuths[a],d.frequencies[f],out[a,f]) for a in range(2) for f in range(3)])
    for r in range(6):
        a,f=divmod(r,3)
        s.exported(f'Az {a+1}, f {f+1}','Pixel level',f'=10*LOG10(AVERAGE({refs[(a,0,f,0)]},{refs[(a,1,f,0)]}))',rows,r,'level_dbsm','dBsm',file='image_values')
    s.finish()


def pbp():
    s=Spec('pbp','PBP','Separate per-dataset min/max envelopes of individual azimuth cuts.',settings='Frequency plot; Individual cuts; PBP; linear power; two datasets.',scope='Magnitude envelopes and valid-cut counts. Circular phase arcs and GUI rendering are separate cases.',category='Plot')
    plans=[]; refs={}
    for k,scale in enumerate([1,10]):
        p=np.array([[0,2,8],[3,4,16],[1,np.nan,4.]])*scale
        d=grid(p,az=[-30,0,30],freq=[8,9,10]); s.save(f'input_{k+1}',d)
        plans.append(PlotPlan(f'input_{k+1}',d,tuple(np.arange(n) for n in d.shape)))
        for a,f in np.ndindex(p.shape): refs[k,a,f]=s.input(f'p{k}_{a}_{f}',f'Dataset {k+1}, az index {a}, frequency {8+f}',p[a,f],'m^2')
    result=prepare_sweep(PlotRequest('frequency',plans[0].dataset,tuple(plans),scale='linear',options={'mode':'individual','percentile':50,'pbp':True}))
    records=[]
    for k,(_,env) in enumerate(result.envelopes):
        lo,hi,n=env.result()
        records.extend([k,f,lo[f],hi[f],n[f]] for f in range(3))
    rows=exported_table(s,'bands',['dataset','frequency_index','lower','upper','count'],records)
    s.check('Both datasets','Envelope count','=2',len(result.envelopes),'count','GRIM prepare_sweep returned envelopes',tolerance=0)
    s.check('Both datasets','Exported frequency-band rows','=2*3',len(rows),'count','bands.csv data-row count',tolerance=0)
    for r,row in enumerate(rows):
        k,f=int(row['dataset']),int(row['frequency_index']); terms=','.join(refs[k,a,f] for a in range(3))
        for col,fn in [('lower','MIN'),('upper','MAX'),('count','COUNT')]: s.exported(f'Dataset {k+1}, f {f+1}',col,f'={fn}({terms})',rows,r,col,'count' if col=='count' else 'm^2',file='bands')
    result.close();s.finish()


def cdf_sector():
    p=[0.,1.,1.,4.,9.,np.nan]; d=grid(p,az=[0,30,60,90,120,150])
    s=Spec('cdf','CDF','Inclusive empirical probabilities preserve zeros and exclude missing samples.',settings='Sample weighting; one run; linear power; CDF and exceedance.',scope='Sample-weighted population including ties. Coordinate-weighted quadrature is not exercised.',category='Plot')
    s.save('input',d); refs=sample_inputs(s,p)
    config=dist.options(); curve=dist.prepare_curve('input',d,tuple(np.arange(n) for n in d.shape),'All selected coordinates',config,scale='linear')
    rows=exported_table(s,'probabilities',['level','cdf','exceedance'],zip(curve['levels'],curve['cdf'],curve['exceed']))
    # These four distinct support levels are independently defined by the
    # authored inputs, never read back from the GRIM curve as an oracle.
    level_refs=[refs[i] for i in (0,1,3,4)]
    s.check('Complete distribution','Distinct level count','=4',len(rows),'count','probabilities.csv data-row count',tolerance=0)
    for i,(level,row) in enumerate(zip(level_refs,rows)):
        s.exported(f'Level {i+1}','Support coordinate',f'={level}',rows,i,'level','m^2',file='probabilities')
        for col,op in [('cdf','<='),('exceedance','>=')]:
            # COUNTIFS uses six contiguous raw-power cells (resolved range endpoints).
            formula=f'=100*COUNTIFS({refs[0]}:{refs[-1]},"{op}"&{level})/COUNT({refs[0]}:{refs[-1]})'
            # Missing sentinels are text and are explicitly excluded by numeric condition.
            formula=f'=100*COUNTIFS({refs[0]}:{refs[-1]},"{op}"&{level},{refs[0]}:{refs[-1]},">=0")/COUNT({refs[0]}:{refs[-1]})'
            s.exported(f'Threshold {row["level"]}',col,formula,rows,i,col,'%',file='probabilities')
    s.finish()
    s=Spec('sector_stats','Sector Stats','Independent linear-power statistics in two tiled azimuth sectors.',settings='Enter tiled sectors 0:90:180; first [0,90), final [90,180]; P90.',category='Plot')
    s.save('input',d); refs=sample_inputs(s,p)
    rec=[]
    sectors=plot_common.parse_sectors('0:90:180',d.azimuths)
    for k,sector in enumerate(sectors):
        members=sector.contains(d.azimuths)
        values=sector_stats_mode._sector_linear_statistics(np.array(p),members,90.)
        rec.append([k]+[values[x] for x in ['count','mean','median','min','max','percentile']])
    rows=exported_table(s,'sector_values',['sector','count','mean','median','min','max','percentile'],rec)
    s.check('Tiled sectors','Sector count','=2',len(rows),'count','GRIM sector parser output',tolerance=0)
    for k,sector in enumerate(sectors):
        for a,member in enumerate(sector.contains(d.azimuths)):
            angle=s.input(f'angle{k}_{a}',f'Sector {k+1}: sample {a+1} azimuth',[0.,30.,60.,90.,120.,150.][a],'deg')
            upper='<' if k==0 else '<='
            s.check(f'Sector {k+1}, sample {a+1}','Coordinate membership',f'=IF(AND({angle}>={k*90},{angle}{upper}{(k+1)*90}),1,0)',int(member),'included','GRIM parsed sector.contains()',tolerance=0)
    for k,(start,end) in enumerate([(0,2),(3,5)]):
        rg=f'{refs[start]}:{refs[end]}'
        for metric,fn in [('count','COUNT'),('mean','AVERAGE'),('median','MEDIAN'),('min','MIN'),('max','MAX'),('percentile','PERCENTILE.INC')]:
            s.exported(f'Sector {k+1}',metric,f'={fn}({rg}'+(',0.9)' if metric=='percentile' else ')'),rows,k,metric,'count' if metric=='count' else 'm^2',file='sector_values')
    s.finish()


def delta():
    a=np.array([[1,4,0],[9,16,np.nan]],float); b=np.array([[2,2,3],[9,4,5]],float)
    d1=grid(a,az=[0,90],freq=[8,9,10]);d2=grid(b,az=[0,90],freq=[8,9,10])
    s=Spec('delta_map','Delta Map','Matched logarithmic level difference, first dataset minus second.',settings='Azimuth/frequency plane; elevation 0; VV.',category='Plot')
    s.save('input_a',d1);s.save('input_b',d2)
    out=delta_map_mode.prepare([('A',d1),('B',d2)],reference=d1,selections={'azimuth':[0,90],'elevation':[0],'frequency':[8,9,10]},polarization='VV')
    records=[]
    for az in range(2):
        for f in range(3): records.append([az,f,out.delta_db[f,az]])
    rows=exported_table(s,'delta_values',['az_index','f_index','delta_db'],records)
    for r,(az,f) in enumerate(np.ndindex(a.shape)):
        pa=s.input(f'a{r}',f'A power {r+1}',a[az,f],'m^2'); pb=s.input(f'b{r}',f'B power {r+1}',b[az,f],'m^2')
        formula=f'=IF(OR({pa}="missing",{pb}="missing"),"missing",IF(OR({pa}<=0,{pb}<=0),"missing",10*LOG10({pa}/{pb})))'
        s.exported(f'Cell {r+1}','Difference',formula,rows,r,'delta_db','dB',file='delta_values')
    s.finish()


def native_export():
    s=Spec('save_load_export','Save Load Export','Native archive and CSV preserve response samples and coordinate units.',settings='Float64 input; power zero, missing phase, missing power, positive and negative phases.',scope='Numerical file round trip. Save All rollback, cancellation, dialogs and recovery require separate manual checks.',category='Storage')
    p=[0.,1.,1e-12,1e6,np.nan]; ph=[np.nan,179.,-179.,45.,np.nan]
    d=grid(p,ph,az=[-180,-90,0,90,180],freq=[9.5])
    rows=s.save('input',d)
    d2=RcsGrid.load(VALIDATION/s.data['files'][0]['native'],storage='memory'); out=s.save('result',d2)
    freq=s.input('frequency','Authored frequency',9.5,'GHz')
    elev=s.input('elevation','Authored elevation',0.,'deg')
    pol=s.input('polarization','Authored polarization','VV','label')
    angle_unit=s.input('angle_unit','Authored angle unit','deg','unit')
    freq_unit=s.input('frequency_unit','Authored frequency unit','GHz','unit')
    s.check('Complete file','Exported data rows','=5',len(out),'count','result.csv data-row count',tolerance=0)
    for i in range(5):
        v=s.input(f'p{i}',f'Authored power {i+1}',p[i],'m^2')
        s.exported(f'Sample {i+1}','Power',f'={v}',out,i,'magnitude_power_linear','m^2')
        v=s.input(f'ph{i}',f'Authored phase {i+1}',ph[i],'deg')
        s.exported(f'Sample {i+1}','Phase',f'={v}',out,i,'phase_deg','deg')
        v=s.input(f'az{i}',f'Authored azimuth {i+1}',[-180.,-90.,0.,90.,180.][i],'deg')
        for metric,column,expected,unit in [
            ('Azimuth','azimuth',v,'deg'),('Elevation','elevation',elev,'deg'),
            ('Frequency','frequency',freq,'GHz'),('Polarization','polarization',pol,'label'),
            ('Azimuth unit','azimuth_unit',angle_unit,'unit'),('Elevation unit','elevation_unit',angle_unit,'unit'),
            ('Frequency unit','frequency_unit',freq_unit,'unit')]:
            s.exported(f'Sample {i+1}',metric,f'={expected}',out,i,column,unit)
    s.finish()


def generate():
    line_plots();sweep_plots();waterfall();pbp();cdf_sector();delta();native_export()

if __name__=='__main__': generate()
