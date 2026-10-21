"""Independent spreadsheet formulas for the RF Compare magnitude metrics."""
from __future__ import annotations
import csv
from pathlib import Path
from common import Spec, grid, VALIDATION, number
from GRIM_Backend.plotting.modes.compare_mode import rf_agreement_statistics


def component(s, prefix, label, left, right, left_db, right_db):
    """Separate means and centered moments expose each statistical formula."""
    n=len(left)
    def ccc(tag,l,r):
        ml=s.calc(prefix+tag+'_ml',label+tag+' left mean','=AVERAGE('+','.join(l)+')')
        mr=s.calc(prefix+tag+'_mr',label+tag+' right mean','=AVERAGE('+','.join(r)+')')
        vl=s.calc(prefix+tag+'_vl',label+tag+' left population variance','=('+'+'.join(f'({x}-{ml})^2' for x in l)+f')/{n}')
        vr=s.calc(prefix+tag+'_vr',label+tag+' right population variance','=('+'+'.join(f'({x}-{mr})^2' for x in r)+f')/{n}')
        cov=s.calc(prefix+tag+'_cov',label+tag+' population covariance','=('+'+'.join(f'({a}-{ml})*({b}-{mr})' for a,b in zip(l,r))+f')/{n}')
        den=s.calc(prefix+tag+'_den',label+tag+' CCC denominator',f'={vl}+{vr}+({ml}-{mr})^2')
        return s.calc(prefix+tag+'_ccc',label+tag+' CCC',f'=IF({den}=0,1,MAX(-1,MIN(1,2*{cov}/{den})))')
    linear=ccc('_linear',left,right)
    log=ccc('_log',left_db,right_db)
    mse=s.calc(prefix+'_mse',label+' mean squared linear error','=('+'+'.join(f'({a}-{b})^2' for a,b in zip(left,right))+f')/{n}')
    pooled=s.calc(prefix+'_pooled',label+' pooled mean squared power','=('+'+'.join(f'({a}^2+{b}^2)/2' for a,b in zip(left,right))+f')/{n}')
    nrmse=s.calc(prefix+'_nrmse',label+' normalized RMS error',f'=SQRT({mse}/{pooled})')
    score=s.calc(prefix+'_score',label+' component score',f'=100*(@weight_linear*MAX({linear},0)+@weight_log*MAX({log},0)+@weight_error*(1-MIN({nrmse}/SQRT(2),1)))','0..100')
    return dict(linear_ccc=linear,log_ccc=log,linear_nrmse=nrmse,score=score)


def generate():
    s=Spec('rf_compare','RF Compare','Independent concordance, normalized error, dB differences and sector-weighted agreement.',
           settings='Magnitude mode; six matched azimuths per case, yielding two contiguous three-sample sectors.',
           scope='Validates the defined metric arithmetic on positive synthetic powers. The RF index is a chosen scoring policy, not a universal physical-quality certificate.',category='Plot numerical comparison')
    for key,label,value in [('weight_linear','Linear CCC weight',.45),('weight_log','Log CCC weight',.30),
                             ('weight_error','Normalized-error weight',.25),('global_weight','Global score weight',.75),
                             ('sector_weight','Lower-quartile sector score weight',.25)]:
        s.input(key,label,value)
    cases=[('identical',[1.,2.,4.,8.,16.,32.],[1.,2.,4.,8.,16.,32.]),
           ('gain_x4',[1.,2.,4.,8.,16.,32.],[4.,8.,16.,32.,64.,128.]),
           ('shape_change',[1.,2.,4.,8.,16.,32.],[2.,1.,8.,4.,32.,16.]),
           ('equal_constant',[2.]*6,[2.]*6),
           ('unequal_constant',[2.]*6,[4.]*6)]
    export_rows=[];expected=[]
    x=[0.,10.,20.,30.,40.,50.]
    for name,left,right in cases:
        s.save(name+'_left',grid(left,az=x))
        s.save(name+'_right',grid(right,az=x))
        ll=[s.input(f'{name}_l{i}',f'{name} left power at {x[i]:g} deg',v,'m2') for i,v in enumerate(left)]
        rr=[s.input(f'{name}_r{i}',f'{name} right power at {x[i]:g} deg',v,'m2') for i,v in enumerate(right)]
        xx=[s.input(f'{name}_x{i}',f'{name} matched azimuth {i+1}',v,'deg') for i,v in enumerate(x)]
        ld=[s.calc(f'{name}_ld{i}',f'{name} left dBsm {i+1}',f'=10*LOG10({v})','dBsm') for i,v in enumerate(ll)]
        rd=[s.calc(f'{name}_rd{i}',f'{name} right dBsm {i+1}',f'=10*LOG10({v})','dBsm') for i,v in enumerate(rr)]
        delta=[s.calc(f'{name}_delta{i}',f'{name} left minus right dB {i+1}',f'={a}-{b}','dB') for i,(a,b) in enumerate(zip(ld,rd))]
        absolute=[s.calc(f'{name}_abs{i}',f'{name} absolute dB error {i+1}',f'=ABS({d})','dB') for i,d in enumerate(delta)]
        whole=component(s,name+'_all',name+' all',ll,rr,ld,rd)
        sectors=[]
        for i,indices in enumerate([slice(0,3),slice(3,6)]):
            sectors.append(component(s,f'{name}_s{i}',f'{name} sector {i+1}',ll[indices],rr[indices],ld[indices],rd[indices]))
        score_refs=[s.calc(f'{name}_sector_score{i}',f'{name} sector {i+1} score','='+v['score'],'0..100') for i,v in enumerate(sectors)]
        lower=s.calc(name+'_sector_quartile',name+' lower-quartile sector score',f'=PERCENTILE.INC({score_refs[0]}:{score_refs[-1]},0.25)','0..100')
        formulas=dict(linear_ccc='='+whole['linear_ccc'],log_ccc='='+whole['log_ccc'],linear_nrmse='='+whole['linear_nrmse'],
                      median_db_delta=f'=MEDIAN({delta[0]}:{delta[-1]})',mae_db=f'=AVERAGE({absolute[0]}:{absolute[-1]})',
                      p95_db=f'=PERCENTILE.INC({absolute[0]}:{absolute[-1]},0.95)',
                      peak_offset=f'=INDEX({xx[0]}:{xx[-1]},MATCH(MAX({ll[0]}:{ll[-1]}),{ll[0]}:{ll[-1]},0))-INDEX({xx[0]}:{xx[-1]},MATCH(MAX({rr[0]}:{rr[-1]}),{rr[0]}:{rr[-1]},0))',
                      score=f'=MIN(100,MAX(0,@global_weight*{whole["score"]}+@sector_weight*{lower}))')
        actual=rf_agreement_statistics(x,left,right)
        for metric,formula in formulas.items():
            export_rows.append(dict(case=name,metric=metric,value=getattr(actual,metric)))
            expected.append((name,metric,formula))
        for i,sector in enumerate(actual.sectors):
            for metric,formula,value in [('score','='+score_refs[i],sector.score),('start','='+xx[i*3],sector.start),
                                          ('stop','='+xx[i*3+2],sector.stop),('sample_count','=3',sector.sample_count)]:
                key=f'sector_{i+1}_{metric}'
                export_rows.append(dict(case=name,metric=key,value=value));expected.append((name,key,formula))
    folder=VALIDATION/'datasets'/'rf_compare'
    path=folder/'metrics.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case','metric','value']);writer.writeheader();writer.writerows(export_rows)
    with path.open(newline='',encoding='utf-8') as handle:observed=list(csv.DictReader(handle))
    s.data['files'].append(dict(name='metrics',csv='datasets/rf_compare/metrics.csv',rows=len(observed)))
    for row_index,((case,metric,formula),row) in enumerate(zip(expected,observed)):
        assert (row['case'],row['metric'])==(case,metric)
        unit='dB' if metric in ['median_db_delta','mae_db','p95_db'] else 'deg' if metric=='peak_offset' or metric.endswith(('_start','_stop')) else ''
        s.check(case,metric,formula,float(row['value']),unit,f'metrics.csv row {row_index+2}: {metric}')
    s.data['notes'].append('metrics.csv contains the direct numerical output of GRIM RF Compare; worksheet actuals are read back from that file.')
    s.data['notes'].append('The six-point sectors are [0,10,20] and [30,40,50] deg. Equal constant traces have CCC=1; unequal constants have CCC=0.')
    invalid=grid([0.,1.,0.],az=[0.,10.,20.]);s.save('input_insufficient_positive',invalid)
    s.reject('Insufficient positive pairs',lambda:rf_agreement_statistics([0.,10.,20.],[0.,1.,0.],[1.,2.,0.]),contains='at least two')
    value=s.finish();print(f'rf_compare: {len(value["checks"])} comparisons')
    return [value]


if __name__=='__main__':generate()
