"""Choose compact worked examples from the complete numerical plot evidence."""
from pathlib import Path
import json

V=Path(__file__).resolve().parents[1]

def build(slug):
    s=json.loads((V/'specs'/f'{slug}.json').read_text())
    inputs={x['id']:x for x in s['inputs']}
    def v(key):return {'id':key,'value':inputs[key]['value']}
    out={k:s[k] for k in ('slug','title','description','settings','scope')}
    out.update(rawHeaders=['Azimuth (deg)','Elevation (deg)','Frequency (GHz)','Pol','RCS (m²)','Phase (deg)'],rawRows=[],selectedCheckIndices=[],labelOverrides={})
    choose=lambda predicate:[i for i,c in enumerate(s['checks']) if predicate(c)]
    if slug in ('azimuth_plot','polar_plot','phase_plot'):
        out['rawRows']=[[v(f'az{i}'),0,10,'VV',v(f'p{i}'),v(f'phi{i}')] for i in range(6)]
        metrics={'Level','Phase'}|({'Plotted azimuth'} if slug=='polar_plot' else set())
        out['selectedCheckIndices']=choose(lambda c:c['metric'] in metrics)
        out['settings']='One VV cut at elevation 0°, frequency 10 GHz.'
    elif slug in ('frequency_plot','elevation_plot'):
        es=[0] if slug=='frequency_plot' else [0,1]
        fs=range(4) if slug=='frequency_plot' else [0]
        out['rawHeaders']=out['rawHeaders'][:5]
        out['rawRows']=[[az,[0,30][e],8+f,'VV',v(f'p{a}_{e}_{f}_0')] for a,az in enumerate([-20,0,20]) for e in es for f in fs]
        out['selectedCheckIndices']=choose(lambda c:c['metric']=='Level' and ', cut 1,' in c['case'] and (slug=='elevation_plot' or c['case'].startswith('mean') or c['case'].endswith('sample 2')))
        out['settings']='Reduce the three azimuths in linear power, then convert the result to dBsm.'
        out['description']='Mean, median, P90 and maximum of a small azimuth sweep.'
        out['labelOverrides']={str(i):c['case'].split(',')[0].capitalize()+' at '+(str(7+int(c['case'].split('sample ')[1]))+' GHz' if slug=='frequency_plot' else str([0,30][int(c['case'].split('sample ')[1])-1])+'°')+' (dBsm)' for i,c in enumerate(s['checks']) if i in out['selectedCheckIndices']}
    elif slug=='waterfall':
        out['rawHeaders']=out['rawHeaders'][:5]
        out['rawRows']=[[az,el,8+f,'VV',v(f'p{a}_{e}_{f}_0')] for a,az in enumerate([0,90]) for e,el in enumerate([0,30]) for f in range(3)]
        out['selectedCheckIndices']=list(range(6))
        out['settings']='Mean over the two elevations; display azimuth × frequency in dBsm.'
        out['labelOverrides']={str(i):f'Az {0 if i<3 else 90}°, {8+i%3} GHz (dBsm)' for i in range(6)}
    elif slug=='pbp':
        out['rawHeaders']=['Dataset','Azimuth (deg)','Elevation (deg)','Frequency (GHz)','Pol','RCS (m²)']
        out['rawRows']=[[f'input_{k+1}',az,0,8+f,'VV',v(f'p{k}_{a}_{f}')] for k in range(2) for a,az in enumerate([-30,0,30]) for f in range(3)]
        out['selectedCheckIndices']=choose(lambda c:c['metric'] in ('lower','upper'))
        out['settings']='Two datasets; each band is the minimum and maximum of its three azimuth cuts.'
        out['labelOverrides']={str(i):c['case'].replace('f 1','8 GHz').replace('f 2','9 GHz').replace('f 3','10 GHz')+' '+c['metric']+' (m²)' for i,c in enumerate(s['checks']) if i in out['selectedCheckIndices']}
    elif slug in ('cdf','sector_stats'):
        out['rawHeaders']=out['rawHeaders'][:5]
        out['rawRows']=[[a,0,10,'VV',v(f'p{i}')] for i,a in enumerate([0,30,60,90,120,150])]
        out['selectedCheckIndices']=choose(lambda c:c['metric'] in (('cdf','exceedance') if slug=='cdf' else ('mean','median','min','max','percentile')))
        out['settings']='Inclusive empirical CDF and exceedance; missing values are excluded.' if slug=='cdf' else 'Sectors [0°, 90°) and [90°, 180°]; percentile = P90.'
    elif slug=='delta_map':
        out['rawHeaders']=['Azimuth (deg)','Frequency (GHz)','A RCS (m²)','B RCS (m²)']
        out['rawRows']=[[0 if i<3 else 90,8+i%3,v(f'a{i}'),v(f'b{i}')] for i in range(6)]
        out['selectedCheckIndices']=list(range(6))
        out['settings']='A minus B in dB; elevation 0°, VV. Zero or missing power has no finite dB difference.'
        out['labelOverrides']={str(i):f'Az {0 if i<3 else 90}°, {8+i%3} GHz (dB)' for i in range(6)}
    elif slug=='save_load_export':
        out['rawRows']=[[v(f'az{i}'),0,9.5,'VV',v(f'p{i}'),v(f'ph{i}')] for i in range(5)]
        out['selectedCheckIndices']=choose(lambda c:c['metric']=='Power' or (c['metric']=='Phase' and c['case'] in ('Sample 2','Sample 3')))
        out['settings']='Save native input, reload it, and export power and phase to CSV.'
    elif slug=='rf_compare':
        out['rawHeaders']=['Azimuth (deg)','A RCS (m²)','B RCS (m²)','A (dBsm, calc.)','B (dBsm, calc.)','Difference (dB)','|Difference| (dB)']
        calculations={x['id']:x for x in s['calculations']}
        def calculated(key):return {'id':key,'formula':calculations[key]['formula']}
        out['rawRows']=[[v(f'gain_x4_x{i}'),v(f'gain_x4_l{i}'),v(f'gain_x4_r{i}'),calculated(f'gain_x4_ld{i}'),calculated(f'gain_x4_rd{i}'),calculated(f'gain_x4_delta{i}'),calculated(f'gain_x4_abs{i}')] for i in range(6)]
        wanted={'linear_ccc','linear_nrmse','median_db_delta','mae_db','p95_db','peak_offset'}
        out['selectedCheckIndices']=choose(lambda c:c['case']=='gain_x4' and c['metric'] in wanted)
        labels={'linear_ccc':'Linear CCC','linear_nrmse':'Normalized RMS error','median_db_delta':'Median A − B (dB)','mae_db':'Mean absolute difference (dB)','p95_db':'P95 absolute difference (dB)','peak_offset':'Peak azimuth offset (deg)'}
        out['labelOverrides']={str(i):labels[s['checks'][i]['metric']] for i in out['selectedCheckIndices']}
        out['description']='Compare two six-sample traces with a fourfold power difference.'
        out['settings']='Magnitude comparison using gain_x4_left and gain_x4_right; 10 GHz, elevation 0°, VV.'
    else:raise ValueError(slug)
    assert out['selectedCheckIndices']
    p=V/'compact_specs'/f'{slug}.json';p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n')
    return len(out['selectedCheckIndices'])

def generate():
    slugs=['azimuth_plot','polar_plot','phase_plot','frequency_plot','elevation_plot','waterfall','pbp','cdf','sector_stats','delta_map','save_load_export','rf_compare']
    for slug in slugs: print(slug,build(slug))

if __name__=='__main__':generate()
