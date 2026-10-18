"""Thickness-only frequency envelopes, individual layers and full joint grid.

Every selected thickness combination is evaluated; retained storage scales with
layers × polarizations × angles × frequencies, not the number of combinations.
"""
from dataclasses import asdict
import hashlib
import itertools
import math
from pathlib import Path
import platform
import re

import numpy as np

from .compute import make_frequency_sweep, make_sweep, layer_material_label
from .tolerance_config import ENVELOPE_MODE, validate_tolerances, StopToleranceAnalysis
from .tolerance_analysis import Parameter, PreparedStudy
from .sweep_admission import sweep_count, validate_grid

MAX_COMBINATIONS = 250_000
MAX_EVALUATED_POINTS = 50_000_000


def explicit_axis(text, label):
    if not str(text).strip(): return None
    tokens = re.split(r'[,;\s]+', str(text).strip())
    if len(tokens)>20001: raise ValueError(f'{label}: too many requested points.')
    try: values = sorted(set(float(t) for t in tokens))
    except ValueError: raise ValueError(f'{label}: enter numbers separated by commas or spaces.') from None
    if not all(math.isfinite(v) for v in values): raise ValueError(f'{label}: values must be finite.')
    if label == 'Frequency' and values[0]<=0: raise ValueError('Frequencies must be positive GHz values.')
    if label == 'Angle' and (values[0]<0 or values[-1]>=90):
        raise ValueError('Incidence angles must be from 0 to less than 90°, measured from the surface normal.')
    return values


def axis_counts(setup):
    counts=[]
    for prefix, key, label in [('f','frequency_list','Frequency'),('a','angle_list','Angle')]:
        axis=explicit_axis(setup.get(key,''),label)
        if axis is not None:
            counts.append(len(axis)); continue
        try: start,stop,step=(float(setup[prefix+'_'+s]) for s in ('start','stop','step'))
        except (ValueError,TypeError): raise ValueError(f'{label} sweep must be numeric.') from None
        n=sweep_count(start,stop,step)
        if prefix=='f' and start<=0: raise ValueError('Frequency start must be positive.')
        if prefix=='a' and (start<0 or stop>=90): raise ValueError('Incidence angles must satisfy 0 ≤ start ≤ stop < 90°.')
        counts.append(n)
    return counts


def validate_envelope_setup(state):
    if state['mode'] != ENVELOPE_MODE: raise ValueError('Choose Thickness tolerance envelopes.')
    try:
        n=int(state['envelope_points']);target=float(state['target'])
    except (TypeError,ValueError): raise ValueError('Enter integer thickness points and a numeric reflection limit.') from None
    if not 3<=n<=101 or n%2==0: raise ValueError('Use an odd number of thickness points from 3 to 101, including nominal and both bounds.')
    if not math.isfinite(target) or not -200<=target<=0: raise ValueError('Reflection limit must be between −200 and 0 dB.')
    if state['polarization'] not in ('TE','TM','Both'): raise ValueError('Choose TE, TM or Both.')
    nf,na=axis_counts(state)
    if nf>20001 or na>1001: raise ValueError('Use at most 20,001 frequencies and 1,001 incidence angles.')
    validate_grid(nf,na,metric_count=2,retained_grids=4,label='Thickness envelopes')
    return state


def envelope_axes(setup):
    nf,na=axis_counts(setup)
    freqs=explicit_axis(setup['frequency_list'],'Frequency')
    angles=explicit_axis(setup['angle_list'],'Angle')
    if freqs is None: freqs=make_frequency_sweep(*(float(setup['f_'+k]) for k in ('start','stop','step')))
    if angles is None: angles=make_sweep(*(float(setup['a_'+k]) for k in ('start','stop','step')))
    return freqs,angles,['te','tm'] if setup['polarization']=='Both' else [setup['polarization'].lower()]


def thickness_parameters(configs):
    if not configs: raise ValueError('Add a stack before computing thickness envelopes.')
    bulk=[i for i,c in enumerate(configs) if not c.is_sheet]
    if not bulk: raise ValueError('Thickness envelopes need at least one bulk material layer; sheets have no thickness.')
    params=[]
    for i in bulk:
        c=configs[i]
        if not math.isfinite(c.thickness_in) or c.thickness_in<=0: raise ValueError(f'Layer {i+1}: thickness must be positive.')
        spec=validate_tolerances(c.tolerances,False).get('thickness')
        if spec and spec['bound']>0:
            width=spec['bound']*c.thickness_in/100 if spec['units']=='%' else spec['bound']
            if c.thickness_in-width<=0: raise ValueError(f'Layer {i+1}: thickness tolerance must stay strictly positive.')
            params.append(Parameter(i,'thickness',**spec))
    return bulk,params


def envelope_workload(configs,setup):
    validate_envelope_setup(setup)
    bulk,params=thickness_parameters(configs)
    nf,na=axis_counts(setup);npoles=2 if setup['polarization']=='Both' else 1
    n=int(setup['envelope_points']);joint=n**len(params)
    if joint>MAX_COMBINATIONS:
        raise ValueError(f'All-layer grid has {joint:,} combinations (limit {MAX_COMBINATIONS:,}). Reduce thickness points or the number of varying layers.')
    # Individual-axis and nominal cases also belong to the Cartesian product;
    # evaluate them only once and reuse them in both envelopes.
    points=joint*nf*na*npoles
    if points>MAX_EVALUATED_POINTS:
        raise ValueError(f'Thickness study has {points:,} response points (limit {MAX_EVALUATED_POINTS:,}). Reduce thickness/frequency/angle points.')
    memory=validate_grid(nf,na,layer_count=len(configs),metric_count=2,
                        retained_grids=(len(bulk)+2)*npoles*4,label='Thickness envelopes')
    return dict(evaluations=joint, response_points=points, grid_points=nf*na*npoles,
                individual_cases=n,active_thicknesses=len(params),array_mib=memory/2**20)


def run_thickness_envelopes(layers,configs,setup,*,stop_requested=lambda:False,progress=lambda *_:None):
    from .tolerance_config import validate_setup
    from .io import layer_config_to_dict
    from .inverse_workflow import search_identity
    from .compute import UncertaintyConfig
    from .ui_options import INVERSE_SCORE_WHOLE_BAND
    setup=validate_setup(setup)
    work=envelope_workload(configs,setup)
    bulk,params=thickness_parameters(configs)
    freqs,angles,pols=envelope_axes(setup)
    target=float(setup['target'])
    identity=search_identity(configs,freqs,angles,setup['polarization'],UncertaintyConfig(False,0,0,0),INVERSE_SCORE_WHOLE_BAND,target)
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if len(layers)!=len(configs): raise ValueError('Loaded stack and layer inputs differ.')
    engine=PreparedStudy(layers,freqs,angles,pols,params,stop_requested)
    offsets=np.linspace(-1,1,int(setup['envelope_points']))
    center=len(offsets)//2
    nominal=engine.response(np.zeros(len(params)))
    panels=[dict(layer_index=i,label=f'Layer {i+1} · {layer_material_label(configs[i])}',
                 nominal_thickness_in=configs[i].thickness_in,
                 min_db=nominal.copy(),max_db=nominal.copy(),cases=1,
                 thickness_values_in=[configs[i].thickness_in]) for i in bulk]
    panel_by_layer={p['layer_index']:p for p in panels}
    lower=nominal.copy();upper=nominal.copy();done=1
    progress(done,work['evaluations'],'Nominal stack')
    for j,p in enumerate(params):
        panel=panel_by_layer[p.layer]
        width=p.bound*configs[p.layer].thickness_in/100 if p.units=='%' else p.bound
        panel['thickness_values_in']=(configs[p.layer].thickness_in+offsets*width).tolist()
        for k,offset in enumerate(offsets):
            if k==center: continue
            deviation=np.zeros(len(params));deviation[j]=offset
            values=engine.response(deviation)
            np.minimum(panel['min_db'],values,out=panel['min_db'])
            np.maximum(panel['max_db'],values,out=panel['max_db'])
            np.minimum(lower,values,out=lower);np.maximum(upper,values,out=upper)
            done+=1;panel['cases']+=1
            progress(done,work['evaluations'],f'Layer {p.layer+1} alone')
    # Stream combinations; nominal and one-at-a-time cases were already included.
    for indices in itertools.product(range(len(offsets)),repeat=len(params)):
        if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
        if sum(i!=center for i in indices)<=1: continue
        values=engine.response(offsets[list(indices)])
        np.minimum(lower,values,out=lower);np.maximum(upper,values,out=upper)
        done+=1;progress(done,work['evaluations'],'All layers varied independently')
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if identity!=search_identity(configs,freqs,angles,setup['polarization'],UncertaintyConfig(False,0,0,0),INVERSE_SCORE_WHOLE_BAND,target):
        raise ValueError('Material files changed during the study; rerun with stable inputs.')
    panels.append(dict(layer_index=None,label='All layers varied independently',min_db=lower,max_db=upper,cases=done))
    assert done==work['evaluations']
    digest=hashlib.sha256()
    for name in ('thickness_envelopes.py','tolerance_analysis.py','tolerance_config.py','compute.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return dict(kind='thickness_envelopes',setup=dict(setup),layers=[layer_config_to_dict(c) for c in configs],
                frequencies_ghz=freqs,angles_deg=angles,polarizations=pols,target_db=target,
                nominal_reflection_db=nominal,panels=panels,parameters=[asdict(p) for p in params],
                offsets=offsets.tolist(),evaluated=done,workload=work,source_identity=identity,
                versions=dict(python=platform.python_version(),numpy=np.__version__,implementation_sha256=digest.hexdigest()),
                interpretation='Pointwise minimum/maximum PEC reflection over the complete selected thickness grid. Other material properties and sheet resistances stay nominal. Each layer varies independently in the joint grid; statistical distributions/groups are not applied. These are sampled performance envelopes, not percentiles, yield, or certified continuous bounds. Incidence angles are measured from the surface normal.')


def export_envelope_csv(path,result):
    import csv
    from .io import _atomic_text_file
    if result.get('kind')!='thickness_envelopes': raise ValueError('Run thickness envelopes before exporting curves.')
    with _atomic_text_file(path, newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','layer_index_1based','polarization','incidence_deg','frequency_ghz',
                         'nominal_reflection_db','minimum_reflection_db','maximum_reflection_db','thickness_combinations'])
        for panel in result['panels']:
            for pi,pol in enumerate(result['polarizations']):
                for ai,angle in enumerate(result['angles_deg']):
                    for fi,freq in enumerate(result['frequencies_ghz']):
                        writer.writerow([panel['label'],'' if panel['layer_index'] is None else panel['layer_index']+1,
                            pol.upper(),angle,freq,result['nominal_reflection_db'][pi,ai,fi],
                            panel['min_db'][pi,ai,fi],panel['max_db'][pi,ai,fi],panel['cases']])
