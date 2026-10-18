"""Thickness-only frequency envelopes, individual layers and full joint grid.

Every selected thickness combination is evaluated; retained storage scales with
layers × polarizations × angles × frequencies, not the number of combinations.
"""
from dataclasses import replace
from decimal import Decimal
import hashlib
import itertools
import math
from pathlib import Path
import platform
import re

import numpy as np

from .compute import make_frequency_sweep, make_sweep, layer_material_label, INCH_TO_M
from .tolerance_config import ENVELOPE_MODE, StopToleranceAnalysis
from .tolerance_analysis import PreparedStudy
from .inverse_grid import NumericChoices
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
        target=float(state['target'])
    except (TypeError,ValueError): raise ValueError('Enter a numeric reflection limit.') from None
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


def thickness_axes(configs):
    """Use exactly the bulk-layer choices already defined for inverse design."""
    if not configs: raise ValueError('Add a stack before computing thickness envelopes.')
    bulk=[i for i,c in enumerate(configs) if not c.is_sheet]
    if not bulk: raise ValueError('Thickness envelopes need at least one bulk material layer; sheets have no thickness.')
    axes=[]
    for i in bulk:
        c=configs[i]
        if not math.isfinite(c.thickness_in) or c.thickness_in<=0: raise ValueError(f'Layer {i+1}: thickness must be positive.')
        low,high,step=c.inv_t_min_in,c.inv_t_max_in,c.inv_t_accuracy_in
        if (low is None)!=(high is None):
            raise ValueError(f'Layer {i+1}: set both thickness limits, or select Fixed.')
        if low is None: low=high=c.thickness_in
        axes.append(NumericChoices.build(low,high,step,f'Layer {i+1} thickness'))
    return bulk,axes


def nominal_index(axis, nominal):
    """Find a nominal solver value without expanding the choices."""
    index=0 if axis.count==1 else int(round((Decimal(str(nominal))-axis.start)/axis.step))
    return index if 0<=index<axis.count and axis.value(index)==nominal else None


def envelope_workload(configs,setup):
    validate_envelope_setup(setup)
    bulk,axes=thickness_axes(configs)
    nf,na=axis_counts(setup);npoles=2 if setup['polarization']=='Both' else 1
    joint=math.prod(axis.count for axis in axes)
    if joint>MAX_COMBINATIONS:
        raise ValueError(f'All-layer grid has {joint:,} combinations (limit {MAX_COMBINATIONS:,}). Increase layer thickness steps or narrow their ranges. No combinations have been skipped.')
    included=[nominal_index(axis,configs[i].thickness_in) is not None for i,axis in zip(bulk,axes)]
    other_nominal=[all(included[:j]+included[j+1:]) for j in range(len(axes))]
    individual_extra=[axis.count-int(inside) for axis,inside in zip(axes,included)]
    reused=int(all(included))+sum(n for n,others in zip(individual_extra,other_nominal) if others)
    evaluations=1+sum(individual_extra)+joint-reused
    points=evaluations*nf*na*npoles
    if points>MAX_EVALUATED_POINTS:
        raise ValueError(f'Thickness study has {points:,} response points (limit {MAX_EVALUATED_POINTS:,}). Increase layer steps or reduce frequency/angle points.')
    memory=validate_grid(nf,na,layer_count=len(configs),metric_count=2,
                        retained_grids=(len(bulk)+2)*npoles*4,
                        extra_bytes=sum(axis.count for axis in axes)*32,label='Thickness envelopes')
    return dict(evaluations=evaluations, joint_combinations=joint,
                response_points=points, grid_points=nf*na*npoles,
                individual_cases=[axis.count for axis in axes],
                active_thicknesses=sum(axis.count>1 for axis in axes),array_mib=memory/2**20)


def run_thickness_envelopes(layers,configs,setup,*,stop_requested=lambda:False,progress=lambda *_:None):
    from .tolerance_config import validate_setup
    from .io import layer_config_to_dict
    from .inverse_workflow import search_identity
    from .compute import UncertaintyConfig
    from .ui_options import INVERSE_SCORE_WHOLE_BAND
    setup=validate_setup(setup)
    work=envelope_workload(configs,setup)
    bulk,axes=thickness_axes(configs)
    freqs,angles,pols=envelope_axes(setup)
    target=float(setup['target'])
    identity=search_identity(configs,freqs,angles,setup['polarization'],UncertaintyConfig(False,0,0,0),INVERSE_SCORE_WHOLE_BAND,target)
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if len(layers)!=len(configs): raise ValueError('Loaded stack and layer inputs differ.')
    engine=PreparedStudy(layers,freqs,angles,pols,[],stop_requested)
    base=[c.thickness_in for c in configs]
    done=0
    def evaluate(thicknesses,phase):
        nonlocal done
        # Properties are prepared once. Replace only this private engine's layer
        # list; never mutate the caller's nominal stack or loaded material data.
        engine.layers=[layer if layer.is_sheet else replace(layer,thickness_m=thicknesses[i]*INCH_TO_M)
                       for i,layer in enumerate(layers)]
        values=engine.response([])
        done+=1
        progress(done,work['evaluations'],phase)
        return values
    nominal=evaluate(base,'Nominal reference')
    included=[nominal_index(axis,base[i]) is not None for i,axis in zip(bulk,axes)]
    other_nominal=[all(included[:j]+included[j+1:]) for j in range(len(axes))]
    panels=[dict(layer_index=i,label=f'Layer {i+1} · {layer_material_label(configs[i])}',
                 nominal_thickness_in=configs[i].thickness_in,
                 min_db=np.full_like(nominal,np.inf),max_db=np.full_like(nominal,-np.inf),
                 cases=axis.count,nominal_in_grid=inside,
                 thickness_values_in=[axis.value(k) for k in range(axis.count)])
            for i,axis,inside in zip(bulk,axes,included)]
    lower=np.full_like(nominal,np.inf);upper=np.full_like(nominal,-np.inf)
    for j,panel in enumerate(panels):
        i=panel['layer_index']
        for thickness in panel['thickness_values_in']:
            if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
            if thickness==base[i]: values=nominal
            else:
                varied=list(base);varied[i]=thickness
                values=evaluate(varied,f'Layer {i+1} alone')
            np.minimum(panel['min_db'],values,out=panel['min_db'])
            np.maximum(panel['max_db'],values,out=panel['max_db'])
            if other_nominal[j]:
                np.minimum(lower,values,out=lower);np.maximum(upper,values,out=upper)
    # Reuse one-at-a-time cases only if they belong to this exact joint grid.
    for choices in itertools.product(*(p['thickness_values_in'] for p in panels)):
        if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
        if sum(t!=base[i] for i,t in zip(bulk,choices))<=1: continue
        varied=list(base)
        for i,t in zip(bulk,choices): varied[i]=t
        values=evaluate(varied,'All layers varied independently')
        np.minimum(lower,values,out=lower);np.maximum(upper,values,out=upper)
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if identity!=search_identity(configs,freqs,angles,setup['polarization'],UncertaintyConfig(False,0,0,0),INVERSE_SCORE_WHOLE_BAND,target):
        raise ValueError('Material files changed during the study; rerun with stable inputs.')
    panels.append(dict(layer_index=None,label='All layers varied independently',min_db=lower,max_db=upper,
                       cases=work['joint_combinations'],nominal_in_grid=all(included)))
    assert done==work['evaluations']
    digest=hashlib.sha256()
    for name in ('thickness_envelopes.py','inverse_grid.py','tolerance_analysis.py','tolerance_config.py','compute.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return dict(kind='thickness_envelopes',setup=dict(setup),layers=[layer_config_to_dict(c) for c in configs],
                frequencies_ghz=freqs,angles_deg=angles,polarizations=pols,target_db=target,
                nominal_reflection_db=nominal,panels=panels,
                evaluated=done,workload=work,source_identity=identity,
                method='layer-min-max-step-v1',
                versions=dict(python=platform.python_version(),numpy=np.__version__,implementation_sha256=digest.hexdigest()),
                interpretation='Pointwise minimum/maximum PEC reflection over every thickness defined by each layer minimum, maximum and step. Values start at minimum and advance by step without exceeding maximum; an off-step maximum is excluded. Layers without limits stay at nominal. Individual panels hold other layers nominal; the combined panel uses every defined combination. Nominal is a separate reference and is included in bounds only when it belongs to the defined grid. Material properties, sheet resistances and statistical tolerance settings do not vary. These are discrete-grid envelopes, not certified continuous bounds. Incidence angles are measured from the surface normal.')


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
