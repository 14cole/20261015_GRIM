"""Thickness-only frequency envelopes, individual layers and full joint grid.

Every selected thickness combination is evaluated; retained storage scales with
layers × polarizations × angles × frequencies, not the number of combinations.
"""
from dataclasses import replace
from decimal import Decimal
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np

from .compute import make_frequency_sweep, make_sweep, layer_material_label, INCH_TO_M
from .tolerance_config import ENVELOPE_MODE, StopToleranceAnalysis
from .tolerance_analysis import (PreparedStudy, effective_material_hash,
                                  requirement_margin, REQUIREMENT_TOL_DB)
from .inverse_grid import NumericChoices
from .sweep_admission import sweep_count, validate_grid

MAX_COMBINATIONS = 250_000
MAX_EVALUATED_POINTS = 50_000_000


def axis_counts(setup):
    counts=[]
    for prefix, label in [('f','Frequency'),('a','Angle')]:
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
    except (TypeError,ValueError): raise ValueError('Enter a numeric reflection target.') from None
    if not math.isfinite(target) or not -200<=target<=0: raise ValueError('Reflection Target must be between −200 and 0 dB.')
    if state['polarization'] not in ('TE','TM','Both'): raise ValueError('Choose TE, TM or Both.')
    nf,na=axis_counts(state)
    if nf>20001 or na>1001: raise ValueError('Use at most 20,001 frequencies and 1,001 incidence angles.')
    validate_grid(nf,na,metric_count=2,retained_grids=4,label='Thickness envelopes')
    return state


def envelope_axes(setup):
    axis_counts(setup)
    freqs=make_frequency_sweep(*(float(setup['f_'+k]) for k in ('start','stop','step')))
    angles=make_sweep(*(float(setup['a_'+k]) for k in ('start','stop','step')))
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


def envelope_workload(configs,setup, *, retain_layer_metrics=False):
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
    # Bounds, witness indices, two realizable stacks per panel, plus optional
    # individual-layer metric grids for the existing detailed sweep viewer.
    extra = sum(axis.count for axis in axes)*32 + joint*80
    if retain_layer_metrics:
        extra += (sum(axis.count for axis in axes)+2)*nf*na*npoles*8*8
    memory=validate_grid(nf,na,layer_count=len(configs),metric_count=2,
                        retained_grids=(len(bulk)+2)*npoles*6,
                        extra_bytes=extra,label='Thickness envelopes')
    return dict(evaluations=evaluations, joint_combinations=joint,
                response_points=points, grid_points=nf*na*npoles,
                individual_cases=[axis.count for axis in axes],
                active_thicknesses=sum(axis.count>1 for axis in axes),array_mib=memory/2**20)


def run_thickness_envelopes(layers,configs,setup,*,stop_requested=lambda:False,progress=lambda *_:None,
                            retain_layer_metrics=False):
    from .tolerance_config import validate_setup
    from .io import layer_config_to_dict
    setup=validate_setup(setup)
    work=envelope_workload(configs,setup,retain_layer_metrics=retain_layer_metrics)
    bulk,axes=thickness_axes(configs)
    freqs,angles,pols=envelope_axes(setup)
    target=float(setup['target'])
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if len(layers)!=len(configs): raise ValueError('Loaded stack and layer inputs differ.')
    engine=PreparedStudy(layers,freqs,angles,pols,[],stop_requested)
    identity=effective_material_hash(layers,engine.properties)
    captured_layers=[layer_config_to_dict(c) for c in configs]
    source_identity=hashlib.sha256(json.dumps(dict(layers=captured_layers,
        frequencies_ghz=freqs,angles_deg=angles,polarizations=pols,target_db=target,
        effective_material_sha256=identity),sort_keys=True,allow_nan=False).encode()).hexdigest()
    base=[c.thickness_in for c in configs]
    done=0
    metric_rows = None
    compute_metrics = engine.compute_metrics
    def capture_metrics(*args, **kwargs):
        result = compute_metrics(*args, **kwargs)
        if metric_rows is not None:
            metric_rows.append(result)
        return result
    engine.compute_metrics = capture_metrics
    def evaluate(thicknesses,phase,capture=False):
        nonlocal done
        nonlocal metric_rows
        # Properties are prepared once. Replace only this private engine's layer
        # list; never mutate the caller's nominal stack or loaded material data.
        engine.layers=[layer if layer.is_sheet else replace(layer,thickness_m=thicknesses[i]*INCH_TO_M)
                       for i,layer in enumerate(layers)]
        metric_rows=[] if capture else None
        values=engine.response([])
        metrics = ({key: np.asarray([row[key] for row in metric_rows]).reshape(values.shape)
                    for key in metric_rows[0] if key not in ('freq_ghz','angle_deg')}
                   if metric_rows else None)
        metric_rows=None
        done+=1
        progress(done,work['evaluations'],phase)
        return values,metrics
    nominal,nominal_metrics=evaluate(base,'Nominal reference',retain_layer_metrics)
    included=[nominal_index(axis,base[i]) is not None for i,axis in zip(bulk,axes)]
    other_nominal=[all(included[:j]+included[j+1:]) for j in range(len(axes))]
    panels=[dict(layer_index=i,label=f'Layer {i+1} · {layer_material_label(configs[i])}',
                 nominal_thickness_in=configs[i].thickness_in,
                 min_db=np.full_like(nominal,np.inf),max_db=np.full_like(nominal,-np.inf),
                 cases=axis.count,nominal_in_grid=inside,
                 thickness_values_in=[axis.value(k) for k in range(axis.count)])
            for i,axis,inside in zip(bulk,axes,included)]
    panels.append(dict(layer_index=None,label='All layers varied independently',
                       min_db=np.full_like(nominal,np.inf),max_db=np.full_like(nominal,-np.inf),
                       cases=work['joint_combinations'],nominal_in_grid=all(included)))
    for panel in panels:
        panel.update(min_case_index=np.full(nominal.shape,-1,dtype=np.int64),
                     max_case_index=np.full(nominal.shape,-1,dtype=np.int64),
                     best_stack=None,worst_stack=None,passing_cases=0)
    def accumulate(panel,values,index,thicknesses):
        # Deterministic ties: lowest case index wins even when evaluation reuse
        # visits the cases in a different order from the Cartesian product.
        for key,compare in (('min',np.less),('max',np.greater)):
            bound=panel[key+'_db']; witness=panel[key+'_case_index']
            mask=compare(values,bound) | ((values==bound)&((witness<0)|(index<witness)))
            witness[mask]=index; np.copyto(bound,values,where=mask)
        worst=float(values.max()); margin=requirement_margin(target,worst)
        panel['passing_cases']+=int(margin>=0)
        for name,better in (('best_stack',lambda a,b:a<b),('worst_stack',lambda a,b:a>b)):
            old=panel[name]
            if old is None or better(worst,old['worst_reflection_db']) or (worst==old['worst_reflection_db'] and index<old['case_index']):
                pi,ai,fi=np.unravel_index(int(values.argmax()),values.shape)
                panel[name]=dict(case_index=index,thicknesses_in=list(thicknesses),
                    worst_reflection_db=worst,margin_db=margin,reflection_db=values.copy(),
                    limiting_point=dict(polarization=pols[pi],angle_deg=angles[ai],frequency_ghz=freqs[fi]))
    def joint_index(thicknesses):
        index=0
        for i,axis in zip(bulk,axes):
            position=nominal_index(axis,thicknesses[i])
            if position is None: return None
            index=index*axis.count+position
        return index
    joint_seen=set()
    def include_joint(values,thicknesses):
        index=joint_index(thicknesses)
        if index is not None and index not in joint_seen:
            accumulate(panels[-1],values,index,thicknesses)
            joint_seen.add(index)
    for j,panel in enumerate(panels[:-1]):
        i=panel['layer_index']
        if retain_layer_metrics:
            panel['metrics']={key:np.empty((panel['cases'],*nominal.shape)) for key in nominal_metrics}
        for case,thickness in enumerate(panel['thickness_values_in']):
            if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
            varied=list(base);varied[i]=thickness
            if thickness==base[i]: values,metrics=nominal,nominal_metrics
            else:
                values,metrics=evaluate(varied,f'Layer {i+1} alone',retain_layer_metrics)
            accumulate(panel,values,case,varied)
            if retain_layer_metrics:
                for key in metrics: panel['metrics'][key][case]=metrics[key]
            if other_nominal[j]:
                include_joint(values,varied)
    # Reuse one-at-a-time cases only if they belong to this exact joint grid.
    for choices in itertools.product(*(p['thickness_values_in'] for p in panels[:-1])):
        if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
        if sum(t!=base[i] for i,t in zip(bulk,choices))<=1: continue
        varied=list(base)
        for i,t in zip(bulk,choices): varied[i]=t
        values,_=evaluate(varied,'All layers varied independently')
        include_joint(values,varied)
    if stop_requested(): raise StopToleranceAnalysis('Thickness study stopped; previous results retained.')
    if len(joint_seen)!=work['joint_combinations']:
        raise RuntimeError('Incomplete thickness grid; no result will be published.')
    for panel in panels:
        panel['margin_loss_db']=requirement_margin(target,nominal.max())-panel['worst_stack']['margin_db']
    assert done==work['evaluations']
    digest=hashlib.sha256()
    for name in ('thickness_envelopes.py','inverse_grid.py','tolerance_analysis.py','tolerance_config.py','compute.py','requirements.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return dict(kind='thickness_envelopes',setup=dict(setup),layers=captured_layers,
                frequencies_ghz=freqs,angles_deg=angles,polarizations=pols,target_db=target,
                nominal_reflection_db=nominal,panels=panels,
                evaluated=done,workload=work,source_identity=source_identity,
                method='layer-min-max-step-v2',effective_material_sha256=identity,
                requirement_comparison_tolerance_db=REQUIREMENT_TOL_DB,
                versions=dict(python=platform.python_version(),numpy=np.__version__,implementation_sha256=digest.hexdigest()),
                interpretation='Pointwise minimum/maximum PEC reflection over every thickness defined by each layer minimum, maximum and step. Values start at minimum and advance by step without exceeding maximum; an off-step maximum is excluded. Layers without limits stay at nominal. Individual panels hold other layers nominal; the combined panel uses every defined combination. Nominal is a separate reference and is included in bounds only when it belongs to the defined grid. Material properties, sheet resistances and statistical tolerance settings do not vary. These are discrete-grid envelopes, not certified continuous bounds. Incidence angles are measured from the surface normal.')


def case_thicknesses(result,panel,case_index):
    """Decode a captured witness without consulting the editable stack."""
    values=[float(c['thickness_in']) for c in result['layers']]
    individuals=result['panels'][:-1]
    if panel['layer_index'] is not None:
        values[panel['layer_index']]=panel['thickness_values_in'][int(case_index)]
    else:
        index=int(case_index)
        for individual in reversed(individuals):
            index,choice=divmod(index,individual['cases'])
            values[individual['layer_index']]=individual['thickness_values_in'][choice]
    return values


def export_envelope_csv(path,result,polarization=None):
    import csv
    import json
    from .io import _atomic_text_file
    if result.get('kind')!='thickness_envelopes': raise ValueError('Run thickness envelopes before exporting curves.')
    if polarization is not None and polarization.upper() not in [p.upper() for p in result['polarizations']]:
        raise ValueError('That polarization was not analyzed in this study.')
    with _atomic_text_file(path, newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['case','layer_index_1based','polarization','incidence_deg','frequency_ghz',
                         'nominal_reflection_db','minimum_reflection_db','maximum_reflection_db','thickness_combinations',
                         'minimum_case_index_0based','maximum_case_index_0based',
                         'minimum_stack_thicknesses_in','maximum_stack_thicknesses_in'])
        for panel in result['panels']:
            for pi,pol in enumerate(result['polarizations']):
                if polarization is not None and pol.upper() != polarization.upper(): continue
                for ai,angle in enumerate(result['angles_deg']):
                    for fi,freq in enumerate(result['frequencies_ghz']):
                        low=int(panel['min_case_index'][pi,ai,fi]); high=int(panel['max_case_index'][pi,ai,fi])
                        writer.writerow([panel['label'],'' if panel['layer_index'] is None else panel['layer_index']+1,
                            pol.upper(),angle,freq,result['nominal_reflection_db'][pi,ai,fi],
                            panel['min_db'][pi,ai,fi],panel['max_db'][pi,ai,fi],panel['cases'],low,high,
                            json.dumps(case_thicknesses(result,panel,low)),json.dumps(case_thicknesses(result,panel,high))])


def export_envelope_files(path, result):
    from .export_paths import polarization_paths
    paths = polarization_paths(path, result['polarizations'])
    for pol, destination in paths.items():
        export_envelope_csv(destination, result, polarization=pol)
    return list(paths.values())
