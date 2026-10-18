"""Shared manufacturing scenarios and independent candidate verification.

Bounds are sampled, never certified. Ranking shares scenarios across designs;
verification uses disjoint seeds and refines the operating grid independently.
"""
from dataclasses import replace, asdict
import hashlib
import itertools
import math
import platform
from pathlib import Path

import numpy as np

from .compute import (LoadedLayer, build_uncertainty_scales, compute_angle_metrics_many,
                      UncertaintyConfig)
from .io import read_material_table, constant_material_from_layer, layer_config_to_dict
from .tolerance_analysis import PreparedStudy, parameters_from_layers, sampled_deviations, StopToleranceAnalysis, repeatability
from .tolerance_config import validate_tolerances
from .sweep_admission import validate_grid
from .ui_options import INVERSE_SCORE_YIELD, inverse_polarizations

from .robust_config import ROBUST_DEFAULTS, validate_robust_setup


def bound_scenarios(count, dimensions, seed):
    """Uniform joint interior probes plus nominal, axis ends and box corners.

    The rectangle deliberately ignores distribution/group restrictions: it is a
    conservative input support, distinct from statistical manufacturing trials.
    """
    if not dimensions:
        return np.empty((1, 0))
    from scipy.stats import qmc
    interior = 2*qmc.Sobol(dimensions, scramble=True, seed=seed).random_base2(int(math.log2(count)))-1
    axes = np.eye(dimensions)
    corners = (np.asarray(list(itertools.product((-1., 1.), repeat=dimensions)))
               if dimensions <= 8 else np.asarray([np.ones(dimensions), -np.ones(dimensions)]))
    return np.vstack((np.zeros((1, dimensions)), axes, -axes, corners, interior))


def manufacturing_scenarios(configs, setup, mode):
    setup = validate_robust_setup(setup)
    params = parameters_from_layers(configs)
    if mode == INVERSE_SCORE_YIELD:
        scenarios = np.vstack(list(sampled_deviations(params, setup['samples'], setup['seed'])))
    else:
        scenarios = bound_scenarios(setup['samples'], len(params), setup['seed'])
    return params, scenarios


def manufacturing_case_count(configs, setup, mode):
    setup = validate_robust_setup(setup)
    p = len(parameters_from_layers(configs))
    return (1 + setup['samples'] if mode == INVERSE_SCORE_YIELD else
            1 + 2*p + (2**p if p <= 8 else 2) + setup['samples'])


def manufacturing_response(layers, freqs, angles, pol, params, stop_requested):
    return PreparedStudy(layers, freqs, angles, inverse_polarizations(pol), params, stop_requested)


def score_manufacturing(layers, freqs, angles, pol, params, scenarios, mode, target, stop_requested):
    engine = manufacturing_response(layers, freqs, angles, pol, params, stop_requested)
    nominal = engine.response(np.zeros(len(params)))
    means = []
    worst = float(nominal.max())
    failed = 0
    for deviation in scenarios:
        response = nominal if not np.any(deviation) else engine.response(deviation)
        peak = float(response.max())
        worst = max(worst, peak)
        failed += int(peak > target)
        means.append(float(response.mean()))
    score = 100*failed/len(scenarios) if mode == INVERSE_SCORE_YIELD else worst-target
    return score, float(nominal.mean()), max(means), float(np.mean(means)), min(means)


def load_design(configs):
    """Load a captured design without consulting live widgets."""
    cache = {}
    def table(path):
        if path not in cache:
            cache[path] = read_material_table(Path(path))
        return cache[path]
    return [LoadedLayer(0., False, 0., None, None, True, c.sheet_resistance) if c.is_sheet else
            LoadedLayer(c.thickness_in*.0254, c.anisotropic, c.polarization_deg,
                        constant_material_from_layer(c) if c.is_constant else table(c.file_0deg),
                        table(c.file_90deg) if c.anisotropic else None) for c in configs]


def candidate_configs(configs, candidate):
    return [replace(c, sheet_resistance=candidate.sheet_resistance_ohm[i]) if c.is_sheet else
            replace(c, thickness_in=candidate.thickness_in[i]) for i, c in enumerate(configs)]



def refine_axis(axis):
    values = np.asarray(axis, dtype=float)
    if len(values) <= 1:
        return values.tolist()
    return np.sort(np.concatenate((values, (values[:-1]+values[1:])/2))).tolist()


def verify_candidates(configs, candidates, frequencies, angles, polarization, uncertainty,
                      target, setup=None, *, continuous=True, stop_requested=lambda: False,
                      progress=lambda *_: None):
    """Verify retained candidates on nested grids and fresh joint scenarios.

    Every configured refinement is evaluated. A small final change is evidence
    of sampled stability only, never a proof. Discrete frequency targets are
    preserved; intervening frequencies are tested only for a band sweep.
    """
    from .inverse_workflow import search_identity
    from .ui_options import INVERSE_SCORE_WHOLE_BAND
    setup = validate_robust_setup(setup)
    if not candidates:
        raise ValueError('Retain at least one candidate before verification.')
    if not math.isfinite(target) or not -200 <= target <= 0:
        raise ValueError('Verification reflection target must be between −200 and 0 dB.')
    for values, label in ((frequencies, 'frequencies'), (angles, 'angles')):
        if not values or not all(math.isfinite(v) for v in values) or any(b<=a for a,b in zip(values,values[1:])):
            raise ValueError(f'Verification {label} must be finite, nonempty and strictly increasing.')
    if min(frequencies)<=0 or min(angles)<0 or max(angles)>=90:
        raise ValueError('Use positive frequencies and incidence angles from 0 to less than 90°.')
    has_parameters = any(any(v['bound'] > 0 for v in validate_tolerances(c.tolerances, c.is_sheet).values()) for c in configs)
    params = parameters_from_layers(configs) if has_parameters else []
    systematic = uncertainty.enabled and any((uncertainty.thickness_pct, uncertainty.eps_pct, uncertainty.mu_pct))
    if params and systematic:
        raise ValueError('Verification uses either per-layer manufacturing bounds or shared systematic scales. Disable shared uncertainty to verify per-layer bounds; this avoids silently double-counting material errors.')
    pols = inverse_polarizations(polarization)
    max_nf = (len(frequencies)-1)*2**setup['levels']+1 if continuous else len(frequencies)
    max_na = (len(angles)-1)*2**setup['levels']+1
    validate_grid(max_nf, max_na, layer_count=len(configs), metric_count=2, retained_grids=2*len(pols), label='Candidate verification')
    # Bound runtime before starting, in addition to the existing allocation guard.
    support_samples = setup['samples']*2**setup['levels']
    support_count = (1+2*len(params)+(2**len(params) if len(params)<=8 else 2)+support_samples) if params else support_samples+48
    work = len(candidates)*max_nf*max_na*len(pols)*(support_count*(setup['levels']+1)+setup['samples']*setup['repeats'])
    if work > 50_000_000:
        raise ValueError('Verification exceeds 50 million response points. Verify fewer candidates or reduce samples/refinements.')
    source_identity = search_identity(configs, frequencies, angles, polarization, uncertainty, INVERSE_SCORE_WHOLE_BAND, target)
    captured_layers = [layer_config_to_dict(c) for c in configs]
    # Separate ranking (seed), support verification (seed+1), and yield repeats.
    seeds = [(setup['seed']+2+r) % 2**32 for r in range(setup['repeats'])]
    rows = []
    for ci, candidate in enumerate(candidates):
        if stop_requested(): raise StopToleranceAnalysis('Verification stopped; no partial result published.')
        design = candidate_configs(configs, candidate)
        layers = load_design(design)
        freqs, sampled_angles = list(frequencies), list(angles)
        history = []
        worst_condition = None
        for level in range(setup['levels']+1):
            support = bound_scenarios(setup['samples']*2**level, len(params) if params else 3, (setup['seed']+1)%2**32)
            if level:
                freqs = refine_axis(freqs) if continuous else freqs
                sampled_angles = refine_axis(sampled_angles)
            engine = manufacturing_response(layers, freqs, sampled_angles, polarization, params, stop_requested)
            if params:
                cases = support
                def response(case): return engine.response(case)
            else:
                deltas = np.asarray([uncertainty.thickness_pct, uncertainty.eps_pct, uncertainty.mu_pct])/100 if uncertainty.enabled else np.zeros(3)
                cases = np.unique(np.vstack((np.asarray(build_uncertainty_scales(uncertainty)), 1+support*deltas)), axis=0)
                def response(case):
                    out = []
                    for p in pols:
                        plane = []
                        for a in sampled_angles:
                            if stop_requested(): raise StopToleranceAnalysis('Verification stopped.')
                            plane.append(compute_angle_metrics_many(freqs, a, layers, p,
                                thickness_scale=case[0], eps_scale=case[1], mu_scale=case[2])['metal_loss_db'])
                        out.append(plane)
                    return np.asarray(out)
            peak = -math.inf
            for case_index, case in enumerate(cases):
                values = response(case)
                if not np.all(np.isfinite(values)):
                    raise ValueError('Verification produced nonfinite reflection.')
                value = float(values.max())
                if value > peak:
                    peak = value
                    pi, ai, fi = np.unravel_index(int(values.argmax()), values.shape)
                    worst_condition = dict(frequency_ghz=freqs[fi], angle_deg=sampled_angles[ai], polarization=pols[pi],
                                           deviations_or_scales=case.tolist(), case_index=case_index)
            history.append(dict(level=level, frequencies=len(freqs), angles=len(sampled_angles),
                                cases=len(cases), worst_reflection_db=peak, margin_db=target-peak))
            progress(ci, len(candidates), f'Candidate {ci+1}: refinement {level}/{setup["levels"]}')
        fractions = []
        if params:
            for seed in seeds:
                passing = 0
                for batch in sampled_deviations(params, setup['samples'], seed):
                    for case in batch:
                        passing += int(float(engine.response(case).max()) <= target)
                fractions.append(100*passing/setup['samples'])
                progress(ci, len(candidates), f'Candidate {ci+1}: independent yield repeat {len(fractions)}/{len(seeds)}')
        change = abs(history[-1]['margin_db']-history[-2]['margin_db'])
        rows.append(dict(candidate_index=ci, design=[layer_config_to_dict(c) for c in design],
                         worst_margin_db=history[-1]['margin_db'], worst_condition=worst_condition,
                         refinement_history=history, last_change_db=change,
                         sampled_stable=bool(len(history)>=3 and change<=.05 and abs(history[-2]['margin_db']-history[-3]['margin_db'])<=.05),
                         yield_estimate=repeatability(fractions) if fractions else None))
        progress(ci+1, len(candidates), 'Verification')
    if stop_requested(): raise StopToleranceAnalysis('Verification stopped; no partial result published.')
    if search_identity(configs, frequencies, angles, polarization, uncertainty, INVERSE_SCORE_WHOLE_BAND, target) != source_identity:
        raise ValueError('Material files changed during verification; rerun with stable inputs.')
    implementation = hashlib.sha256()
    for name in ('robust_design.py', 'robust_config.py', 'tolerance_analysis.py', 'tolerance_config.py', 'compute.py'):
        implementation.update(Path(__file__).with_name(name).read_bytes())
    import scipy
    return dict(setup=setup, layers=captured_layers, source_identity=source_identity,
                target_db=target, polarizations=list(pols), frequency_scope='continuous band refinement' if continuous else 'discrete targets only',
                uncertainty=asdict(uncertainty), parameters=[asdict(p) for p in params],
                support_model='Per-layer rectangular bounds; group constraints ignored for conservative support sampling' if params else 'Shared systematic scales',
                yield_seeds=seeds if params else [], candidates=rows,
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, implementation_sha256=implementation.hexdigest()),
                interpretation='Verification ranks retained candidates only. Nested frequency/angle grids and joint interior tolerance samples are not certified continuous bounds. Modeled yield uses independent scrambled trials on the finest grid; sampling variability excludes material-model and unsampled-grid error.')
