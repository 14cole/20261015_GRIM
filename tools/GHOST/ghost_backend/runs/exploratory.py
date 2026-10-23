"""Select one of certification's initial meshes without running a comparison."""
from copy import deepcopy
import math


MESH_MODES = ('certification_base', 'certification_fine', 'geometry')


def select_exploratory_mesh(snapshot, frequency_ghz, *, mesh_mode='certification_base',
                            geometry_units='inches', material_base_dir=None,
                            factorization='adaptive', ram_budget_gib=None):
    """Return an independent geometry, execution options, and selection record.

    Base/fine refer to the initial candidates, not an accepted or subsequently
    refined certification result. Eligible inputs use the same coarsened P2/P3
    pair as certification. Other inputs retain its linear base/fine pair.
    Neither selection solves a system, compares fields, or certifies accuracy.
    """
    from ghost_backend.execution.options import efficient_defaults, validate_options
    from ghost_backend.runs.quality import validate_mesh_convergence_policy
    from ghost_backend.twod.adaptive_geometry import candidate_meshes, eligible_snapshot
    from ghost_backend.twod.preparation import prepare_geometry

    if mesh_mode not in MESH_MODES:
        raise ValueError('MESH_MODE must be one of: ' + ', '.join(MESH_MODES))
    frequency = float(frequency_ghz)
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError('Frequency must be finite and positive in GHz.')
    options = validate_options(dict(efficient_defaults(), factorization=factorization,
                                    mesh_strategy='global', basis_order=1,
                                    ram_budget_gib=ram_budget_gib))
    candidate = deepcopy(snapshot)
    record = dict(requested=mesh_mode, candidate='geometry', basis_order=1,
                  mesh_strategy='global', coarsening_factor=1.,
                  certification_performed=False)
    if mesh_mode != 'geometry':
        _, _, materials, scale = prepare_geometry(
            snapshot, material_base_dir, geometry_units)
        factor = validate_mesh_convergence_policy()['fine_factor']
        candidates = candidate_meshes(snapshot, materials, factor, True,
                                      [frequency], scale)
        phase = 'base' if mesh_mode == 'certification_base' else 'fine'
        _, candidate, degree = next(item for item in candidates if item[0] == phase)
        candidate = deepcopy(candidate)
        polynomial = '_2d_hp_coarsening' in candidate
        strategy = 'local' if polynomial else 'global'
        reason = ('Initial higher-order candidate used by mesh certification.' if polynomial else
                  eligible_snapshot(snapshot, materials, [frequency], scale)[1])
        options.update(mesh_strategy=strategy, basis_order=degree)
        record.update(candidate=phase, basis_order=degree, mesh_strategy=strategy,
                      coarsening_factor=float(candidate.get('_2d_hp_coarsening', 1.)),
                      fine_factor=factor, selection_reason=reason,
                      higher_order_candidate=polynomial)
    return candidate, options, record
