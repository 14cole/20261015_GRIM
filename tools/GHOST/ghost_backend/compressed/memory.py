"""compressed RAM forecasts, separate from retained-payload limits."""
import math
import time
import numpy as np
from ghost_backend.execution.metrics import timed_stage

MIB = 1024**2
GIB = 1024**3
SAMPLES_PER_BAND = 4
SMALL_DENSE_DOFS = 8192


def inverse_storage(n):
    """Maximum retained numeric tree and construction live set, in bytes."""
    cache = {}
    def visit(size):
        if size in cache:
            return cache[size]
        if size <= 512:

            final = 16*size*size + 8*size
            result = final, 2*final + 16*MIB
        else:
            left, right = size//2, size-size//2
            rank = min(256, left//2)
            lf, lp = visit(left)
            rf, rp = visit(right)
            uv = 32*size*rank
            small = 16*(2*rank)**2 + 8*(2*rank)
            final = lf + rf + uv + small


            peak = max(64*size*rank + 32*MIB,
                       uv + lp, uv + lf + rp,
                       final + 16*size*rank + 80*max(left,right)*rank + small)
            result = final, peak
        cache[size] = result
        return result
    final, peak = visit(max(1, int(n)))
    return final + 40*n, peak + 40*n


def sample_operator(oracle, coordinates, tile=512, checkpoint=None, pilot_identity=None):
    """Sample spatial-separation bands without enumerating a quadratic grid."""
    from ghost_backend.compressed.operator import StreamedOperator
    checkpoint = checkpoint or (lambda: None)
    shell = StreamedOperator(oracle, coordinates, tile=tile, assemble=False,
        budget=max(16*MIB, 128*oracle.n), checkpoint=checkpoint)
    lengths = np.asarray([len(ids) for ids in shell.groups], dtype=np.int64)
    groups = len(lengths)
    diagonal = 16*int(np.dot(lengths, lengths))
    expected = allowance = float(shell.bytes + diagonal)
    bands = []
    rng = np.random.RandomState(1904)
    low = 1
    while low < groups:
        checkpoint()
        high = min(2*low, groups)
        gaps = np.arange(low, high, dtype=np.int64)
        cumulative = np.cumsum(groups-gaps)
        population = 2*int(cumulative[-1])

        chosen = set()
        for upper in range(population-min(SAMPLES_PER_BAND,population), population):
            pick = int(rng.randint(0, upper+1, dtype=np.int64))
            chosen.add(upper if pick in chosen else pick)
        ratios = []
        for picked in sorted(chosen):
            checkpoint()
            reverse = picked >= cumulative[-1]
            position = int(picked % cumulative[-1])
            index = int(np.searchsorted(cumulative, position, side='right'))
            gap = int(gaps[index])
            i = position-int(cumulative[index-1] if index else 0)
            j = i+gap
            if reverse:
                i, j = j, i
            rows, cols = shell.groups[i], shell.groups[j]
            if hasattr(oracle, 'prepare_columns'):
                oracle.prepare_columns(cols)
            raw, tail = oracle.get_with_error(rows, cols)
            if (raw.shape != (len(rows),len(cols)) or tail.shape != raw.shape or
                    not np.all(np.isfinite(raw)) or not np.all(np.isfinite(tail)) or np.any(tail<0)):
                raise ValueError('Invalid coefficient tile in compressed memory forecast.')
            raw_bytes = raw.nbytes
            compressed = shell.compress_tile(i, j, raw, tail)
            payload = compressed[3]
            ratios.append(sum(a.nbytes for a in payload if a is not None)/float(raw_bytes))
            from ghost_backend.compressed.pilots import save
            save(pilot_identity, rows, cols, compressed)
            raw = tail = payload = compressed = None


        prefix = np.r_[0, np.cumsum(lengths)]
        indices = np.arange(groups)
        starts = np.minimum(groups, indices+low)
        stops = np.minimum(groups, indices+high)
        entries = 2*int(np.dot(lengths, prefix[stops]-prefix[starts]))
        mean = float(np.mean(ratios))
        sem = float(np.std(ratios, ddof=1)/math.sqrt(len(ratios))) if len(ratios)>1 else 0.


        upper = min(1., max(mean+2*sem, 1.15*mean, 2./max(1,int(lengths.max()))))
        if len(ratios) == population:
            upper = mean
        expected += 16*entries*mean
        allowance += 16*entries*upper
        bands.append(dict(first_gap=low, last_gap=high-1, samples=len(ratios),
                          mean_ratio=mean, allowance_ratio=upper))
        low = high
    return dict(method='sampled_spatial_tiles', operator_bytes=int(math.ceil(expected)),
        operator_allowance_bytes=int(math.ceil(allowance)), samples=sum(b['samples'] for b in bands),
        tile=tile, groups=groups, bands=bands, sampled=True)


def hmatrix_storage(mesh, infos, formulation, system_dofs, layout=None):
    """Geometry-only hierarchical forecast for a system above its threshold, or None.

    Thin layers with a magnetic contrast (twice the nodes) keep the tile
    operator (``hmatrix.unsupported``) and so its pricing.
    """
    from ghost_backend.compressed import hmatrix
    from ghost_backend.compressed.runtime import coordinates
    if not system_dofs or not hmatrix.selected(system_dofs) or hmatrix.configuration_unsupported() is not None:
        return None
    if formulation == 'thin_dielectric_layer' and system_dofs != len(mesh.nodes):
        return None
    if layout is not None:
        from ghost_backend.twod.formulations.regions import dof_coordinates
        xy = dof_coordinates(mesh, layout)
    else:
        xy = coordinates(mesh, system_dofs)
    if len(xy) != system_dofs:
        return None
    waves = [complex(value) for info in infos for value in (info.k_plus, info.k_minus) if value is not None]
    wavenumber = max(abs(value) for value in waves) if waves else 0.
    return hmatrix.forecast_storage(xy, hmatrix.support_radius(mesh, xy), wavenumber)


@timed_stage('compressed_memory_sampling')
def geometry_storage(mesh, infos, pol, kind, k0, layer=None, dofs=None):
    from ghost_backend.compressed.coefficients import NativeOracle
    from ghost_backend.compressed.regional_coefficients import PreparedOracle
    from ghost_backend.compressed.runtime import coordinates, checkpoint
    from ghost_backend.twod.assembly.session import current_session
    from ghost_backend.twod.formulations.regions import dof_coordinates
    started = time.perf_counter()
    from ghost_backend.compressed.pilots import assembled_partner, key as pilot_key
    previous = assembled_partner(mesh, infos, pol, kind, dofs)
    if previous is not None:
        return previous
    if dofs is not None and dofs <= SMALL_DENSE_DOFS:
        size = 16*dofs**2 + 96*dofs
        return dict(method='small_dense_ceiling', operator_bytes=size,
            operator_allowance_bytes=size, samples=0, sampled=False, seconds=0.)
    session = current_session()
    key = pilot_key(mesh, [] if kind=='thin_dielectric_layer' else infos or [],
                    pol, kind, k0, layer)
    cached = getattr(session, 'memory_storage', {}) if session is not None else {}
    if key in cached:
        return dict(cached[key])
    checkpoint()
    if kind == 'multi_region':
        oracle = PreparedOracle(mesh, infos, pol, cut=32)
        xy = dof_coordinates(mesh, oracle.layout)
    else:
        native_kind = {'te_robin':'robin', 'robin':'robin', 'single_dielectric':'dielectric',
                       'sheet':'sheet', 'mixed_sheet_pec':'sheet', 'thin_dielectric_layer':'thin'}[kind]
        oracle = NativeOracle(mesh, infos, pol, k0, native_kind, layer=layer)
        xy = coordinates(mesh, oracle.n)
    if oracle.n <= SMALL_DENSE_DOFS:

        size = 16*oracle.n**2 + 96*oracle.n
        result = dict(method='small_dense_ceiling', operator_bytes=size,
            operator_allowance_bytes=size, samples=0, sampled=False)
    else:
        result = sample_operator(oracle, xy, min(512,getattr(oracle,'maximum_tile',512)), checkpoint,
                                 pilot_key(mesh, infos, pol, kind, k0, layer))
    result['seconds'] = time.perf_counter()-started
    if session is not None:
        if len(cached) >= 8:
            cached.clear()
        cached[key] = dict(result)
        session.memory_storage = cached
    return result


def forecast(n, d, count, batch, threads, storage_limit, resources=None, safety=1., floor_gb=0.):
    """One polarization is resident; its spooled partner is a disk allowance.

    Safety applies only to sampled operator storage. Structural inverse limits
    and phase workspaces are not multiplied again. floor_gb is a minimum total
    process reservation, not another copy of interpreter/geometry overhead.
    """
    if not math.isfinite(safety) or safety < 1 or not math.isfinite(floor_gb) or floor_gb < 0:
        raise ValueError('Compressed memory safety must be finite and >=1; floor must be finite and >=0.')
    resources = resources or {}
    model = resources.get('hmatrix_storage')
    from ghost_backend.compressed import hmatrix
    if model is not None and hmatrix.selected(d):
        return hmatrix_forecast(n, d, count, batch, threads, resources, model, floor_gb)
    sample = resources.get('compressed_storage')
    if sample is None:
        size = min(storage_limit, 16*d*d+96*d)
        sample = dict(method='dimensions_only_ceiling', operator_bytes=size,
            operator_allowance_bytes=size, sampled=False, samples=0)
    expected = sample['operator_bytes']
    operator = max(sample['operator_allowance_bytes'], int(math.ceil(expected*safety))) if sample['sampled'] else expected
    inverse, construction = inverse_storage(d)

    resident = min(storage_limit, operator+inverse)
    inverse_growth = max(0, construction-inverse)
    tile = min(512,max(1,(8*MIB)//max(16*n,1))) if resources.get('formulation')=='thin_dielectric_layer' else 512
    groups = int(sample.get('groups',2**int(math.ceil(math.log(max(1.,float(d)/tile),2)))))


    tile_metadata = 1280*groups**2
    from ghost_backend.twod.polynomial_quadrature import MOMENT_CACHE_BYTES
    moment_cache = resources.get('moment_cache_bytes', MOMENT_CACHE_BYTES if resources.get('basis_width', 2) > 2 else 0)
    from ghost_backend.compressed.pilots import CACHE_BYTES as PILOT_BYTES
    from ghost_backend.compressed.worker_pool import forecast_bytes
    worker_bytes = forecast_bytes(d, threads, groups=groups)
    from ghost_backend.twod.assembly.polynomial_pair import retained_bytes as polynomial_retained_bytes
    polynomial_bytes=polynomial_retained_bytes('compressed')
    overhead = 128*MIB + 6144*n + count*4096 + tile_metadata + moment_cache + PILOT_BYTES + worker_bytes + polynomial_bytes

    from ghost_backend.twod.operators import _ASSEMBLY_TILE
    from ghost_backend.execution.options import option
    _ASSEMBLY_TILE = option('assembly_tile', _ASSEMBLY_TILE)
    kernel_work = 32*MIB
    if _ASSEMBLY_TILE > 0:


        kernel_work = max(kernel_work,312*min(n,_ASSEMBLY_TILE)**2)
    # Reciprocal traversal keeps at most two directional tile destinations.
    assembly_work = 96*MIB + max(1,threads)*kernel_work + 16*512*min(n,1024)


    from ghost_backend.twod.assembly.kernels import PROJECTION_CACHE_BYTES
    solve_work = 32*MIB + 16*12*d*batch
    phases = dict(assembly=min(storage_limit,operator)+assembly_work+overhead,
                  factorization=resident+inverse_growth+overhead,
                  solve=resident+solve_work+PROJECTION_CACHE_BYTES+overhead)
    peak = max(float(floor_gb)*GIB, max(phases.values()))
    return dict(model='compressed_phase_v1', peak_bytes=int(math.ceil(peak)),
        phase_bytes={k:int(v) for k,v in phases.items()}, operator_bytes=int(expected),
        operator_allowance_bytes=int(operator), inverse_ceiling_bytes=int(inverse),
        process_geometry_bytes=int(overhead), rhs_workspace_bytes=int(solve_work),
        projection_cache_bytes=PROJECTION_CACHE_BYTES,
        retained_worker_allowance_bytes=worker_bytes,
        retained_polynomial_operator_bytes=polynomial_bytes,
        moment_cache_bytes=moment_cache,
        assembly_workspace_bytes=int(assembly_work),
        tile_metadata_bytes=int(tile_metadata),
        storage_limit_bytes=int(storage_limit), temporary_disk_bytes=int(sample['operator_allowance_bytes']),
        temporary_disk_semantics='one partner estimate; exact payload depends on its material equations',
        storage_method=sample['method'], sampled=sample['sampled'], samples=sample['samples'],
        safety=float(safety), floor_gb=float(floor_gb), forecast_is_hard_limit=False)


def hmatrix_forecast(n, d, count, batch, threads, resources, model, floor_gb=0.):
    """Phases of a hierarchical (compressed.hmatrix) solve; both polarizations' operators resident.

    The TE assembly builds the TM operator in the same traversal and keeps it
    for the TM solve when both fit the hierarchical storage cap; otherwise
    each polarization is assembled alone. The inverse is the model's.
    """
    from ghost_backend.compressed.runtime import hmatrix_storage_budget
    limit = hmatrix_storage_budget()
    operator, inverse, construction = model['operator_bytes'], model['inverse_bytes'], model['inverse_peak_bytes']
    resident = 2*operator if 2*operator + inverse <= limit else operator
    from ghost_backend.twod.polynomial_quadrature import MOMENT_CACHE_BYTES
    moment_cache = resources.get('moment_cache_bytes', MOMENT_CACHE_BYTES if resources.get('basis_width', 2) > 2 else 0)
    overhead = 128*MIB + 6144*n + count*4096 + moment_cache
    from ghost_backend.twod.assembly.kernels import PROJECTION_CACHE_BYTES
    solve_work = 32*MIB + 16*12*d*batch
    phases = dict(assembly=resident + (resident//operator)*model['assembly_workspace_bytes'] + overhead,
                  factorization=resident + construction + overhead,
                  solve=resident + inverse + solve_work + PROJECTION_CACHE_BYTES + overhead)
    peak = max(float(floor_gb)*GIB, max(phases.values()))
    return dict(model='compressed_hmatrix_v1', peak_bytes=int(math.ceil(peak)),
        phase_bytes={k: int(v) for k, v in phases.items()}, operator_bytes=int(operator),
        operator_allowance_bytes=int(operator), resident_operator_bytes=int(resident),
        inverse_ceiling_bytes=int(inverse), inverse_construction_bytes=int(construction),
        process_geometry_bytes=int(overhead), rhs_workspace_bytes=int(solve_work),
        projection_cache_bytes=PROJECTION_CACHE_BYTES, moment_cache_bytes=moment_cache,
        assembly_workspace_bytes=int(model['assembly_workspace_bytes']),
        storage_limit_bytes=int(limit), storage_fits=bool(operator + inverse <= limit),
        temporary_disk_bytes=0, storage_method=model['method'], sampled=False, samples=0,
        coefficients=int(model['coefficients']), inverse_rank=int(model['inverse_rank']),
        safety=1., floor_gb=float(floor_gb), forecast_is_hard_limit=False)
