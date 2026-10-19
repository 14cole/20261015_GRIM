"""Combined regional potentials on closed, continuous PEC/transmission contours.

Each regional field uses S*phi + gamma*D*phi. gamma changes sign with the
region side so the auxiliary impedance condition has consistent orientation.

The exterior (illuminated) region may also close as a union of open arcs that
meet at split junction nodes (partial coatings: TYPE 2+3+4 junctions). Its D/W
density is then continuous: at each junction the slave end's D/W columns route
to the master end's coefficient, while S/K' keep both values. The element-wise
Maue form of W is exact for that continuous density except on the two
endpoint test rows of each junction, which receive the endpoint term
``-eps*F(p)`` (``_junction_flux_entries``). Bounded regions bounded by open
interfaces keep the single-layer representation: their unbounded complement
has no interior eigenvalues, and combining them raised the condition estimate
without an accuracy gain. Polynomial meshes use the same endpoint routing,
with the full polynomial density derivative in the junction flux term.

A closed one-sided (PEC/impedance) contour may carry a spatially varying Robin
law, including PEC zones, several impedance zones and tapers. The boundary
equation is ``dn(u) + alpha(x)*u`` (or ``u`` on TM PEC nodes) applied to the
same closed-contour density, so W is never weighted by alpha and its Maue form
keeps no endpoint terms; alpha only weights S and the D trace. Uniqueness of
``S + gamma*D`` follows from the interior impedance problem with ``1/gamma`` and
does not involve alpha. Rows are therefore routed per node exactly as the base
S/K' assembly routes them.
"""
import numpy as np
from ghost_backend.twod.constants import EPS


def _union_junctions(mesh, interfaces, rid, indices):
    """Junctions of a region whose interfaces close as a union of open arcs.

    Junction nodes are split per interface signature, so the region's arcs meet
    at coincident but distinct mesh nodes. Returns ``None`` unless every
    geometric endpoint cluster (mesh snap tolerance) holds exactly two element
    ends with one arc ending and the other starting in the region's orientation
    (element tangent times the side sign). Endpoint nodes are the first two
    local nodes for both linear and polynomial elements.
    Otherwise the region keeps the former single-layer representation (open
    strips, cracks, touching loops).
    """
    from scipy.spatial import cKDTree
    from ghost_backend.twod.operators import NODE_SNAP_TOLERANCE
    ends = []
    for mi in indices:
        sign = 1 if interfaces[mi]['r_m'] == rid else -1
        for ei in interfaces[mi]['eids']:
            element = mesh.elements[ei]
            for position in (0, 1):
                # +1 where the region's oriented contour arrives at this end;
                # last: +1 where the element's own direction arrives.
                ends.append((mi, int(element.node_ids[position]), int(ei),
                             1 if (sign > 0) == (position == 1) else -1,
                             1 if position == 1 else -1))
    if not ends:
        return None
    points = np.array([mesh.nodes[end[1]].xy for end in ends], float)
    parent = list(range(len(ends)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in sorted(cKDTree(points).query_pairs(4.0*NODE_SNAP_TOLERANCE)):
        parent[root(i)] = root(j)
    clusters = {}
    for i in range(len(ends)):
        clusters.setdefault(root(i), []).append(ends[i])
    junctions = []
    for members in clusters.values():
        if len(members) != 2 or members[0][3] + members[1][3] != 0:
            return None
        a, b = sorted(members)
        if (a[0], a[1]) == (b[0], b[1]):
            continue
        junctions.append(dict(point=np.array(mesh.nodes[a[1]].xy, float), ends=(a, b),
                              master=(a[0], a[1]), slave=(b[0], b[1])))
    return sorted(junctions, key=lambda j: (j['master'], j['slave']))


def _node_dof(layout, mi, rid, node):
    interface = layout['ifaces'][mi]
    offset, _ = layout['dof_map'][mi, 'minus' if interface['r_m'] == rid else 'plus']
    return offset + int(np.searchsorted(interface['nodes'], node))


def couplings(mesh, layout):
    if 'combined_couplings' in layout:
        return layout['combined_couplings']
    interfaces = layout['ifaces']
    closed = []
    for interface in interfaces:
        counts = {}
        for ei in interface['eids']:
            for node in mesh.elements[ei].node_ids[:2]:
                counts[node] = counts.get(node, 0) + 1
        closed.append(bool(counts) and all(count == 2 for count in counts.values()))
    result = {}
    for rid, indices in layout['region_ifaces'].items():
        if not all(closed[mi] for mi in indices):
            # Only the unbounded (illuminated) region is combined here: its
            # complement is the bounded outline whose Dirichlet eigenvalues
            # were the observed resonances. A simply connected bounded region
            # has an unbounded complement and no such eigenvalues; bounded
            # regions with holes formed by open arcs remain unprotected, as
            # before. Combined on the bounded coating of the partial-coating
            # case, the condition estimate reached 1.5e6 at ka=40 with no
            # accuracy gain.
            if not layout['region_props'][rid]['has_incident']:
                continue
            junctions = _union_junctions(mesh, interfaces, rid, indices)
            if not junctions:
                continue
            layout.setdefault('combined_junctions', {})[rid] = junctions
        k = layout['region_props'][rid]['k']
        for mi in indices:
            result[mi, rid] = (1 if interfaces[mi]['r_m'] == rid else -1)/(1j*k)
    layout['combined_couplings'] = result
    return result


def _source_columns(layout, mi, rid, col_map, n):
    """Mesh source node -> system column of the region's D/W density.

    On a union-closed region the D/W density is continuous: the slave end of
    each junction carries its master's coefficient. The single-layer density
    keeps both junction values.
    """
    interface = layout['ifaces'][mi]
    offset, count = layout['dof_map'][mi, 'minus' if interface['r_m'] == rid else 'plus']
    cc = np.full(n, -1, int)
    cc[interface['nodes']] = col_map[offset:offset+count]
    for junction in layout.get('combined_junctions', {}).get(rid, ()):
        slave_interface, slave_node = junction['slave']
        if slave_interface == mi:
            cc[slave_node] = col_map[_node_dof(layout, junction['master'][0], rid, junction['master'][1])]
    return cc


_JUNCTION_GAUSS = np.polynomial.legendre.leggauss(32)


def _segment_green(p0, p1, point, k, derivative_degree=None):
    """int_e G_k(point, y) ds_y on a straight panel.

    A log singularity at a panel end is removed by t = u**4; other panels use
    composite 32-point rules subdivided by their distance to the point.
    With ``derivative_degree``, return int G*dphi/ds ds for every local
    polynomial basis function; the length cancels between ds and dphi/ds.
    """
    from ghost_backend.twod.operators import _green_2d_array, NODE_SNAP_TOLERANCE
    x, w = (_JUNCTION_GAUSS[0] + 1)/2, _JUNCTION_GAUSS[1]/2
    p0, p1, point = np.asarray(p0, float), np.asarray(p1, float), np.asarray(point, float)
    length = float(np.linalg.norm(p1 - p0))
    at_start = np.linalg.norm(p0 - point) <= 4*NODE_SNAP_TOLERANCE
    at_end = np.linalg.norm(p1 - point) <= 4*NODE_SNAP_TOLERANCE
    if at_start or at_end:
        t = x**4 if at_start else 1 - x**4
        weights = w*4*x**3
    else:
        d = p1 - p0
        s = np.clip(np.dot(point - p0, d)/max(float(np.dot(d, d)), 1e-300), 0, 1)
        distance = float(np.linalg.norm(p0 + s*d - point))
        pieces = int(np.clip(np.ceil(2*length/max(distance, 1e-300)), 1, 64))
        t = ((np.arange(pieces)[:, None] + x[None, :])/pieces).ravel()
        weights = np.tile(w/pieces, pieces)
    y = p0[None, :] + t[:, None]*(p1 - p0)[None, :]
    weighted = weights*_green_2d_array(k, np.linalg.norm(y - point[None, :], axis=1))
    if derivative_degree is not None:
        from ghost_backend.twod.basis import values
        return weighted @ values(t, derivative_degree, derivative=True)
    return complex(np.sum(weighted)*length)


def _junction_flux_entries(mesh, layout, mi, rid, cc, flux_map, flux_weight):
    """Endpoint terms of hypersingular rows tested by half-hats at junctions.

    The element-wise Maue form of W is exact for a continuous density and a
    continuous test function on a closed contour. A junction row is tested by
    an endpoint basis function that ends (eps=+1) or starts (eps=-1) at p in
    its element's direction; integrating its tangential derivative by parts
    leaves ``W = W_Maue - eps*F(p)``, ``F(p) = int G(p,y) dpsi/ds dy``. Each
    source panel enters in its own direction and normal, exactly as its Maue
    block does: the side sign is part of gamma in the row weight, so the sum
    over the region's arcs is the closed-contour term. Returns (rows, columns,
    values) of source interface ``mi``; the continuous density psi already
    routes through ``cc``.
    """
    junctions = layout.get('combined_junctions', {}).get(rid)
    if not junctions:
        return None
    interface = layout['ifaces'][mi]
    k = layout['region_props'][rid]['k']
    cache = layout.setdefault('combined_junction_green', {})
    elements = [mesh.elements[ei] for ei in interface['eids']]
    degree = len(elements[0].node_ids)-1
    n0 = np.array([e.node_ids[0] for e in elements], int)
    n1 = np.array([e.node_ids[1] for e in elements], int)
    rows, columns, values = [], [], []
    for index, junction in enumerate(junctions):
        tests = [(end[1], end[4]) for end in junction['ends'] if flux_map[end[1]] >= 0]
        if not tests:
            continue
        key = rid, mi, index
        if key not in cache:
            if degree == 1:
                cache[key] = np.array([_segment_green(e.p0, e.p1, junction['point'], k)/e.length
                                            for e in elements])
            else:
                cache[key] = np.array([_segment_green(e.p0, e.p1, junction['point'], k,
                                                     derivative_degree=degree) for e in elements])
        coefficient = cache[key]
        for node, eps in tests:
            scale = -eps*flux_weight[node]
            if degree > 1:
                nodes = np.asarray([e.node_ids for e in elements]).ravel()
                rows.append(np.full(len(nodes), flux_map[node]))
                columns.append(cc[nodes])
                values.append(scale*coefficient.ravel())
                continue
            for nodes, factor in ((n1, 1.), (n0, -1.)):
                rows.append(np.full(len(nodes), flux_map[node]))
                columns.append(cc[nodes])
                values.append(factor*scale*coefficient)
    if not rows:
        return None
    rows, columns, values = np.concatenate(rows), np.concatenate(columns), np.concatenate(values)
    keep = columns >= 0
    return rows[keep], columns[keep], values[keep]


def _route_conductor_rows(layout, target, dest, gamma, trace_map, trace_weight, flux_map, flux_weight):
    """Route D and W per node on a one-sided interface.

    TM nodes without impedance carry the Dirichlet trace equation and receive
    ``gamma*D``. Every other node carries the flux (Neumann/Robin) equation and
    receives ``-gamma*W`` plus ``gamma*alpha*D``. The PEC test is the base
    assembly's own (nodal alpha within EPS), so both parts of one row always
    describe the same equation. Uniform contours reduce to the former
    whole-interface selection.
    """
    nodes, dest = np.asarray(target['nodes']), np.asarray(dest)
    alpha = np.asarray(target['robin_alpha'])
    impedance = np.abs(alpha) > EPS
    dirichlet = ~impedance if layout['polarization'] == 'TM' else np.zeros(len(nodes), bool)
    trace_map[nodes[dirichlet]], trace_weight[nodes[dirichlet]] = dest[dirichlet], gamma
    flux = ~dirichlet
    flux_map[nodes[flux]], flux_weight[nodes[flux]] = dest[flux], -gamma
    robin = flux & impedance
    trace_map[nodes[robin]], trace_weight[nodes[robin]] = dest[robin], gamma*alpha[robin]
    return nodes[robin], alpha[robin]


def _trace_jump_mass(mesh, interface, robin_nodes, robin_alpha):
    """Mass rows of the +-1/2 D-trace jump for one source interface.

    Dirichlet and transmission rows use the plain mass. A Robin row is later
    scaled by ``gamma*alpha_node``; its jump must instead be the element-weighted
    ``sum_e alpha_e*M_e`` that the base assembly already uses for ``alpha*S``,
    so those rows hold ``M_alpha/alpha_node``. With a uniform law both forms
    coincide; with a taper or zones the nodal form alone left a second-order
    error proportional to the steepness of the law.
    """
    from scipy.sparse import diags
    from ghost_backend.twod.assembly.mass import sparse_mass
    mass = sparse_mass(mesh, interface['mask'])
    if not len(robin_nodes):
        return mass
    weighted = sparse_mass(mesh, np.asarray(interface['robin_alpha_elements'])*interface['mask'])
    keep, scale = np.ones(len(mesh.nodes), complex), np.zeros(len(mesh.nodes), complex)
    keep[robin_nodes], scale[robin_nodes] = 0., 1./np.asarray(robin_alpha)
    return (diags(keep) @ mass + diags(scale) @ weighted).tocsr()


def add_corrections(matrix, mesh, layout, obs_order=8, src_order=8, rows=None, columns=None,
                    geometry=None):
    """Add D traces and -W fluxes in bounded native coefficient queries."""
    import ghost_backend.twod.solver as rcs
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.assembly.scatter import SystemScatter
    n, size = len(mesh.nodes), layout['n_dof']
    rows = np.arange(size) if rows is None else np.asarray(rows)
    columns = np.arange(size) if columns is None else np.asarray(columns)
    row_map, col_map = np.full(size,-1,int), np.full(size,-1,int)
    row_map[rows], col_map[columns] = np.arange(len(rows)), np.arange(len(columns))
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    for (mi, rid), gamma in couplings(mesh, layout).items():
        interface = interfaces[mi]
        side = 'minus' if interface['r_m'] == rid else 'plus'
        cc = _source_columns(layout, mi, rid, col_map, n)
        source_nodes = np.flatnonzero(cc >= 0)
        if not len(source_nodes):
            continue
        trace_map, flux_map = np.full(n,-1,int), np.full(n,-1,int)
        trace_weight, flux_weight = np.zeros(n,complex), np.zeros(n,complex)
        own_robin = (np.empty(0, int), np.empty(0, complex))
        for observer in layout['region_ifaces'][rid]:
            target = interfaces[observer]
            rm, rp, nodes = target['r_m'], target['r_p'], target['nodes']
            if rm < 0 or rp < 0:
                start, width = dofs[observer,'plus' if rm < 0 else 'minus']
                routed = _route_conductor_rows(layout, target, row_map[start:start+width], gamma,
                                               trace_map, trace_weight, flux_map, flux_weight)
                if observer == mi:
                    own_robin = routed
            else:
                flux, width = dofs[observer,'minus']
                trace, _ = dofs[observer,'plus']
                trace_map[nodes], flux_map[nodes] = row_map[trace:trace+width], row_map[flux:flux+width]
                trace_weight[nodes] = gamma * (1 if rid == rm else -1)
                flux_weight[nodes] = -gamma * (1 if rid == rm else -_inverse_beta(layout,target,layout['polarization']))
        k = layout['region_props'][rid]['k']
        endpoint = _junction_flux_entries(mesh, layout, mi, rid, cc, flux_map, flux_weight)
        if endpoint is not None:
            np.add.at(matrix, (endpoint[0], endpoint[1]), endpoint[2])
        mass = _trace_jump_mass(mesh, interface, *own_robin)
        for kind, mapping, weight in (('D', trace_map, trace_weight), ('W', flux_map, flux_weight)):
            active_rows = np.flatnonzero(mapping >= 0)
            output = SystemScatter(matrix,n,active_rows,source_nodes,[(mapping,cc,weight)])
            for start in range(0,len(active_rows),64):
                rr = active_rows[start:start+64]
                if kind == 'D':
                    _, operator = rcs._assemble_linear_operator_matrices_multi(mesh,k,False,
                        [interface['mask']], obs_order=obs_order,src_order=src_order,
                        compute_single_layer=False, output_node_ids_many=[([],source_nodes)],
                        double_layer_output_node_ids_many=[(rr,source_nodes)],
                        prepared_geometry=geometry)[0]
                    values = (operator.values.copy() if hasattr(operator,'values') else
                              operator[np.ix_(rr,source_nodes)].copy())
                    values += (.5 if side == 'minus' else -.5)*mass[rr,:][:,source_nodes].toarray()
                else:
                    operator = rcs._assemble_linear_hypersingular_matrix(mesh,k,
                        obs_order=obs_order,src_order=src_order,source_element_mask=interface['mask'],
                        output_node_ids=(rr,source_nodes),prepared_geometry=geometry)
                    values = operator.values
                output.scatter_add(rr[:,None],source_nodes[None,:],values)


def fused_outputs(matrix, mesh, layout, rows=None, columns=None):
    """D/W routes keyed by (wavenumber, source interface), plus trace jumps.

    The outputs write directly to the owned dense matrix or compressed tile.
    No rectangular D/W matrices or row-strip queries are retained.
    """
    from ghost_backend.twod.formulations.regions import _inverse_beta
    from ghost_backend.twod.assembly.scatter import SystemScatter
    n, size = len(mesh.nodes), layout['n_dof']
    rows = np.arange(size) if rows is None else np.asarray(rows)
    columns = np.arange(size) if columns is None else np.asarray(columns)
    row_map, col_map = np.full(size, -1, int), np.full(size, -1, int)
    row_map[rows], col_map[columns] = np.arange(len(rows)), np.arange(len(columns))
    interfaces, dofs = layout['ifaces'], layout['dof_map']
    grouped = {}
    masses = layout.setdefault('combined_sparse_mass', {})
    for (mi, rid), gamma in couplings(mesh, layout).items():
        interface = interfaces[mi]
        side = 'minus' if interface['r_m'] == rid else 'plus'
        cc = _source_columns(layout, mi, rid, col_map, n)
        source_nodes = np.flatnonzero(cc >= 0)
        if not len(source_nodes):
            continue
        trace_map, flux_map = np.full(n, -1, int), np.full(n, -1, int)
        trace_weight, flux_weight = np.zeros(n, complex), np.zeros(n, complex)
        own_robin = (np.empty(0, int), np.empty(0, complex))
        for observer in layout['region_ifaces'][rid]:
            target = interfaces[observer]
            rm, rp, nodes = target['r_m'], target['r_p'], target['nodes']
            if rm < 0 or rp < 0:
                start, width = dofs[observer, 'plus' if rm < 0 else 'minus']
                routed = _route_conductor_rows(layout, target, row_map[start:start+width], gamma,
                                               trace_map, trace_weight, flux_map, flux_weight)
                if observer == mi:
                    own_robin = routed
            else:
                flux, width = dofs[observer, 'minus']
                trace, _ = dofs[observer, 'plus']
                trace_map[nodes], flux_map[nodes] = row_map[trace:trace+width], row_map[flux:flux+width]
                trace_weight[nodes] = gamma*(1 if rid == rm else -1)
                flux_weight[nodes] = -gamma*(1 if rid == rm else -_inverse_beta(layout, target, layout['polarization']))
        k = layout['region_props'][rid]['k']
        endpoint = _junction_flux_entries(mesh, layout, mi, rid, cc, flux_map, flux_weight)
        if endpoint is not None:
            np.add.at(matrix, (endpoint[0], endpoint[1]), endpoint[2])
        routes = grouped.setdefault((k, mi), [[], []])
        for index, (mapping, weight) in enumerate(((trace_map, trace_weight), (flux_map, flux_weight))):
            if np.any(mapping >= 0):
                routes[index].append((mapping, cc, weight))
        if np.any(trace_map >= 0):
            # Robin rows follow the layout's own law, never the queried subset.
            if mi not in masses:
                masses[mi] = _trace_jump_mass(mesh, interface, *own_robin).tocoo()
            mass = masses[mi]
            output = SystemScatter(matrix, n, np.flatnonzero(trace_map >= 0), source_nodes,
                                   [(trace_map, cc, trace_weight)])
            output.scatter_add(mass.row, mass.col, (.5 if side == 'minus' else -.5)*mass.data)
    result = {}
    for key, pair in grouped.items():
        outputs = []
        for routes in pair:
            rr = np.unique(np.concatenate([np.flatnonzero(r >= 0) for r, c, w in routes])) if routes else []
            cc = np.unique(np.concatenate([np.flatnonzero(c >= 0) for r, c, w in routes])) if routes else []
            outputs.append(SystemScatter(matrix, n, rr, cc, routes))
        result[key] = tuple(outputs)
    return result


def exterior_double_density(mesh, layout):
    coefficients = {key: gamma for key, gamma in couplings(mesh,layout).items()
                    if layout['region_props'][key[1]]['has_incident']}
    if not coefficients:
        return None
    junctions = layout.get('combined_junctions', {})
    def density(solution):
        result = np.zeros((len(mesh.nodes),solution.shape[1]),complex)
        for (mi,rid), gamma in coefficients.items():
            interface = layout['ifaces'][mi]
            offset,count = layout['dof_map'][mi,'minus' if interface['r_m']==rid else 'plus']
            result[interface['nodes']] += gamma*solution[offset:offset+count]
            for junction in junctions.get(rid, ()):
                if junction['slave'][0] == mi:
                    # The continuous D density carries the master value.
                    node = junction['slave'][1]
                    master = _node_dof(layout, junction['master'][0], rid, junction['master'][1])
                    result[node] += gamma*(solution[master] - solution[offset + int(
                        np.searchsorted(interface['nodes'], node))])
        return result
    return density


# Union-closed exterior systems mix the exterior's W rows (~gamma/h) with the
# single-layer densities of the bounded regions (S, K' and the beta-weighted
# flux jump). The single-pass row/column max equilibration of the condition
# estimate leaves those density blocks two to three orders under-scaled: on the
# 3 GHz airfoil (8608 unknowns) it reported 1.5e6 against 2.8e4 for the former
# equation although the fields agree to 2e-5. A Sinkhorn balance of the
# equilibrated matrix's block sums (one scalar per DOF block, i.e. per
# interface side) removes that artificial imbalance: 0.5 GHz airfoil 1-norm
# condition 2.8e3 (former equation 3.4e3 with the same balance, 5.6e3 without).
# A diagonal scaling cannot remove a genuine near-null vector; the former
# equation at the partial coating's resonance keeps its large estimate.
CONDITION_BALANCE_ITERATIONS = 50


def condition_balance_blocks(layout):
    """Contiguous DOF block starts for the balanced condition estimate, or None."""
    if not layout.get('combined_junctions'):
        return None
    return np.asarray(sorted(offset for offset, _ in layout['dof_map'].values()), int)


def _balance(partial, starts, row_scale, col_scale):
    """Block Sinkhorn scalars from per-column, per-row-block equilibrated sums."""
    n = partial.shape[1]
    sizes = np.diff(np.append(starts, n))
    labels = np.repeat(np.arange(len(starts)), sizes)
    sums = np.stack([np.add.reduceat(row, starts) for row in partial])
    rows, columns = np.ones(len(starts)), np.ones(len(starts))
    for _ in range(CONDITION_BALANCE_ITERATIONS):
        rows /= np.sqrt((sums*columns[None, :]).sum(1)*rows/sizes)
        columns /= np.sqrt((rows[:, None]*sums).sum(0)*columns/sizes)
    rows, columns = rows/np.max(rows), columns/np.max(columns)
    column_sums = columns[labels]*(rows[:, None]*partial).sum(0)
    return row_scale/rows[labels], col_scale/columns[labels], float(np.max(column_sums))


def balanced_condition_scaling(a_mat, scaling, starts, max_block_bytes=16*1024*1024):
    """Refine (row_scale, col_scale, norm_1) by a block Sinkhorn balance.

    ``scaling`` is the max equilibration (the equilibrated matrix is
    ``A / row_scale[:, None] / col_scale[None, :]``). One column-blocked pass
    accumulates, for every column, the sums of its equilibrated magnitudes per
    row block; block scalars R, C then balance the block sums per entry, and the
    returned scales and 1-norm describe ``R*A_eq*C`` exactly.
    """
    row_scale, col_scale, _ = scaling
    a = np.asarray(a_mat)
    n = a.shape[0]
    starts = np.asarray(starts, int)
    partial = np.zeros((len(starts), n))
    width = max(1, min(n, int(max_block_bytes) // (8*n)))
    for start in range(0, n, width):
        stop = min(n, start + width)
        block = np.abs(a[:, start:stop])
        block /= row_scale[:, None]
        block /= col_scale[None, start:stop]
        partial[:, start:stop] = np.add.reduceat(block, starts, axis=0)
    return _balance(partial, starts, row_scale, col_scale)


def balanced_streamed_scaling(operator, row_scale, col_scale, starts):
    """``balanced_condition_scaling`` for a compressed tile operator (one tile pass).

    Column groups run on threads as in ``StreamedOperator.equilibrate``."""
    from ghost_backend.compressed.operator import _run_groups, MATMUL_WORKERS
    starts = np.asarray(starts, int)
    n = operator.n
    labels = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, n)))
    partial = np.zeros((len(starts), n))
    by_column = {}
    for i, j in operator.tiles:
        by_column.setdefault(j, []).append(i)

    def group(j):
        cols = np.asarray(operator.groups[j])
        total = np.zeros((len(starts), len(cols)))
        for i in by_column[j]:
            operator.checkpoint()
            left, right = operator.tiles[i, j]
            rows = np.asarray(operator.groups[i])
            magnitude = np.abs(left if right is None else left@right)
            weights = np.zeros((len(starts), len(rows)))
            weights[labels[rows], np.arange(len(rows))] = 1./row_scale[rows]
            total += weights@magnitude
        partial[:, cols] = total/col_scale[None, cols]
    _run_groups(by_column, group, MATMUL_WORKERS)
    return _balance(partial, starts, row_scale, col_scale)
