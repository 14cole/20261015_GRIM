"""Shared sheet and sheet/PEC matrix assembly, including open-edge conditions.

Sheets keep their single-potential equations (TM: S current; TE: D jump with
the hypersingular W). On a sheet+PEC scene every closed, consistently oriented
TYPE 2 loop additionally carries a second potential on the same density, so
the conductor's fictitious interior resonances are removed:

* TM: u_s = S x + gamma*D x_pec, gamma = sigma/(i*k0) (as the regional
  combined potential), rows u = 0 on PEC and the sheet law on sheets;
* TE: u_s = D x + eta*S x_pec, eta = sigma*i*k0 (as the dielectric exterior),
  rows dn(u) = 0 on PEC and the sheet law on sheets.

sigma = +1 where the exterior lies on the panel's minus side. The sheet rows
receive the conductor's second potential through its (disjoint, regular)
kernels; no sheet may touch a conductor (rejected upstream). Open TYPE 2
strips, polynomial bases and the compressed engine keep the former equations;
the solver then reports that the conductor is not resonance-protected.
"""
import numpy as np
from ghost_backend.twod.assembly.mass import add_mass


def pec_combination(mesh, infos, pol=None):
    """Element signs of the closed conductor loops of a sheet+PEC scene, or None."""
    from ghost_backend.compressed.runtime import enabled
    if enabled() or not mesh.elements or any(len(e.node_ids) != 2 for e in mesh.elements):
        return None
    conductor = np.array([int(info.seg_type) != 1 for info in infos], bool)
    if not np.any(conductor) or np.all(conductor):
        return None
    sigma = np.zeros(len(infos))
    arrivals, departures = {}, {}
    for ei in np.flatnonzero(conductor):
        info, element = infos[ei], mesh.elements[ei]
        if (info.minus_region < 0) == (info.plus_region < 0):
            return None
        sigma[ei] = 1. if info.plus_region < 0 else -1.
        departures.setdefault(int(element.node_ids[0]), []).append(ei)
        arrivals.setdefault(int(element.node_ids[1]), []).append(ei)
    nodes = sorted(set(arrivals) | set(departures))
    node_sigma = np.zeros(len(mesh.nodes))
    for node in nodes:
        into, out = arrivals.get(node, []), departures.get(node, [])
        if len(into) != 1 or len(out) != 1 or sigma[into[0]] != sigma[out[0]]:
            return None
        node_sigma[node] = sigma[into[0]]
    return dict(mask=conductor, sigma=sigma, node_sigma=node_sigma, nodes=np.asarray(nodes, int))


def _add_combined_conductor(matrix, mesh, pol, k0, combination, obs_order, src_order):
    import ghost_backend.twod.solver as rcs
    for sign in (1., -1.):
        mask = combination['mask'] & (combination['sigma'] == sign)
        if not np.any(mask):
            continue
        if pol == 'TM':
            gamma = sign/(1j*float(k0))
            _, operator = rcs._assemble_linear_operator_matrices(mesh, k0, False,
                obs_order=obs_order, src_order=src_order, source_element_mask=mask,
                compute_single_layer=False)
            matrix += gamma*operator
            # D trace from the exterior: +1/2 where it lies on the minus side.
            add_mass(matrix, mesh, .5*sign*gamma, mask.astype(float))
        else:
            eta = sign*1j*float(k0)
            _, operator = rcs._assemble_linear_operator_matrices(mesh, k0, True,
                obs_order=obs_order, src_order=src_order, source_element_mask=mask,
                compute_single_layer=False)
            # Rows hold -dn(u_s); dn(S) from the exterior is K' - sign/2.
            matrix -= eta*operator
            add_mass(matrix, mesh, .5*sign*eta, mask.astype(float))
        operator = None


def field_options(mesh, infos, pol, k0, combination=None):
    """Second far-field potential of the combined conductor density, or {}."""
    combination = pec_combination(mesh, infos, pol) if combination is None else combination
    if combination is None:
        return {}
    weight = combination['node_sigma']*(1/(1j*float(k0)) if pol == 'TM' else 1j*float(k0))
    return dict(second_potential='DLP' if pol == 'TM' else 'SLP',
                second_density_builder=lambda solution: weight[:, None]*solution[:len(mesh.nodes)],
                second_element_mask=combination['mask'])


def assemble_system(mesh, infos, pol, k0, obs_order=8, src_order=8):
    import ghost_backend.twod.solver as rcs
    from ghost_backend.compressed.runtime import enabled, native
    endpoints = rcs._geometric_sheet_endpoint_nodes(mesh, infos) if pol == 'TE' else np.empty(0, dtype=int)
    if enabled():
        operator,oracle=native(mesh,infos,pol,k0,'sheet',obs_order,src_order)
        return operator,oracle.endpoints
    z = np.asarray([complex(i.robin_impedance) if int(i.seg_type) == 1 else 0.0 for i in infos])
    if pol == 'TM':
        operator, _ = rcs._assemble_linear_operator_matrices(mesh, k0, False,
            obs_order=obs_order, src_order=src_order, compute_double_layer=False)
        coefficient = z / (1j * float(k0) * rcs.ETA0)
    else:
        destination = np.zeros((len(mesh.nodes), len(mesh.nodes)), complex, order='F')
        operator = rcs._assemble_linear_hypersingular_matrix(mesh, k0,
            obs_order=obs_order, src_order=src_order, destination=destination)
        coefficient = (1j * float(k0) / rcs.ETA0) * z
    matrix = operator
    operator = None
    if np.any(coefficient != 0):
        add_mass(matrix, mesh, -1., coefficient)
    combination = pec_combination(mesh, infos, pol)
    if combination is not None:
        _add_combined_conductor(matrix, mesh, pol, k0, combination, obs_order, src_order)
    if endpoints.size:
        matrix[endpoints] = 0
        matrix[endpoints, endpoints] = 1
    return matrix, endpoints


def rhs_many(mesh, k0, angles, pol, endpoints):
    from ghost_backend.twod.assembly.kernels import incident_loads
    bu, bdn = incident_loads(mesh, k0, angles, want_u=pol == 'TM', want_dn=pol == 'TE')
    rhs = -bu if pol == 'TM' else bdn
    if endpoints.size:
        rhs[endpoints] = 0
    return rhs
