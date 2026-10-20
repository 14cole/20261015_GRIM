"""Checked HODLR inverse of a hierarchical operator, gated by exact sample rows.

The inverse tree is the compressed backend's (``compressed.inverse``): every
node keeps the LU of its Woodbury correction. Its off-diagonal blocks are
unions of hierarchical blocks, compressed by the adaptive randomized range
finder (``linalg.hierarchical.compress_sampled``) through products with the
retained factors, so the inverse never reads a coefficient twice.

Solves keep the compressed backend's acceptance (refinement to a normwise
backward error of 3e-15 against the retained operator, GMRES on stalled
columns, the 1e-12 physical bound with dropped-route errors) and add the
measurement that replaces a per-row compression bound: the residual of every
solution column on the exactly evaluated sample rows must have a normwise
backward error within the same 1e-12. A failure rebuilds the operator once at
a tenth of its tolerance, then rejects the solve.
"""
import numpy as np
import ghost_backend.linalg.hierarchical as hf
from ghost_backend.compressed.inverse import CompressedSystem
from ghost_backend.compressed.factor import (CompressedFactor, SOLVE_BACKWARD_ERROR_LIMIT,
                                             CONDITION_PROBE_BACKWARD_ERROR_LIMIT)
from ghost_backend.compressed.hmatrix import HMatrixRejected
from ghost_backend.execution.metrics import timed_stage

# Hierarchical inverse ranks of electrically large 2-D bodies exceed the 256
# of the tile backend's cross approximation; storage still caps the build.
MAX_INVERSE_RANK = 2048
PROBES = 8


class _RangeBlock:
    """Off-diagonal inverse block as permuted ranges of the hierarchical operator."""
    def __init__(self, operator, r0, r1, c0, c1, checkpoint):
        self.operator, self.r0, self.r1, self.c0, self.c1 = operator, r0, r1, c0, c1
        self.shape = (r1 - r0, c1 - c0)
        self.checkpoint = checkpoint

    def matmul(self, rhs):
        return self.operator.range_matmul(self.r0, self.r1, self.c0, self.c1, np.asarray(rhs, complex))

    def project(self, basis):
        return self.operator.range_matmul(self.r0, self.r1, self.c0, self.c1, np.asarray(basis, complex),
                                          adjoint=True).conj().T


def _span(positions):
    start = int(positions[0])
    if not np.array_equal(positions, np.arange(start, start + len(positions))):
        raise ValueError('Inverse blocks must be contiguous clusters of the operator tree.')
    return start, start + len(positions)


class HMatrixSystem(CompressedSystem):
    """CompressedSystem whose off-diagonal blocks come from range products."""
    def __init__(self, operator, coordinates, tolerance, budget, checkpoint):
        # CompressedSystem's construction. BLAS stays on one thread (as for
        # its ACA builder): range products run on a thread team instead, and
        # the many small Woodbury products lose to a widened BLAS team.
        from ghost_backend.compressed.inverse import PreparedAccess
        if not np.isfinite(tolerance) or not 0 < tolerance < 1 or not np.isfinite(budget) or budget <= 0:
            raise ValueError('Invalid tolerance or storage budget.')
        self.operator = operator
        self.range_blocks = 0
        self.inverse_only = True
        self.n, self.budget = operator.n, budget
        self.checkpoint = checkpoint or (lambda: None)
        self.checkpoint()
        self.tolerance, self.leaf = tolerance, operator.leaf
        self.builder, self.inverse_threads = 'hmatrix_range_sampling', None
        self.bytes = 0; self.ranks = []; self.leaves = 0
        self.sampled_blocks = self.sampled_fallbacks = 0
        self.permutation = hf.spatial_order(np.asarray(coordinates, float), np.arange(operator.n), self.leaf)
        if not np.array_equal(self.permutation, operator.order):
            raise ValueError('Inverse and operator trees differ.')
        self.reserve(self.permutation.nbytes)
        from ghost_backend.execution.options import single_thread_blas
        with single_thread_blas(), operator.product_scope():
            self.root = self.build(PreparedAccess(operator), self.permutation)
        self.evidence = dict(bytes=self.bytes, max_rank=max(self.ranks or [0]), leaves=self.leaves,
            accesses=operator.entries, oracle_calls=operator.calls, max_query_entries=operator.max_entries,
            relative_inf_error_bound=None, relative_one_error_bound=None, storage_budget=budget,
            inverse_only=True, inverse_builder=self.builder, range_blocks=self.range_blocks,
            max_inverse_rank=MAX_INVERSE_RANK, inverse_blas_threads=1)

    def compress_block(self, oracle, rows, cols):
        operator = self.operator
        r0, r1 = _span(operator.position[np.asarray(rows, np.int64)])
        c0, c1 = _span(operator.position[np.asarray(cols, np.int64)])
        block = _RangeBlock(operator, r0, r1, c0, c1, self.checkpoint)
        rng = np.random.default_rng((hf.SAMPLE_SEED, r1 - r0, c1 - c0, r0, c0))
        probes = rng.standard_normal((c1 - c0, PROBES)) + 1j*rng.standard_normal((c1 - c0, PROBES))
        norm = float(np.linalg.norm(block.matmul(probes)))/np.sqrt(2*PROBES)
        u, v, error = hf.compress_sampled(block, max(norm, 1e-300)*self.tolerance*.25, MAX_INVERSE_RANK, rng)
        self.range_blocks += 1
        self.ranks.append(u.shape[1])
        return u, v, error


class HMatrixFactor(CompressedFactor):
    """CompressedFactor for ``HMatrixOperator`` with the exact sample-row gate."""

    def __init__(self, operator, *args, storage_budget_bytes=None, **kwargs):
        if storage_budget_bytes is None:
            from ghost_backend.compressed.runtime import hmatrix_storage_budget
            storage_budget_bytes = hmatrix_storage_budget()
        self.storage_allowance = int(storage_budget_bytes)
        super().__init__(operator, *args, storage_budget_bytes=storage_budget_bytes, **kwargs)

    def _refresh_inverse_budget(self):
        from ghost_backend.twod.assembly.polynomial_pair import retained_bytes
        self.budget = (self.storage_allowance - self.a.bytes
                     - getattr(self.a, 'reserved_partner_bytes', 0) - retained_bytes('compressed'))
        if self.budget <= 0:
            raise MemoryError('No compressed inverse storage remains after the operator rebuild.')

    def _build(self, tolerance):
        self.factor = None
        self.checkpoint()
        self.factor = timed_stage('factorization')(HMatrixSystem)(self.a, self.a.coordinates, tolerance,
                                                                  self.budget, self.checkpoint)
        self.tolerance = tolerance; self.event['factorizations'] += 1
        self.event['preconditioners'].append(self.factor.evidence)

    def sampled_errors(self, x, b):
        """Normwise backward errors of ``x`` on the exactly evaluated sample rows."""
        residual, _ = self.a.sample_residual(x, b)
        norm = self.a.operator_norm
        denominator = np.maximum(norm*np.max(abs(x), axis=0) + np.max(abs(b), axis=0), 1e-300)
        return np.max(abs(residual), axis=0)/denominator

    def inverse(self, rhs, trans=0, return_residual=False, limit=SOLVE_BACKWARD_ERROR_LIMIT):
        b = np.asarray(rhs, complex); vector = b.ndim == 1
        if vector:
            b = b[:, None]
        x, residual = super().inverse(b, trans, True, limit)
        if trans == 0:
            errors = self.sampled_errors(x, b)
            if not np.all(np.isfinite(errors)) or np.max(errors) > limit:
                self.event['sampled_rejection'] = float(np.max(errors))
                if getattr(self, 'tightened', False):
                    raise HMatrixRejected('Hierarchical solve failed its exact sample-row backward error '
                                          '({:.3g} > {:.3g}).'.format(float(np.max(errors)), limit))
                self.tightened = True
                self.factor = None
                del x, residual
                self.a.rebuild(self.a.tolerance/10)
                self._refresh_inverse_budget()
                self.matrix_inf = max(float(np.max(self.a.row_norm - self.a.row_error)), 0.)
                self.event['compressed'] = self.a.evidence
                self.event['tightened_operator_tolerance'] = self.a.tolerance
                self._fresh_build()
                return self.inverse(rhs, trans, return_residual, limit)
            key = 'max_sampled_backward_error' if limit <= SOLVE_BACKWARD_ERROR_LIMIT else 'max_probe_sampled_backward_error'
            self.event[key] = max(self.event.get(key, 0.), float(np.max(errors)))
            self.event['sampled_rows'] = len(self.a.sample_rows)
        solution = x[:, 0] if vector else x
        if return_residual:
            return solution, residual[:, 0] if vector else residual
        return solution

    @timed_stage('condition_estimation')
    def _condition(self):
        blocks = self.diagnostics.get('condition_balance_blocks')
        if blocks is None:
            return super()._condition()
        # Block Sinkhorn balance from one pass over the retained blocks.
        self.diagnostics.pop('condition_balance_blocks')
        rows, columns, _ = self.a.equilibrate()
        rows, columns, norm = self.a.balanced_scaling(rows, columns, blocks)
        from scipy.sparse.linalg import LinearOperator, onenormest
        probe = CONDITION_PROBE_BACKWARD_ERROR_LIMIT
        inverse = LinearOperator(self.a.shape,
            matvec=lambda z: columns*self.inverse(rows*np.asarray(z).reshape(-1), limit=probe),
            rmatvec=lambda z: rows*self.inverse(columns*np.asarray(z).reshape(-1), trans=2, limit=probe),
            matmat=lambda z: columns[:, None]*self.inverse(rows[:, None]*np.asarray(z), limit=probe),
            rmatmat=lambda z: rows[:, None]*self.inverse(columns[:, None]*np.asarray(z), trans=2, limit=probe),
            dtype=complex)
        estimate = norm*float(onenormest(inverse))
        if not np.isfinite(estimate):
            raise hf.HierarchicalRejected('Nonfinite hierarchical condition estimate.')
        self.diagnostics.update(condition_est=estimate, condition_label=self.label,
            condition_method='equilibrated_1norm_hmatrix_refined_inverse_block_balanced')

    @timed_stage('linear_solve')
    def solve(self, rhs):
        import ghost_backend.twod.solver as rcs
        b = np.asarray(rhs, complex); vector = b.ndim == 1
        if vector:
            b = b[:, None]
        x, residual = self.inverse(b, return_residual=True)
        self.relative_residual = self.relative_errors(x, b, residual)
        self.event['rhs_batches'] += 1
        self.event['max_rhs_columns'] = max(self.event['max_rhs_columns'], b.shape[1])
        self.event['max_relative_residual'] = max(self.event['max_relative_residual'], float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics.update(linear_backward_error=max(self.event['max_backward_error'],
                                                              self.event.get('max_sampled_backward_error', 0.)),
                                    linear_backward_error_limit=SOLVE_BACKWARD_ERROR_LIMIT,
                                    linear_backward_error_scope='retained operator and exact sample rows')
        rcs._record_dense_backend_event(requested='cpu', used='cpu_hmatrix', n=len(self.a), label=self.label,
            factorizations=self.event['factorizations']-self.reported, rhs_columns=b.shape[1], compressed=self.event,
            sweep_compression=self.event.get('sweep_compression'), condition_method=(self.diagnostics or {}).get('condition_method'))
        self.reported = self.event['factorizations']
        return x[:, 0] if vector else x
