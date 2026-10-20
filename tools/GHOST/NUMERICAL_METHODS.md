# Numerical methods and verification scope

The September 2026 changes preserve the outgoing-wave and amplitude-version-2
conventions. Independent cylinder and sphere references and complex-field
comparisons are necessary when changing a boundary representation: a small
linear residual alone does not establish physical accuracy.

## 2D transmission

A single dielectric body uses an exterior `D + i*k0*S` potential and an
interior single-layer potential. The exterior field projection includes both
terms. With `W = -normal_derivative(D)`, the exterior matrix blocks are
`K + M/2 + i*k0*S` and `W - i*k0*(K' - M/2)`. This removes the fictitious
Neumann resonances of the former exterior double-layer-only representation.

Closed Robin contours, including standalone PEC and IBC bodies in both
polarizations, and closed continuous regional PEC/transmission contours use side-oriented
`S + gamma*D` regional potentials, including the corresponding trace jumps,
hypersingular flux terms, and exterior field projection. Dense and compressed
coefficient queries implement the same equations.

The exterior region of a body with open material junctions (a partial coating:
TYPE 2 bare arc, TYPE 3 coating outline and caps, TYPE 4 coated core) is closed
only as a union of open arcs that meet at split junction nodes. It now uses the
same `S + gamma*D` potential. Its D/W density is continuous: at each junction
the slave end's D/W columns carry the master end's coefficient, while S/K'
keep both values. The element-wise Maue form of W is exact for a continuous
density except on the two half-hat test rows of a junction; integrating the test
function's tangential derivative by parts leaves `W = W_Maue - eps*F(p)`, with
`F(p) = int G(p,y) dpsi/ds dy` and `eps = +1/-1` where the test panel's own
direction ends/starts at p. Without that term the split-disk error below grew
by about 30%; without the continuous D density (independent junction values)
the fields were wrong at O(1). Bounded
regions bounded by open interfaces keep the single-layer representation: a
simply connected bounded region has an unbounded complement and no interior
eigenvalues. Combined on the coating as well, the condition estimate reached
1.5e6 at ka = 40 with no accuracy gain. A bounded region with holes formed by
open arcs, and polynomial (degree > 1) meshes of such regions, remain
unprotected.

Partial coating (PEC core r = 0.07 m, lower half coated to 0.1 m, eps 3; ka
for a = 0.1 m), against a 1824-panel solution of the new equation (agrees with
912 panels to 0.001 dB):

| ka | panels | former VV 0/90 deg error (dB) | new | former / new condition |
|---|---|---|---|---|
| 2.8688 | 114 | +0.90 / +0.79; HH 0 deg +3.97 | -0.003 / +0.03; HH +0.06 | 3.3e4 / 512 |
| 2.869 | 186 | +0.175 / -0.455; HH -0.16 | +0.001 / +0.001 | 7.0e4, 2.1e5 / 1.2e3, 3.5e3 |
| 4.536 | 186 | +0.12 | +0.003 | 5.6e4 / 1.5e3 |
| 6.43 | 186 | -0.28 / +0.32 | +0.001 / 0.000 | 2.3e5 / 2.4e3 |
| 2.0 | 120 | 0.000 / -0.003 | 0.000 / -0.002 | 318, 529 / 681, 2.1e3 |

Fine sweeps (2.85-2.89 step 0.001, 4.50-4.58 step 0.002, 5.6-6.6 step 0.01)
have a flat condition estimate (at most 3.5e3, 4.3e3 and 7.5e3) where the
former equation peaked at 2.1e5, 5.6e4 and 2.3e5. With exact references, a
dielectric disk and a coated PEC cylinder split into two half regions of the
same material by a TYPE 5 diameter reproduce the former accuracy at 12
frequencies (ka 0.5-6) and at the polygon's interior Dirichlet zeros. The
condition estimate off resonance is 2-6x the former (as for closed combined
regions) and grows faster than linearly under extreme refinement (2.7e4 at 912
panels, 9.4e4 at 1824 panels at ka = 2). Closed-region systems are unchanged
bit for bit.

The single-pass row/column max equilibration of the condition estimate
misjudges these systems: the exterior's W rows (about gamma/h) dominate the
transmission flux rows, so the bounded regions' single-layer density blocks
stay two to three orders under-scaled. On the 3 GHz airfoil survey (8608
unknowns, lossy high-index coating) it reported 1.47e6 (TE) where the former
equation gave 2.8e4, with fields that agree to 2e-5 of the peak. For systems
with union junctions only, the dense and compressed estimators now refine the
equilibration by a Sinkhorn balance of the equilibrated block sums (one scalar
per interface side; `condition_method` ends in `_block_balanced`). The
airfoil's TE/TM estimates become 1.1e4 / 3.3e4 at 3 GHz and 2.0e4 / 5.8e4 at
6 GHz (former equation 2.8e4 / 7.5e3 and 3.1e4 / 1.1e4). A diagonal scaling
cannot hide a genuine near-null vector: the former equation at the partial
coating's resonance keeps 1.05e6 balanced (1.16e6 unbalanced). The gate limit
is unchanged. Under extreme refinement of the partial coating, where the weak
direction sits at the junctions, the balanced estimate still grows (6.8e5 HH
at 3648 panels, ka = 2).

The closed-contour law may vary along the contour: PEC plus impedance zones,
several impedance zones, tapers, and separate bodies with different laws. The
boundary equation is `dn(u) + alpha(x) u` (or `u` on TM PEC nodes) for one
continuous closed-contour density. W is never weighted by `alpha`, so its Maue
form keeps no endpoint terms, and the uniqueness argument for `S + gamma*D`
(an interior impedance problem with `1/gamma`) does not involve `alpha`. The D/W
correction is therefore routed per node exactly as the base S/K' assembly:
TM PEC nodes receive `gamma*D`; every other node receives `-gamma*W` plus
`gamma*alpha*D`. The +-1/2 trace jump of a Robin row uses the element-weighted
mass `sum_e alpha_e M_e`, matching the element-weighted `alpha*S`; weighting
that jump by the nodal average alone left a second-order error proportional
to the steepness of the law (0.81% instead of 0.09% against the former equation
for a 20+5j to 377 ohm TM taper at 48 panels per wavelength).

A change of impedance flag along a closed PEC/impedance loop no longer splits
the mesh node: it is a change of boundary law, not a break in the contour, and a
hypersingular operator has no meaning across a split density. Open chains and
every other signature difference (type, materials, regions, boundary kind) keep
separate nodes. The former equation and the combined one converge to the same
fields under refinement (0.001% to 0.09% apart at 3,328 panels); at the former
discrete resonance of a PEC|75-20j ohm circle the 25 dB defect, a condition
estimate of 7e5 and a passing quality gate become a smooth response with a
condition estimate of 30. In TM every node that touches a PEC element carries
the Dirichlet row (`u = 0` on that element), in the regional layout as in the
standalone Robin assembler; the nodal coefficient is set to zero there. The
first round-5 draft classified a PEC/impedance junction node by its averaged
coefficient instead, which applies the flux equation over the PEC half of the
test function: 2.5% against 0.9% for the former equation at 24 panels per
wavelength. With the shared rule the two equations agree to 0.025% / 0.014% /
0.007% at 104 / 208 / 416 panels. At the junction the TM boundary condition
changes type (`u = 0` to a Robin law) and the density behaves like `r**(-1/2)`,
so uniform linear panels converged at first order in both equations (0.97%,
0.49%, 0.24%, 0.12% at 104 to 832 panels against a graded 3328-panel
reference). `_build_panels` now splits the panel on each side of such a
junction of an opaque conductor (TYPE 2 and TYPE 4 segment ends)
geometrically, four levels of ratio 1/2: eight extra panels per junction,
adjacent length ratios of at most 2 (inside the calibrated far-rule range). The
same input then gives 0.24%, 0.058%, 0.016%, 0.0055%, the level of the smooth
contour and of TE (0.23%, 0.057%, 0.014%, 0.0034%). Four fixed levels only
shrink the first-order term (the innermost panel is `h/16`): against a finer
reference the TM error ratio per doubling decays 4.1, 3.5, 2.4, 2.0 from 104 to
1664 panels (0.0071% and 0.0036% at the fine end). Since round 12 the mesher
adds a level for every doubling of the density beyond 32 panels per wavelength
(`_junction_grading_levels`; the innermost panel stays two decades above the
1e-9 m node snap): 0.238 / 0.060 / 0.0148 / 0.0035 / 0.0007 % at 104 to 1664
panels, ratios 3.9 to 5.1 and the TE values at the fine end, for 4 to 16 more
panels. Meshes of up to 32 panels per wavelength are unchanged.

The junction is classified from the boundary condition the field solve
evaluates, not from the flag (Codex's review of round 10: a positive flag
carrying 0 ohms was left ungraded, 0.97% instead of 0.24% at 104 panels and
fields 1.04% away from the identical flag-0 input). `_boundary_condition_junctions`
evaluates each conductor segment end (`arc_s` 0 or 1, as a taper is sampled) at
every frequency the mesh serves, with the solver's own `|Zs| <= EPS` meaning
PEC, and grades where the complex jump `|Z1 - Z2|` exceeds `1 - 1/4` of the
larger magnitude (`impedance_jump_is_graded`, `JUNCTION_GRADING_CONTRAST = 4`):
a factor of four for laws in phase, about 44 degrees for laws of equal
magnitude. PEC next to any impedance always does; two PEC
definitions meeting do not; a table that reaches zero grades the meshes of the
frequencies where it does (every frequency of a fixed reference mesh). The
contrast rule exists because exact zero is not the only case: a Robin
coefficient that is large at the panel scale acts as `u = 0` until the panels
resolve `1/|alpha|`. Uniform against graded error at 104 / 208 / 416 panels,
law | `75-20j` on the same circle:

| Law, polarization | Uniform | Graded |
|---|---|---|
| 0 ohm law, TM (as flag 0) | 0.97 / 0.48 / 0.24 % | 0.24 / 0.058 / 0.015 % |
| 1 ohm, TM | 0.36 / 0.17 / 0.089 % | 0.24 / 0.056 / 0.013 % |
| 5 ohm, TM | 0.29 / 0.13 / 0.061 % | 0.25 / 0.063 / 0.015 % |
| 20 ohm, TM (contrast 3.9) | 0.20 / 0.057 / 0.015 % | 0.26 / 0.065 / 0.016 % |
| 1e5 ohm, TE | 1.98 / 1.07 / 0.57 % | 0.13 / 0.031 / 0.010 % |
| 3770 ohm, TE | 1.04 / 0.32 / 0.087 % | 0.20 / 0.046 / 0.010 % |
| 200 ohm, TE (contrast 2.6) | 0.24 / 0.059 / 0.014 % | 0.23 / 0.058 / 0.014 % |

Against a `377` ohm partner, `20` ohm (contrast 19) loses 1.17 / 0.42 / 0.13 %
to 0.27 / 0.065 / 0.015 % in TM, `50` ohm (7.5) 0.53 / 0.16 / 0.043 %, and
`100` or `1500` ohm (3.8, 4.0) are within 1.5 times the graded error in both
polarizations. Below a contrast of four the uniform mesh is second order, so
the steps of a staircase that approximates a taper are not graded. Phase counts
since round 12, because a phase-only contrast costs a constant factor although
not the order: laws of magnitude 77.6 ohms against `75-20j`, TM, uniform over
graded error at 416 panels, 1.2 at 30 degrees apart, 1.3 at 45, 1.5 at 60, 1.8
at 75 and 2.4 at 105 (`+77.6j`: 0.59 / 0.15 / 0.038 % against 0.26 / 0.065 /
0.016 %); TE is indifferent there and mirrors it against a 377 ohm partner.
Combined contrasts (a factor of three with 45 degrees) gain up to 1.9 and are
graded too. The measure is a threshold on one number, not a model of the
singularity: at equal jump, `25.9` ohms at +30 degrees gains nothing and at -60
degrees 1.9. Above the threshold
grading never lost more than the 0.06 percentage points it costs at 104 panels
(it breaks the uniform circle's error cancellation). Both polarizations share
the graded mesh, so their systems stay interchangeable. Solves, backend
selection and HPC scheduling pass the same material library and frequencies to
`_build_panels` and count the same panels; a caller without a library has only
inline laws evaluated, and a table is then graded only against PEC. The rule is
a threshold: a table crossing it between two frequencies of a sweep changes
that junction's mesh by eight panels there. Free strip ends, triple junctions
and TYPE 1 sheet joints are not graded. The real HPC driver was run end to end
on such bodies (plan without submission, then the worker in an interpreter
where importing Qt raises): for a rectangle with two PEC/impedance junctions (24 and
36 panels per side at 1 GHz, 180 to 540 per wavelength, so seven to nine levels)
the schedule forecasts 126 base and 176 certification nodes, the worker solves
and attests 176 panels, and a zero-ohm law gives 96 and 144 in both (the
certification mesh gives each chain `max(B + 1, ceil(1.5 B))` panels, longest
panels first, never fewer than the base on any primitive).
The BoR mesher applies the same rule to the chains of a conductor (below).
Monostatic, bistatic and boundary-density entry points use this same selection.
The discrete 104-panel TE circle resonance at `ka=5.13718385499639` is a
regression case; merely checking the nearby smooth-circle eigenvalue misses
the former narrow error band.

Production regional assembly evaluates S, K', D and W in one traversal per
wavenumber and scatters directly into the system matrix or compressed tile.
The fused W contribution uses a calibrated far rule of 8 points for degree 1
and 9 for degrees 2 and 3 (10 before the 22 September audit) for passive
wavenumbers with `1e-4 <= |k| Lmax <= 3`, panel length ratio at least 0.001,
separation at least three maximum panel lengths; the worst W block error is
1.5e-14 to 4.6e-14. Outside that range it retains the 16-point floor; explicit
higher orders and `far_grading=False` retain the conservative rule. Calibration
compares the completed W block, including cancellation, with independent
40/48-point rules. The graded far orders of S, K' and D depend on the
polynomial degree, with a separate table for strongly lossy wavenumbers
(`arg k < -45 degrees`); every bin is within 1e-12 (S relative to the block
maximum, K' and D to the integral of the kernel magnitude). A panel end that
hovers within half a panel length over another panel's interior uses the
adaptive near rule (worst K' 1.3e-8 to 7.4e-12). Self blocks with `|k l| > 8`
are built from sub-intervals (1.5e-14; the generic fallback was 4.4e-2 at
`|k l| = 8.1`). Kernel tables accept a check tolerance of
`max(2e-13, 8 eps |k| r)` and stop at their 4096-interval budget (about 377
wavelengths), beyond which the far expansion applies; they used to be refused
beyond about 163 wavelengths.
Near pairs reuse the same checked Green and derivative integrals.
The separate operator implementation remains an independent assembly reference.
This removes repeated 64-row correction queries without reducing quadrature
accuracy. Resource estimates include the additional routes and basis moments.

The combined equation needs S, K', D and W where the former equations needed a
single operator, and a conductor's TM system cannot be derived from its TE
system (TE uses K' and W, TM uses S and D), so the co-polarized solve of a
conductor had paid two full traversals. The regional layout takes nothing
polarization-specific from the element records, and the fused engine applies
masks and routes to finished tile accumulators, so the TE step of a
co-polarized dense solve now assembles both systems in one traversal
(`assemble_pair`) wherever TM would otherwise be assembled fresh (combined
systems whose conductor rows exceed one eighth of the unknowns). The TM step
finds its finished matrix as the retained one. The second matrix is resident
during the TE solve, so the pairing is admitted only when it fits on top of that
solve's own estimate; otherwise, and for single-polarization calls, bistatic
entries and the compressed backend, nothing changes. Both matrices equal the
separately assembled ones to rounding (1e-13 relative), and the retained matrix
no longer needs restoring from the residual spool. Measured on the whole
co-polarized solve at 4,320 panels (`ka = 60`): PEC 33.8 to 27.7 s, uniform
impedance 50.7 to 31.7 s, PEC|impedance 50.4 to 31.7 s; a coated PEC core
(1,440 panels, 23 % conductor rows) 24.0 to 15.4 s; a dielectric cylinder,
whose TM system is still derived from the TE one, 6.0 s both ways. A conductor
gains less than an impedance body because each of its polarizations already
needs both Hankel functions and only two of the four operators. The combined
equation therefore remains more expensive than the unprotected one it replaced;
what is left is the four-operator contraction and W's far rule.

Plain dielectric assembly shares the exterior S/K'/D/W traversal. Combined
regional TE-to-TM reuse retains trace rows and rescales the appropriate flux
blocks, while independently rebuilding conductor/Robin equations through
bounded row-strip queries. Those queries cost several times more per row than
the fused assembly, so reuse is attempted only while conductor rows are at most
one eighth of the unknowns (measured break-even between 9% and 20% on coated
cylinders); otherwise TM is assembled fresh into the same owned buffer. A
standalone PEC/IBC body has nothing to reuse: routing it through the row-strip
path made the co-polarized solve slower than its two polarizations run
separately (4,320 nodes: PEC 22.6 s against 12.1 s, 75-20j ohm IBC 50.9 s
against 17.7 s, identical fields). The combined equation itself still costs
about twice the former single-operator assembly for such bodies. Unequal
observer/source rules use the original assembly path. These optimizations
preserve the previously introduced combined-potential formulation.

The outer tile scheduler owns 2D assembly concurrency. Native far integration
does not start a nested four-thread pool. Endpoint-touching pairs (one
1e-9 m touching tolerance, the node-snap width, plus a same-node rule) use a
graded `t^4` corner rule with 20 points per direction and corner-stable
distances: per block 2.6e-13 collinear and 1.3e-10 (K') at a 170 degree reflex
corner, where the former Duffy batches erred by up to 6.3e-4; on a circle the
quadrature error was O(h) (5e-5 to 2.4e-6) and total convergence is now clean
O(h^2). Far tiles are computed on threads but scattered in a fixed tile order,
so thread counts give bitwise identical matrices, and weighted tile
preparation occurs outside the matrix scatter lock.

Dense double-precision LU can spool an exclusively owned original matrix to
temporary disk storage and overwrite its RAM buffer with LU. Every residual
product still uses every original row; condition scaling is computed before
overwrite. Original TE coefficients needed for TM assembly are restored from
the spool into the same buffer after solving. `dense_residual_storage` accepts `auto`, `memory`, or `disk` in the
2D execution options. The admission estimates price the original next to its
LU, so `auto` keeps it in memory whenever that copy fits: it spools an eligible
matrix of at least 512 MiB only when the memory available as the matrix is
factored cannot hold the copy (plus an eighth, at least 256 MiB), and falls
back to memory if the spool cannot be created. Every residual product of a
spooled matrix re-reads the whole file (on the certified 10 GHz airfoil, 64 s
of 207 s for a 7.8 GB system; kept in memory the run took 145 s within its
16.7 GB estimate). Non-owned arrays, mixed precision, and hierarchical factors
retain their existing storage. Spools close on completion or failure. BoR modal
factors follow the same policy: a mode worker whose copy does not fit factors a
system of 512 MiB or more in place and computes its residuals from the spool.

A dense double-precision system of at least 10,000 unknowns
(`linalg.hierarchical.HIERARCHICAL_MIN_UNKNOWNS`; the environment variable
`GHOST_HIERARCHICAL_MIN_UNKNOWNS` overrides it, 0 keeps LU) is factored as a
HODLR inverse under the `dense` and `auto` factorizations. Its off-diagonal
blocks are compressed by an adaptive randomized range finder (products of the
block with 32 Gaussian probes at a time, re-orthogonalized, until fresh
samples leave less than `1e-10 ||A||_inf`; then a truncated SVD through the QR
of the projected block), and the factor is accepted exactly as before: every
solve is refined against the original matrix to a normwise backward error of
3e-15, the result must then pass the dense backward-error gate, and a factor
that does not converge is rebuilt once at `1e-12`, then replaced by LU. On the
certified airfoil's systems (three batches of 256 right-hand sides, eight
cores) the factor tied LU at 9,082 unknowns (4.2 against 4.7 s) and was 1.7
times faster at 13,618 (8.0 against 13.7 s, factor 182 MB against a 2,967 MB
LU); below the threshold LU is faster (7,348 unknowns: 3.4 against 2.8 s).
The block tolerance sets the cost: at `1e-9` refinement needed two steps, at
`1e-6` seven, each a product with the original matrix per batch. A cluster
tree over the coordinates keeps coincident unknowns of different densities in
the same leaf; the natural order would split them at the top level, where one
block then reached rank 1,024. Memory forecasts price such a system as the
matrix and its factor budget (0.65 of it) instead of the matrix and an LU
copy; an LU fallback that finds no room for its copy spools the original
(`dense` and `auto`). End to end, the certified airfoil at 6 GHz takes 40.5 s
instead of 51.7 s with LU (factorizations 12.9 against 25.3 s, peak 5.1
against 7.6 GB, RCS equal to 1.7e-12 of the largest amplitude); at 10 GHz,
where the LU copy did not fit the planner's margin on a 31 GB workstation and
the compressed backend ran (90-92 s, 4.3 GB), the dense path is now admitted
and takes 88 s (11.5 GB).

## Hierarchical 2-D operator

From 8,192 unknowns (`compressed.hmatrix.HMATRIX_MIN_UNKNOWNS`;
`GHOST_HMATRIX_MIN_UNKNOWNS` overrides it, 0 disables it) the compressed
backend assembles an H-matrix instead of evaluating every coefficient tile.
Below the threshold, and whenever the automatic policy admits dense storage,
nothing changes: results are bit-identical, and the automatic policy keeps
dense LU/HODLR wherever it is admitted (the hierarchical backend is for the
systems dense storage cannot hold). The threshold is where the tile operator
stops being practical: at 10,000 PEC unknowns its retained payload exceeded the
2-GiB default cap, while the hierarchical operator kept 15% of the dense
storage (dense was 1.2 times faster there; from 15,000 unknowns the
hierarchical solve of a PEC cylinder was faster than dense, 43 against 50 s).

*Structure.* Unknowns are clustered by the bisection of `spatial_order`
(the inverse tree's permutation), leaves holding about 128 distinct points
times the number of densities sharing a point. Bounding boxes are inflated by
each cluster's longest adjacent panel; two clusters at one level are
admissible when `min(diam) <= dist`. Inadmissible leaf blocks are evaluated
exactly by the formulation's coefficient oracle, so near, junction, thin-gap
and acute-corner quadrature is unchanged. The admissible partners of one row
cluster form its interaction list, compressed as one low-rank block: sampled
columns (spread, random and high-weight), a pivoted-QR skeleton of the long
rows, and a least-squares fit over more samples than the rank (an
interpolating cross would fit the coefficients' rounding noise, about 3e-13
of a block, into unseen rows), validated by fresh random rows and columns,
then recompressed by QR/SVD. The target is a mean row-sum error of `1e-12
||A||_inf` per list, estimated from probes. Lists of one level are compressed
concurrently (single-threaded coefficient queries on all allocated logical
CPUs; results do not depend on the thread count), finest level first, a pilot
wave of eight lists calibrating each level's rank guess. TE and TM share one
kernel traversal when both are built together; the TM operator is kept for
its solve.

*Error measurement.* No per-row deterministic bound is claimed. Exactly
evaluated rows are retained within a 256 MiB coefficient-storage allowance.
There is one middle row per leaf plus random rows up to at least 64 while
that allowance permits. When there are too many leaves, a spread subset of
leaf middle rows is used. Evidence records the total and sampled leaf counts,
coverage fraction, whether coverage is limited, and retained sample-row bytes.
After assembly the complete row errors `sum_j |A_ij - H_ij|` of these rows
must stay below `1e-11 ||A||_inf` (measured 1-3e-12), otherwise the tile
operator is used. Every field solve must then pass the backend's existing
gates against the retained operator (refinement to 3e-15, 1e-12 physical
bound) and, in addition, a normwise backward error within 1e-12 of the dense
gate on the exact sample rows (measured 1e-14 to 3e-13). A failure rebuilds the
operator at a tenth of its tolerance once, then rejects the solve. Field
changes against dense LU are therefore conditioning limited: 2e-12 to 4e-12 of
the peak on 5,000-20,000-unknown PEC cylinders, 1e-10 to 4e-10 on the
6 GHz airfoil, 6e-9 on a dielectric of condition 6e3 (where the tile backend
itself differs from LU by 1e-9) and 1.4e-8 on the 0.001-inch coat
(condition 6e5).

*Factorization.* The compressed backend's inverse tree (Woodbury HODLR at
`1e-8`, refinement and GMRES) is built from range products with the
retained factors (adaptive randomized sampling, ranks up to 2,048), so no
coefficient is read twice. Condition estimates use exact one-pass
equilibration over the retained blocks. A factorization owns one lazily started
thread team for its range products; it joins that team on success, failure or
cancellation. Adjoint products reduce bounded batches of block contributions
in a fixed order, avoiding a complete tree's temporary outputs.

The retained-storage ledger includes numeric geometry, tree and block indexes,
sample rows and their diagnostics, and norm/error vectors as well as block
coefficients. Shared buffers may be charged to both operators of a TE/TM pair;
Python containers and transient work remain in the separate RAM forecast.

*Planning.* Memory and work forecasts come from the cluster tree and
interaction lists of the actual mesh with a calibrated rank model (12 +
0.21 kD per density for lists; 0.25 kD at the inverse root and 0.09 kD below,
plus margins), so no coefficient is evaluated. Both polarizations' operators,
the inverse and concurrent assembly workspaces are priced, and the
forecast is used by the solver's admission gate, the automatic planner, the
HPC planner and the GUI setup. At the 2,048-MiB default (and at 0) the
hierarchical storage cap is sized automatically from the solve RAM limit.

*Coverage.* Every formulation reached through the compressed coefficient
oracles is covered: closed PEC/IBC combined potentials, single dielectric
bodies, coated and multi-region systems (both polarizations, lossy and
magnetic media, two bodies, thin gaps, acute and reentrant corners), sheets.
Thin layers with a magnetic contrast keep the tile operator (each coefficient
column is a solve with the inverse mass matrix), as do the experimental
`verified_cur` far method and the polynomial fine-to-coarse projection
(those systems are assembled on the coarse mesh directly). The fallback is
recorded as `hmatrix_fallback` in the factor evidence.

*Measurements* (8-core Windows workstation, 91 angles, both polarizations):

| case | unknowns | before | after |
|---|---|---|---|
| PEC cylinder | 20,000 | dense 129 s, 7.5 GB | dense unchanged; hierarchical 82 s, 2.8 GB |
| PEC cylinder | 40,000 | dense refused (25.6 GB/matrix) | hierarchical 387 s, 8.5 GB |
| airfoil 10 GHz (plain) | 28,466 | dense refused; tile backend over its cap | hierarchical 100 s, 2.5 GB |
| airfoil 6 GHz (plain) | 17,120 | dense 26 s, 12.2 GB | dense unchanged; hierarchical 44 s, 1.4 GB |
| 0.001-in coat, 20-in cylinder | 15,078 | dense 42 s, 11.1 GB | dense unchanged; hierarchical 95 s, 1.8 GB |

In the high-frequency regime list ranks grow about linearly with cluster size
(145 at 2,500 PEC unknowns per cluster at 10 GHz, 265 at 5,000), so the
coarsest levels keep coefficient work near 14% of `N^2` and assembly time
grows faster than `N^2 log N`; each coefficient query also costs several
dense-assembly entries (no reciprocal reuse). Multi-region systems pay most:
their dense assembly shares TE/TM and reciprocal kernels, so the hierarchical
backend is 1.7-2.3 times slower than dense where dense fits, while it uses 6 to 9
times less memory.

The dense HODLR condition estimate now accepts probe solves at a normwise
backward error of 1e-9, measured against the exact matrix (usually the
unrefined factor: one exact product per probe instead of two), and repeats with
fully refined probes when `kappa*1e-9` exceeds 1e-2, as the compressed backend
does. On the certified coated cylinder (1280-gon, one angle) the condition
estimate fell from 1.9 to 1.3 s of 29 s; fields are bit-identical and the
estimate changed by 4e-6 relative.

## BoR formulations and geometry

Closed uniform IBC dispatch selects CFIE for reactive and resistive impedance.
The direct `solve_bor` API also defaults to automatic selection: CFIE for
closed opaque bodies, EFIE for open shells and transmitting sheets.
Physical resistance does not remove EFIE representation resonances. Closed
bodies with a spatially varying impedance (a PEC zone next to an impedance
band, several impedance zones, tapers) use the same CFIE. Its magnetic-field
part contains the EFIE operator acting on `M = -Zs n x J`, whose charge term
needs a continuous source: a jump of `Zs` between two elements would be a
magnetic line charge that the Gauss-point tables cannot represent. `M` is
therefore expanded in the nodal basis with the average impedance of the two
adjacent elements at each node (the convention the partial-coating bare pieces
already use), which spreads a jump over two elements and reduces to the uniform
law exactly (a uniform array reproduces the scalar result bit for bit). The
electric-field part and the far field keep the element-wise `Zs`.

No exact reference exists for a varying law, so the CFIE was compared with the
EFIE away from its resonances on a 0.1 m sphere at `ka = 3.3`: a smooth taper
agrees to 0.0009 dB at 240 elements and 0.0065 dB at 60; a PEC half next to
`100+50j` to 0.015 dB (0.09 dB at 60); two zones `50 | 300+100j` to 0.048 dB
(0.42 dB at 60). At an abrupt jump both equations converge at first order on
uniform elements and the CFIE needs about 1.5 times the EFIE's density. The
dispatcher therefore grades the generatrix of a conductor toward every chain
junction where the evaluated impedance jumps, with the 2D rule
(`impedance_jump_is_graded`, the law at the chain end at that frequency) and
the 2D levels (four, ratio 1/2: eight more elements per junction).
`_mark_impedance_junctions` marks the chain ends; one breakpoint function
(`_primitive_breaks`) serves the mesher and the element count, so preview,
near-pair count and solve see the same mesh, and each frequency record carries
`graded_impedance_junctions`. CFIE, `ka = 3.3`, against graded 240-element
references, 30 / 60 / 120 elements: PEC half | `100+50j` 0.20 / 0.11 / 0.055 dB
uniform, 0.058 / 0.015 / 0.004 dB graded (three levels 0.072 / 0.021 / 0.007,
five 0.050 / 0.011 / 0.002); zones `50 | 300+100j` 0.97 / 0.48 / 0.23 dB
uniform and 0.17 / 0.066 / 0.024 dB with three levels; the EFIE behaves alike
(0.15 / 0.074 / 0.036 to 0.057 / 0.015 / 0.004 dB). This had been refused by
the near angular rule, whose failure was a rounding floor (see Certification).
Only the conductor kind is graded: bare pieces of partial and banded coatings,
material junctions and sheets are not. The BoR HPC driver was run end to end on
a graded sphere (plan, then the headless worker): 32 elements, one graded
junction, IBC-CFIE, and 24 with a zero-ohm law. Around the first cavity resonance
(`ka = 2.70..2.80`, 60 elements) the EFIE departs from its neighbors by 1.5 dB
(PEC half) and 16.3 dB (taper); the CFIE stays within 0.0012 dB. Explicit
`formulation='efie'` remains a diagnostic option with no resonance guarantee.

Full PEC cores use a combined electric/magnetic boundary equation in the
coating medium, including corresponding cross-surface terms. In the direct
single-coating notation, the core row is
`[Tco + etaL*R*Pco, -eta0*Pco + eta0/etaL*R*Tco, -Tcc - etaL*Kcc]`,
where `R` rotates tangential test rows and `Kcc` includes the MFIE jump.
The full multilayer path uses the same core equation. Partial core surfaces
meeting material junctions retain their existing junction formulation.

Preview, certified dispatch and direct full-coating APIs require every inner
interface/core to lie strictly inside its enclosing surface, without crossing
or touching it. Same-surface separated pairs route to checked near integration
when the gap is below twice the larger panel length, as well as the existing
angular-resolution threshold. Preview memory includes these geometric pair
counts; a fixed index stencil alone cannot price tight grooves accurately.

## TE sheets

Adjacent impedance flags and virtual sheet regions assigned to segment names
no longer split a TE sheet trace. Each nonbranching sheet is oriented before
nodes are shared and before polynomial enrichment. Real material regions remain
distinct, and TM interface splitting is unchanged.

A TE sheet attached to a PEC body requires a coupled junction condition that
is not implemented. Such a joint now raises a specific error instead of
pinning both coincident nodes and returning a misleading certified field.
Model a physical fin as a closed, finite-thickness PEC contour. Branching TE
sheets are also rejected. Separated sheets and PEC bodies remain supported.

On a separated sheet + PEC scene every closed, consistently oriented TYPE 2
loop carries a second potential on its density: TM `S x + gamma*D x_pec`,
`gamma = sigma/(i*k0)`, and TE `D x + eta*S x_pec`, `eta = sigma*i*k0`, where
sigma is the panel side of the exterior; the sheet rows receive the conductor's
second potential through regular kernels and the far field projects both
potentials. For a PEC circle (a = 0.1 m) with a separated strip the condition
peaks of the former equations (3.6e5 at ka = 2.4057 TM, 3.3e4 at 3.054 TE,
9e4 at 3.833) fall to at most 121; fields agree with the former equations to
1e-4 of the amplitude away from resonance, and the error against the exact
cylinder (transparent strip) is unchanged. Open TYPE 2 strips, polynomial
bases and the compressed engine keep the single-potential equations, and the
solve then warns that the conductor is not resonance-protected.
The desktop co-polarized workflow requires both VV (TE) and HH (TM), so the
attached-sheet guard prevents that entire run, even though a standalone TM
solve is supported. The solver selection tooltip explains this limitation.

## 2-D to 3-D line phase investigation (2026-10-17)

Production now converts native version-2 B with `psi_tm_deg=180` and
`psi_te_deg=0`, consistently in low-level line expansion, Assembly and wing
expansion. The investigation below establishes this convention mapping;
it does not certify the local-line model for arbitrary features.

For current GHOST amplitude-version-2 coefficients, the convention conversion
is **TM: 180 degrees; TE: 0 degrees**. These are local section polarizations,
not fixed global radar channel names. `PSI_HH_DEG` presently holds the TM
setting because a 2-D file labels TM as HH; for a cylinder along global z
viewed broadside, local TM instead contributes to radar VV.

### Derivation

Both solvers use `exp(+j*omega*t)`, with outgoing `exp(-j*k*r)` and coming-from
look direction d. The 2-D kernel is `G2 = +j*H0^(2)(k*rho)/4`, and its stored
coefficient B omits that kernel prefactor. The standard outgoing 3-D scalar
kernel is `G3 = exp(-j*k*r)/(4*pi*r)`. Integrating G3 along an infinite axis
gives `-j*H0^(2)(k*rho)/4 = -G2`. This follows by substituting
`z = rho*sinh(t)` into the
[Hankel integral representation, NIST DLMF 10.9.11](https://dlmf.nist.gov/10.9.E11).
Consequently, extending the same scalar source density over length L gives
the broadside amplitude `-L*B/(4*pi)` in the infinite-cylinder approximation.

TM's scalar is axial electric field, so that minus sign is the electric-field
mapping. TE's scalar is axial magnetic field: the incident wave travels along
`-d`, while the scattered wave travels along `+d`. The Maxwell relation
`H = propagation_direction x E / eta` therefore adds another minus sign when
converting the normalized backscattered magnetic field to electric field in
the same radar basis. Thus:

```text
F_TM = -B_TM * line_integral / (4*pi)
F_TE = +B_TE * line_integral / (4*pi)
line_integral = integral exp(+2j*k*d.r(s)) ds
```

There is no complex conjugation and no additional universal 45- or 90-degree
offset. The Hankel spreading phase is already accounted for in the conversion.
The legacy defaults differ by -13.1 degrees (TM) and -9.2 degrees (TE).
Those residual rotations have no convention-based justification.

### Numerical evidence and limits

The local investigation compared analytic PEC-circle coefficients, actual
exported 2-D circle coefficients, and CFIE BoR finite cylinders at 1 GHz. Ten
finite-cylinder solves covered radii 0.15 and 0.30 wavelengths, lengths 2, 4,
8 and 16 wavelengths, seven aspects from 80 to 100 degrees, and three mesh
refinement pairs. All modal tails converged, all touching self-checks were
below 2e-8 (limit 1e-5), and linear backward errors were below 2e-13.

For radius 0.15 wavelengths, the absolute broadside phase discrepancy between
the BoR field and the derived 180/0-degree line mapping decreased with length:

| Length / wavelength | TM discrepancy | TE discrepancy |
|---|---:|---:|
| 2 | 1.101 degrees | 4.372 degrees |
| 4 | 0.367 degrees | 2.377 degrees |
| 8 | 0.112 degrees | 1.239 degrees |
| 16 | 0.028 degrees | 0.633 degrees |

The radius-0.30, length-16 case differed by 0.449 degrees TM and 0.166 degrees
TE. Doubling the mesh density changed the two length-4 cases by less than
0.033 degrees across all seven aspects. Refining the radius-0.15, length-16
case changed broadside phase by less than 0.001 degrees. Exported 192-node
2-D circles agreed with the analytic complex series within 0.020 degrees.
Independent numerical integration of G3 reproduced `-G2` within 5e-11 relative
error. Translation tests reproduced `exp(+2j*k*d.delta_r)` within 2.5e-9 for
BoR and 4e-16 for line expansion.

These comparisons validate the convention mapping, not a general finite-body
approximation. At 10 degrees off broadside, the radius-0.15, length-8 line
model still had phase discrepancies of about 15-16 degrees. The ordinary line
path here used the original-frequency broadside section coefficient; it did
not exercise the wing tool's optional reduced-frequency oblique correction.
Finite-end diffraction, curvature and body-feature coupling are separate
modeling issues and must not be folded into a universal phase constant.

The existing curved groove fixture illustrates the distinction. Using 180/0
instead of the legacy preset changed isolated-delta normalized complex RMS
from 0.354 to 0.489 for VV and 0.737 to 0.558 for HH; combined RMS changed from
0.392 to 0.494. Whole-body errors remained only about 1.4-3.5 percent, because
the clean body dominates. A small whole-body error therefore does not certify
the feature's phase. The preset happens to reduce one component of this
fixture's model error; that is not evidence for a universal convention.
Its existing regression allows 40 degrees of per-channel phase RMS, so passing
that test cannot establish the correctness of a 9-13 degree convention offset.

### Conversion and historical compatibility

`SeamCoefficients.convention` defaults to `native_2d_B`. Native 2-D file loaders
require the current B convention. For analytic or preconverted electric-line
coefficients, construct with `convention="electric_line"`; automatic expansion
then uses 0/0. `coefficients.as_electric_line()` changes TM's sign and marks the
result; calling it again does nothing. It retains the line-integral/(4*pi)
normalization. It does not rotate local polarizations into the radar basis.

Low-level and public field APIs use `psi_tm_deg=None, psi_te_deg=None` for
automatic conversion. Explicit numbers remain absolute rotations of the
supplied coefficients, so historical calls with 166.9/-9.2 reproduce their
previous behavior. To apply separate empirical calibration, first convert
with `as_electric_line()`, then supply the residual TM/TE rotations. Automatic
conversion and explicit conversion agree for co-pol and cross-pol fields.

New GUI assemblies use the native 180/0 preset. The legacy empirical preset
remains available. Recipe version 7 distinguishes these new defaults: loading
version 5 or 6 preserves explicit settings and materializes missing settings
as historical 166.9/-9.2, with a visible compatibility notice. Saved output
provenance records each line's input coefficient convention and applied angles.

Reproducible scripts and numerical JSON records from this investigation are
local, git-ignored files under `audit_codex_review/phase_*`.

## Certification

The certified BoR entry point checks the touching-pair integration diagnostics
on both the base and fine meshes. Failed, nonfinite or unperformed reported
self-checks reject certification even when mesh-to-mesh fields agree. Survey
solves retain their numerical warnings without claiming mesh certification.

Sharp-feature mesh grading is not a separate accuracy certificate. The 2-D
material-interface angular problem uses positive-real coefficients `1/|mu|`
and `1/|eps|`; for lossy or negative-index media this is a surrogate rather
than the full complex-material eigenproblem. BoR conical tips use the root
`P_nu_D(-cos(theta)) = 0` through the calibrated grading exponent
`min(1, 2*nu_D)`, capped at eight levels. Independent mesh-convergence checks
remain necessary for both choices.

Mesh refinement checks discretization changes; it cannot detect an equation
that converges to the wrong model. Quality gates check the solved linear system
and mesh evidence, not general correctness of every supported physical model.
The default certification tolerances have not been tightened as a substitute
for the formulation fixes. Informational mesh-cache/junction notices are
separate from warnings counted by the quality gate.

The certified result is the fine solve, but the comparison measures the
base-vs-fine change. With refinement ratio r and order p the published error is
about `delta/(r**p - 1)`. 2-D certificates now carry `estimated_published_error`
(and flat `estimated_published_complex_max_normalized` etc.) with the
conservative p = 1 (twice the change at r = 1.5); the limits still gate the
measured change. On the audit's edged bodies the true published error was
1.19-1.31x (square, p = 1.35) and 1.99x (strip, p = 1.0) the change; the
estimate is 1.01-1.68x the true error. The equivalent BoR record belongs in
`bor/dispatch.py` beside `realized_mesh_refinement`, with r the smallest
realized surface ratio.

The 2-D condition gate is frequency aware for the combined-coupling
formulations (dielectric two-density, combined Robin and regional potentials,
combined sheet + PEC). Their condition estimate grows like 1/(k L) below k L ~ 1
(cond*ka constant from ka = 0.1 to 1e-5: dielectric 300-840, coated PEC
34-95, PEC/IBC 13.5) while the error stays at the mesh level, so ka = 3e-4
dielectric runs (condition 2.5e6) used to fail the 1e6 limit with a 4e-4
error. The gate compares `cond*min(1, k_max*L)`, with L the half-diagonal of
the mesh bounding box and k_max the largest region wavenumber (a fictitious
resonance of a high-index region needs k_region*L of order 2, so it is never
relaxed), floored at 1e-6. Single-layer formulations and every system with
k_max*L >= 1 keep the raw limit. `condition_est_max` remains the raw estimate;
`condition_est_frequency_scaled_max` and the gate's `condition_est_gate_value`
report the compared value.

A reference mesh frequency can only maintain or refine the discretization:
the single-frequency path now includes both the solve and reference frequencies
when choosing the shortest material wavelength.

The near ANGULAR rule (sinh core plus Gauss-Legendre tail, orders raised until
two results agree to 2e-8) used to stop with "did not converge at the maximum
order" on very small elements, which is what had made BoR mesh grading
impossible. The cause was not the order: the sampled MFIE/IBC brackets form
`rho_p - rho_q*cos(xi)` and products such as `tr_p*tz_q - tz_p*tr_q*cos(xi)`,
and for two points `d` apart on one element those differences are of order
`rho*xi**2` and `d` while their operands are of order `rho`. Rounding then
leaves 2e-9 of the kernel at `d = 1e-5 rho`, 1e-8 at `5e-6 rho` and 3e-7 at
`1e-6 rho`, above the tolerance, in the native sampler and in NumPy alike.
`kernels._stable_brackets` writes every such term with `h = 2 sin(xi/2)**2`
(the `h**2` parts cancel analytically; `R**2 = d**2 + 2 rho_p rho_q h`), agrees
with the sampled forms to 1e-13 where those are accurate, and stays within 4e-10
down to `d = 1e-6 rho`. At 20 elements per wavelength the self and adjacent
terms of an equatorial element still converge at `ka = 400`; elements of `h/64`
now solve.

Since the 22 September audit the near rule is graded (a sinh core, ratio-2
geometric panels up to `s = 0.25` and one tail panel, each sized from its own
phase) and runs natively (`near_green_rule`, `near_brackets_rule`); the native
bracket rule evaluates every point with the stable forms at no extra cost, and
the NumPy fallback switches to them below `d = 1e-4 max(rho)`. A coarse/fine
comparison is accepted only when every order actually grew: the former check
could accept two identical saturated tail orders (4096) and returned kernels
wrong by up to 1.1e-3 at `d/a = 1e-8` without an error (526 of 920 points of a
tiny-element self cell above tolerance, worst 2.3e-2). The kernels now agree
with independent references to 4e-14 for `d/a` from 1e-9 to 0.5.

BoR self and adjacent meridian quadrature used a fixed graded rule until
17 October 2026 (quadtree cells toward the singular set, 4 x 5 Gauss points
per cell, depth `near_depth`; every touching pair now uses the polar rule
described below). Its self-term error was measured in the 22 September audit: the rule misses the
model integral of `log|s - s'|` by 8e-3 and leaves about 6e-4 relative
far-field error at 27 elements per wavelength, decreasing as O(h). A rule
graded in `|s - s'|` only (each triangle mapped to `u = |s - s'|` and the
along-diagonal coordinate, geometric intervals in `u`) removes that error (5e-6
on the model integral, 2-3e-6 in the far field) with a third of the points,
but against exact series for PEC (CFIE and EFIE, `ka` 1 to 10), impedance and
dielectric spheres at 10 to 80 elements per wavelength its end-to-end error was
not smaller at practical densities, and mostly slightly larger (ka = 3 CFIE:
0.0248 against 0.0199 dB at 20 elements per wavelength): the cell rule's
integration error partly offsets the flat-segment geometry error. It also cost
about 10% more, so the cell rule was kept then (see below for why it no
longer is).

Far element pairs (at least two element lengths apart), the plane-wave
excitation and the far-field projection use `kernels.FAR_GAUSS_ORDER = 3`
Gauss-Legendre points per element since 24 September (four before; the
`gauss_order` argument still selects any order). Against four points the
amplitudes moved 2e-9 to 2e-8 relative on PEC (CFIE and EFIE), impedance
(`75 - 20j`), lossy and lossless dielectric (`4 - 0.1j`, `2.56`) and coated
spheres at `ka = 5` and 20 elements per (material) wavelength, and the errors
against the exact series were unchanged to 1e-4 dB (0.005 to 0.02 dB); the
4 GHz ogive moved 2e-7 against a 9e-3 discretization error. Far tables shrink
by (3/4)^2 and the far build by about a third. Mirror-symmetric bodies stay
symmetric to 3e-8 (4e-10 with four points): element pairs on the near/far
routing boundary can be routed differently on the two sides by rounding, and
the far rule's error on those pairs is what shows.

Element pairs of two different surfaces that touch at a junction point had
their own rule (`JUNCTION_CELL_ORDER = 10`, `JUNCTION_CELL_DEPTH = 7`, replaced
by the polar rule below). The
order-4, depth-4 cells they used before left 1.2e-3 relative far-field error at
partial, layered-patch and banded junctions against an order-12, depth-8
reference; the refined rule is within 2e-5 and costs nothing measurable (a
junction has a few such pairs). A vacuum coating, which must be invisible,
reproduces the exact PEC sphere as well as or better than before (ka = 3, 24
elements per hemisphere: 0.0095 dB, the plain PEC mesh's error; ka = 1:
unchanged). Reverse cross-surface operators are derived from the forward ones
by reciprocity, which is exact only when both directions use the same rule;
comparing the derived and independently integrated reverse operators is how
the junction error was found.

Sharp meridian corners use a dedicated rule (`bor/sharp_pairs.py`). At a small
included angle the kernel is nearly singular along the whole element pair, and
on thin steep elements near the axis the modal kernel changes character within
one cell; the fixed order-4, depth-4 cells left a mesh-independent error that
extra depth could not remove (MFIE block error 2.5e-2 at a 15 degree rim, 0.50
at 1 degree; needle-tip self pairs up to 1.0). Touching pairs with an included
angle under 60 degrees, and steep elements within 30 degrees of the axis with
2*rho_max/L < 8 (their self pair and both neighbours), are integrated with a
polar rule graded dyadically toward the complex pole and the ring scale rho,
with z taken relative to the touching point. Blocks agree with a deeper
reference to 6e-8 (rims) and 7e-7 (needles) down to 0.25 degrees, and the rule
is reciprocal to 1e-12.

Since 17 October every touching pair uses that rule: self, adjacent and
junction pairs of ordinary bodies too, so no pair keeps the fixed cells (the
former per-pair assembly fallbacks, which were unreachable, are removed).
Against a deep reference (32 points per piece, every graded stop four times
deeper, ring stops graded for the mode cap) the largest relative block error
per category, modes to 8 / 30 / 60, went from the cells' to:

| pairs | fixed cells | polar rule |
| --- | --- | --- |
| sphere self/adjacent | 3e-3 / 1e-2 / 2e-2 | 5e-9 / 5e-9 / 1e-8 |
| corners 30..150 deg (incl. self) | 2e-3..5e-3 / to 2e-2 / to 4e-2 | 4e-8 max |
| collinear (180 deg) | 6e-6 / 4e-5 / 2e-4 | 9e-8 / 2e-9 / 3e-9 |
| sphere pole, cone tips, disc centre | 0.2..0.3 / 0.6 / 0.7 | 1e-8 / 3e-8 / 3e-8 |
| junctions 45..180 deg (rotated PV) | to 8e-5; 3e-4 beside an L/8 element | 9e-8 max |
| thin junction (steep, near the axis) | 1e-6 / 1e-5 / 6e-5 | 5e-9 |

(maxima over 10 and 40 elements per wavelength, air, lossy `3 - 0.1j` and
high-contrast `10 - 1j` media for the 8-mode column; EFIE, MFIE and IBC
brackets). Self pairs use 16 Gauss points per piece, touching pairs 12; ring
stops deepen above `m_max` = 16 (self) and 8 (touching). A self pair whose
ring is at least an element length away is integrated by the two triangles
about the diagonal in `|s - s'|` rather than about its end: same accuracy,
40% cheaper near kernels (fewer samples at tiny separations). Junction pairs
are classified like same-surface ones, so a steep element near the axis at
a junction is `thin` by its geometry, whatever the angle.

Ordinary results move by the old rule's error: 1e-4 to 1e-2 dB on spheres
(0.025 dB on the coarse 30 degree cone-sphere), and against the exact series
at a fixed coarse mesh not always toward it, as the 22 September audit found:
ka = 1 PEC and the `ka = 1.5` dielectric sphere get closer (0.0157 to
0.0149 dB, 0.0570 to 0.0472 dB), ka = 3 PEC, impedance and coated spheres
farther (0.0068 to 0.0096 dB at 48 elements), ka = 10 is unchanged. Under
refinement the new errors fall by 4.0 per halving and Richardson
extrapolation lands on the series (ka = 1, 3, 10 PEC: 1e-6, 4e-6, 1e-5 dB;
impedance 7e-5; dielectric 1e-5; coated 8.5e-4 from two meshes), where the
cells' did not (5e-5, 5e-4 with a sign change of the error between 96 and
192 elements, 8e-6; 4e-4; 3e-3; 5e-3): the cells added an error of lower
order that partly cancelled the facet error on practical meshes (on the
cone-sphere, cylinder and flat-based cone the change halves per halving of
the mesh: 0.025, 0.012, 0.006, 0.003 dB). Near
preparation dominates small solves, and the change costs 12-32% of the solve
time (sphere ka = 10 at 320 elements +12%, coated sphere +32%), the self-check
included.

Every solve re-integrates a sample of the touching pairs with a finer rule
(`CHECK_POINTS` more points per piece, every graded stop one level deeper):
every fourth acute or thin pair, every sixteenth regular one
(`REGULAR_CHECK_STRIDE`), and the most severe pair of each class (smallest
angle; smallest ring ratio, which for regular pairs is the axis pole), in the
same batch as their production blocks, in process workers too. The result is
`near_quadrature['touching_pairs']` (`touching_pair_quadrature` for material
layouts) and, when acute or thin pairs exist, `near_quadrature['sharp_pairs']`
(`sharp_pair_quadrature`); a difference above 1e-5 raises a warning.
Production measures 1e-9..3e-8. The EFIE near-block reciprocity diagnostic is
now rounding (1e-15): the rule is reciprocal by construction. Included meridian corners
below 0.25 degrees, the smallest angle validated, are rejected before solving
(1e-6 degree rounding tolerance). A generatrix node other than the two end
points lying on the axis is rejected too: an element on the axis sweeps no
area and the system loses rank (LAPACK info = 1, condition 1e37 on graded
needle tips whose inner nodes the mesher had snapped onto the axis within
1e-6 of the body size). Kept off the axis, needle and cone tips graded by
ten levels (innermost radius 1e-8 wavelengths) solve without warnings and
agree with four levels to 0.002 dB. What remains at a thin rim is ordinary first-
order mesh convergence (the edge current is singular), about 0.1 dB at N = 40,
which mesh certification measures.

## BoR material interfaces: weighted region equations

Every BoR material interface combines the equations of its two regions. PMCHWT
adds them with equal weights. The continuous PMCHWT equations have no real
resonance, but on a lossless interface the coarse discrete sum had spurious
singularities at isolated real frequencies: a 20-element
`eps_r = 3` sphere of radius 0.1 m (10 elements per medium wavelength) reached
a condition estimate of 2e8 at 0.95127 and 1.09777 GHz, with +6.7 dB axial and
10 dB off-axis error, a 3e-13 residual and a band about 1e-4 wide; the same
mesh cost a coated sphere 8.4 dB. The near-null vector is a `phi`-directed
node-to-node alternation at the two poles in `|m| = 1`. Its charge,
`j*m*J_phi/rho`, involves no derivative, so its discrete charge is as small as
a smooth current's; the exterior operator is then net capacitive on it and the
interior one net inductive (`-4.815e-3j` and `+4.823e-3j`), and equal weights
let them cancel. Using the exact axis relation (`t_rho` instead of its sign) or
refining the self/adjacent quadrature only moves the crossings by about 0.1%.
That rules those two repairs out as cures; it does not show that no other
discretization of the pole would remove the mode. The artifact weakens under
refinement (largest condition estimate 2e8 at 20 elements, 4e6 at 40), as a
property of the discretization should. What follows is therefore a numerical
stabilization of the discrete system, not a correction of the formulation.

The regions' equations are therefore added with unit-modulus weights
`c_R = exp(-j*15 deg * rank_R)`, regions ranked by `Re(eps_r*mu_r)` and the
exterior normalized to 1 (`_region_equation_weights`). Mautz and Harrington's
uniqueness condition constrains the coefficients `alpha` and `beta` that
multiply ONE region's electric and magnetic equations: `alpha*conj(beta)` real
and positive (PMCHWT has `alpha = beta = 1`). Both equations of a region share
its weight here, so that product is `|c_R|**2 = 1` whatever the phase; the
condition holds, and the phase between regions is free to make the reactive
cancellation impossible. (An earlier version of this section stated the
condition for the product of two regions' weights, which a relative phase would
violate; Codex's review corrected it.) Unequal weights no longer cancel the identity terms, so each
interface row receives `+sigma_R*c_R/2 * <W, n x M>` (electric) and
`-sigma_R*c_R/2 * <W, n x J>` (magnetic). The denser region takes the negative
phase because loss rotates its term the same way: with the opposite sign a
lossy medium can restore the cancellation (condition 1e5 at `2-2j`, +30 deg),
with this one loss only improves it. Regions of equal `Re(eps_r*mu_r)` keep
equal weights; their reactive parts have the same sign.

Against the exact series over 0.8 to 1.6 GHz (19 frequencies, 5 aspects, both
polarizations; maximum / median error in dB and largest condition estimate):

| Body | PMCHWT | Weighted |
|---|---|---|
| dielectric 3, 20 elements | 10.24 / 0.077 / 2e8 | 0.17 / 0.064 / 3e4 |
| dielectric 3, 40 elements | 0.045 / 0.016 / 4e6 | 0.043 / 0.015 / 8e4 |
| dielectric 3-0.3j, 40 | 0.186 / 0.030 / 1e5 | 0.173 / 0.031 / 6e4 |
| dielectric 10, 40 | 0.470 / 0.072 / 2e7 | 0.472 / 0.077 / 7e4 |
| dielectric 0.5, 40 | 0.023 / 0.006 / 4e7 | 0.023 / 0.006 / 2e5 |
| coated 3, 20 | 8.44 / 0.185 / 3e8 | 0.37 / 0.161 / 3e4 |
| coated 3, 40 | 0.083 / 0.034 / 4e6 | 0.085 / 0.035 / 9e4 |
| two layers 4 over 2, 40 | 0.079 / 0.038 / 4e6 | 0.077 / 0.039 / 2e5 |
| two layers 2 over 4, 40 | 0.093 / 0.033 / 4e6 | 0.097 / 0.034 / 2e5 |

The condition column is the largest estimate over each sweep: the weights
bound the worst case, they do not lower the estimate at every frequency.
Frequency by frequency on 20-element spheres (`eps_r` from 0.5 to 10, five
frequencies each, 45 cases) the weighted estimate is lower in 31 cases and
higher in 14, all but one of those at `|eps_r - 1| <= 0.1` and by up to 11
times (`eps_r = 0.99` at 0.7 GHz: 3.9e4 to 4.3e5, found by Codex), with the
error unchanged there (0.282 to 0.277 dB). The largest weighted estimate of
that survey is 4.3e5 against 2.3e8.

The identity terms give the rows a second-kind part, which costs a few
thousandths of a dB in the median where PMCHWT was healthy (15 degrees is the
smallest angle tried; 90 degrees doubles that cost). The same weights apply to
the dielectric, coated, partial-coating and multi-region solvers and to the
dense, streamed and compressed paths. A coarse mesh still has coarse-mesh
accuracy (0.17 dB at 10 elements per wavelength); the default 20 per medium
wavelength is unchanged.

Neither form resolves a vanishing contrast. The scattered field is then a small
difference of order-one equivalent currents, and the relative error grows like
`1/|eps_r*mu_r - 1|`: on the 20-element sphere at 0.7 GHz, 0.04 dB at
`eps_r = 1.1`, 0.28 dB at 1.01 and 12 dB at 1.0001 (exact RCS 8.6e-11 m^2;
10.3 and 8.5 dB at 40 and 80 elements, 9.6 and 8.0 dB weighted). The sphere
comparisons above do not establish accuracy in that regime.
Codex's review added magnetic and lossy media (`mu_r` up to 3, `3-3j | 1-1j`:
0.006 to 0.056 dB at 48 elements) and three-layer coatings (0.005 and 0.008 dB
at 40 elements per surface) to the evidence for the weighted form.

## October 2 solver review changes

Fixed-reference 2-D sweeps now retain the complete request's sizing-frequency
set across frequency-local execution and checkpoint resume. Both polarizations
and base/fine certification phases retain their own mesh topology; matrices
and factors still live for one frequency. Low references and dispersive
materials can therefore produce a finer mesh than earlier canonical sweeps.

Run forecasts use digest keys and a 16 MiB byte budget. Reused checkpoints
verify their digest/header before solving and decode their sample arrays once
at final merge. Failed checkpoint writes preserve computed results within a
bounded allowance; if further work cannot fit, the GUI marks the result
partial and does not automatically export it. BoR desktop sweeps now use
frequency checkpoints with BoR source, material and option identities.

LU fallback checks remaining host and solve-reservation memory before copying
the original matrix. Failed factor construction tracebacks are released before
replacement. Batched compressed GMRES retires completed columns independently.
Original-matrix residual gates remain unchanged.

BoR `quadrature_check='refine'` is an optional same-mesh comparison, exposed as
"Compare refined integration" in the desktop BoR options. It increases the
self/adjacent and cross-surface junction grading depth by one, compares each
frequency/channel's complex fields at requested angles, and returns the
refined result only when maximum and RMS normalized changes are at most
0.002 and 0.001. It adds a second solve and is off by default. This checks
integration sensitivity, not continuous angular interpolation or the accuracy
of the drawn piecewise-linear shape. Refined integration need not reduce total
error when integration and faceting errors previously offset each other.

Far-block ACA now checks spread rows and up to three random unused rows before
accepting its stopping estimate. A failed probe becomes a new pivot. This is
stronger sampled evidence, not a deterministic full-block error certificate.

Backend timing reuse retains the conservative initial cost prior. Repeated
paired measurements may also inform a nearby request on the same geometry,
materials, source, host and execution settings, subject to tight frequency,
DOF and angular-grid bounds, stable timing samples and an uncertainty margin.
Normal solves do not perform speculative extra backend solves for calibration.

Earlier component benchmarks were measured separately from complete solves;
their speedups are not claims of the same improvement in complete solve time.
The published [measurement summary](../../experiments/solver_review_20261002/results.json)
records the later airfoil and subtraction comparisons, their scope and the
status of research prototypes. Local baseline copies and raw benchmark logs
are not part of the published repository.

The subsequent 2-D sweep improvements retain admission-sampled compressed
tiles within a 32 MiB run cache. Reuse requires identical geometry, evaluated
materials, polarization, quadrature settings and row/column DOF lists. Cached
tiles enter the normal assembly queue so error-bound sums keep their original
order. A completed polarization partner supplies an exact payload size instead
of being sampled again. Run-owned spawned assembly workers refresh their
coefficient data between operators and close on completion or cancellation;
idle worker allowances remain in the memory forecast during factorization.

Desktop checkpointed sweeps use up to two frequency workers when CPU, RAM and
predicted work permit, with nested assembly pools disabled. Expensive units
run first and final samples retain request order. CPU reservations and worker
interpreter memory are included in admission. This changes scheduling, not
mesh certification or the equations solved.

The stage cost model separates near/far assembly, compression, factorization
and RHS work. Nearby calibration requires the same geometry, materials,
source, host, options and factorization regime, compatible observed meshes,
at least three stable samples for each backend and an uncertainty margin.
Bounds are 15% in frequency, 20% in DOFs and angle count, and 50% in stage work.
Fallback or repeated-factorization runs do not train the stage model.

`compressed_far_method='verified_cur'` is experimental and off by default.
It proposes factors from selected exact rows and columns only for separated
support boxes. Every coefficient is still evaluated to validate the proposal;
its complete error enters the existing row/column error bounds. Rejected
proposals fall back to ordinary QR compression. This is not a sampled-only
error certificate or an asymptotically faster production assembly path.

Each frequency builds its own operator and inverse. The experimental inverse
cache has been removed. Older execution profiles with
`frequency_preconditioner='off'` are normalized by dropping the retired field;
explicit requests to enable it are rejected with a migration message.

Native fixed-width far-kernel loops preserve the arithmetic order. The earlier
frequency-scheduling comparison is included in the published measurement
summary. Native-only component measurements and rejected local experiments
do not establish complete-solve improvements.

On the measured Windows host, certified `airfoil.geo` sweeps at 1, 1.5, 2,
2.5 and 3 GHz with 181 angles and the same four-CPU/12-GiB budget averaged
40.765 seconds sequentially and 32.845 seconds with two frequency workers
(two runs each, 19.4% less elapsed time). Meshes matched and the largest
complex-field change, normalized by each channel's peak, was 4.92e-13.
One certified 10 GHz before/after pair measured 94.279 and 91.296 seconds
(3.2% less time); memory sampling fell from 3.668 to 2.017 seconds and the
largest normalized field change was 3.96e-14. That pair used the same optimized
native binary on both sides to isolate the Python/backend changes. The 3 GHz
dense solve's internal time was essentially unchanged. Parent-process peak
working set was essentially unchanged; simultaneous worker-inclusive peak RAM
was unavailable, so these measurements establish no RAM saving. The complete
1-18 GHz sweep has not been timed, and component or small-sweep gains should
not be extrapolated to it.

The optional inverse-reuse experiment did not improve the measured cases.
A fixed-mesh coated case at 1 and 1.01 GHz (384 panels, 181 angles) avoided two
factorizations but required extra GMRES work: 4.244 seconds without reuse and
4.692 seconds with reuse, with 16.1 MiB more peak working set. Its largest
peak-normalized field change was 3.14e-12 and physical backward error remained
below 2.86e-15. The fixed-mesh airfoil case at 3-3.1 GHz produced inverses
larger than the 128 MiB cache; the redundant reuse run was stopped and is not a
timing comparison. This evidence supports keeping the option off by default.

The latest production changes widen the sampled column-space proposals used
by ordinary QR tile compression and trim their projected rank before storage.
Every accepted proposal is checked against the complete original tile at the
existing tolerance; its full coefficient error still enters the operator's
error bounds. Rejected proposals fall back to the existing compression path.
This enabled optimization is separate from the optional `verified_cur` path.
Regional coefficient assembly also prepares shared index maps once per
operator and reuses selections within each tile query, avoiding redundant
whole-mesh mapping work. It does not reuse meshes or coefficients across
frequencies or change material routing, geometry or certification thresholds.

Fresh-process certified airfoil comparisons used 181 angles, both
polarizations, four CPU threads and a 12 GiB budget. Relative to the already
optimized backend, the 3 GHz median changed from 15.31 to 15.21 seconds
(essentially unchanged), the 10 GHz median from 92.12 to 79.37 seconds
(13.8% lower), and the 18 GHz measurement from 236.75 to 196.12 seconds
(17.2% lower). The 3 and 10 GHz figures use two runs per version; 18 GHz uses
one paired measurement. Assembly time fell about 23-29% at 10 and 18 GHz,
while sampled aggregate process-tree peak working set fell about 1.5%.
Summed working sets may double-count shared pages; this is a sampled process
metric, not a measurement of unique physical RAM consumption.
All certification checks passed, and the largest whole-airfoil complex-field
change was below 1.3e-13 of the corresponding pattern peak. These are separate
single-frequency comparisons; their gains must not be added to earlier
measurements or extrapolated to a complete frequency sweep.

A one-time coherent-subtraction comparison found quadratic/cubic changes
below 0.066% of the feature-response peak for standard PEC grooves at 3, 10
and 15 GHz, and 0.683% for a synthetic 0.01-inch groove at 10 GHz. A finer
cubic mesh changed the latter response by 0.236%. After the production
optimizations, that narrow-gap subtraction was unchanged in VV and changed
by at most 8.04e-11 of its feature peak in HH. These are observed comparisons,
not error bounds for other geometries or individual deep nulls. The standard
coupons have frequency-designed remote backing, so their three frequencies
do not constitute a sweep of one fixed coupon. The published
[results](../../experiments/solver_review_20261002/results.json) and
[subtraction plot](../../experiments/solver_review_20261002/subtraction_comparison.png)
record this evidence; the production subtraction workflow is unchanged.

Same-frequency quadratic/cubic assembly reuse remains a small dense research
prototype: its setup-inclusive comparison measured a 27.1% reduction, but
does not establish a gain for the full compressed, certified airfoil solver.
Directional far-kernel interpolation is also a prototype, compared with
direct SciPy kernels rather than GHOST's native assembled operator.
Higher-order BoR remains a design proposal without an implementation or
measured saving. None of these three research directions is enabled in the
production solver.

## Surface-wave and geometry prerequisites (October 2026 follow-up)

Mesh certification checks actual polynomial intervals per bound wavelength
for both opaque impedance boundaries (TYPE 2/4) and transmitting impedance
sheets (TYPE 1). With the `exp(+j omega t)` convention and transverse decay
`exp(-kappa y)`, opaque boundaries use
`kappa/k0 = -j eps_r Zs/ETA0` (TE) or `-j ETA0 mu_r/Zs` (TM).
The two-sided sheet operator in air instead gives `-2j Zs/ETA0` (TE) and
`-j ETA0/(2 Zs)` (TM). Thin dielectric layer models keep their separate laws.
The physical constants must satisfy `Re(kappa) > 0` and
`abs(Im(beta)) <= abs(Re(beta))`, where
`(beta/k0)^2 = eps_r mu_r + (kappa/k0)^2`. Testing these conditions before
accounting for a complex material index can select the wrong bound waves.

Automatic sizing remains capped at 16 times the material-based density.
Explicit panel requests remain explicit. A mesh that does not resolve the
bound wave can be used for a survey with an advisory, but cannot receive a
mesh-convergence certificate merely because two coarse fields agree.
The normal 5% panel-rounding allowance still applies to the nominal
20 polynomial intervals per wavelength. The new finite-strip regressions
compare certified fields against an independently refined numerical field;
they do not establish an analytic error bound for arbitrary geometries.

Shared geometry and mesh welding now require every member of an endpoint
cluster to lie within the physical 1 nm tolerance of every other member.
Rounded spatial cells only locate candidates. Distinct points in the same
cell and chains of nearby points cannot silently collapse a longer element.
BoR junction construction follows this same tolerance; radius snapping uses
the existing numerical axis-zero tolerance rather than drawing proximity.
Both geometry validation searches bound their broad-phase work and support
cancellation. Winding area sums use coordinates relative to a contour vertex.

Direct BoR regional descriptors require one air exterior, unique valid
boundary references, opposite signs across dielectric interfaces, and one
material side for conductor surfaces. Invalid incidence is rejected before
operator construction. Dielectric IDs 900000 and above are reserved for
solver-owned virtual sheet regions; user dielectric IDs must be 1–899999.
This prevents a material definition from being silently interpreted as air.

## Running tests

Install the `test` and (for desktop coverage) `gui` extras, then run
`python ghost_backend/tests/run_suite.py`. This runs the headless suite together
and Qt-related modules in separate interpreters, continues after a failed group,
and exits nonzero if any group fails. This separates Qt module lifetimes from
numerical workers and completed successfully on Windows. The root cause of the
original order-dependent native abort has not been proven.

The new audit regressions cover the exact narrow cylinder-resonance inputs,
sheet segmentation/orientation, compressed/dense coefficients, conservative
mesh references, BOR storage and planning, mode retries, and grouped FFT
coefficients and complex fields.
