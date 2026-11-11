# P1 contract proposal: exact 1D piecewise-linear Poisson step

**State: independently reviewed and approved for this bounded implementation. The API is implemented and its software validation is recorded in `todo.md`; no new PDE benchmark was run.**

## 1. Decision and intended value

This is a **GO only as a narrow, independently replayable engineering certificate/oracle** for the exact 1D model below. It is a **NO-GO as a claimed new general P1 estimator, new convergence theorem, or research novelty claim**. Standard finite-element assembly already gives the formulas; novelty is not supplied by converting them to `Fraction`s.

The limited value over existing code is an end-to-end source-of-enclosure path: exact source, mesh, current state, and correction values determine the continuous weak-form directional quantities, rather than asking a caller to provide `c/q` centers and radii. This would independently exercise the boundary between analytic continuous integrals and the existing finite-dimensional Ritz verifier. The value is worthwhile only if used as a trusted small oracle for the float64 estimator and future enclosure work. If the project needs only such cross-checks and no production API, the narrower choice is a test-only oracle; do not add a public solver feature.

## 2. Mathematical contract and signs

Supported problem:

\[
-u''=f\quad\text{on }(0,1),\qquad u(0)=u(1)=0,
\]

with weak form

\[
a(u,v)=\int_0^1u'v'\,dx=\ell(v)=\int_0^1fv\,dx,
\]

and energy

\[
J(v)=\tfrac12a(v,v)-\ell(v).
\]

The current state `u` and direction `phi` are continuous, conforming, piecewise-linear functions on the same finite mesh and satisfy homogeneous endpoint values. The source is piecewise constant on those cells. Define

\[
c=\ell(\phi)-a(u,\phi),\qquad q=a(\phi,\phi).
\]

For `q>0`, the exact line minimizer is `alpha=c/q` (signed), and

\[
J(u)-J(u+\alpha\phi)=\alpha c-\tfrac12\alpha^2q=\frac{c^2}{2q}.
\]

For `c>0`, `alpha>0`; for `c<0`, `alpha<0`. Both give the same positive exact decrease. If a downstream policy forbids negative coefficients, it must reject/flip the direction explicitly; this certificate must not silently take `abs(c)` while retaining the original `phi`. If `c=0` and `q>0`, return a verified zero-gain record with `alpha=0` and gain `0`. If `q=0`, the direction is zero in `H_0^1` (given zero endpoints), so reject it; do not divide or report a positive certificate.

## 3. Exact input semantics and supported domain

Proposed public input is the existing `P1Poisson1DData` plus state and correction nodal arrays of equal length. Contract requirements:

- Domain endpoints are exactly the binary64 values `0.0` and `1.0`; node coordinates are finite, strictly increasing binary64 values. Interpret each input bit pattern as its exact IEEE-754 binary64 dyadic rational. Thus the mesh is the mesh encoded by those exact dyadics, not by an intended decimal spelling such as `0.1`.
- There is one finite binary64 cell-source value per interval. Interpret each bit pattern as its exact dyadic rational; it is constant on that cell.
- `u` and `phi` are finite binary64 nodal values, interpreted exactly as dyadic rationals. First and last values must be exactly zero as numbers. The piecewise-linear interpolants are the mathematical functions used by the certificate.
- No quadrature rule, float64 dot product, neural training history, or caller-declared `delta/epsilon` is accepted as evidence. The routine derives exact `c` and `q` from the canonical raw input.
- Unsupported: nonzero Dirichlet data, nonuniform coefficient `k(x)`, non-cellwise-constant source, higher-order or discontinuous trial functions, nonlinear operators, multidimensional domains, arbitrary callbacks, and non-finite values. These must produce a typed fail-closed result, never an estimate labeled verified.

The binary64 interpretation is intentional: the certificate is exact for the supplied bits. It does not certify that those bits equal an intended real-valued decimal input or a physical measurement.

## 4. Cell formulas and an independent objective check

For cell `I_i=[x_i,x_{i+1}]`, let `h_i=x_{i+1}-x_i>0`, `du_i=u_{i+1}-u_i`, and `dphi_i=phi_{i+1}-phi_i`. Since the interpolants have constant derivatives on the cell and `f_i` is constant,

\[
c_i=f_i h_i\frac{\phi_i+\phi_{i+1}}2-\frac{du_i\,d\phi_i}{h_i},\qquad
q_i=\frac{(d\phi_i)^2}{h_i},\qquad
c=\sum_i c_i,\quad q=\sum_i q_i.
\]

Every operation above is over exact rationals reconstructed from the input binary64 bits. A second calculation must independently form `J(u)-J(u+alpha phi)` from exact cell integrals and require equality to `c^2/(2q)`; this guards against a shared error in the line-step identity implementation. This is a finite exact identity check, not a numerical convergence check.

## 5. Record, hash, and replay protocol

Proposed record fields (schema and algorithm IDs versioned separately):

- canonical input payload: nodes, cell-source values, `u`, `phi`, each represented by fixed-width binary64 hex; endpoint/domain and sign convention identifiers;
- SHA-256 of the canonical input payload, including schema/algorithm identifiers;
- canonical exact rational `c`, `q`, `alpha`, and gain as reduced `[numerator, positive_denominator]` pairs;
- actual exact-rational operation count, bounded by the cap and independently recomputed during replay;
- result status/reason and the declared supported-model tag.

Serialization must use a fixed UTF-8 JSON encoding, sorted keys, compact separators, no NaN/Infinity, no duplicate keys, and canonical reduced rational pairs. `-0.0` may be retained in the raw-input hash to preserve provenance, while exact arithmetic interprets it as rational zero. The verifier reparses and canonicalizes input, recomputes all cell values and hashes, recomputes the independent exact energy difference, and compares every derived field. A serialized `status` or hash alone is not proof. Verification does not update a `SolverState` or accept arbitrary state mutation callbacks.

Any future record schema/algorithm change gets a new identifier; old records are rejected or handled by a separately versioned verifier. Hashes provide integrity/binding, not authenticity or proof of input provenance.

## 6. Resource contract and failure behavior

Enforced initial caps:

| Resource | Proposed limit |
|---|---:|
| mesh cells | 4,096 |
| exact rational arithmetic operations | 200,000 |
| maximum numerator/denominator component size | 16,384 bits |
| record UTF-8 size | 4,000,000 bytes |
| JSON nesting depth | 64 |
| result reason text | 256 characters |

The exact operation-count bound for the present formula path is `36 * cells + 11` (14/cell for `c/q` assembly, 20/cell for two independent energies, 2/cell for the trial, plus 11 fixed operations); at 4,096 cells this is 147,467, below the 200,000 cap. Implementation review must also account for parser memory and preflight work. Count/byte/depth/input-bit checks must precede expensive parsing or `Fraction` construction where feasible. Every intermediate rational operation increments a deterministic counter and checks component-bit limits. If bounds cannot be guaranteed from these caps, lower the caps or reject before work.

Results distinguish at least `verified_positive_gain`, `verified_zero_gain`, `invalid_input`, and `unsupported_resource_limit`. For any invalid/unsupported case, derived certificate values are absent; the implementation does not return a partial positive result. Unexpected arithmetic/parser failures map to a fixed fail-closed reason without embedding attacker-controlled payloads. The function is pure with respect to solver state.

## 7. Differentiation from existing APIs

| Existing component | What it already does | Contract difference | Boundary |
|---|---|---|---|
| [`ritz_certificate.py`](../src/problems/ritz_certificate.py) | Exact-rational verification of a finite SPD quadratic assembled as a matrix; recomputes discrete `c/q`, validates caller-declared enclosure centers/radii, and verifies a discrete trial/decrease. It deliberately does not validate enclosure provenance or certify a continuous PDE. | This contract starts from mesh/source/nodal function bits and derives the **continuous** weak-form `c/q` directly by cell integration; it has no caller-supplied enclosure radii. A finite matrix result may be cross-checked downstream, but cannot replace the continuous integration step. | It certifies only this restricted 1D continuous energy identity, not a general PDE or solver acceptance policy. |
| [`polynomial_poisson_1d.py`](../src/problems/polynomial_poisson_1d.py) | Exact rational assembly/replay for global rational polynomial forcing/basis data on `(0,1)`, including a correctly rounded float64 discrete problem and separate continuous/dyadic targets, with explicit degree/dimension/work limits. | A mesh P1 function is generally only piecewise polynomial and has derivative jumps at interior nodes; the current global-polynomial basis schema does not encode arbitrary FE knot locations as such. This contract assembles directly from local cells and binary64 dyads. | Both are narrow exact analytic families. Do not claim this contract generalizes or supersedes the polynomial family. If existing code can be cleanly generalized without weakening its proof, prefer that reuse over duplicated arithmetic. |
| [`estimators_1d.py`](../src/estimators_1d.py) | On the same zero-Dirichlet, piecewise-constant-source, conforming P1 setting, evaluates an equilibrated-flux formula with exact cell-integration expressions in float64 and returns an `eta_eq` intended to equal the H1-seminorm error in exact arithmetic. It explicitly labels the result `numerical_bound`, not an outward-rounded interval. | This contract answers a different query: exact directional residual `c`, curvature `q`, optimal signed step, and exact energy decrease for a supplied correction. It derives them in rational arithmetic from the exact input bits and emits replay evidence. The existing estimator neither computes this candidate gain nor proves a rounding enclosure. | Scope overlap is high. The new value is an exact oracle/provenance path, not a “better equilibrated estimator.” Compare outputs in tests; do not silently promote `numerical_bound` to verified status. |

**Novelty/value conclusion:** mathematically, the cell formulas and line minimization are standard; the implementation concept is not a new theorem. It is distinct from the existing APIs because it closes a narrow *input-to-continuous-gain provenance* gap that neither the arbitrary-matrix verifier nor the float64 error estimator closes. The bounded contract has practical value only as an independently replayed oracle/fixture. Treat any stronger contribution as **NO-GO** unless a separate proof shows a genuinely new bound or cost advantage beyond exact FE assembly.

## 8. Modular integration point (if this contract is approved)

- Keep the implementation in a role-specific module such as `src/problems/p1_poisson_1d.py`; export the data/result/build/verify names from `src/problems/__init__.py`. Preserve the repository's direct `core`, `problems`, `enrichment`, and `strategies` layout; do not create an umbrella `cem_framework` package.
- Accept/reuse the validated `P1Poisson1DData` contract but re-read its stored arrays as exact binary64 bits; do not consume its float64 estimator result as an interval input. State/correction validation belongs to the certificate module.
- Keep the certificate independent of `CEMSolver.step_pool`, candidate selection, and mutation. A later adapter may compare this result against `RitzCandidateScorer` and `EquilibratedFlux1DEstimator`; it must retain distinct fields/statuses and no implicit acceptance.
- Keep the global-polynomial record schema separate unless a refactor establishes that both families share one exact, reviewed canonical integral layer without widening either accepted input domain.

## 9. Acceptance review and implementation evidence

The independent review approved the mathematical scope. The implementation was checked against these acceptance points on the final source state:

1. A test oracle implemented independently from production helpers using `fractions.Fraction` and direct exact energy evaluation.
2. Uniform and nonuniform meshes; exact endpoints; positive and negative `c`; `c=0`; `phi=0`; nonzero end values; malformed lengths; non-finite values; duplicate/decreasing nodes; and boundaries at extreme allowed dyadic spacings.
3. Signed zero, smallest subnormal, and largest finite binary64 inputs, plus cancellation and intermediate-bit/operation/record-size/depth caps. Signed-zero coverage compares `-0.0` and `+0.0` domain endpoints: their exact rational outputs agree, raw-bit input hashes differ, and the negative-zero endpoint record replays. The largest-finite regression uses `f=[max_float,max_float]`, nodes `[0,0.5,1]`, `u=0`, and `phi=[0,1,0]`; it compares exact `c`, `q`, signed `alpha`, gain, and direct energy decrease against the independent `Fraction` oracle. Cap failures must be deterministic and fail before unbounded work.
4. For each positive-curvature case, exact equality between assembled formula and independent direct `J(u)-J(u+alpha phi)`; positive decrease for either sign of `c`; zero gain only for `c=0`.
5. Canonical serialization, duplicate-key/noncanonical-input rejection, altered input/hash/output rejection, deterministic replay, and proof that failed verification never mutates inputs or solver state.
6. Cross-check the float64 equilibrated-flux estimator on an analytically solvable P1 example (e.g. `f=1`, mesh `{0,1/2,1}`, exact squared H1-seminorm error `1/48`) while preserving its `numerical_bound` status. A nonzero function cannot lie in both the global-polynomial trial space (with zero endpoints) and the conforming P1 space: a global polynomial that is linear on each cell is globally affine, hence zero under both endpoint constraints. Therefore a nonzero same-input cross-check against `polynomial_poisson_1d.py` is mathematically unavailable; document the distinct domains and keep its tests independently green rather than fabricating a shared case. These checks do not prove either route is a general machine-certified H1 error bound.
7. Exact work limits documented from operation accounting: the current operation count is exactly `36*cells + 11` and the maximum-cell case was exercised. Focused tests, the full existing package tests/examples, compileall, and a clean direct-import wheel installation were then run on the final source state.

## 10. Explicit nonclaims

Even if every item above passes, the result establishes only exact `c/q/alpha/gain` for the finite, one-dimensional, constant-coefficient, zero-Dirichlet, piecewise-constant-load P1 model encoded by the input binary64 bits. It does **not** establish a general continuous-PDE solution-error estimate, a general P1 candidate certificate, training/quadrature provenance for neural candidates, candidate-pool or continuous-dictionary coverage, K1 weak-greedy conditions, P2/P3/P4, convergence, multidimensional validity, comparative efficiency, physical-data validity, or research novelty. Stage 2 and general P1 remain **OPEN**.
