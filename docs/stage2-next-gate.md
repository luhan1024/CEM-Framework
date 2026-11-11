# Stage 2 next gate: deterministic P1 hat weak learner

**Status: bounded proposal-only implementation approved; local implementation complete, final source review pending.**

This note records the approved narrow software gate selected from the current
research plan, the literature-to-API audit, and the CEM package. The approved
adapter and proposal-only learner are implemented; this does not change the
manuscript or research plan, integrate the learner into the solver, or report a
PDE benchmark.

## Decision

The next unique deliverable should be a **deterministic, exact-gain weak learner
over a fixed 1D conforming P1 hat dictionary** for the existing bounded Poisson
model. The learner scans the inactive interior hat functions, chooses the
largest exact single-direction energy decrease, and returns one ordinary CEM
`Candidate` with a replayable `P1PoissonStepCertificate` in its provenance.

This is the smallest available route from the current generic candidate
container toward a reproducible candidate proposer: unlike `AnalyticLearner`
and `LocalBasisLearner`, which return a caller-supplied basis, the proposed
learner derives its proposal from the current state and a typed P1 problem.
The protocol's supplied residual is checked exactly for consistency but is not
used as a substitute for exact weak-form integration. It remains a deliberately
narrow finite-dictionary method.
The exact P1 integral formula and finite-element hat basis are standard; the
deliverable would be software integration and auditability, not a new theorem
or a novelty claim.

Do **not** select Gaussian/RBF quadrature as this gate. It is the more relevant
later route toward heterogeneity and continuous parameter families, but the
current API has no validated source for its general `delta/epsilon`. First
establish a functional proposal API with zero-radius exact provenance on the
smallest compatible candidate family.

## Current implementation boundary

- `Candidate` stores an immutable correction vector and caller-declared
  support, parameters, provenance, and proposal cost. `WeakLearner.propose`
  already defines the proposal interface; the built-in analytic/local learners
  use fixed supplied vectors.
- `CandidatePool` hashes finite batches and `BestPredictedGainSelector`
  reports a ratio relative to the supplied finite pool. It says nothing about
  unenumerated or continuous candidates.
- `certify_p1_poisson_step` proves exact `c`, `q`, signed `alpha`, and ideal
  line gain for binary64 data interpreted as dyadic rationals, but it neither
  generates candidates nor updates solver state.
- `polynomial_poisson_1d.py` covers a bounded global-polynomial model, not a
  local P1 hat dictionary or reusable general candidate generation. The
  equilibrated-flux estimator remains a float64 numerical estimate, not an
  outward-rounded enclosure.

Accordingly, the proposed learner is an adapter around the exact P1 contract,
not a general `delta/epsilon` provider. Its mathematical scope is restricted
to unit-coefficient Poisson on `(0,1)`, homogeneous zero Dirichlet data,
cellwise-constant source, a finite increasing binary64 mesh, and a conforming
P1 state on that same mesh.

## Candidate family and exact selection rule

Let `0=x_0<...<x_N=1` be the supplied mesh, `h_i=x_{i+1}-x_i`, and let
`phi_j` be the usual nodal hat at each eligible interior node `j`. Interpret
all supplied binary64 values as their exact dyadic rationals. For state values
`u_i` and cell source `f_i`, the hat is supported only on cells `j-1` and `j`.
Its exact directional quantities are

```text
c_j = f[j-1]*h[j-1]/2 + f[j]*h[j]/2
      - (u[j]-u[j-1])/h[j-1] + (u[j+1]-u[j])/h[j]
q_j = 1/h[j-1] + 1/h[j]
alpha_j = c_j/q_j
Delta_j = c_j*c_j/(2*q_j)
```

Thus `q_j>0`, and for this exact data/model class the enclosure radii are
`delta=epsilon=0`; they are zero because the integrals are evaluated exactly,
not because an empirical tolerance is assumed. The learner scans all eligible
inactive hats and selects the largest `Delta_j`; ties use ascending stable
element ID. If `N=1`, no interior hat exists even with an empty active set, so
return the typed `empty_dictionary` outcome. If `N>1` but all hats are
recognized as active, return `dictionary_exhausted` immediately after active
ID validation. That path precedes residual recomputation and scan-budget
reservation. If at least one hat is inactive but every exact `Delta_j=0`,
return `no_positive_gain`. All three outcomes are raised as P1-specific typed
exceptions, not returned as fake candidates.

Only when `K>0` and `max_j Delta_j>0` is selected, the proposal is the exact
maximizer of `|c_j|/sqrt(q_j)` and has finite-dictionary weakness `tau=1` by
construction. For an empty/exhausted dictionary or an all-zero gain scan, no
candidate is returned and a ratio/weakness value is **undefined**, not zero or
one. This statement applies only to the fixed finite inactive P1-hat set, not
all P1 functions, local polynomials, an RBF family, or a continuous dictionary.

The selected hat is passed once through the existing full-domain
`certify_p1_poisson_step` verifier. This checks that the two-cell search result
matches the exact continuous weak-form certificate and direct energy identity.
The unselected hats should be evaluated by the local two-cell formulas; do not
call the O(mesh-size) full-domain verifier for every hat, which would turn the
scan into quadratic work.

## Approved deliverable and API shape

The approved implementation consists of a typed P1 Poisson Ritz
adapter plus one `P1PoissonHatWeakLearner` compatible with
`WeakLearner.propose(problem, state, residual, budget)`. The learner:

1. Accepts only `P1Poisson1DRitzProblem`, built from one validated
   `P1Poisson1DData`. Its `SolverState.values` is the `N-1` vector of interior
   nodal coefficients; the unique full state is `[0, *values, 0]`. The problem
   adapter builds the standard matrix-free P1 stiffness action and cell-load
   vector from that exact data. It binds a schema/version, raw-bit mesh/source
   hashes, assembly identity, and dimension in a recomputable
   `problem_fingerprint`; it rejects non-finite/unrepresentable float assembly.
   The state fingerprint separately domain-separates absent outer coefficients
   from a present vector and binds its dtype and shape; an empty float64 vector
   is not encoded as `None`.
   A one-cell input is accepted as data but the proposer returns
   `empty_dictionary` before constructing the adapter operator or a
   zero-dimensional Ritz solve.
2. For a nonempty inactive dictionary, treats the protocol `residual` as a
   freshness/identity guard only. It must
   be a native, finite float64 vector of shape `(N-1,)` and its bytes must
   exactly equal a fresh `problem.residual(state_snapshot)` result computed by
   the same deterministic adapter code. Comparison is bitwise, with no
   tolerance; shape/dtype/value or signed-zero bit mismatch raises a typed
   `residual_mismatch`. The learner does not use this rounded vector to form
   `c_j`: it recomputes exact weak-form `c_j/q_j` from the bound data and
   extended nodal state. Stale state, a different source/mesh/problem
   fingerprint, or a caller-supplied residual from another problem fails
   closed before candidate scan. The proposer validates active IDs before
   residual replay; after that validation, an exhausted
   dictionary returns immediately without recomputing the residual or doing
   exact local preprocessing; a scan-work reservation cannot make that known
   empty remainder fail with `budget_exceeded`.
3. Enumerates the bounded inactive interior-hat dictionary deterministically;
   it uses the supplied state (not a manufactured solution) to calculate each
   exact local `c_j/q_j`.
4. Validates `state.active_element_ids` before scanning. IDs are
   `p1hat-v1:<problem_fingerprint>:<mesh_sha256>:node:<j>` for
   `j=1,...,N-1`; the active tuple must contain unique strings and every entry
   must belong to this exact mesh/data dictionary. Duplicate, malformed, or
   foreign IDs raise `active_set_invalid`; they are never silently ignored.
   Active valid hats are excluded. With `N>1`, if all hats are active, raise
   `dictionary_exhausted` immediately after this validation; do not reserve
   `6*N` work, recompute residual, or start scan preflight. `N=1` is instead
   `empty_dictionary`.
5. Returns exactly one best-gain `Candidate`, with a one-hot interior nodal
   correction, the ID above, local two-cell support, source/state/problem
   fingerprints, selected exact certificate record/hash, dictionary size and
   tie rule, and proposal-work counts in existing candidate fields.
6. Preserves `WeakLearner.propose(...) -> Candidate` without widening the
   generic solver protocol: all no-candidate and failed-preflight outcomes
   raise `P1HatProposalError(status, reason_code, work_audit)`. `status` and
   `reason_code` are finite enums, not caller text. Required statuses are
   `empty_dictionary`, `no_positive_gain`, `dictionary_exhausted`, `budget_exceeded`,
   `problem_state_mismatch`, `residual_mismatch`, and `active_set_invalid`;
   reason codes identify the finite cause (e.g. `all_gains_zero`,
   `all_hats_active`, `hat_cap`, `fraction_operation_cap`, or
   `foreign_active_id`). The immutable audit reports work completed and the
   preflight bounds. The solver's generic return type is unchanged; the
   exception is tested directly, and this gate does not call the solver.

The residual check is intentionally strict because the proposed adapter
defines one deterministic residual operation sequence. It is not a claim that
the rounded discrete residual equals every exact continuous functional value;
the independently recomputed exact `c_j` is authoritative for selection.

The learner can be exercised as a **proposal-only** implementation of the
existing WeakLearner protocol. Do not hook it into `CEMSolver.step` or claim a
certified accepted update in this gate: the current solver applies a
float64-rounded `score.alpha`, whereas the P1 oracle certifies the exact
rational line minimizer. A later solver-integration gate would need an exact
certificate for the realized rounded state transition and an explicit policy
that rejects any unverified/non-decreasing transition.

### Preflight and live-work audit

The problem adapter must use O(N)-storage tridiagonal P1 stiffness actions,
not a dense `(N-1) x (N-1)` matrix. It stores a build audit including `N` cell
visits, stiffness/load contribution updates, operator rows, input/assembly
bytes hashed, and model-fingerprint construction. That cost is reported once
with the learner call that consumes the adapter; it is neither free nor
repeatedly charged as if assembly were rerun.

For a proposal let `N` be the cell count and `K` the inactive interior hats
after active-set validation (`0 <= K <= N-1`). After validating the budget
mapping's shape, keys, and value types and validating active IDs, handle
`N=1` as `empty_dictionary` and `N>1,K=0` as `dictionary_exhausted`
immediately. These empty-dictionary paths do not enter scan-budget reservation.
Only when `K>0`, before scanning any hat, check all caller limits and reserve:

- `6*N` exact arithmetic operations to precompute per-cell widths, source-load
  halves, state slopes, and inverse widths;
- at most `7*K` exact arithmetic operations for each hat's local `c`, `q`,
  squared score and cross-product comparison used for exact gain ranking;
- if `K>0`, one selected-candidate full certificate with at most `36*N+11`
  rational operations, from the existing P1 certificate contract;
- at most `3*N+2` binary64-to-Fraction conversions for local mesh/source/state
  values and, if a selected certificate is reserved, another `4*N+3` for its
  nodes/source/state/direction snapshot;
- caps for hat evaluations, local-cell visits (`2*K`), Fraction component
  bits, certificate/record bytes, hash bytes, residual operator rows, and the
  `N-1` candidate-vector elements and metadata bytes.

The `budget` argument is either `None` or a mapping containing only the keys
listed below. `None` and omitted keys use the listed per-call defaults. Every
value must be a non-boolean, nonnegative integer; booleans, floats, strings,
negative values, non-mappings, and unknown keys are rejected before any hat
scan with `P1HatProposalError(status="budget_exceeded", reason_code=...)`.
Unknown keys use `unknown_budget_key`; malformed mappings/values use
`invalid_budget_shape`/`invalid_budget_value`. A caller may lower a default
but may not raise a hard upper bound. If a lowered limit cannot fit the complete
preflight reservation, the call fails before scanning; it never scans a prefix
and never reports a partial-dictionary maximizer.

The proposed budget fields inherit or bound these limits. Dynamic defaults
refer to this call's validated `N` and complete inactive-hat count `K`:

| Field | Default / hard maximum | Source and meaning |
|---|---:|---|
| `max_cells` | 4,096 | Inherit `MAX_P1_CELLS`; `N` cannot exceed it. |
| `max_hat_evaluations` | `N-1` (at most 4,095) | Must fit all `K` inactive hats; one evaluation means computing one hat's exact `c_j`, `q_j`, and gain. No partial scan is allowed. |
| `max_local_cell_visits` | `2*K` (at most 8,190) | Each evaluated interior hat visits its two adjacent cells once. |
| `max_adapter_cell_visits` | `N` (at most 4,096) | One matrix-free P1 adapter build audit; each source cell is visited once. |
| `max_adapter_operator_rows` | `N-1` (at most 4,095) | Interior rows represented by the matrix-free adapter, counted once at build. |
| `max_residual_operator_rows` | `N-1` (at most 4,095) | One fresh residual recomputation for the bytewise freshness check. |
| `max_operator_row_visits` | `2*(N-1)` (at most 8,190) | Sum of adapter row construction and the one residual action; excludes neither category and is not multiplied by `K`. |
| `max_fraction_ops_total` | 250,000 | Proposed aggregate learner cap for precompute + local scan + selected replay. |
| `max_certificate_ops` | 200,000 | Inherit `MAX_EXACT_RATIONAL_OPERATIONS` for the single selected full certificate. |
| `max_fraction_conversions` | 0 if `K=0`; otherwise `3*N+2` plus `4*N+3` (at most 28,677) | Local exact input conversions plus one reserved selected-certificate snapshot. |
| `max_fraction_component_bits` | 16,384 | Inherit `MAX_FRACTION_COMPONENT_BITS` and enforce on every created exact rational. |
| `max_certificate_record_bytes` | 4,000,000 | Inherit `MAX_RECORD_JSON_BYTES`; preflight the selected input-bound certificate. |
| `max_candidate_metadata_bytes` | 65,536 | Proposed cap for canonical candidate provenance/support/parameters/work audit. |
| `max_candidate_vector_elements` | `N-1` (at most 4,095) | One interior one-hot correction; construction is preflighted. |
| `max_hash_bytes` | `max_certificate_record_bytes + 8*(4*N+3) + max_candidate_metadata_bytes` (at most 4,196,632) | Upper bound for certificate record, binary64 certificate snapshots, and canonical candidate metadata; count actual input bytes to every hash. |

Every row's upper bound is a hard maximum for that key, including proposed
aggregate, metadata, and derived hash caps. Callers cannot increase any key;
they may omit it to use the displayed default or lower it. A lower value is
accepted only if it covers the full reservation for the path actually taken.

The only budget keys are exactly the field names in this table. A missing key
uses its dynamic or fixed default above. A supplied `max_hat_evaluations` or
`max_local_cell_visits` below the complete `K`/`2*K` requirement, or any other
cap below its reserved work on a nonempty scan, fails before scanning with
`hat_evaluations=0`. The caller cannot use a small cap to truncate the
dictionary. For `N>1,K=0`, the proposer raises `dictionary_exhausted`
immediately after active-ID validation: exact rational operations,
conversions, hat evaluations, local-cell visits, residual operator rows,
certificate reservation, and certificate replays are all zero. It does not
compare numeric scan-work caps, so a low cap cannot turn known exhaustion into
`budget_exceeded`. For `N=1`, `empty_dictionary` likewise occurs before
adapter construction or residual evaluation and all those actual work
counters are zero.

The aggregate exact-arithmetic preflight is zero when `K=0`, because the
proposer returns `dictionary_exhausted` before exact local preprocessing. For
`K>0`, it is `6*N + 7*K + (36*N+11)`. At `N=4096`, `K=4095`, this is 200,708
operations, below the 250,000 aggregate cap. No single certificate may exceed
the inherited 200,000-operation cap. Every Fraction arithmetic wrapper
increments the live counter and checks component-bit limits. Input conversions,
gain comparisons, assembly, residual evaluation, hashing, array allocation,
serialization and candidate construction use separate explicit counters and
their own caps. Fraction comparisons use exact integer cross-products: count
one comparison per hat and include the compared numerator/denominator bit
lengths in the same 16,384-bit check; do not hide these operations in an
uncounted float conversion.

When constructed for a nonempty proposal, the adapter build audit counts
exactly `N` local-cell visits, `4*N` stiffness contribution updates, `2*N`
load contribution updates, `N-1` interior operator rows represented, the
actual binary64 scalar updates, and bytes hashed to form its fingerprint. Any
adapter build that preceded a terminal empty-dictionary proposal is prior
setup and remains separately attributed; that proposal itself does not rebuild
the adapter. Each freshness check counts exactly one residual recomputation
over `N-1` operator rows; for `n=N-1>=1`, this means `3*n-2` tridiagonal
multiplications, `2*(n-1)` neighbor additions, and `n` residual subtractions,
plus the exact bytes compared. (For `N=1` or an exhausted `K=0` proposal, no
residual is evaluated.) The local scan visits exactly `2*K` cell-hat
incidences and evaluates exactly `K` inactive hats when it reaches the scan.
The selected
candidate counts its `N-1`-element zero/one vector allocation/copy, metadata
serialization bytes, and bytes hashed. The selected full-domain replay counts
its actual certificate operations and record bytes. The scan performs no
full-domain replay per hat.

Metadata size is preflighted from a canonical maximum-counter skeleton before
the scan. The candidate metadata serialization loop is bounded by the
preflight allowance; every actual JSON dump contributes its observed byte
length to the serialization ledger. If the predicted length differs, the
ledger is reconciled before retrying; failure to converge within the bound is
a typed `budget_exceeded` outcome. The final embedded audit records the exact
serialized metadata length. Actual and reserved serialization/hash bytes are
separate fields, and a low metadata cap fails before any hat evaluation.

Before scanning, validate caller limits and check that the preflight bound fits
each requested cap. A caller may lower defaults but may not raise inherited
certificate, bit-length, record-size, or cell-count caps. If any reservation
fails, raise `P1HatProposalError(status="budget_exceeded", ...)` with
`hat_evaluations=0`. Each operation then ticks the same live counter; if a live
counter would exceed the reserve/cap, abort with a finite partial-work audit.
For `K>0`, the single full-certificate reservation is `36*N+11` rational
operations, bounded by the inherited `max_certificate_ops=200,000`. It is a
reservation before selection, not a claim that the replay always executes: an
all-zero-gain scan returns `no_positive_gain`, executes zero certificate
replays, and reports the unused reservation separately from actual work. For
`K=0`, there is no certificate reservation. A positive-gain winner executes
exactly one replay. Keep reserved and actual
operation/conversion/byte totals as separate audit fields.

The deterministic work budget is not a wall-clock guarantee. Report actual
elapsed time and peak memory separately, and make no speed claim based on the
count alone.

No hat is scanned until the full aggregate reserve, byte caps, and output
allocation fit the caller budget. If live counts exceed preflight, abort with
`budget_exceeded` and a finite partial-work audit. A rejected preflight records
`hat_evaluations=0`; an exhausted dictionary has `K=0` and zero evaluated hats;
an all-zero scan records `K` evaluated hats and zero full-certificate replays.
The implementation must derive exact counters from the operations it actually
performs and test preflight against the live totals; these formulas are a
design constraint, not yet implementation evidence.

### CandidatePool decision

Do not route the hat dictionary through `CandidatePool`/`step_pool` in this
gate. The learner already exhaustively scans its specified finite dictionary
and returns its exact maximizer; placing that single winner in a pool would
make a pool ratio of one tautological. Placing all hats in `CandidatePool`
would require an exact-rational selector and proof-bearing score path, whereas
the current pool scorer contract stores a floating `predicted_reduction`.
Keep any per-hat scan report as learner provenance/work audit, not as a
continuous-pool guarantee. A later heterogeneous multi-family comparison may
use CandidatePool only with a clearly finite, predeclared family-winner pool
and explicitly finite-pool interpretation.

## Alternative comparison: exact local FE vs Gaussian/RBF intervals

| Option | Reliable source for `c/q` and `delta/epsilon` | Cost/accounting | Main limitation | Decision |
|---|---|---|---|---|
| **P1 interior hats (selected)** | Exact dyadic cell integrals for piecewise-constant `f`, P1 `u/phi`, unit coefficient. `delta=epsilon=0`; selected direction independently replayed by the existing full-domain oracle. | Two local cells per hat for the search plus one full-domain selected-candidate replay. Count exact rational operations, hats scanned, mesh/problem assembly, candidate hashing and replay; no training or quadrature. | Standard finite-element basis; only a fixed finite dictionary and one model. No continuous-family coverage or new FE theorem. | **GO to review as the next narrow proposal gate.** |
| Local piecewise-polynomial/FEM-patch with degree `p>1` | Exact rational polynomial integration is possible when geometry, coefficient, source and basis coefficients have exact polynomial/rational representations; otherwise coefficient/data approximation must be enclosed. | Cell count times polynomial degree/quadrature or exact-convolution cost, conditioning and support assembly. | Requires new domain/data/basis contracts beyond current P1 state/direction API. | Defer until P1 hats demonstrate a useful reproducible learner. |
| Analytic Gaussian/RBF candidates | For specially bounded parameters and simple 1D data, derive closed forms involving `exp/erf` and evaluate them with validated interval special functions; alternatively use cell-split interval quadrature with a proved remainder, derivative bounds, coefficient/source errors and parameter uncertainty. These terms must produce explicit `delta` and `epsilon` with `qhat>epsilon`. | Record interval evaluations, subdivisions, function/derivative evaluations, special-function enclosure work, parameter search and verification. Narrow widths amplify derivatives and may force many subdivisions. | Stable interval `exp/erf`, endpoint/tail control, lower scale bounds, and uniform candidate-parameter bounds are not implemented; trained NN/RBF errors are additional terms. | Later P1 provenance gate, after parameter class and interval backend are specified and independently validated. |

The options are not interchangeable. Exact integration removes quadrature error
only for the named polynomial/P1 model; it does not establish useful
heterogeneous representation or a cost advantage. A Gaussian formula evaluated
with ordinary floating `erf` is not an interval proof. A quadrature tolerance
or agreement between two rules is not a bound unless the remainder and all
input/parameter errors are enclosed.

## Relation to the audited literature

These are mappings from the existing bounded audit, not new source re-reviews:

- The audit's Xu & Xu entry describes WOGA weakness relative to a specified
  dictionary and its randomized discrete dictionary result with sample-size/
  coverage and uniform quadrature hypotheses. The proposed `tau=1` statement
  is only an exhaustive finite P1-hat maximum; it supplies no coverage or
  continuous-dictionary inequality. See
  [`stage2-literature-gap-audit.md`, item 1 and cross-cutting conclusion 2](stage2-literature-gap-audit.md).
- The audit's Mishra–Molinaro entry describes solution-error control from
  residual, quadrature/generalization and PDE stability assumptions. It is not
  a candidate-wise directional `c/q` enclosure or a gain-ranking theorem.
  See [audit item 8](stage2-literature-gap-audit.md).
- The equilibrated-flux/posterior-estimator sources concern reliable solution
  error with reconstruction and regularity hypotheses. They do not certify
  every candidate's `c/q`; the current float64 flux adapter is expressly not
  outward-rounded. See [audit items 9 and 17](stage2-literature-gap-audit.md).
- The surveyed local FEM, adaptive basis, subspace correction and finite-neuron
  methods are prior art. The P1 learner must be presented as a transparent
  reproducible software bridge, not as the invention of FEM hats, greedy
  selection, adaptive basis enrichment, or Ritz minimization.

## Acceptance criteria and local validation

The numbered criteria below define the approved software/API gate and remain
the reviewer recheck list. They are exercised by the focused P1 tests and the
package-level suite; the exact final-source rerun is recorded at handoff.

1. Independent test-only `Fraction` oracle computes every hat's `c`, `q`,
   `alpha`, gain and direct energy difference without production helpers.
2. Local two-cell formulas equal the existing whole-domain P1 certificate for
   every hat on small uniform and nonuniform meshes; selected index equals a
   separately enumerated exact argmax, including deterministic ties and
   `c=0`/all-zero cases.
3. Protocol outcomes are exact: positive-gain success returns a `Candidate`;
   zero gain, exhausted dictionary, invalid typed inputs, residual mismatch,
   active-set mismatch and low budget raise `P1HatProposalError` with an
   allowed finite status/reason and immutable partial-work audit. No generic
   solver return type changes. Include an `N=1` entry test that returns
   `empty_dictionary` with operator-assembly, residual, candidate, and
   certificate work counters zero. Data/problem identity hashing performed at
   adapter construction remains separately reported and may be nonzero. Tests
   inspect both exception type and fields.
4. Residual semantics are tested explicitly: adapter and learner bind the
   same data/problem fingerprint and interior-state mapping; the supplied
   residual must match fresh adapter output byte-for-byte. One-bit, stale-state,
   wrong-source, wrong-mesh, signed-zero-byte and wrong-shape residual variants
   reject before any hat scan. Independent exact `c_j` never trusts this rounded
   residual.
5. Stable IDs bind schema, problem/data fingerprint, mesh fingerprint and
   node index. Test active-hat exclusion, duplicate IDs, unknown/foreign IDs
   (reject, do not ignore), malformed IDs, the `N=1` empty-dictionary path,
   and all-hats-active exhaustion. For `N>1,K=0`, assert active-set validation
   is followed directly by `dictionary_exhausted` with no scan, residual
   recomputation, exact local preprocessing, or certificate replay.
6. Verify the typed matrix-free P1 adapter's stiffness/load against independent
   exact FE assembly, and verify dimensions/state mapping/problem identity.
   Reject unbound matrices even if their shape matches.
7. Assert the preflight formula and every live counter on small and maximum
   supported meshes. Low-budget cases reject before scanning (`hat_evaluations`
   remains zero); every successful scan counts cell visits, exact conversions,
   rational operations/comparisons, one selected full certificate, adapter
   assembly, residual recomputation, hash bytes, candidate allocations and
   serialization. Assert metadata byte length equals a canonical re-encoding,
   and actual hash/serialization counters never exceed their reservations.
   Demonstrate O(N) storage/work accounting and no per-hat full-domain replay.
8. Do not call `CandidatePool`, `step_pool`, or `CEMSolver.step` in this gate.
   A later solver hookup is a separate review gate requiring exact verification
   of the realized float64 state transition and atomic rejection of a rounded
   non-decrease.
9. Test `tau=1` only for a nonempty inactive set (`K>0`) whose maximum exact
   gain is positive. Assert that `tau` and the weakness ratio are undefined
   for `N=1`/empty dictionary, `K=0`/exhausted dictionary, and all-zero gains;
   they must not be encoded as zero or one.

The approved implementation was validated from the repository root in the
TensorFlow Conda Python 3.11.13 environment. Re-run the following against the
exact source tree under review; do not resolve dependencies from the network:

   ```powershell
   $py = 'C:\Users\LuHan\miniconda3\envs\tensorflow\python.exe'
   $env:PYTHONPATH = 'src'
   & $py -m unittest discover -s tests -q
   & $py -m compileall -q src examples tests
   & $py examples\poisson_smoke.py
   & $py examples\poisson_ritz_regression.py
   & $py examples\heterogeneous_ritz_pool.py
   git diff --check
   git rev-parse HEAD
   git status --porcelain
   ```

   The following clean offline wheel check creates a fresh task-specific
   directory under repository `tmp/`, uses it for `TMP`/`TEMP`, disables package
   indexes, installs to an isolated target, imports only from that target, and
   removes only that verified temporary directory:

   ```powershell
   $stage = Join-Path (Get-Location) 'tmp\stage2_p1_hat_validation'
   if (Test-Path -LiteralPath $stage) { throw "Stage already exists: $stage" }
   $wheelDir = Join-Path $stage 'wheel'
   $installDir = Join-Path $stage 'installed'
   $tempDir = Join-Path $stage 'temp'
   New-Item -ItemType Directory -Path $wheelDir,$installDir,$tempDir | Out-Null
   $env:TMP = $tempDir
   $env:TEMP = $tempDir
   $env:PIP_NO_INDEX = '1'
   & $py -m pip wheel --no-deps --no-build-isolation --wheel-dir $wheelDir .
   if ($LASTEXITCODE -ne 0) { throw 'offline wheel build failed' }
   $wheelPath = (Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' |
       Select-Object -First 1).FullName
   if (-not $wheelPath) { throw 'wheel artifact missing' }
   & $py -m pip install --no-deps --target $installDir $wheelPath
   if ($LASTEXITCODE -ne 0) { throw 'isolated wheel install failed' }
   $env:PYTHONPATH = $installDir
   & $py -c "import core, problems, enrichment, strategies, p1_poisson_hat; from problems import certify_p1_poisson_step"
   if ($LASTEXITCODE -ne 0) { throw 'isolated direct-import failed' }
   Get-FileHash -Algorithm SHA256 $wheelPath
   ```

   The evidence manifest binds `git rev-parse HEAD`, the expected working-tree
   status and changed-file diff, a SHA-256 manifest over the exact source/test/
   documentation inputs, exact commands/results, Python and NumPy versions,
   and wheel SHA-256. A wheel/test pass is software/API evidence, not scientific
   PDE validation. No PDE benchmark or manuscript change is required for this
   gate.

## Evidence level, claims and No-Go

**Expected evidence level:** a mathematical identity for the bounded exact
P1 finite dictionary, independent rational-oracle agreement, deterministic
software behavior and audited resource accounting. This is not empirical PDE
performance evidence and does not show a cost advantage.

**Allowed claim after a future passing implementation review:** the package has
a deterministic, replayable weak learner that selects the exact best
single-hat direction from the specified finite P1 dictionary for the named
1D Poisson inputs. If solver transition integration has not separately passed,
make no claim that the CEM state update itself is certified.

**No claim:** general P1 candidate enclosure; arbitrary `delta/epsilon`
provenance; RBF/NN family gains; coverage of continuous or unenumerated
dictionaries; WOGA/K1 convergence; a Mishra–Molinaro solution-error bound;
flux-certified neural candidates; P2/P3/P4/L1; a new FEM/greedy theorem;
novelty or speed/accuracy advantage.

**No-Go / redesign conditions:** local and full-domain exact results disagree;
the chosen candidate is not the independent finite-dictionary maximizer;
input-to-problem/residual mapping is not bitwise bound and replayable; malformed
or foreign active IDs are silently ignored; the typed no-candidate/budget
outcomes violate the WeakLearner return contract; preflight differs from live
work or runs after scanning starts; implementation performs a full-domain
exact replay per hat; or a result is labeled certified after falling back to
float quadrature/heuristics. Any later accepted solver update must be rejected
as a separate subgate unless its realized rounded transition is exactly
verified. If exact search/certification is not useful under the honest work
ledger, record that No-Go and design the Gaussian/RBF interval path separately
rather than broadening this one.

## Current action

**The bounded proposal-only implementation is complete and awaits final source
review.** Required checks are the P1 focused suite, full unittest discovery,
compileall, example smoke runs, wheel build/install/import, and `git diff
--check` on the final reviewed source. Do not hook this learner into
`CEMSolver`, run a PDE benchmark, edit `研究计划.md`, alter the manuscript,
commit, or push under this gate.
