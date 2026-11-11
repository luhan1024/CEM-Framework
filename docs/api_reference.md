# API reference

## Module organization

The `src/` directory is the import-isolation boundary. Public imports are
provided by the responsibility packages `core`, `problems`, `enrichment`, and
`strategies`; `solver.py` and the limited `estimators_1d.py` adapter are source
root modules. For example, import `CEMSolver` from `solver`, problem types from
`problems`, candidate types from `enrichment`, and scoring policies from
`strategies`. There is intentionally no umbrella import namespace.
The distribution name `cem-framework` is a packaging identifier; it does not
define a `cem_framework` import package. Generated `.egg-info` metadata is not
an import package.

These short top-level names are generic and can collide with unrelated
modules. A dedicated environment or an isolated `src/` path is recommended.

## `LinearPDEProblem`

`LinearPDEProblem(operator, source, weights=None, feasible=None)`

- `operator`: finite dense matrix or matrix-free object exposing `shape` and
  `apply(v)`.
- `source`: finite residual/source vector of length `shape[0]`.
- `weights`: optional positive finite vector of length `shape[0]`.
- `feasible(values)`: optional admissibility predicate.
- `apply_operator(values)`, `residual(state)`, `objective(state)`,
  `admissible(values)`, and `validate_state(state)` are public operations.

## `NonlinearPDEProblem`

`NonlinearPDEProblem(residual_fn, merit_fn, feasible=None)` uses callable
residual and merit functions. It cannot be used with
`FullyCorrectivePolicy`.

## `Candidate` and learners

`Candidate(element_id, family, correction, operator_image, support,
parameters, seed, provenance, proposal_cost={}, version="1")` validates and
freezes its arrays and metadata. `WeakLearner.propose(problem, state_snapshot,
residual, budget)` returns a Candidate. `AnalyticLearner` and
`LocalBasisLearner` are built-in examples. `element_id` uniqueness is enforced
within the solver active set; audit workflows should not recycle identifiers.

### Finite P1 hat learner and `CEMSolver` integration

Import `P1PoissonHatWeakLearner` and its concrete error `P1HatProposalError`
from the top-level `p1_poisson_hat` module. The structured base exception
`ProposalFailure` is exported by `enrichment`. `P1Poisson1DRitzProblem` is
exported by `problems`, and `P1Poisson1DData` by `estimators_1d`.
`P1Poisson1DRitzProblem(data)` is a matrix-free, immutable P1 Ritz adapter for
the bounded zero-Dirichlet 1D Poisson data represented by
`P1Poisson1DData`. `P1PoissonHatWeakLearner.propose(problem, state, residual,
budget=None)` searches the finite inactive interior-hat dictionary using exact
rational local gains and returns one best positive-gain `Candidate`. The
supplied residual is a strict float64 shape/dtype/byte freshness check. The
candidate provenance includes the selected full-domain certificate record and
hash, fingerprints, deterministic tie rule, and actual/reserved work audit.
The state fingerprint domain-separates `outer_coefficients=None` from a
present vector and binds the present vector's dtype and shape, so an empty
float64 array is a distinct state from `None`.

No-candidate and rejected-preflight outcomes raise `P1HatProposalError` with
finite `status`, `reason_code`, and immutable partial `work_audit`; statuses
include `empty_dictionary`, `dictionary_exhausted`, `no_positive_gain`,
`budget_exceeded`, `problem_state_mismatch`, `residual_mismatch`, and
`active_set_invalid`. The learner composes with `CEMSolver` using
`RitzCandidateScorer` and a Stagewise or Fully-Corrective policy. A typed
`ProposalFailure` is caught by `CEMSolver.step()`: the pre-attempt state and
active-candidate registry are restored, `step()` returns `None`, and a
`proposal_failed` `CorrectionRecord` is appended with `proposal_status`,
`reason_code`, and a typed-failure audit envelope when it fits the size limits.
Budget overrides may lower documented caps but cannot raise them; metadata
size, serialization bytes, and hash bytes are preflighted and recorded as
actual/reserved quantities. This finite 1D integration does not establish
continuous-PDE error bounds, convergence, or performance claims.

## Scoring and policy

`LinearCandidateScorer.score(problem, state, candidate) -> Score` computes a
weighted linear trial and validates its operator image. `NonlinearCandidateScorer.score(problem,
state, candidate, alpha=1.0) -> Score` evaluates a nonlinear merit trial.
`MeritDecreasePolicy`, `StagewisePolicy`, and `FullyCorrectivePolicy` accept
only finite strict merit decreases according to their documented mode.

`Score` fields are `accepted_shape`, `finite`, `q`, `correlation`,
`predicted_reduction`, `alpha`, `before`, `after`, and `reason`. A scorer must
return this shape-compatible result; custom policies consume
`decide(score) -> bool` and custom scorers may implement
`score(problem, state, candidate[, alpha])`.

## State and ledger

`SolverState(values, active_element_ids=(), outer_coefficients=None)` owns the
mutable solution vector and decomposition metadata. `snapshot()` returns a
read-only `StateSnapshot(values, active_element_ids, outer_coefficients)`;
learners receive a distinct snapshot. The solver keeps a private immutable
rollback baseline and detects snapshot/state mutation around learner, scorer,
and policy calls. `SolverState.hash()` currently uses
`STATE_HASH_ALGORITHM_VERSION == 2`: SHA-256 is domain-separated by
`cem.solver_state.v2`, and binds each array's dtype, shape, and C-order bytes,
the active IDs, and whether `outer_coefficients` is absent or present. A
present empty vector is distinct from `None`.

`CorrectionRecord` contains `element_id`, `family`, `support`, `parameters`,
`proposal_cost`, `scoring_cost`, `predicted_reduction`, `before`, `after`,
`accepted`, `status`, before/after state hashes, and touched/frozen IDs. Current
records also expose `record_schema_version` (2),
`state_hash_algorithm_version` (2 for newly written records), `proposal_status`,
`reason_code`, `proposal_audit_json`, and `proposal_audit_sha256`. The record
schema version describes the serialized fields; the state-hash algorithm
version describes the algorithm used for `state_hash_before/after` and is a
separate version number. For an audited P1 candidate, `proposal_status` is its
certificate status (for example `verified_positive_gain`) and
`reason_code` is the scorer reason or `None`. For a typed proposal failure,
`proposal_status` is the exception status (for example `dictionary_exhausted`)
and `reason_code` is its structured reason (for example `all_hats_active`).
The audit JSON and digest are `None` when no audit envelope was produced; an
oversize fallback also omits both.

`CorrectionLedger.from_dicts()` reads legacy v1 and current v2 CorrectionRecord
and PoolAuditRecord rows. A legacy row without version fields is read as schema
v1/hash v1; missing v2 proposal-audit fields are filled with `None`. V2 rows
must carry a supported state-hash version (1 or 2). Unknown schema versions,
unsupported hash versions, and unknown fields are rejected. Loading a ledger
does not by itself verify proposal-audit JSON or its digest.

For proposal-audit integrity, `proposal_audit_json` is canonical JSON (sorted
keys, compact separators, ASCII escaping, non-finite values forbidden), and
`proposal_audit_sha256` is SHA-256 of its exact UTF-8 bytes. The public
module-level verifier is `verify_proposal_audit` from `core.ledger`:
`verify_proposal_audit(payload_json, payload_sha256, problem=None, state=None)`.
It returns a boolean. For a `verified_candidate` envelope, supplying both
`problem` and the pre-attempt `state` snapshot additionally checks the P1
candidate binding against that problem and state; supply both or neither. A
`typed_failure` envelope has no candidate to bind, so its verification covers
the canonical payload, digest, schema, and failure fields. After loading ledger
rows, callers should explicitly verify every non-null audit pair. The audit
envelope is capped at
8,000,000 UTF-8 bytes and the embedded P1 certificate at 4,000,000 bytes. An
oversize attempt is recorded with `reason_code="proposal_audit_oversize"` and
a bounded terminal record (at most 16,384 bytes, 32 referenced work-event IDs,
each at most 128 UTF-8 bytes); the audit payload is omitted from that fallback
record.

`CorrectionLedger.records` preserves records in order and `to_dicts()` exports
JSON-friendly data. Export maps non-finite floating values to JSON `null`
without changing in-memory records; unsupported objects are not silently
discarded.

## `CEMSolver`

`CEMSolver(problem, learners, scorer, policy, state, ledger=None)` owns the
state, active registry, and correction ledger. `step(budget=None)` returns an
accepted `Score` or `None`. Rejections return `None` and append a rejected
record; learner/scorer/policy exceptions are caught, state/registry are
restored, and a failure/rejection record is appended. In particular, a typed
`ProposalFailure` raised by the integrated P1 learner is recorded with its
status/reason and audit envelope; it is not treated as an accepted candidate.
FC construction requires
`active_element_ids=()` and `outer_coefficients=None` or empty; every non-empty
coefficient array is rejected for both Stagewise and FC policies, and no policy
supports checkpoint registry restoration. Before learners
run, `step()` rejects and logs residuals that are not finite one-dimensional
arrays. `patch(append=(), replace=None, remove=())` is FC-only,
atomic, and records accepted/rejected patch outcomes. It raises `ValueError`
for wrong policy mode, duplicate/unknown IDs, stale operator images, or an
invalid trial; failed patches restore state and registry.
The `learners` tuple is an ordered proposal sequence: `step()` stops at the
first policy-accepted candidate rather than arbitrating over all candidates.
Custom policies may customize `decide(score)`, but `mode` must be omitted,
`"stagewise"`, or `"fully_corrective"`; unknown modes are rejected.

## Discrete Ritz/energy adapter

`DiscreteRitzProblem(K, source, feasible=None)` is a first finite-dimensional
adapter for an explicit dense real SPD matrix. It validates finite values,
symmetry, shape, and positive definiteness; it does not densify arbitrary
operators and is not a weak-form or continuous-PDE energy integrator. Its
objective is `J(u)=0.5*u.T@K@u-source.T@u`, with residual `source-K@u`.
Both objective evaluation and fully-corrective refitting use the instrumented
operator path, so dense and matrix-free operator-application events remain
countable under the same ledger semantics.
`RitzCandidateScorer` uses `q=phi.T@K@phi`, `c=residual.T@phi`,
`alpha=c/q`, and checks predicted reduction against the actual objective
difference. Fully-corrective mode solves the Ritz Gram system on active
corrections and fails closed for rank-deficient or ill-conditioned sets. These
are algebraic/software checks only, not a convergence theorem or PDE energy
certificate; `LinearPDEProblem` strong-residual semantics are unchanged.
`MatrixFreeRitzProblem(operator, source, spd_assumption="caller_declared")`
provides the same finite-dimensional objective through the existing
`Operator.shape/apply` protocol without materializing `K`. The declaration is
an assumption, not a global SPD proof. The core Ritz path does not request a
transpose action or run transpose probes; callers must provide any external
symmetry/SPD evidence themselves. Matrix-free FC applies the operator to active columns to build
the Gram matrix and fails closed on nonfinite, nonsymmetric, or ill-conditioned
systems.

## Exact dyadic 1D P1 Poisson direction oracle

`certify_p1_poisson_step(data, state_values, direction_values)` and
`verify_p1_poisson_step_certificate(record, data, state_values,
direction_values)` are a separate, pure exact-arithmetic path for the narrow
`P1Poisson1DData` model: `-u''=f` on `(0,1)`, zero Dirichlet endpoints,
cellwise-constant source, and conforming piecewise-linear state/direction on
the same mesh. Finite native binary64 inputs are interpreted as their exact
dyadic values. The record contains exact `c`, `q`, signed `alpha=c/q`,
`c²/(2q)`, an input hash, and the checked rational-operation count. Verification
recomputes the record and direct energy difference; it does not update
`CEMSolver` or accept caller-supplied enclosure radii.

This is an exact directional energy identity for the encoded 1D model, not a
general PDE estimator, solution-error certificate, training/quadrature
certificate, convergence result, or candidate-selection guarantee. Invalid
inputs and resource-limit cases fail closed. For the existing float64
equilibrated-flux error estimate see `estimators_1d`; do not treat its
`numerical_bound` status as an outward-rounded interval proof.

## Finite-pool selection (opt-in)

`CandidatePool(candidates, generation_metadata=None, proposal_failures=(),
active_element_ids=())` is immutable and rejects duplicate or currently active
IDs. Its `pool_id` hashes candidate semantic content, generation metadata, and
the normalized active-element set. Candidate arrays and provenance mappings are
owned snapshots; post-construction mutation attempts fail rather than silently
changing candidate hashes.
NumPy integer seeds are normalized to Python `int`; boolean and unsupported
seed types are rejected at `Candidate` construction.
the best-gain selector is order-independent. `PoolCandidateResult` preserves
proposal/scoring status and an optional `Score`. `BestPredictedGainSelector`
implements only finite positive predicted-gain maximization with deterministic
`element_id`/full-content-hash tie-breaking. `CEMSolver.step_pool(selector,
pool=None, budget=None)` is opt-in; with no supplied pool it proposes all
learners from one snapshot/residual and scores each at most once. Selector,
acceptance, and update remain separate; acceptance rejection does not retry a
runner-up. `CorrectionLedger.to_dicts()` exports the `PoolAuditRecord`,
including pool ID, rule, candidate statuses, gains, ratio/unavailable reason,
proposal cost metadata, and scoring statuses. P0 does not measure scoring
cost; it records score outcomes and failure reasons. The ratio is an empirical finite-pool
diagnostic only; no gamma selector or continuous-dictionary/WOGA guarantee is
implemented. Each pool audit also stores frozen, JSON-friendly
`generation_metadata`, the caller-declared `budget`, and separate stable
`selector_identity`/`selector_rule` fields. These are provenance declarations,
not verified claims about randomness, dictionary coverage, or weak-greedy theory;
budget is not measured work (use `WorkLedger` for measured calls and spans).

## Evidence and scope

This package provides no convergence theorem, production solver guarantee,
benchmark claim, built-in SciPy sparse adapter, built-in PyTorch adapter, or
BlockPolicy. Users may supply their own adapters/components using those
libraries.

## Raw work ledger

`WorkLedger` is append-only and stores measured `CostEvent` records. Call
quantities (`*_calls`, unit `call`) and wall time (`wall_time`, unit `second`).
CPU time, memory, deployment amortization, and policy allocation are not
implemented. `observed_state` distinguishes observed, estimated metadata, and
derived summaries. Events carry run/step/selection/attempt IDs, exclusive
timing semantics, status, and optional candidate attribution.
Event metadata is recursively copied and frozen at append time; only JSON-like
scalar values, string-key mappings, sequences, and NumPy values are accepted.
NumPy arrays/scalars are converted to immutable tuples/Python scalars. Cycles,
nonfinite numbers, non-string mapping keys, and unsupported objects fail before
the event is appended. `to_dicts()` returns detached JSON-compatible data.
`CorrectionRecord.work_event_ids` and `PoolAuditRecord.work_event_ids`
reference the raw events. The ledger does not emit an opaque total-cost scalar;
instrumentation overhead and timer precision are measurement limitations.
The public `CEMSolver.patch()` operation also receives its own step/attempt
and exclusive wall span; its operator, objective, and refit events are linked
to the accepted or rollback record.
Optional transpose actions may be supplied through `TransposeOperator`, but
the current core does not require or call them.
