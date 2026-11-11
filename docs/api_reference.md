# API reference

## Module organization

The `src/` directory is the import-isolation boundary. Public imports are
provided by the responsibility packages `core`, `problems`, `enrichment`, and
`strategies`; `solver.py` and the limited `estimators_1d.py` adapter are source
root modules. For example, import `CEMSolver` from `solver`, problem types from
`problems`, candidate types from `enrichment`, and scoring policies from
`strategies`. There is intentionally no umbrella import namespace.

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
learners receive this snapshot and cannot mutate solver state. `hash()` covers
values, IDs, and coefficients.

`CorrectionRecord` contains `element_id`, `family`, `support`, `parameters`,
`proposal_cost`, `scoring_cost`, `predicted_reduction`, `before`, `after`,
`accepted`, `status`, before/after state hashes, and touched/frozen IDs.
`CorrectionLedger.records` preserves records in order and `to_dicts()` exports
JSON-friendly data. Export maps non-finite floating values to JSON `null`
without changing in-memory records; unsupported objects are not silently
discarded.

## `CEMSolver`

`CEMSolver(problem, learners, scorer, policy, state, ledger=None)` owns the
state, active registry, and correction ledger. `step(budget=None)` returns an
accepted `Score` or `None`. Rejections return `None` and append a rejected
record; learner/scorer/policy exceptions are caught, state/registry are
restored, and an exception record is appended. FC construction requires
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
