# API reference

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
JSON-friendly data.

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

## Evidence and scope

This package provides no convergence theorem, production solver guarantee,
benchmark claim, built-in SciPy sparse adapter, built-in PyTorch adapter, or
BlockPolicy. Users may supply their own adapters/components using those
libraries.
Optional transpose actions may be supplied through `TransposeOperator`, but
the current core does not require or call them.
