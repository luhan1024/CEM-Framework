# CEM Framework user guide

This guide describes only the APIs implemented in this package. It is a small
NumPy-first correction-element scaffold, not a production PDE package and it
makes no convergence, accuracy, or performance guarantee.

## Minimal linear composition

```python
import numpy as np
from cem_framework import (CallableOperator, LinearPDEProblem, SolverState,
    AnalyticLearner, LinearCandidateScorer, MeritDecreasePolicy, CEMSolver)

operator = CallableOperator((2, 2), lambda x: np.array([x[0] + 2*x[1], 3*x[0] - x[1]]))
problem = LinearPDEProblem(operator, np.array([1., 0.]))
solver = CEMSolver(problem,
    (AnalyticLearner("basis", np.array([1., 0.]), "basis"),),
    LinearCandidateScorer(), MeritDecreasePolicy(), SolverState(np.zeros(2)))
score = solver.step()
assert score is not None
```

`LinearPDEProblem` accepts a dense 2-D NumPy array (wrapped by
`NumpyMatrixOperator`) or an object with `shape == (m, n)` and
`apply(values) -> array(m,)`. `CallableOperator` is the convenience adapter.
The optional `TransposeOperator` capability is not required by the current
solver. Operator outputs must be finite and correctly shaped.

## Learners, scorer, and policies

Implement a learner with a `family` attribute and
`propose(problem, state_snapshot, residual, budget) -> Candidate`. The
snapshot arrays are read-only. A `Candidate` owns an immutable correction,
optional operator image, support/parameter metadata, and a stable
`element_id`. Linear problems use `LinearCandidateScorer`; nonlinear problems
use `NonlinearCandidateScorer`. `StagewisePolicy` appends a new correction;
`FullyCorrectivePolicy` refits all active outer coefficients.

The solver has a general restart boundary: no policy currently supports
active-set checkpoint restoration. An initial `state.values` may be any finite
baseline, but `active_element_ids=()` and `outer_coefficients=None` or an
empty array are required for every policy. Any non-empty coefficient array,
including all-zero values, fails fast rather than being silently discarded.
Checkpoint registry restoration is not implemented. Fully-corrective mode also
requires an actual `LinearPDEProblem` instance (including its matrix-free
operator support).

Custom policies may provide their own `decide(score) -> bool`, but their
`mode` must be absent (defaulting to stagewise), `"stagewise"`, or
`"fully_corrective"`; an unknown mode such as `"block"` fails fast and cannot
create implicit block semantics. Learners are attempted in the order supplied
to `CEMSolver`; the first accepted candidate is committed, and later learners
are not called during that `step()`.

### Custom learner contract

```python
import numpy as np
from cem_framework import (Candidate, CallableOperator, LinearPDEProblem,
    SolverState, LinearCandidateScorer, MeritDecreasePolicy, CEMSolver)

class ResidualDirectionLearner:
    family = "custom-residual"
    def propose(self, problem, state_snapshot, residual, budget):
        direction = np.asarray(residual, dtype=float).copy()
        if direction.shape != state_snapshot.values.shape or not np.all(np.isfinite(direction)):
            raise ValueError("learner direction must be finite and match state shape")
        image = problem.apply_operator(direction) if hasattr(problem, "apply_operator") else None
        return Candidate("custom-residual-0", self.family, direction, image,
                         {"support": "global"}, {"budget": dict(budget)},
                         budget.get("seed"), {"source": "ResidualDirectionLearner"},
                         {"proposal": 1.0})

operator = CallableOperator((2, 2), lambda x: np.array([x[0] + 2*x[1], 3*x[0] - x[1]]))
problem = LinearPDEProblem(operator, np.array([1., 0.]))
solver = CEMSolver(problem, (ResidualDirectionLearner(),),
                   LinearCandidateScorer(), MeritDecreasePolicy(), SolverState(np.zeros(2)))
assert solver.step() is not None
```

The learner receives a read-only `StateSnapshot`, a finite residual array, and
the caller's budget mapping. It must return a finite one-dimensional correction
with a stable unique `element_id`; if supplied, `operator_image` must equal
`problem.apply_operator(correction)` and have the operator output shape.
`element_id` must be unique within the solver's active set; audit workflows
should not recycle identifiers. Support/parameter/provenance metadata is
immutable after construction.
The solver checks that the problem residual is one-dimensional and finite
before calling learners; invalid residuals are rejected and logged. A custom
problem/operator remains responsible for producing a numerically meaningful
residual within that boundary.

## Fully-corrective patching

`solver.patch(append=(candidate,), replace={id: candidate}, remove=(id,))`
is available only for a fully-corrective solver. IDs must be stable; append
IDs must be new, replacement IDs must already exist, and removed IDs must be
active. The patch refits the active set and records before/after hashes,
objective values, touched/frozen IDs, and estimated work in the ledger.
Failures restore the prior state and registry and add a rejected ledger
record. Stagewise patching is rejected.

## Nonlinear boundary and unsupported features

Nonlinear problems use `NonlinearPDEProblem`, `NonlinearCandidateScorer`, and
stagewise acceptance. Fully-corrective refitting is intentionally restricted
to `LinearPDEProblem`. `BlockPolicy` is not implemented or exported; no
implicit block semantics should be inferred.

## Failure diagnosis

Inspect `solver.ledger.records` after every proposal. Rejected records retain
the status/reason and state hashes. Common statuses include shape mismatch,
stale operator image, inadmissible trial, zero/nonfinite operator image,
exception, and patch rejection. `CorrectionLedger.to_dicts()` is the stable
JSON-friendly export path.
