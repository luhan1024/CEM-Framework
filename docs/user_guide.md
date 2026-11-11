# CEM Framework user guide

This guide describes only the APIs implemented in this package. It is a small
NumPy-first correction-element scaffold, not a production PDE package and it
makes no convergence, accuracy, or performance guarantee.

## Source and import organization

The `src/` directory is the installation and import-isolation boundary. Its
responsibility packages are `core` (state and audit), `problems` (operators
and discrete problem definitions), `enrichment` (learners, candidate pools,
and serialization), and `strategies` (scoring and acceptance policies).
`solver.py` coordinates the iterative lifecycle, while `estimators_1d.py`
contains the narrowly scoped 1D adapter. Import from the corresponding package
or module, such as `from solver import CEMSolver` or
`from enrichment import Candidate`.

These short top-level package names are generic and can conflict with modules
from other projects on a shared import path. Use a dedicated environment or
isolate this project's `src/` directory.

## Minimal linear composition

```python
import numpy as np
from core import SolverState
from enrichment import AnalyticLearner
from problems import CallableOperator, LinearPDEProblem
from solver import CEMSolver
from strategies import LinearCandidateScorer, MeritDecreasePolicy

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
requires an actual `LinearPDEProblem`, `DiscreteRitzProblem`, or
`MatrixFreeRitzProblem` instance.

`DiscreteRitzProblem` plus `RitzCandidateScorer` is a separate finite-dimensional
energy path for an explicit dense SPD matrix. It minimizes
`J(u)=0.5*u.T@K@u-source.T@u`, uses the Ritz curvature/correlation identity,
and solves the Ritz Gram system in fully-corrective mode. It deliberately does
not reinterpret the strong-residual scorer, densify arbitrary operators, or
claim a continuous weak-form/PDE energy certificate; tests are algebraic and
software-level only.

### Bounded polynomial Poisson enclosure (1D)

`build_polynomial_poisson_1d_enclosure` is a separate bounded family path for
weak `-u''=f` on `(0,1)` with homogeneous Dirichlet data, exact rational
polynomial forcing/basis coefficients, and finite float32/float64 state and
direction coefficients. Coefficients are listed in ascending powers. The
builder computes the rational weak-form stiffness/load, rounds those entries
deterministically to binary64, constructs the matching `DiscreteRitzProblem`,
and returns a full replay record with the enclosure and embedded discrete
certificate. Float32 vectors are promoted exactly to binary64; all other
vector dtypes are rejected.

```python
from fractions import Fraction
import numpy as np
from problems import (
    build_polynomial_poisson_1d_enclosure,
    verify_polynomial_poisson_1d_enclosure,
)

psi = (Fraction(0), Fraction(1), Fraction(-1))  # x(1-x)
problem, record = build_polynomial_poisson_1d_enclosure(
    (Fraction(1),), (psi,), np.array([0.0]), np.array([1.0]))
assert verify_polynomial_poisson_1d_enclosure(record)
```

The replay checks both the continuous polynomial Ritz functional and the
rounded discrete quadratic for that exact canonical input. Its scope is
limited by degree, dimension, coefficient-size, operation, and record-size
caps. It is not a generic quadrature engine, an Ern--Vohralík equilibrated-flux
estimator, or a weak-greedy convergence result; it does not close P1 for other
families or general enclosure provenance.

`MatrixFreeRitzProblem` accepts a square `Operator`/`CallableOperator` and
requires the explicit string `spd_assumption="caller_declared"`; it never
materializes `K`, and that declaration is not a global SPD proof. Fully-
corrective mode applies the operator to each active correction column to form
the Gram matrix and rejects nonfinite, nonsymmetric, or ill-conditioned
systems without ridge or pseudoinverse.

## Narrow 1D equilibrated-flux estimator

`P1Poisson1DData` and `EquilibratedFlux1DEstimator` are an optional, isolated
verification adapter for `-u''=f` on `(0,1)` with zero Dirichlet data, a
continuous conforming P1 nodal solution, and a piecewise-constant source. The
adapter uses the stated exact cell-integration formulas in float64 and returns
an immutable `EstimatorRecord` with data/solution hashes, assumptions,
integration rule, equilibrium defect and its acceptance tolerance, and status.
By default the tolerance is a scale-aware, cell-count-dependent float64
roundoff allowance based on a `gamma_n` operation-count model. It is only a
numerical acceptance threshold, not a strict error bound; callers may provide
an explicit relative threshold. A record with a defect above its threshold
fails closed. Invalid solution inputs return `status="failed"`, a failure
reason, and no `eta_eq`; invalid problem data are rejected at construction.

In this one-dimensional special case, imposing both `sigma'=-f` and
`integral(sigma)=u(1)-u(0)=0` selects the exact flux `sigma=u'`. Therefore
`eta_eq=||sigma-u_h'||_L2` equals the exact H1-seminorm error in exact
arithmetic. It is not a nontrivial efficiency improvement over the exact
error, and this result does not extend here to higher dimensions or general
discretizations. Prager--Synge still gives the reliability identity; its
`||sigma-u'||` term is zero in this 1D case. The reported quantity is the
Prager--Synge estimator value computed numerically in float64. The
`numerical_bound` status does not mean machine-certified: the value is not an
outward-rounded interval, and the roundoff acceptance tolerance is not a
rigorous error bound. The adapter is not connected to solver policies or mesh
marking.

Custom policies may provide their own `decide(score) -> bool`, but their
`mode` must be absent (defaulting to stagewise), `"stagewise"`, or
`"fully_corrective"`; an unknown mode such as `"block"` fails fast and cannot
create implicit block semantics. Learners are attempted in the order supplied
to `CEMSolver`; the first accepted candidate is committed, and later learners
are not called during that `step()`.

### Custom learner contract

```python
import numpy as np
from core import SolverState
from enrichment import Candidate
from problems import CallableOperator, LinearPDEProblem
from solver import CEMSolver
from strategies import LinearCandidateScorer, MeritDecreasePolicy

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

## Opt-in finite candidate pools

`CandidatePool` is an immutable batch with a stable `pool_id` covering
candidate semantic contents, generation metadata, and the normalized active
element set. Candidate arrays and nested provenance mappings are owned
snapshots; post-construction mutation attempts fail. `step_pool(selector)` is
opt-in; it proposes every configured learner from one solver snapshot and one
residual, scores each candidate at most once, and records proposal/score
failures. `BestPredictedGainSelector` considers only finite, valid,
positive-predicted-gain candidates and uses `element_id` plus full candidate
hash for deterministic ties. It is order-independent. A selected candidate is
still passed through the normal acceptance policy; rejection does not retry a
runner-up. The pool ledger records candidate statuses, proposal cost metadata
(not measured runtime), selected/max gain, and
`finite_pool_realized_ratio` (or an unavailable reason). Scoring cost is not
measured in P0; score outcomes/statuses and failure reasons are recorded
instead. This ratio
is an empirical finite-pool diagnostic, not a continuous-dictionary or WOGA
guarantee. No gamma selector is implemented.
Pool audits retain frozen, JSON-friendly generation metadata and the
caller-declared budget, plus separate stable selector identity and rule fields.
These are provenance declarations rather than verified randomness,
dictionary-coverage, or weak-greedy guarantees; budgets are not measured work.

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
JSON-friendly export path. During export only, non-finite floating values
(NaN and +/-Inf) become JSON `null`; in-memory diagnostic values are
unchanged, and unsupported objects still fail serialization.

When a `WorkLedger` is supplied to `CEMSolver`, measured proposal, score,
operator, objective, refit, failure, and wall-span events are appended
separately. Estimated proposal metadata is not measured work. The built-in
`LinearPDEProblem` emits operator/objective counters when its writable
`work_ledger` field is attached; custom problem plugins without that hook are
not automatically instrumented.
Cost-event metadata is recursively snapshotted on append. Nested mappings and
sequences cannot be changed through caller aliases or through the event, while
`to_dicts()` returns a detached export that callers may edit independently.
Unsupported objects, cycles, nonfinite numbers, and non-string mapping keys
are rejected before an event is recorded.
The public `patch()` operation is instrumented as a separate step/attempt with
an exclusive wall span, and its measured events are linked to the accepted or
rollback record.
