# CEM Framework

This is an isolated, NumPy-first composable API for correction-element
solvers. It is an engineering scaffold, not a claim of a new algorithm or
convergence theorem. Existing Case 1--6 scripts and assets are intentionally
not imported or modified.

## Quick start

From the project root:

```powershell
python -m pip install -e . -r requirements.txt
python examples/poisson_smoke.py
python examples/poisson_ritz_regression.py
python examples/heterogeneous_ritz_pool.py
python -m unittest discover -s tests -v
```

`heterogeneous_ritz_pool.py` is a deterministic finite-dimensional SPD Ritz
tutorial. It composes an `AnalyticLearner` and a `LocalBasisLearner` with
`DiscreteRitzProblem`, `RitzCandidateScorer`, `CandidatePool`,
`BestPredictedGainSelector`, `FullyCorrectivePolicy`, `SolverState`,
`CEMSolver`, and `WorkLedger`. The output shows candidate family/support/
provenance, finite-pool selections, objective decrease, score-prediction
identity checks, and stable ledger event IDs. It is only a software
composition demo: it is not a PDE benchmark, convergence evidence, or a
novelty claim, and it does not run Julia/Arb or alter Case data.

`poisson_ritz_regression.py` is a separate fixed-grid 1D Poisson software
regression: a 3-node finite-difference SPD matrix with `h=1/4` and
`f=ones(3)`, coordinate/hat candidates, finite-pool selection, and fully
corrective refits. It checks objective monotonicity, score-prediction
identity, the offline oracle `np.linalg.solve(K, f) =
[3/32, 1/8, 3/32]`, final residual/value tolerances, JSON serialization, and
ledger event references. It deliberately uses one fixed grid only; it is not
a continuous-PDE certificate, mesh-convergence or speed comparison, research
benchmark, or novelty evidence.

`P1Poisson1DData` and `EquilibratedFlux1DEstimator` provide a separate narrow
verification adapter for zero-Dirichlet 1D Poisson with a continuous P1
solution and piecewise-constant source. With the boundary-constrained
equilibrated flux, exact arithmetic gives `eta_eq` equal to the exact
H1-seminorm error in this special case. The float64 output is labeled
`numerical_bound`, not machine-certified; it is neither a higher-dimensional
estimator nor a CEM accuracy/efficiency claim. See `docs/user_guide.md`.

Define a `LinearPDEProblem` with `(A, f, W)`, assemble any number of
`WeakLearner` implementations, choose a scorer and `MeritDecreasePolicy`, and
construct `CEMSolver`. `Candidate` separates learner-internal `parameters`
from outer coefficients held by `SolverState`. A rejected candidate leaves
state, active IDs and coefficients unchanged but appends a ledger record with
cost, hashes and the failure reason.

The linear operator is not required to be a dense matrix. A NumPy matrix is
wrapped by `NumpyMatrixOperator`; matrix-free code can provide a
`CallableOperator((m, n), apply_fn, transpose_fn=None)` or any object exposing
`shape` and `apply(v)`. The core solver only calls `problem.apply_operator`;
`transpose_apply` is an optional `TransposeOperator` capability for extensions,
not part of the minimum contract.
For example:

```python
import numpy as np
from core import SolverState
from enrichment import AnalyticLearner
from problems import CallableOperator, LinearPDEProblem
from solver import CEMSolver
from strategies import LinearCandidateScorer, MeritDecreasePolicy

op = CallableOperator((2, 2), lambda x: np.array([x[0] + 2*x[1], 3*x[0] - x[1]]))
problem = LinearPDEProblem(op, np.array([1., 0.]))
solver = CEMSolver(problem,
    (AnalyticLearner("basis", np.array([1., 0.]), "basis"),),
    LinearCandidateScorer(), MeritDecreasePolicy(), SolverState(np.zeros(2)))
solver.step()
```

`StagewisePolicy` and `FullyCorrectivePolicy` are explicit semantic markers:
stagewise does not refit old elements; fully corrective is allowed to refit
outer coefficients. `solver.patch(append=..., replace=..., remove=...)` is
available only with `FullyCorrectivePolicy`; it refits the active set and logs
the before/after hashes, objective, touched IDs and estimated work. “frozen”
means not touched by that update, not that the PDE solution is spatially
unchanged outside support.

No policy currently supports active-set checkpoint restart. An initial
`state.values` may be any finite baseline, but `active_element_ids` must be
empty and `outer_coefficients` must be `None` or empty for every policy.

NumPy is the only core dependency. There are currently no built-in SciPy or
PyTorch adapters; users may use those libraries inside their own operator,
learner, or scorer components. Install locally with
`python -m pip install -e . -r requirements.txt`. Python 3.10+ is supported;
`src/` is the installation and import isolation boundary. Inside `src/`,
`core`, `problems`, `enrichment`, and `strategies` are the responsibility
groups; `solver.py` and the narrow `estimators_1d.py` adapter sit at the source
root. Imports use those entry points directly, for example
`from solver import CEMSolver`, `from problems import DiscreteRitzProblem`,
and `from enrichment import Candidate`.

These top-level names (`core`, `problems`, `solver`, and others) are generic
and may collide with unrelated installed modules if multiple projects share
one import path. Use a dedicated virtual environment or isolate the intended
`src/` directory when importing the framework.

The current public package deliberately does not expose a BlockPolicy: block
selection/refit semantics require a separate, explicit block candidate
contract and are not silently treated as stagewise updates.

Detailed usage and API contracts are in
[`docs/user_guide.md`](docs/user_guide.md) and
[`docs/api_reference.md`](docs/api_reference.md).

Learners are tried in the tuple order supplied to `CEMSolver`; the first
candidate accepted by the policy is committed and `step()` returns. The solver
does not score all learners and select a best candidate. Custom policies may
customize `decide(score)`, but `mode` must be omitted (defaulting to
stagewise), `"stagewise"`, or `"fully_corrective"`; unknown modes are rejected.
