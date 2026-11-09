# `cem_framework`

This is an isolated, NumPy-first composable API for correction-element
solvers. It is an engineering scaffold, not a claim of a new algorithm or
convergence theorem. Existing Case 1--6 scripts and assets are intentionally
not imported or modified.

## Quick start

From the project root:

```powershell
$env:PYTHONPATH = "$PWD\Code"
python Code/cem_framework/examples/poisson_smoke.py
python -m unittest discover -s Code/cem_framework/tests -v
```

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
from cem_framework import (CallableOperator, LinearPDEProblem, SolverState,
    AnalyticLearner, LinearCandidateScorer, MeritDecreasePolicy, CEMSolver)

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

NumPy is the required core dependency. SciPy and PyTorch adapters are optional
extras and are not imported by this package. Install locally with
`python -m pip install -e Code/cem_framework` or run directly with
`PYTHONPATH=Code`. Python 3.10+ is supported.

The current public package deliberately does not expose a BlockPolicy: block
selection/refit semantics require a separate, explicit block candidate
contract and are not silently treated as stagewise updates.
