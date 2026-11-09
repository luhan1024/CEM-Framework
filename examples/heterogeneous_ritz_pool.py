"""Deterministic finite-dimensional heterogeneous candidate-pool tutorial.

This is a software-composition demonstration on a tiny SPD Ritz problem.  It
is not a PDE benchmark, convergence study, novelty claim, or evidence about
the Case experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Make ``python Code/cem_framework/examples/heterogeneous_ritz_pool.py`` work
# from the repository root without requiring an environment variable.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cem_framework import (  # noqa: E402
    AnalyticLearner,
    BestPredictedGainSelector,
    CandidatePool,
    CEMSolver,
    DiscreteRitzProblem,
    FullyCorrectivePolicy,
    LocalBasisLearner,
    RitzCandidateScorer,
    SolverState,
    WorkLedger,
)


def _candidate_summary(candidate):
    return {
        "element_id": candidate.element_id,
        "family": candidate.family,
        "support": dict(candidate.support),
        "provenance": dict(candidate.provenance),
    }


def run_demo() -> dict:
    """Run the deterministic finite-dimensional composition example."""

    # Diagonal SPD data keeps the scorer identity and fully-corrective update
    # exactly aligned while remaining a nontrivial two-component Ritz target.
    K = np.diag([2.0, 3.0])
    source = np.array([2.0, 3.0])
    work = WorkLedger(run_id="heterogeneous-ritz-tutorial")
    problem = DiscreteRitzProblem(K, source, work_ledger=work)
    learners = (
        AnalyticLearner(
            family="analytic_global",
            basis=np.array([1.0, 0.0]),
            element_id="analytic-e1",
            support={"kind": "global", "coordinate": 0},
        ),
        LocalBasisLearner(
            family="local_numeric",
            basis=np.array([0.0, 1.0]),
            element_id="local-e2",
            support={"kind": "local", "coordinate": 1, "radius": 1},
        ),
    )
    solver = CEMSolver(
        problem=problem,
        learners=learners,
        scorer=RitzCandidateScorer(),
        policy=FullyCorrectivePolicy(),
        state=SolverState(np.zeros(2)),
        work_ledger=work,
    )
    selector = BestPredictedGainSelector()
    pools = []
    objective_history = [problem.objective(solver.state)]

    for step_index in range(2):
        residual = problem.residual(solver.state)
        remaining = tuple(
            learner for learner in learners
            if learner.element_id not in solver.state.active_element_ids
        )
        candidates = tuple(
            learner.propose(
                problem,
                solver.state.snapshot(),
                residual.copy(),
                {"seed": 100 + step_index},
            )
            for learner in remaining
        )
        pool = CandidatePool(
            candidates,
            generation_metadata={
                "source": "tutorial_manual_finite_pool",
                "state_hash": solver.state.hash(),
                "step": step_index,
            },
            active_element_ids=solver.state.active_element_ids,
        )
        before = problem.objective(solver.state)
        score = solver.step_pool(
            selector,
            pool=pool,
            budget={"seed": 100 + step_index},
        )
        if score is None:
            raise AssertionError("finite pool should select a positive-gain candidate")
        after = problem.objective(solver.state)
        if not after < before:
            raise AssertionError("real Ritz objective must decrease")
        if not np.isclose(
            score.before - score.after,
            score.predicted_reduction,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError("Ritz score prediction identity failed")
        record = solver.ledger.records[-1]
        selected_id = solver.state.active_element_ids[-1]
        pools.append(
            {
                "pool_id": pool.pool_id,
                "candidates": [_candidate_summary(c) for c in pool.candidates],
                "selected_element_id": selected_id,
                "selection_rule": record.rule,
                "score_prediction": float(score.predicted_reduction),
                "score_before": float(score.before),
                "score_after": float(score.after),
                "objective_before": float(before),
                "objective_after": float(after),
                "ledger_event_ids": list(record.work_event_ids),
            }
        )
        objective_history.append(after)

    return {
        "demo_scope": "finite-dimensional SPD Ritz software composition only",
        "not_a_pde_benchmark": True,
        "not_convergence_evidence": True,
        "not_a_novelty_claim": True,
        "objective_history": objective_history,
        "pools": pools,
        "final_active_element_ids": list(solver.state.active_element_ids),
        "final_values": solver.state.values.tolist(),
        "ledger_event_ids": [event.event_id for event in work.events],
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
