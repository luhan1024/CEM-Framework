"""Fixed-grid 1D Poisson Ritz regression/tutorial.

This is a single finite-difference grid software regression. It is not a
continuous-PDE certificate, mesh-convergence study, speed comparison,
research benchmark, or novelty claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cem_framework import (  # noqa: E402
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


def run_regression() -> dict:
    h = 1.0 / 4.0
    K = (1.0 / h**2) * np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    )
    source = np.ones(3)
    oracle = np.linalg.solve(K, source)
    expected_oracle = np.array([3.0 / 32.0, 1.0 / 8.0, 3.0 / 32.0])
    if not np.allclose(oracle, expected_oracle, rtol=0.0, atol=1e-15):
        raise AssertionError("offline Poisson oracle changed")

    work = WorkLedger(run_id="fixed-grid-poisson-ritz")
    problem = DiscreteRitzProblem(K, source, work_ledger=work)
    learners = tuple(
        LocalBasisLearner(
            family="finite_difference_hat",
            basis=np.eye(3)[node],
            element_id=f"hat-node-{node + 1}",
            support={"grid": "fixed-3-interior-node", "node": node + 1},
        )
        for node in range(3)
    )
    solver = CEMSolver(
        problem, learners, RitzCandidateScorer(), FullyCorrectivePolicy(),
        SolverState(np.zeros(3)), work_ledger=work,
    )
    selector = BestPredictedGainSelector()
    objective_history = [problem.objective(solver.state)]
    pool_records = []

    for step_index in range(3):
        residual = problem.residual(solver.state)
        remaining = tuple(
            learner for learner in learners
            if learner.element_id not in solver.state.active_element_ids
        )
        candidates = tuple(
            learner.propose(problem, solver.state.snapshot(), residual.copy(),
                            {"seed": 200 + step_index})
            for learner in remaining
        )
        pool = CandidatePool(
            candidates,
            generation_metadata={
                "source": "fixed-grid-poisson-regression",
                "state_hash": solver.state.hash(),
                "grid_nodes": 3,
                "step": step_index,
            },
            active_element_ids=solver.state.active_element_ids,
        )
        before = problem.objective(solver.state)
        score = solver.step_pool(selector, pool=pool, budget={"seed": 200 + step_index})
        if score is None:
            raise AssertionError("fixed-grid pool should select a positive-gain hat")
        after = problem.objective(solver.state)
        if not after <= before + 1e-13:
            raise AssertionError("Ritz objective is not monotone decreasing")
        if not np.isclose(score.before - score.after, score.predicted_reduction,
                          rtol=1e-11, atol=1e-13):
            raise AssertionError("Ritz score prediction identity failed")
        record = solver.ledger.records[-1]
        pool_records.append({
            "pool_id": pool.pool_id,
            "candidate_ids": [candidate.element_id for candidate in pool.candidates],
            "selected_element_id": solver.state.active_element_ids[-1],
            "objective_before": float(before),
            "objective_after": float(after),
            "predicted_reduction": float(score.predicted_reduction),
            "work_event_ids": list(record.work_event_ids),
        })
        objective_history.append(after)

    residual = problem.residual(solver.state)
    final_values = solver.state.values.copy()
    if not np.allclose(final_values, oracle, rtol=0.0, atol=1e-12):
        raise AssertionError("fixed-grid Ritz state does not match offline oracle")
    if not np.linalg.norm(residual, ord=np.inf) <= 1e-12:
        raise AssertionError("fixed-grid Ritz residual is outside tolerance")
    ledger_json = solver.ledger.to_dicts()
    json.dumps(ledger_json, allow_nan=False)
    top_event_ids = {event.event_id for event in work.events}
    if not all(set(record["work_event_ids"]).issubset(top_event_ids)
               for record in pool_records):
        raise AssertionError("pool ledger references are not present in WorkLedger")
    return {
        "scope": "fixed 3-node finite-difference SPD Poisson regression only",
        "not_continuous_pde_certificate": True,
        "not_mesh_convergence_or_speed_comparison": True,
        "not_research_benchmark_or_novelty_claim": True,
        "oracle": oracle.tolist(),
        "objective_history": objective_history,
        "final_values": final_values.tolist(),
        "residual_inf_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "active_element_ids": list(solver.state.active_element_ids),
        "pool_records": pool_records,
        "ledger_event_ids": [event.event_id for event in work.events],
    }


def main() -> None:
    print(json.dumps(run_regression(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
