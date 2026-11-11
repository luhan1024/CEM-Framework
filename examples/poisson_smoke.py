import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from cem_framework import CEMSolver, SolverState, LinearPDEProblem, AnalyticLearner, LocalBasisLearner, LinearCandidateScorer, MeritDecreasePolicy

def main():
    # -u''=1 on (0,1), u(0)=u(1)=0, three interior nodes, h=1/4.
    h = 0.25; A = (1.0 / h**2) * np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 2.]])
    p = LinearPDEProblem(A, np.ones(3))
    learners = (AnalyticLearner("analytic", np.array([1.,0,0.]), "a0"), LocalBasisLearner("local_numeric", np.array([0.,1,0.]), "l0", {"nodes":[1]}))
    s = CEMSolver(p, learners, LinearCandidateScorer(), MeritDecreasePolicy(), SolverState(np.zeros(3)))
    s.step({"seed": 1}); print("1D Poisson smoke/tutorial", s.state.values, "records=", len(s.ledger.records))
if __name__ == "__main__": main()
