"""Composable, auditable correction-element solver primitives."""
from .state import SolverState, StateTransaction
from .problem import LinearPDEProblem, NonlinearPDEProblem
from .operators import Operator, TransposeOperator, NumpyMatrixOperator, CallableOperator
from .learner import Candidate, WeakLearner, AnalyticLearner, LocalBasisLearner
from .scorer import LinearCandidateScorer, NonlinearCandidateScorer
from .policy import AcceptancePolicy, MeritDecreasePolicy, StagewisePolicy, FullyCorrectivePolicy
from .ledger import CorrectionLedger
from .solver import CEMSolver
from .serialization import candidate_to_json, candidate_from_json

__all__ = ["SolverState", "StateTransaction", "LinearPDEProblem", "NonlinearPDEProblem", "Operator", "TransposeOperator", "NumpyMatrixOperator", "CallableOperator", "Candidate", "WeakLearner", "AnalyticLearner", "LocalBasisLearner", "LinearCandidateScorer", "NonlinearCandidateScorer", "AcceptancePolicy", "MeritDecreasePolicy", "StagewisePolicy", "FullyCorrectivePolicy", "CorrectionLedger", "CEMSolver", "candidate_to_json", "candidate_from_json"]
