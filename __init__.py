"""Composable, auditable correction-element solver primitives."""
from .state import SolverState, StateTransaction
from .problem import LinearPDEProblem, DiscreteRitzProblem, MatrixFreeRitzProblem, NonlinearPDEProblem
from .operators import Operator, TransposeOperator, NumpyMatrixOperator, CallableOperator
from .learner import Candidate, WeakLearner, AnalyticLearner, LocalBasisLearner
from .scorer import LinearCandidateScorer, RitzCandidateScorer, NonlinearCandidateScorer
from .policy import AcceptancePolicy, MeritDecreasePolicy, StagewisePolicy, FullyCorrectivePolicy
from .ledger import CorrectionLedger
from .pool import CandidatePool, PoolCandidateResult, BestPredictedGainSelector, SelectionResult
from .cost import CostEvent, WorkLedger
from .solver import CEMSolver
from .serialization import candidate_to_json, candidate_from_json

__all__ = ["SolverState", "StateTransaction", "LinearPDEProblem", "DiscreteRitzProblem", "MatrixFreeRitzProblem", "NonlinearPDEProblem", "Operator", "TransposeOperator", "NumpyMatrixOperator", "CallableOperator", "Candidate", "WeakLearner", "AnalyticLearner", "LocalBasisLearner", "LinearCandidateScorer", "RitzCandidateScorer", "NonlinearCandidateScorer", "AcceptancePolicy", "MeritDecreasePolicy", "StagewisePolicy", "FullyCorrectivePolicy", "CorrectionLedger", "CandidatePool", "PoolCandidateResult", "BestPredictedGainSelector", "SelectionResult", "CostEvent", "WorkLedger", "CEMSolver", "candidate_to_json", "candidate_from_json"]
