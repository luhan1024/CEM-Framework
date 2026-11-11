"""Correction candidates, finite pools, and their serialization."""

from .learner import AnalyticLearner, Candidate, LocalBasisLearner, WeakLearner
from .pool import (
    BestPredictedGainSelector,
    CandidatePool,
    PoolCandidateResult,
    SelectionResult,
)
from .serialization import candidate_from_json, candidate_to_json

__all__ = [
    "Candidate",
    "WeakLearner",
    "AnalyticLearner",
    "LocalBasisLearner",
    "CandidatePool",
    "PoolCandidateResult",
    "BestPredictedGainSelector",
    "SelectionResult",
    "candidate_to_json",
    "candidate_from_json",
]
