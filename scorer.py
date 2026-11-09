from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Score:
    accepted_shape: bool
    finite: bool
    q: float
    correlation: float
    predicted_reduction: float
    alpha: float
    before: float
    after: float
    reason: str = ""

class LinearCandidateScorer:
    def __init__(self, q_epsilon=1e-14): self.q_epsilon = float(q_epsilon)
    def score(self, problem, state, candidate):
        if candidate.correction.shape != state.values.shape: return Score(False, False, 0, 0, 0, 0, np.nan, np.nan, "shape_mismatch")
        image = problem.apply_operator(candidate.correction)
        if candidate.operator_image is not None and (candidate.operator_image.shape != image.shape or not np.allclose(candidate.operator_image, image, rtol=1e-10, atol=1e-12)):
            return Score(True, False, 0, 0, 0, 0, problem.objective(state), np.inf, "stale_or_inconsistent_operator_image")
        if image.shape != problem.source.shape or not np.all(np.isfinite(image)): return Score(False, False, 0, 0, 0, 0, np.nan, np.nan, "operator_image_shape_or_finite_mismatch")
        r = problem.residual(state); q = problem.inner(image, image); c = problem.inner(r, image); before = problem.objective(state)
        if not np.isfinite(q) or not np.isfinite(c) or q <= self.q_epsilon: return Score(True, False, q, c, 0, 0, before, before, "zero_or_nonfinite_operator_image")
        alpha = c / q; trial = state.values + alpha * candidate.correction
        if not problem.admissible(trial): return Score(True, False, q, c, 0, alpha, before, np.inf, "inadmissible_trial")
        after = 0.5 * problem.inner(problem.source - problem.apply_operator(trial), problem.source - problem.apply_operator(trial))
        return Score(True, np.isfinite(after), q, c, c*c/(2*q), alpha, before, after, "")

class NonlinearCandidateScorer:
    def score(self, problem, state, candidate, alpha=1.0):
        if candidate.correction.shape != state.values.shape: return Score(False, False, 0, 0, 0, alpha, np.nan, np.nan, "shape_mismatch")
        before = problem.objective(state); trial = state.values + alpha * candidate.correction
        if not problem.admissible(trial): return Score(True, False, 0, 0, 0, alpha, before, np.inf, "inadmissible_trial")
        after = float(problem.merit_fn(trial)); return Score(True, np.isfinite(after), 0, 0, before-after, alpha, before, after, "")
