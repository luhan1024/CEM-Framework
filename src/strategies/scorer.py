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
    objective_geometry = "strong_residual_least_squares"
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

class RitzCandidateScorer:
    objective_geometry = "ritz_energy"
    """Stagewise scorer for the finite-dimensional discrete Ritz objective."""
    def __init__(self, q_epsilon=1e-14, identity_tolerance=1e-10): self.q_epsilon=float(q_epsilon); self.identity_tolerance=float(identity_tolerance)
    def score(self, problem, state, candidate):
        from problems.problem import DiscreteRitzProblem, MatrixFreeRitzProblem
        if not isinstance(problem, (DiscreteRitzProblem, MatrixFreeRitzProblem)): raise TypeError("RitzCandidateScorer requires a Ritz problem")
        if candidate.correction.shape != state.values.shape: return Score(False, False, 0, 0, 0, 0, np.nan, np.nan, "shape_mismatch")
        image=problem.apply_operator(candidate.correction)
        if candidate.operator_image is not None and (candidate.operator_image.shape != image.shape or not np.allclose(candidate.operator_image, image, rtol=1e-10, atol=1e-12)):
            return Score(True, False, 0, 0, 0, 0, problem.objective(state), np.inf, "stale_or_inconsistent_operator_image")
        residual=problem.residual(state); q=float(candidate.correction @ image); c=float(residual @ candidate.correction); before=problem.objective(state)
        if not np.isfinite(q) or not np.isfinite(c) or q <= self.q_epsilon: return Score(True, False, q, c, 0, 0, before, before, "zero_or_nonfinite_ritz_curvature")
        alpha=c/q; trial=state.values+alpha*candidate.correction
        if not problem.admissible(trial): return Score(True, False, q, c, 0, alpha, before, np.inf, "inadmissible_trial")
        after=problem.objective(type("Trial",(),{"values":trial})())
        reduction=before-after; predicted=c*c/(2*q)
        if not np.isfinite(after) or not np.isclose(reduction, predicted, rtol=self.identity_tolerance, atol=self.identity_tolerance*max(1.0,abs(before),abs(after))):
            return Score(True, False, q, c, predicted, alpha, before, after, "ritz_identity_mismatch")
        return Score(True, True, q, c, predicted, alpha, before, after, "")

class NonlinearCandidateScorer:
    def score(self, problem, state, candidate, alpha=1.0):
        if candidate.correction.shape != state.values.shape: return Score(False, False, 0, 0, 0, alpha, np.nan, np.nan, "shape_mismatch")
        before = problem.objective(state); trial = state.values + alpha * candidate.correction
        if not problem.admissible(trial): return Score(True, False, 0, 0, 0, alpha, before, np.inf, "inadmissible_trial")
        after = float(problem.merit_fn(trial)); return Score(True, np.isfinite(after), 0, 0, before-after, alpha, before, after, "")
