from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .operators import as_operator

@dataclass
class LinearPDEProblem:
    operator: object
    source: np.ndarray
    weights: np.ndarray | None = None
    feasible: Callable[[np.ndarray], bool] | None = None
    def __post_init__(self):
        self.operator = as_operator(self.operator); self.source = np.asarray(self.source, float)
        if self.source.ndim != 1 or self.operator.shape[0] != self.source.size:
            raise ValueError("operator must be (m,n) and source must have shape (m,)")
        if not np.all(np.isfinite(self.source)): raise ValueError("source must be finite")
        self.weights = np.ones(self.source.size) if self.weights is None else np.asarray(self.weights, float)
        if self.weights.shape != self.source.shape or np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0): raise ValueError("weights must be finite and positive")
    def apply_operator(self, values): return self.operator.apply(values)
    def residual(self, state): return self.source - self.apply_operator(state.values)
    def validate_state(self, state):
        if state.values.shape != (self.operator.shape[1],): raise ValueError("state shape is incompatible with operator")
        if not self.admissible(state.values): raise ValueError("initial state is inadmissible")
    def inner(self, a, b): return float(np.sum(self.weights * np.asarray(a) * np.asarray(b)))
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state):
        r = self.residual(state); return 0.5 * self.inner(r, r)

@dataclass
class NonlinearPDEProblem:
    residual_fn: Callable[[np.ndarray], np.ndarray]
    merit_fn: Callable[[np.ndarray], float]
    feasible: Callable[[np.ndarray], bool] | None = None
    def residual(self, state): return np.asarray(self.residual_fn(state.values), float)
    def inner(self, a, b): return float(np.dot(a, b))
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state): return float(self.merit_fn(state.values))
