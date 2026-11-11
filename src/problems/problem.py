from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .operators import as_operator


class RitzEnergyProblem:
    """Nominal contract for finite-dimensional Ritz-energy adapters.

    Inheriting this marker is required at solver/scorer gates; matching the
    attribute names alone is insufficient. Each implementation remains
    responsible for its own SPD and objective-consistency guarantees.
    """

    objective_geometry = "ritz_energy"

    def validate_contract(self) -> None:
        shape = getattr(self, "shape", None)
        if (not isinstance(shape, tuple) or len(shape) != 2
                or any(isinstance(size, (bool, np.bool_))
                       or not isinstance(size, (int, np.integer)) for size in shape)
                or int(shape[0]) != int(shape[1]) or int(shape[0]) < 0):
            raise TypeError("RitzEnergyProblem requires a square-compatible shape")
        source = np.asarray(getattr(self, "source", None))
        if source.shape != (int(shape[0]),) or not np.all(np.isfinite(source)):
            raise TypeError("RitzEnergyProblem source must be a finite shape-compatible vector")
        operator = getattr(self, "operator", None)
        operator_shape = getattr(operator, "shape", None)
        if operator_shape != shape:
            raise TypeError("RitzEnergyProblem operator shape must match problem shape")
        if operator is self:
            if not callable(getattr(self, "apply_operator", None)):
                raise TypeError("self operator must implement apply_operator")
        elif not callable(getattr(operator, "apply", None)):
            raise TypeError("RitzEnergyProblem operator must expose apply")
        for method in ("apply_operator", "residual", "objective", "admissible", "validate_state"):
            if not callable(getattr(self, method, None)):
                raise TypeError(f"RitzEnergyProblem requires {method}()")

@dataclass
class LinearPDEProblem:
    objective_geometry = "strong_residual_least_squares"
    operator: object
    source: np.ndarray
    weights: np.ndarray | None = None
    feasible: Callable[[np.ndarray], bool] | None = None
    work_ledger: object | None = None
    def __post_init__(self):
        self.operator = as_operator(self.operator); self.source = np.asarray(self.source, float)
        if self.source.ndim != 1 or self.operator.shape[0] != self.source.size:
            raise ValueError("operator must be (m,n) and source must have shape (m,)")
        if not np.all(np.isfinite(self.source)): raise ValueError("source must be finite")
        self.weights = np.ones(self.source.size) if self.weights is None else np.asarray(self.weights, float)
        if self.weights.shape != self.source.shape or np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0): raise ValueError("weights must be finite and positive")
    def _work(self, quantity, phase, status="completed"):
        if self.work_ledger is not None: return self.work_ledger.record_call(phase, quantity, status=status)
        return None
    def apply_operator(self, values):
        try: result=self.operator.apply(values)
        except Exception:
            self._work("operator_apply_calls", "operator_apply_failed", "failed")
            raise
        self._work("operator_apply_calls", "operator_apply"); return result
    def residual(self, state): return self.source - self.apply_operator(state.values)
    def validate_state(self, state):
        if state.values.shape != (self.operator.shape[1],): raise ValueError("state shape is incompatible with operator")
        if not self.admissible(state.values): raise ValueError("initial state is inadmissible")
    def inner(self, a, b): return float(np.sum(self.weights * np.asarray(a) * np.asarray(b)))
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state):
        try: r = self.residual(state); result=0.5 * self.inner(r, r)
        except Exception:
            self._work("objective_eval_calls", "objective_failed", "failed"); raise
        self._work("objective_eval_calls", "objective"); return result

@dataclass
class DiscreteRitzProblem(RitzEnergyProblem):
    """Finite-dimensional SPD Ritz adapter; this is not a weak-form PDE integrator."""
    K: np.ndarray
    source: np.ndarray
    feasible: Callable[[np.ndarray], bool] | None = None
    work_ledger: object | None = None
    spd_tol: float = 1e-12
    objective_geometry = "ritz_energy"
    def __post_init__(self):
        K=np.asarray(self.K, dtype=float); f=np.asarray(self.source, dtype=float)
        if not np.isfinite(self.spd_tol) or self.spd_tol <= 0: raise ValueError("spd_tol must be finite and positive")
        if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0: raise ValueError("K must be non-empty square")
        if f.ndim != 1 or f.size != K.shape[0]: raise ValueError("source shape is incompatible with K")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(f)): raise ValueError("K and source must be finite")
        if not np.allclose(K, K.T, rtol=1e-12, atol=1e-14): raise ValueError("K must be symmetric")
        try: eig=np.linalg.eigvalsh(K)
        except np.linalg.LinAlgError as exc: raise ValueError("K SPD check failed") from exc
        if eig[0] <= float(self.spd_tol): raise ValueError("K must be positive definite")
        self.K=K.copy(); self.K.setflags(write=False); self.source=f.copy(); self.source.setflags(write=False)
    @property
    def shape(self): return self.K.shape
    @property
    def operator(self): return self
    def _work(self, quantity, phase, status="completed"):
        if self.work_ledger is not None: return self.work_ledger.record_call(phase, quantity, status=status)
        return None
    def apply_operator(self, values):
        values=np.asarray(values, dtype=float)
        if values.shape != self.source.shape: raise ValueError("values shape is incompatible with K")
        try: result=self.K @ values
        except Exception:
            self._work("operator_apply_calls", "ritz_operator_apply_failed", "failed"); raise
        self._work("operator_apply_calls", "ritz_operator_apply"); return result
    def residual(self, state): return self.source - self.apply_operator(state.values)
    def validate_state(self, state):
        if state.values.shape != self.source.shape: raise ValueError("state shape is incompatible with K")
        if not self.admissible(state.values): raise ValueError("initial state is inadmissible")
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state):
        try:
            u=np.asarray(state.values, dtype=float); Ku=self.apply_operator(u); result=0.5*float(u @ Ku)-float(self.source @ u)
        except Exception:
            self._work("objective_eval_calls", "ritz_objective_failed", "failed"); raise
        self._work("objective_eval_calls", "ritz_objective"); return result

@dataclass
class MatrixFreeRitzProblem(RitzEnergyProblem):
    """Matrix-free finite Ritz adapter with caller-declared SPD semantics."""
    operator: object
    source: np.ndarray
    feasible: Callable[[np.ndarray], bool] | None = None
    work_ledger: object | None = None
    spd_assumption: str = "caller_declared"
    objective_geometry = "ritz_energy"
    @property
    def shape(self): return tuple(int(size) for size in self.operator.shape)
    def __post_init__(self):
        if self.spd_assumption != "caller_declared": raise ValueError("spd_assumption must be exactly 'caller_declared'")
        shape=getattr(self.operator, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[0]) != int(shape[1]) or int(shape[0]) <= 0 or not hasattr(self.operator, "apply"):
            raise TypeError("operator must be a nonempty square Operator with apply")
        f=np.asarray(self.source, dtype=float)
        if f.ndim != 1 or f.size != int(shape[0]) or not np.all(np.isfinite(f)): raise ValueError("source must be finite with operator-compatible shape")
        self.source=f.copy(); self.source.setflags(write=False)
    def _work(self, quantity, phase, status="completed"):
        if self.work_ledger is not None: return self.work_ledger.record_call(phase, quantity, status=status)
        return None
    def apply_operator(self, values):
        values=np.asarray(values, dtype=float); n=int(self.operator.shape[1])
        if values.shape != (n,) or not np.all(np.isfinite(values)): raise ValueError("operator input shape or finiteness mismatch")
        try: result=np.asarray(self.operator.apply(values), dtype=float)
        except Exception:
            self._work("operator_apply_calls", "matrix_free_ritz_operator_apply_failed", "failed"); raise
        if result.shape != self.source.shape or not np.all(np.isfinite(result)):
            self._work("operator_apply_calls", "matrix_free_ritz_operator_apply_failed", "failed"); raise ValueError("operator output shape or finiteness mismatch")
        self._work("operator_apply_calls", "matrix_free_ritz_operator_apply"); return result
    def residual(self, state): return self.source - self.apply_operator(state.values)
    def validate_state(self, state):
        if state.values.shape != self.source.shape: raise ValueError("state shape is incompatible with matrix-free Ritz operator")
        if not self.admissible(state.values): raise ValueError("initial state is inadmissible")
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state):
        try:
            u=np.asarray(state.values, dtype=float); Ku=self.apply_operator(u); result=0.5*float(u @ Ku)-float(self.source @ u)
        except Exception:
            self._work("objective_eval_calls", "matrix_free_ritz_objective_failed", "failed"); raise
        self._work("objective_eval_calls", "matrix_free_ritz_objective"); return result

@dataclass
class NonlinearPDEProblem:
    residual_fn: Callable[[np.ndarray], np.ndarray]
    merit_fn: Callable[[np.ndarray], float]
    feasible: Callable[[np.ndarray], bool] | None = None
    work_ledger: object | None = None
    def residual(self, state): return np.asarray(self.residual_fn(state.values), float)
    def inner(self, a, b): return float(np.dot(a, b))
    def admissible(self, values): return True if self.feasible is None else bool(self.feasible(np.asarray(values)))
    def objective(self, state): return float(self.merit_fn(state.values))
