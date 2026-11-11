from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

def _copy(x):
    return np.array(x, copy=True)

@dataclass
class SolverState:
    values: np.ndarray
    active_element_ids: tuple[str, ...] = ()
    outer_coefficients: np.ndarray | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float).copy()
        if self.values.ndim != 1:
            raise ValueError("SolverState.values must be one-dimensional")
        if self.outer_coefficients is not None:
            self.outer_coefficients = np.asarray(self.outer_coefficients, dtype=float).copy()

    def snapshot(self) -> "StateSnapshot":
        return StateSnapshot(_copy(self.values), tuple(self.active_element_ids), None if self.outer_coefficients is None else _copy(self.outer_coefficients))

    def restore(self, snap: "StateSnapshot") -> None:
        self.values = _copy(snap.values); self.active_element_ids = tuple(snap.active_element_ids)
        self.outer_coefficients = None if snap.outer_coefficients is None else _copy(snap.outer_coefficients)

    def hash(self) -> str:
        h = hashlib.sha256(self.values.tobytes()); h.update(repr(self.active_element_ids).encode())
        if self.outer_coefficients is not None: h.update(self.outer_coefficients.tobytes())
        return h.hexdigest()

@dataclass(frozen=True)
class StateSnapshot:
    values: np.ndarray
    active_element_ids: tuple[str, ...]
    outer_coefficients: np.ndarray | None

    def __post_init__(self):
        self.values.setflags(write=False)
        if self.outer_coefficients is not None: self.outer_coefficients.setflags(write=False)

class StateTransaction:
    def __init__(self, state: SolverState):
        self.state = state; self.before = state.snapshot(); self.committed = False
    def commit(self): self.committed = True
    def rollback(self): self.state.restore(self.before); self.committed = False
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.committed: self.rollback()
        return False
