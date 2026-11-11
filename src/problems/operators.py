"""Small operator protocol and adapters used by the linear solver core."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Operator(Protocol):
    shape: tuple[int, int]
    def apply(self, values: np.ndarray) -> np.ndarray: ...

@runtime_checkable
class TransposeOperator(Protocol):
    def transpose_apply(self, values: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class NumpyMatrixOperator:
    matrix: np.ndarray
    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("matrix operator must be finite and 2-D")
        matrix = matrix.copy(); matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
    @property
    def shape(self): return self.matrix.shape
    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.shape[1],): raise ValueError("operator input shape mismatch")
        return self.matrix @ values
    def transpose_apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.shape[0],): raise ValueError("transpose input shape mismatch")
        return self.matrix.T @ values

@dataclass(frozen=True)
class CallableOperator:
    """Matrix-free adapter; ``transpose`` is optional but explicit when needed."""
    shape: tuple[int, int]
    fn: Callable[[np.ndarray], np.ndarray]
    transpose: Callable[[np.ndarray], np.ndarray] | None = None
    def __post_init__(self):
        if len(self.shape) != 2 or any(int(n) <= 0 for n in self.shape): raise ValueError("operator shape must be positive (m,n)")
    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (self.shape[1],): raise ValueError("operator input shape mismatch")
        out = np.asarray(self.fn(values), dtype=float)
        if out.shape != (self.shape[0],) or not np.all(np.isfinite(out)): raise ValueError("operator output shape or finiteness mismatch")
        return out
    def transpose_apply(self, values):
        if self.transpose is None: raise NotImplementedError("transpose action was not supplied")
        values = np.asarray(values, dtype=float)
        if values.shape != (self.shape[0],): raise ValueError("transpose input shape mismatch")
        out = np.asarray(self.transpose(values), dtype=float)
        if out.shape != (self.shape[1],) or not np.all(np.isfinite(out)): raise ValueError("transpose output shape or finiteness mismatch")
        return out

def as_operator(operator) -> Operator:
    if isinstance(operator, np.ndarray): return NumpyMatrixOperator(operator)
    if callable(operator) and hasattr(operator, "shape"):
        return CallableOperator(tuple(operator.shape), operator)
    if not hasattr(operator, "shape") or not hasattr(operator, "apply"):
        raise TypeError("operator must be a 2-D array or expose shape and apply(v)")
    return operator
