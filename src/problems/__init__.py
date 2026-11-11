"""Discrete problem definitions and operator interfaces."""

from .operators import CallableOperator, NumpyMatrixOperator, Operator, TransposeOperator
from .problem import (
    DiscreteRitzProblem,
    LinearPDEProblem,
    MatrixFreeRitzProblem,
    NonlinearPDEProblem,
)

__all__ = [
    "Operator",
    "TransposeOperator",
    "NumpyMatrixOperator",
    "CallableOperator",
    "LinearPDEProblem",
    "DiscreteRitzProblem",
    "MatrixFreeRitzProblem",
    "NonlinearPDEProblem",
]
