"""Discrete problem definitions and operator interfaces."""

from .operators import CallableOperator, NumpyMatrixOperator, Operator, TransposeOperator
from .problem import (
    DiscreteRitzProblem,
    LinearPDEProblem,
    MatrixFreeRitzProblem,
    NonlinearPDEProblem,
)
from .ritz_certificate import (
    DiscreteRitzStepCertificate,
    certify_discrete_ritz_step,
    verify_discrete_ritz_certificate,
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
    "DiscreteRitzStepCertificate",
    "certify_discrete_ritz_step",
    "verify_discrete_ritz_certificate",
]
