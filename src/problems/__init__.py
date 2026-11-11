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
from .polynomial_poisson_1d import (
    PolynomialPoisson1DError,
    PolynomialPoissonEnclosure1D,
    build_polynomial_poisson_1d_enclosure,
    verify_polynomial_poisson_1d_enclosure,
)
from .p1_poisson_1d import (
    P1PoissonStepCertificate,
    certify_p1_poisson_step,
    verify_p1_poisson_step_certificate,
)
from .p1_poisson_ritz import P1Poisson1DRitzProblem

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
    "PolynomialPoisson1DError",
    "PolynomialPoissonEnclosure1D",
    "build_polynomial_poisson_1d_enclosure",
    "verify_polynomial_poisson_1d_enclosure",
    "P1PoissonStepCertificate",
    "certify_p1_poisson_step",
    "verify_p1_poisson_step_certificate",
    "P1Poisson1DRitzProblem",
]
