"""Narrow 1D P1 Poisson equilibrated-flux estimator.

The returned value is computed in floating point from exact cell-integration
formulae. It is not an outward-rounded or machine-certified interval bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


_INTEGRATION_RULE = "exact_piecewise_constant_cell_integrals_float64"
_SCHEMA_VERSION = "1"


def _array_hash(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": canonical.shape}, separators=(",", ":")).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class P1Poisson1DData:
    """Validated zero-Dirichlet 1D Poisson data with cellwise-constant source."""

    nodes: np.ndarray
    cell_source: np.ndarray
    dirichlet_values: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        source = np.asarray(self.cell_source, dtype=np.float64)
        if nodes.ndim != 1 or nodes.size < 2:
            raise ValueError("nodes must be a one-dimensional array with at least two entries")
        if source.ndim != 1 or source.size != nodes.size - 1:
            raise ValueError("cell_source must have one value per mesh cell")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(source)):
            raise ValueError("nodes and cell_source must be finite")
        if nodes[0] != 0.0 or nodes[-1] != 1.0 or np.any(np.diff(nodes) <= 0.0):
            raise ValueError("nodes must be strictly increasing from 0 to 1")
        if len(self.dirichlet_values) != 2 or tuple(map(float, self.dirichlet_values)) != (0.0, 0.0):
            raise ValueError("only homogeneous zero Dirichlet values are supported")
        nodes = nodes.copy()
        source = source.copy()
        nodes.setflags(write=False)
        source.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "cell_source", source)
        object.__setattr__(self, "dirichlet_values", (0.0, 0.0))

    @property
    def data_hash(self) -> str:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "nodes_sha256": _array_hash(self.nodes),
            "cell_source_sha256": _array_hash(self.cell_source),
            "dirichlet_values": self.dirichlet_values,
            "integration_rule": _INTEGRATION_RULE,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EstimatorRecord:
    """Immutable estimator result; failed records never carry an eta value."""

    status: str
    eta_eq: float | None
    data_hash: str | None
    solution_hash: str | None
    integration_rule: str
    assumptions: tuple[str, ...]
    flux_node_values: tuple[float, ...] = ()
    equilibrium_defect: float | None = None
    equilibrium_tolerance: float | None = None
    failure_reason: str | None = None
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        assumptions = tuple(self.assumptions)
        flux_values = tuple(float(x) for x in self.flux_node_values)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "flux_node_values", flux_values)
        for name, value in (("data_hash", self.data_hash), ("solution_hash", self.solution_hash)):
            if value is not None and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest or None")
        if not self.integration_rule or not all(isinstance(item, str) for item in assumptions):
            raise ValueError("integration_rule and assumption strings must be nonempty/valid")
        if any(not math.isfinite(value) for value in flux_values):
            raise ValueError("flux_node_values must be finite")
        if self.status == "numerical_bound":
            if self.eta_eq is None or not math.isfinite(self.eta_eq) or self.eta_eq < 0.0:
                raise ValueError("successful numerical_bound records require finite nonnegative eta_eq")
            if self.data_hash is None or self.solution_hash is None:
                raise ValueError("successful records require data and solution hashes")
            if self.equilibrium_defect is None or not math.isfinite(self.equilibrium_defect) or self.equilibrium_defect < 0.0:
                raise ValueError("successful records require a finite nonnegative equilibrium defect")
            if self.equilibrium_tolerance is None or not math.isfinite(self.equilibrium_tolerance) or self.equilibrium_tolerance < 0.0:
                raise ValueError("successful records require a finite nonnegative equilibrium tolerance")
            if self.equilibrium_defect > self.equilibrium_tolerance:
                raise ValueError("successful records require equilibrium_defect <= equilibrium_tolerance")
            if not flux_values or not assumptions:
                raise ValueError("successful records require flux values and assumption metadata")
            if self.failure_reason is not None:
                raise ValueError("successful records cannot contain a failure reason")
        elif self.status == "failed":
            if self.eta_eq is not None or not self.failure_reason:
                raise ValueError("failed records require a reason and must not contain eta_eq")
            if flux_values or self.equilibrium_defect is not None or self.equilibrium_tolerance is not None:
                raise ValueError("failed records must not contain flux or equilibrium results")
        else:
            raise ValueError("status must be 'numerical_bound' or 'failed'")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-friendly snapshot."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "eta_eq": self.eta_eq,
            "data_hash": self.data_hash,
            "solution_hash": self.solution_hash,
            "integration_rule": self.integration_rule,
            "assumptions": list(self.assumptions),
            "flux_node_values": list(self.flux_node_values),
            "equilibrium_defect": self.equilibrium_defect,
            "equilibrium_tolerance": self.equilibrium_tolerance,
            "failure_reason": self.failure_reason,
        }


class EquilibratedFlux1DEstimator:
    """Compute the 1D equilibrated-flux formula for a conforming P1 solution.

    For exact data and exact arithmetic, the boundary-constrained equilibrated
    flux equals the exact derivative in this 1D Dirichlet special case. Hence
    eta_eq equals, rather than merely bounds, the exact H1-seminorm error.
    Floating-point results remain numerical estimates, not certified intervals.
    """

    assumptions = (
        "domain=(0,1)",
        "zero_dirichlet_boundary_values",
        "continuous_conforming_P1_solution",
        "piecewise_constant_L2_source",
        "exact_piecewise_cell_integration_formula",
        "cell_count_scaled_roundoff_acceptance_tolerance_not_error_bound",
    )

    def __init__(self, equilibrium_rtol: float | None = None) -> None:
        """Set an optional explicit relative tolerance.

        By default, ``estimate`` derives a scale-aware roundoff acceptance
        threshold from the number of cells. An explicit value overrides that
        default and remains a numerical acceptance threshold, not an error
        bound or certificate.
        """
        if equilibrium_rtol is not None and (
            not math.isfinite(equilibrium_rtol) or equilibrium_rtol < 0.0
        ):
            raise ValueError("equilibrium_rtol must be finite and nonnegative, or None")
        self.equilibrium_rtol = None if equilibrium_rtol is None else float(equilibrium_rtol)

    @staticmethod
    def _default_equilibrium_rtol(cell_count: int) -> float:
        """Return a conservative gamma_n-style allowance for accumulated ops."""
        eps = np.finfo(np.float64).eps
        operation_count = cell_count + 4
        eps_n = operation_count * eps
        if eps_n >= 0.5:
            raise FloatingPointError("mesh is too large for the configured float64 roundoff model")
        gamma_n = eps_n / (1.0 - eps_n)
        return 64.0 * gamma_n

    def estimate(self, data: P1Poisson1DData, solution_values: Any) -> EstimatorRecord:
        data_hash = data.data_hash if isinstance(data, P1Poisson1DData) else None
        solution_hash: str | None = None
        try:
            if not isinstance(data, P1Poisson1DData):
                raise TypeError("data must be P1Poisson1DData")
            values = np.asarray(solution_values, dtype=np.float64)
            if values.ndim == 1:
                solution_hash = _array_hash(values)
            if values.ndim != 1 or values.shape != data.nodes.shape:
                raise ValueError("solution_values must have one finite P1 nodal value per mesh node")
            if not np.all(np.isfinite(values)):
                raise ValueError("solution_values must be finite")
            if values[0] != 0.0 or values[-1] != 0.0:
                raise ValueError("solution_values must satisfy zero Dirichlet endpoint values")

            with np.errstate(over="raise", invalid="raise", divide="raise"):
                widths = np.diff(data.nodes)
                source = data.cell_source
                cumulative_source = np.concatenate(([0.0], np.cumsum(source * widths)))
                # Enforce integral(sigma)=u(1)-u(0)=0. Since sigma=C-F,
                # C is the exact integral of the piecewise-linear primitive F.
                flux_constant = float(np.sum(widths * (cumulative_source[:-1] + 0.5 * source * widths)))
                flux_nodes = flux_constant - cumulative_source
                slopes = np.diff(values) / widths
                flux_minus_p1_midpoint = (
                    flux_constant - cumulative_source[:-1] - 0.5 * source * widths - slopes
                )
                # For q(s)=q_mid - f*s_center, integrate q^2 exactly on each cell.
                cell_squared_norms = (
                    flux_minus_p1_midpoint * flux_minus_p1_midpoint * widths
                    + source * source * widths * widths * widths / 12.0
                )
                total_squared_norm = float(np.sum(cell_squared_norms))
                defect = float(np.max(np.abs(np.diff(flux_nodes) + source * widths)))
            if not np.all(np.isfinite(flux_nodes)) or not np.all(np.isfinite(cell_squared_norms)):
                raise FloatingPointError("equilibrated flux or cell integral is nonfinite")
            if not math.isfinite(total_squared_norm) or total_squared_norm < 0.0:
                raise FloatingPointError("squared estimator integral is invalid")
            eta = math.sqrt(total_squared_norm)
            if not math.isfinite(eta) or not math.isfinite(defect):
                raise FloatingPointError("estimator result is nonfinite")
            scale = max(
                float(np.max(np.abs(flux_nodes))),
                float(np.max(np.abs(source * widths))),
                np.finfo(np.float64).tiny,
            )
            relative_tolerance = (
                self._default_equilibrium_rtol(int(source.size))
                if self.equilibrium_rtol is None
                else self.equilibrium_rtol
            )
            equilibrium_tolerance = relative_tolerance * scale
            if defect > equilibrium_tolerance:
                raise FloatingPointError(
                    f"equilibrium defect {defect!r} exceeds tolerance {equilibrium_tolerance!r}"
                )
            return EstimatorRecord(
                status="numerical_bound",
                eta_eq=eta,
                data_hash=data_hash,
                solution_hash=solution_hash,
                integration_rule=_INTEGRATION_RULE,
                assumptions=self.assumptions,
                flux_node_values=tuple(float(x) for x in flux_nodes),
                equilibrium_defect=defect,
                equilibrium_tolerance=equilibrium_tolerance,
            )
        except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            return EstimatorRecord(
                status="failed",
                eta_eq=None,
                data_hash=data_hash,
                solution_hash=solution_hash,
                integration_rule=_INTEGRATION_RULE,
                assumptions=self.assumptions,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
