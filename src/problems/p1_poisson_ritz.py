"""Matrix-free Ritz adapter for the bounded one-dimensional P1 Poisson model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping
from threading import Lock
from types import MappingProxyType

import numpy as np

from estimators_1d import P1Poisson1DData
from .p1_poisson_1d import (MAX_P1_CELLS, P1PoissonStepCertificate,
                           verify_p1_poisson_step_certificate)
from .problem import RitzEnergyProblem


_SCHEMA = "cem.p1_poisson_1d_ritz_problem.v1"
_ASSEMBLY_ID = "p1-tridiagonal-local-assembly.v1"


def _immutable_f64(values: object) -> np.ndarray:
    array = np.asarray(values, dtype="<f8", order="C")
    return np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)


def _hash_update(digest: object, payload: bytes, audit: dict[str, int]) -> None:
    digest.update(payload)
    audit["hash_events"] += 1
    audit["hash_bytes"] += len(payload)


def _array_sha256(values: np.ndarray, audit: dict[str, int]) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, json.dumps(
        {"shape": list(values.shape), "dtype": "binary64-le"},
        sort_keys=True, separators=(",", ":")).encode("ascii"), audit)
    _hash_update(digest, np.asarray(values, dtype="<f8", order="C").tobytes(order="C"), audit)
    return digest.hexdigest()


def _p1_array_sha256(values: np.ndarray, audit: dict[str, int]) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    shape_json = json.dumps({"shape": canonical.shape}, separators=(",", ":")).encode("utf-8")
    _hash_update(digest, shape_json, audit)
    _hash_update(digest, canonical.tobytes(order="C"), audit)
    return digest.hexdigest()


def _p1_data_sha256(nodes: np.ndarray, source: np.ndarray,
                    audit: dict[str, int]) -> str:
    """Mirror P1Poisson1DData.data_hash while accounting for every hash input."""
    nodes_hash = _p1_array_sha256(nodes, audit)
    source_hash = _p1_array_sha256(source, audit)
    payload = {
        "schema_version": "1",
        "nodes_sha256": nodes_hash,
        "cell_source_sha256": source_hash,
        "dirichlet_values": (0.0, 0.0),
        "integration_rule": "exact_piecewise_constant_cell_integrals_float64",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256()
    _hash_update(digest, encoded, audit)
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class P1Poisson1DRitzProblem(RitzEnergyProblem):
    """Bounded matrix-free P1 Ritz problem, bound to one immutable data snapshot.

    ``data`` is retained only as an identity witness. Every operator and
    certificate input uses private copies; mutation of the caller's data is
    detected by :meth:`assert_integrity` before a learner proposal.
    """

    data: P1Poisson1DData
    nodes: np.ndarray
    cell_source: np.ndarray
    diagonal: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    load_full: np.ndarray
    _certificate_data: P1Poisson1DData
    number_of_cells: int
    dimension: int
    mesh_sha256: str
    source_sha256: str
    data_sha256: str
    assembly_sha256: str
    problem_fingerprint: str
    build_audit: MappingProxyType
    _build_audit_claimed: bool
    _build_audit_lock: Lock
    work_ledger: object | None
    objective_geometry: str = "ritz_energy"

    def __init__(self, data: P1Poisson1DData) -> None:
        if type(data) is not P1Poisson1DData:
            raise TypeError("data must be exactly P1Poisson1DData")
        nodes = _immutable_f64(data.nodes)
        source = _immutable_f64(data.cell_source)
        cells = int(source.size)
        if cells > MAX_P1_CELLS:
            raise ValueError("mesh_cell_limit_exceeded")
        hash_audit = {"hash_events": 0, "hash_bytes": 0}
        mesh_hash = _array_sha256(nodes, hash_audit)
        source_hash = _array_sha256(source, hash_audit)
        data_snapshot = P1Poisson1DData(nodes, source)
        data_hash = _p1_data_sha256(nodes, source, hash_audit)

        if cells == 1:
            diagonal = _immutable_f64(np.zeros(0, dtype=np.float64))
            upper = _immutable_f64(np.zeros(0, dtype=np.float64))
            lower = _immutable_f64(np.zeros(0, dtype=np.float64))
            load_full = _immutable_f64(np.zeros(0, dtype=np.float64))
            assembly_digest = hashlib.sha256()
            _hash_update(assembly_digest, b"p1-empty-operator-v1", hash_audit)
            assembly_hash = assembly_digest.hexdigest()
            build_audit = {
                "adapter_cell_visits": 0,
                "stiffness_contribution_updates": 0,
                "load_contribution_updates": 0,
                "adapter_operator_rows": 0,
                "binary64_scalar_updates": 0,
                "hash_bytes": 0,
                "storage_bytes": 0,
            }
        else:
            diagonal_list = [0.0] * (cells + 1)
            upper_list = [0.0] * cells
            lower_list = [0.0] * cells
            load_list = [0.0] * (cells + 1)
            scalar_updates = 0
            for index in range(cells):
                width = float(nodes[index + 1] - nodes[index])
                scalar_updates += 1
                inverse_width = 1.0 / width
                scalar_updates += 1
                half_load = float((float(source[index]) * width) * 0.5)
                scalar_updates += 2
                if not all(math.isfinite(value) for value in (width, inverse_width, half_load)):
                    raise ValueError("nonfinite_or_unrepresentable_float_assembly")

                diagonal_list[index] = float(diagonal_list[index] + inverse_width)
                diagonal_list[index + 1] = float(diagonal_list[index + 1] + inverse_width)
                scalar_updates += 2
                upper_list[index] = -inverse_width
                lower_list[index] = -inverse_width

                load_list[index] = float(load_list[index] + half_load)
                load_list[index + 1] = float(load_list[index + 1] + half_load)
                scalar_updates += 2
                if not all(math.isfinite(value) for value in (
                        diagonal_list[index], diagonal_list[index + 1],
                        load_list[index], load_list[index + 1])):
                    raise ValueError("nonfinite_or_unrepresentable_float_assembly")

            diagonal = _immutable_f64(np.asarray(diagonal_list, dtype=np.float64))
            upper = _immutable_f64(np.asarray(upper_list, dtype=np.float64))
            lower = _immutable_f64(np.asarray(lower_list, dtype=np.float64))
            load_full = _immutable_f64(np.asarray(load_list, dtype=np.float64))
            if not all(np.all(np.isfinite(value)) for value in
                       (diagonal, upper, lower, load_full)):
                raise ValueError("nonfinite_or_unrepresentable_float_assembly")
            assembly_digest = hashlib.sha256()
            for array in (diagonal, upper, lower, load_full):
                _hash_update(assembly_digest,
                             np.asarray(array, dtype="<f8").tobytes(order="C"),
                             hash_audit)
            assembly_hash = assembly_digest.hexdigest()
            build_audit = {
                "adapter_cell_visits": cells,
                "stiffness_contribution_updates": 4 * cells,
                "load_contribution_updates": 2 * cells,
                "adapter_operator_rows": cells - 1,
                "binary64_scalar_updates": scalar_updates,
                "hash_bytes": 0,
                "storage_bytes": diagonal.nbytes + upper.nbytes + lower.nbytes + load_full.nbytes,
            }

        identity = {
            "schema": _SCHEMA,
            "assembly_id": _ASSEMBLY_ID,
            "data_sha256": data_hash,
            "mesh_sha256": mesh_hash,
            "source_sha256": source_hash,
            "cells": cells,
            "dimension": max(cells - 1, 0),
            "assembly_sha256": assembly_hash,
            "boundary": ["0x0.0p+0", "0x0.0p+0"],
        }
        encoded_identity = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True).encode("ascii")
        identity_digest = hashlib.sha256()
        _hash_update(identity_digest, encoded_identity, hash_audit)
        fingerprint = identity_digest.hexdigest()
        build_audit["hash_events"] = hash_audit["hash_events"]
        build_audit["hash_bytes"] = hash_audit["hash_bytes"]
        for name, value in (
            ("data", data), ("nodes", nodes), ("cell_source", source),
            ("diagonal", diagonal), ("upper", upper), ("lower", lower),
            ("load_full", load_full), ("_certificate_data", data_snapshot),
            ("number_of_cells", cells), ("dimension", max(cells - 1, 0)),
            ("mesh_sha256", mesh_hash), ("source_sha256", source_hash),
            ("data_sha256", data_hash), ("assembly_sha256", assembly_hash),
            ("problem_fingerprint", fingerprint),
            ("build_audit", MappingProxyType(dict(build_audit))),
            ("_build_audit_claimed", False), ("_build_audit_lock", Lock()),
            ("work_ledger", None),
            ("objective_geometry", "ritz_energy"),
        ):
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    @property
    def operator(self) -> "P1Poisson1DRitzProblem":
        return self

    @property
    def source(self) -> np.ndarray:
        if self.number_of_cells <= 1:
            return _immutable_f64(np.zeros(0, dtype=np.float64))
        return _immutable_f64(self.load_full[1:-1])

    def attach_work_ledger(self, ledger: object) -> None:
        object.__setattr__(self, "work_ledger", ledger)

    def _work(self, quantity: str, phase: str, status: str = "completed") -> None:
        if self.work_ledger is not None:
            self.work_ledger.record_call(phase, quantity, status=status)

    def apply(self, values: object) -> np.ndarray:
        return self.apply_operator(values)

    def admissible(self, values: object) -> bool:
        vector = np.asarray(values)
        return (vector.dtype == np.dtype(np.float64)
                and vector.ndim == 1
                and vector.shape == (self.dimension,)
                and bool(np.all(np.isfinite(vector))))

    def validate_state(self, state: object) -> None:
        self.assert_integrity()
        values = getattr(state, "values", None)
        if not self.admissible(values):
            raise ValueError("state_shape_or_dtype_mismatch")
        active_ids = getattr(state, "active_element_ids", None)
        if type(active_ids) is not tuple or any(not isinstance(item, str) for item in active_ids):
            raise ValueError("state_active_ids_malformed")

    def _validate_p1_binding_fields(
            self, *, element_id: object, correction: object | None,
            candidate_fingerprint: object, support: object, parameters: object,
            problem_fingerprint: object, certificate_status: object,
            certificate_json: object, certificate_sha256: object,
            state: object) -> bool:
        self.assert_integrity()
        if not isinstance(support, Mapping) or not isinstance(parameters, Mapping):
            return False
        node = support.get("node_index")
        if type(node) is not int or not 1 <= node < self.number_of_cells:
            return False
        cells = support.get("cell_indices")
        if not isinstance(cells, (tuple, list)) or tuple(cells) != (node - 1, node):
            return False
        if (parameters.get("basis") != "interior_p1_hat"
                or type(parameters.get("node_index")) is not int
                or parameters.get("node_index") != node):
            return False
        expected_id = (f"p1hat-v1:{self.problem_fingerprint}:"
                       f"{self.mesh_sha256}:node:{node}")
        if element_id != expected_id or problem_fingerprint != self.problem_fingerprint:
            return False
        if (not isinstance(certificate_json, str)
                or not isinstance(certificate_sha256, str)
                or hashlib.sha256(certificate_json.encode("utf-8")).hexdigest()
                != certificate_sha256):
            return False
        certificate = P1PoissonStepCertificate.from_json(certificate_json)
        if (certificate_status != "verified_positive_gain"
                or certificate.status != certificate_status):
            return False
        expected = np.zeros(self.dimension, dtype=np.float64)
        expected[node - 1] = 1.0
        expected_bytes = expected.tobytes(order="C")
        expected_fingerprint = hashlib.sha256(expected_bytes).hexdigest()
        if candidate_fingerprint != expected_fingerprint:
            return False
        if correction is not None:
            if (type(correction) is not np.ndarray
                    or correction.dtype != np.dtype(np.float64)
                    or correction.ndim != 1
                    or correction.shape != (self.dimension,)
                    or correction.tobytes(order="C") != expected_bytes):
                return False

        values = getattr(state, "values", None)
        if (type(values) is not np.ndarray
                or values.dtype != np.dtype(np.float64)
                or values.shape != (self.dimension,)
                or not np.all(np.isfinite(values))):
            return False
        full_state = np.zeros(self.number_of_cells + 1, dtype=np.float64)
        full_state[1:-1] = values
        direction = np.zeros(self.number_of_cells + 1, dtype=np.float64)
        direction[node] = 1.0
        return verify_p1_poisson_step_certificate(
            certificate, self.certificate_data(), full_state, direction)

    def validate_p1_candidate_binding(self, candidate: object,
                                      state: object) -> bool:
        """Fail closed unless candidate fields match its replayed P1 certificate."""
        provenance = getattr(candidate, "provenance", None)
        if not isinstance(provenance, Mapping):
            return False
        if (provenance.get("learner") != "P1PoissonHatWeakLearner"
                or getattr(candidate, "family", None) != "p1-poisson-hat"
                or provenance.get("problem_fingerprint") != self.problem_fingerprint
                or provenance.get("mesh_sha256") != self.mesh_sha256
                or provenance.get("data_fingerprint") != self.data_sha256
                or not isinstance(provenance.get("work_audit"), Mapping)):
            return False
        correction = getattr(candidate, "correction", None)
        if (type(correction) is not np.ndarray
                or correction.ndim != 1
                or correction.dtype != np.dtype(np.float64)
                or correction.shape != (self.dimension,)
                or not np.all(np.isfinite(correction))):
            return False
        fingerprint = hashlib.sha256(correction.tobytes(order="C")).hexdigest()
        return self._validate_p1_binding_fields(
            element_id=getattr(candidate, "element_id", None),
            correction=correction,
            candidate_fingerprint=fingerprint,
            support=getattr(candidate, "support", None),
            parameters=getattr(candidate, "parameters", None),
            problem_fingerprint=provenance.get("problem_fingerprint"),
            certificate_status=provenance.get("certificate_status"),
            certificate_json=provenance.get("certificate_record_json"),
            certificate_sha256=provenance.get("certificate_sha256"),
            state=state)

    def validate_p1_audit_binding(self, envelope: Mapping[str, object],
                                  state: object) -> bool:
        """Bind a persisted candidate envelope to this problem and state."""
        if (not isinstance(envelope, Mapping)
                or envelope.get("schema") != "cem.p1_proposal_audit.v1"
                or envelope.get("outcome") != "verified_candidate"
                or not isinstance(envelope.get("work_audit"), Mapping)):
            return False
        return self._validate_p1_binding_fields(
            element_id=envelope.get("element_id"), correction=None,
            candidate_fingerprint=envelope.get("candidate_fingerprint"),
            support=envelope.get("support"),
            parameters=envelope.get("parameters"),
            problem_fingerprint=envelope.get("problem_fingerprint"),
            certificate_status=envelope.get("certificate_status"),
            certificate_json=envelope.get("certificate_record_json"),
            certificate_sha256=envelope.get("certificate_sha256"),
            state=state)

    def objective(self, state: object) -> float:
        try:
            self.assert_integrity()
            values = getattr(state, "values", None)
            if not self.admissible(values):
                raise ValueError("state_shape_or_dtype_mismatch")
            applied = self.apply_operator(values)
            result = 0.5 * float(values @ applied) - float(self.source @ values)
            if not math.isfinite(result):
                raise ValueError("nonfinite_ritz_objective")
        except Exception:
            self._work("objective_eval_calls", "p1_ritz_objective", "failed")
            raise
        self._work("objective_eval_calls", "p1_ritz_objective")
        return result

    def assert_integrity(self) -> None:
        """Fail if the caller-owned data or any derived adapter array changed."""
        if (type(self.data) is not P1Poisson1DData
                or type(self.data.nodes) is not np.ndarray
                or type(self.data.cell_source) is not np.ndarray
                or self.data.nodes.dtype != np.dtype(np.float64)
                or self.data.cell_source.dtype != np.dtype(np.float64)
                or self.data.nodes.shape != self.nodes.shape
                or self.data.cell_source.shape != self.cell_source.shape
                or self.data.nodes.tobytes(order="C") != self.nodes.tobytes(order="C")
                or self.data.cell_source.tobytes(order="C") != self.cell_source.tobytes(order="C")):
            raise ValueError("problem_data_fingerprint_mismatch")

    def claim_build_audit(self) -> dict[str, int]:
        """Return setup costs once, when first consumed by a learner proposal."""
        with self._build_audit_lock:
            if self._build_audit_claimed:
                return {key: 0 for key in self.build_audit}
            object.__setattr__(self, "_build_audit_claimed", True)
            return dict(self.build_audit)

    def apply_operator(self, values: object) -> np.ndarray:
        try:
            self.assert_integrity()
            vector = np.asarray(values)
            if (vector.dtype != np.dtype(np.float64) or vector.ndim != 1
                    or vector.shape != (self.dimension,)
                    or not np.all(np.isfinite(vector))):
                raise ValueError("state_must_be_a_finite_binary64_interior_vector")
            result = np.empty(self.dimension, dtype=np.float64)
            cells = self.number_of_cells
            for row, node in enumerate(range(1, cells)):
                total = float(self.diagonal[node] * vector[row])
                if node > 1:
                    total = float(total + self.upper[node - 1] * vector[row - 1])
                if node < cells - 1:
                    total = float(total + self.lower[node] * vector[row + 1])
                if not math.isfinite(total):
                    raise ValueError("nonfinite_operator_action")
                result[row] = total
        except Exception:
            self._work("operator_apply_calls", "p1_ritz_operator_apply", "failed")
            raise
        self._work("operator_apply_calls", "p1_ritz_operator_apply")
        return result

    def residual(self, state: object) -> np.ndarray:
        values = getattr(state, "values", None)
        if (type(values) is not np.ndarray or values.dtype != np.dtype(np.float64)
                or values.ndim != 1 or values.shape != (self.dimension,)
                or not np.all(np.isfinite(values))):
            raise ValueError("state_shape_or_dtype_mismatch")
        applied = self.apply_operator(values)
        return np.asarray(self.load_full[1:-1] - applied, dtype=np.float64)

    def certificate_data(self) -> P1Poisson1DData:
        """Return a detached copy for the full-domain exact certificate call."""
        return P1Poisson1DData(np.array(self.nodes, copy=True),
                               np.array(self.cell_source, copy=True))


__all__ = ["P1Poisson1DRitzProblem"]
