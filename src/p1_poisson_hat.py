"""Proposal-only exact weak learner for finite P1 Poisson hat dictionaries."""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any

import numpy as np

from core.state import SolverState
from problems.p1_poisson_1d import (
    MAX_EXACT_RATIONAL_OPERATIONS,
    MAX_FRACTION_COMPONENT_BITS,
    MAX_P1_CELLS,
    MAX_RECORD_JSON_BYTES,
    P1PoissonStepCertificate,
    certify_p1_poisson_step,
)
from problems.p1_poisson_ritz import P1Poisson1DRitzProblem
from enrichment.learner import Candidate


_STATUS_REASONS = {
    "empty_dictionary": {"no_interior_hat"},
    "dictionary_exhausted": {"all_hats_active"},
    "no_positive_gain": {"all_gains_zero"},
    "problem_state_mismatch": {
        "wrong_problem_type", "problem_data_changed", "state_type",
        "state_shape_or_dtype", "state_nonfinite", "state_changed_during_proposal",
        "problem_integrity_failure",
    },
    "residual_mismatch": {
        "residual_type_or_shape", "residual_nonfinite", "residual_bits_differ",
        "residual_recompute_failed",
    },
    "active_set_invalid": {
        "active_ids_not_tuple", "active_id_not_string", "duplicate_active_id",
        "malformed_active_id", "foreign_active_id",
    },
    "budget_exceeded": {
        "invalid_budget_shape", "unknown_budget_key", "invalid_budget_value",
        "budget_limit_increased", "cell_cap", "hat_cap", "local_cell_cap",
        "adapter_cell_cap", "adapter_operator_row_cap", "residual_operator_row_cap",
        "operator_row_cap", "fraction_operation_cap", "certificate_operation_cap",
        "fraction_conversion_cap", "fraction_component_bit_cap",
        "certificate_record_byte_cap", "candidate_metadata_byte_cap",
        "candidate_vector_cap", "hash_byte_cap", "live_counter_exceeded",
        "certificate_resource_limit", "certificate_invalid", "certificate_identity_mismatch",
        "internal_work_bound_exceeded",
    },
}

_BUDGET_KEYS = (
    "max_cells",
    "max_hat_evaluations",
    "max_local_cell_visits",
    "max_adapter_cell_visits",
    "max_adapter_operator_rows",
    "max_residual_operator_rows",
    "max_operator_row_visits",
    "max_fraction_ops_total",
    "max_certificate_ops",
    "max_fraction_conversions",
    "max_fraction_component_bits",
    "max_certificate_record_bytes",
    "max_candidate_metadata_bytes",
    "max_candidate_vector_elements",
    "max_hash_bytes",
)


def _freeze_mapping(value: Mapping[str, Any]) -> MappingProxyType:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


class P1HatProposalError(RuntimeError):
    """Finite, auditable no-candidate or rejected-proposal outcome."""

    def __init__(self, status: str, reason_code: str,
                 work_audit: Mapping[str, Any]) -> None:
        if status not in _STATUS_REASONS or reason_code not in _STATUS_REASONS[status]:
            raise ValueError("invalid P1 hat proposal error status/reason")
        self.status = status
        self.reason_code = reason_code
        self.work_audit = _freeze_mapping(work_audit)
        super().__init__(f"{status}:{reason_code}")


class _Work:
    def __init__(self, n_cells: int = 0, inactive_hats: int = 0) -> None:
        self.values: dict[str, Any] = {
            "n_cells": n_cells,
            "inactive_hats": inactive_hats,
            "adapter_cell_visits": 0,
            "stiffness_contribution_updates": 0,
            "load_contribution_updates": 0,
            "adapter_operator_rows": 0,
            "binary64_scalar_updates": 0,
            "fraction_ops_reserved": 0,
            "fraction_ops_local_actual": 0,
            "fraction_ops_certificate_reserved": 0,
            "fraction_ops_certificate_actual": 0,
            "fraction_ops_actual_total": 0,
            "fraction_conversions_reserved": 0,
            "fraction_conversions_local_actual": 0,
            "fraction_conversions_certificate_actual": 0,
            "fraction_comparisons": 0,
            "hat_evaluations": 0,
            "local_cell_visits": 0,
            "residual_operator_rows": 0,
            "operator_row_visits": 0,
            "certificate_replays": 0,
            "certificate_record_bytes": 0,
            "certificate_input_bytes": 0,
            "certificate_input_bytes_reserved": 0,
            "candidate_vector_elements": 0,
            "candidate_vector_copy_elements": 0,
            "candidate_metadata_bytes": 0,
            "candidate_metadata_bytes_reserved": 0,
            "preflight_metadata_bytes_serialized": 0,
            "candidate_metadata_serializations": 0,
            "candidate_metadata_serialization_bytes": 0,
            "certificate_input_serializations": 0,
            "hash_events": 0,
            "hash_bytes": 0,
            "hash_events_reserved": 0,
            "hash_bytes_reserved": 0,
            "serialization_bytes": 0,
            "serialization_bytes_reserved": 0,
            "preflight": {},
            "effective_budget": {},
        }

    def snapshot(self) -> MappingProxyType:
        return _freeze_mapping(self.values)


def _error(work: _Work, status: str, reason: str) -> None:
    raise P1HatProposalError(status, reason, work.snapshot())


def _parse_budget_shape(budget: object, work: _Work) -> dict[str, int]:
    if budget is None:
        return {}
    if not isinstance(budget, Mapping):
        _error(work, "budget_exceeded", "invalid_budget_shape")
    try:
        items = list(budget.items())
    except Exception:
        _error(work, "budget_exceeded", "invalid_budget_shape")
    parsed: dict[str, int] = {}
    for key, value in items:
        if not isinstance(key, str) or key not in _BUDGET_KEYS:
            _error(work, "budget_exceeded", "unknown_budget_key")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            _error(work, "budget_exceeded", "invalid_budget_value")
        value = int(value)
        if value < 0:
            _error(work, "budget_exceeded", "invalid_budget_value")
        parsed[key] = value
    return parsed


def _budget_defaults(n: int, k: int) -> dict[str, int]:
    max_record = MAX_RECORD_JSON_BYTES
    max_metadata = 65_536
    return {
        "max_cells": MAX_P1_CELLS,
        "max_hat_evaluations": max(n - 1, 0),
        "max_local_cell_visits": 2 * k,
        "max_adapter_cell_visits": n,
        "max_adapter_operator_rows": max(n - 1, 0),
        "max_residual_operator_rows": max(n - 1, 0),
        "max_operator_row_visits": 2 * max(n - 1, 0),
        "max_fraction_ops_total": 250_000,
        "max_certificate_ops": MAX_EXACT_RATIONAL_OPERATIONS,
        "max_fraction_conversions": 0 if k == 0 else (3 * n + 2) + (4 * n + 3),
        "max_fraction_component_bits": MAX_FRACTION_COMPONENT_BITS,
        "max_certificate_record_bytes": max_record,
        "max_candidate_metadata_bytes": max_metadata,
        "max_candidate_vector_elements": max(n - 1, 0),
        "max_hash_bytes": max_record + 8 * (4 * n + 3) + max_metadata,
    }


def _effective_budget(raw: dict[str, int], n: int, k: int,
                      work: _Work) -> dict[str, int]:
    defaults = _budget_defaults(n, k)
    for key, value in raw.items():
        if value > defaults[key]:
            _error(work, "budget_exceeded", "budget_limit_increased")
    limits = dict(defaults)
    limits.update(raw)
    return limits


def _bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


class _ExactOps:
    def __init__(self, work: _Work, bit_cap: int, operation_cap: int) -> None:
        self.work = work
        self.bit_cap = bit_cap
        self.operation_cap = operation_cap

    def _apply(self, left: Fraction, right: Fraction, operation: str) -> Fraction:
        next_count = self.work.values["fraction_ops_local_actual"] + 1
        if next_count + self.work.values["fraction_ops_certificate_actual"] > self.operation_cap:
            _error(self.work, "budget_exceeded", "live_counter_exceeded")
        if max(_bits(left), _bits(right)) > self.bit_cap:
            _error(self.work, "budget_exceeded", "fraction_component_bit_cap")
        self.work.values["fraction_ops_local_actual"] = next_count
        if operation == "add":
            result = left + right
        elif operation == "sub":
            result = left - right
        elif operation == "mul":
            result = left * right
        elif operation == "div":
            if right == 0:
                _error(self.work, "problem_state_mismatch", "problem_integrity_failure")
            result = left / right
        else:  # pragma: no cover - fixed internal call sites
            raise AssertionError(operation)
        if _bits(result) > self.bit_cap:
            _error(self.work, "budget_exceeded", "fraction_component_bit_cap")
        self.work.values["fraction_ops_actual_total"] = (
            self.work.values["fraction_ops_local_actual"]
            + self.work.values["fraction_ops_certificate_actual"])
        return result

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "add")

    def sub(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "sub")

    def mul(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "mul")

    def div(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "div")

    def from_float(self, value: np.float64) -> Fraction:
        result = Fraction.from_float(float(value))
        self.work.values["fraction_conversions_local_actual"] += 1
        if _bits(result) > self.bit_cap:
            _error(self.work, "budget_exceeded", "fraction_component_bit_cap")
        return result


def _state_snapshot(state: object, dimension: int, work: _Work):
    if type(state) is not SolverState:
        _error(work, "problem_state_mismatch", "state_type")
    if (type(state.values) is not np.ndarray or state.values.dtype != np.dtype(np.float64)
            or state.values.ndim != 1 or state.values.shape != (dimension,)):
        _error(work, "problem_state_mismatch", "state_shape_or_dtype")
    if not np.all(np.isfinite(state.values)):
        _error(work, "problem_state_mismatch", "state_nonfinite")
    if type(state.active_element_ids) is not tuple:
        _error(work, "active_set_invalid", "active_ids_not_tuple")
    for item in state.active_element_ids:
        if not isinstance(item, str):
            _error(work, "active_set_invalid", "active_id_not_string")
    if state.outer_coefficients is not None:
        if (type(state.outer_coefficients) is not np.ndarray
                or state.outer_coefficients.dtype != np.dtype(np.float64)
                or state.outer_coefficients.ndim != 1
                or not np.all(np.isfinite(state.outer_coefficients))):
            _error(work, "problem_state_mismatch", "state_shape_or_dtype")
    try:
        snapshot = state.snapshot()
    except Exception:
        _error(work, "problem_state_mismatch", "state_shape_or_dtype")
    return snapshot


def _state_fingerprint(problem: P1Poisson1DRitzProblem, snapshot: object) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    payloads = (
        b"cem.p1hat.state.v1\0",
        bytes.fromhex(problem.problem_fingerprint),
        np.asarray(snapshot.values, dtype="<f8", order="C").tobytes(order="C"),
        json.dumps(list(snapshot.active_element_ids), separators=(",", ":"),
                   ensure_ascii=True).encode("ascii"),
    )
    if snapshot.outer_coefficients is None:
        payloads += (b"outer_coefficients:none\0",)
    else:
        coefficients = snapshot.outer_coefficients
        descriptor = (b"outer_coefficients:present\0"
                      + coefficients.dtype.str.encode("ascii") + b"\0"
                      + struct.pack(">I", coefficients.ndim)
                      + b"".join(struct.pack(">Q", int(size))
                                for size in coefficients.shape))
        payloads += (
            descriptor,
            np.asarray(coefficients, dtype="<f8", order="C").tobytes(order="C"),
        )
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest(), len(payloads), sum(len(payload) for payload in payloads)


def _state_is_unchanged(state: SolverState, initial: str,
                        problem: P1Poisson1DRitzProblem) -> bool:
    try:
        snapshot = state.snapshot()
        current, _, _ = _state_fingerprint(problem, snapshot)
        return current == initial
    except Exception:
        return False


def _validate_active_ids(state: SolverState, problem: P1Poisson1DRitzProblem,
                         work: _Work) -> set[str]:
    active = state.active_element_ids
    if len(active) != len(set(active)):
        _error(work, "active_set_invalid", "duplicate_active_id")
    valid = {f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:{j}"
             for j in range(1, problem.number_of_cells)}
    for element_id in active:
        if not element_id.startswith("p1hat-v1:") or ":node:" not in element_id:
            _error(work, "active_set_invalid", "malformed_active_id")
        if element_id not in valid:
            _error(work, "active_set_invalid", "foreign_active_id")
    return set(active)


def _certificate_input_json(problem: P1Poisson1DRitzProblem,
                            snapshot: object) -> bytes:
    n = problem.number_of_cells
    full_state = np.zeros(n + 1, dtype=np.float64)
    full_state[1:-1] = snapshot.values
    direction = np.zeros(n + 1, dtype=np.float64)
    direction[1] = 1.0
    payload = {
        "schema": "cem.p1_poisson_1d_step_certificate.v1",
        "algorithm": "exact-dyadic-piecewise-linear-cell-gain.v1",
        "model": "poisson_1d_zero_dirichlet_cellwise_constant_p1_binary64_exact",
        "sign_convention": "c=integral(f*phi)-integral(u_prime*phi_prime);alpha=c/q;signed",
        "nodes_hex": [struct.pack(">d", float(value)).hex() for value in problem.nodes],
        "cell_source_hex": [struct.pack(">d", float(value)).hex()
                            for value in problem.cell_source],
        "state_hex": [struct.pack(">d", float(value)).hex() for value in full_state],
        "direction_hex": [struct.pack(">d", float(value)).hex() for value in direction],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _candidate_metadata_upper_bound(problem: P1Poisson1DRitzProblem, k: int,
                                    limits: dict[str, int], work: _Work,
                                    required: dict[str, int], cert_input_bytes: int,
                                    hash_events_reserved: int,
                                    hash_bytes_reserved: int) -> int:
    """Serialize a max-counter skeleton before scanning to reserve metadata."""
    audit = _plain(work.snapshot())
    audit.update({
        "fraction_ops_local_actual": 6 * problem.number_of_cells + 7 * k,
        "fraction_ops_certificate_actual": required["max_certificate_ops"],
        "fraction_ops_actual_total": required["max_fraction_ops_total"],
        "fraction_conversions_local_actual": 3 * problem.number_of_cells + 2,
        "fraction_conversions_certificate_actual": 4 * problem.number_of_cells + 3,
        "fraction_comparisons": max(k - 1, 0),
        "hat_evaluations": k,
        "local_cell_visits": 2 * k,
        "residual_operator_rows": problem.number_of_cells - 1,
        "operator_row_visits": required["max_operator_row_visits"],
        "certificate_replays": 1,
        "certificate_record_bytes": required["max_certificate_record_bytes"],
        "certificate_input_bytes": cert_input_bytes,
        "candidate_vector_elements": problem.number_of_cells - 1,
        "candidate_vector_copy_elements": problem.number_of_cells - 1,
        "candidate_metadata_bytes": limits["max_candidate_metadata_bytes"],
        "candidate_metadata_bytes_reserved": limits["max_candidate_metadata_bytes"],
        "preflight_metadata_bytes_serialized": limits["max_candidate_metadata_bytes"],
        "candidate_metadata_serializations": 4,
        "candidate_metadata_serialization_bytes": 4 * limits["max_candidate_metadata_bytes"],
        "certificate_input_serializations": 2,
        "hash_events": hash_events_reserved,
        "hash_bytes": hash_bytes_reserved,
        "hash_events_reserved": hash_events_reserved,
        "hash_bytes_reserved": hash_bytes_reserved,
        "serialization_bytes": (2 * cert_input_bytes
                                 + required["max_certificate_record_bytes"]
                                 + 5 * limits["max_candidate_metadata_bytes"]),
        "serialization_bytes_reserved": (2 * cert_input_bytes
                                          + required["max_certificate_record_bytes"]
                                          + 5 * limits["max_candidate_metadata_bytes"]),
        "preflight": required,
        "effective_budget": limits,
    })
    node = problem.number_of_cells - 1
    element_id = (f"p1hat-v1:{problem.problem_fingerprint}:"
                  f"{problem.mesh_sha256}:node:{node}")
    provenance = {
        "learner": "P1PoissonHatWeakLearner",
        "problem_fingerprint": problem.problem_fingerprint,
        "data_fingerprint": problem.data_sha256,
        "mesh_sha256": problem.mesh_sha256,
        "state_fingerprint": "f" * 64,
        "dictionary_size": k,
        "tie_rule": "ascending_stable_element_id",
        "weakness_tau": 1.0,
        "certificate_status": "verified_positive_gain",
        "certificate_sha256": "f" * 64,
        "work_audit": audit,
    }
    metadata = {
        "element_id": element_id,
        "family": "p1-poisson-hat",
        "support": {"node_index": node, "cell_indices": [node - 1, node]},
        "parameters": {"basis": "interior_p1_hat", "node_index": node},
        "seed": None,
        "provenance": provenance,
        "proposal_cost": {"fraction_ops": float(required["max_fraction_ops_total"]),
                          "hat_evaluations": float(k)},
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    work.values["serialization_bytes"] += len(encoded)
    work.values["preflight_metadata_bytes_serialized"] = len(encoded)
    return len(encoded)


def _limits_preflight(limits: dict[str, int], problem: P1Poisson1DRitzProblem,
                      k: int, work: _Work, cert_input_bytes: int,
                      state_hash_bytes: int, state_hash_events: int,
                      build_audit: Mapping[str, int]) -> dict[str, int]:
    n = problem.number_of_cells
    certificate_ops = 36 * n + 11
    fraction_ops = 6 * n + 7 * k + certificate_ops
    conversions = (3 * n + 2) + (4 * n + 3)
    cert_record_bytes = cert_input_bytes + 50_000
    build_cells = int(build_audit["adapter_cell_visits"])
    build_rows = int(build_audit["adapter_operator_rows"])
    residual_rows = n - 1
    operator_rows = build_rows + residual_rows
    hash_events_reserved = (int(build_audit["hash_events"]) + state_hash_events + 2)
    hash_bytes = (int(build_audit["hash_bytes"]) + state_hash_bytes
                  + cert_input_bytes + cert_record_bytes
                  + limits["max_candidate_metadata_bytes"])
    required = {
        "max_cells": n,
        "max_hat_evaluations": k,
        "max_local_cell_visits": 2 * k,
        "max_adapter_cell_visits": build_cells,
        "max_adapter_operator_rows": build_rows,
        "max_residual_operator_rows": residual_rows,
        "max_operator_row_visits": operator_rows,
        "max_fraction_ops_total": fraction_ops,
        "max_certificate_ops": certificate_ops,
        "max_fraction_conversions": conversions,
        "max_fraction_component_bits": MAX_FRACTION_COMPONENT_BITS,
        "max_certificate_record_bytes": cert_record_bytes,
        "max_candidate_metadata_bytes": limits["max_candidate_metadata_bytes"],
        "max_candidate_vector_elements": n - 1,
        "max_hash_bytes": hash_bytes,
    }
    reasons = {
        "max_cells": "cell_cap",
        "max_hat_evaluations": "hat_cap",
        "max_local_cell_visits": "local_cell_cap",
        "max_adapter_cell_visits": "adapter_cell_cap",
        "max_adapter_operator_rows": "adapter_operator_row_cap",
        "max_residual_operator_rows": "residual_operator_row_cap",
        "max_operator_row_visits": "operator_row_cap",
        "max_fraction_ops_total": "fraction_operation_cap",
        "max_certificate_ops": "certificate_operation_cap",
        "max_fraction_conversions": "fraction_conversion_cap",
        "max_fraction_component_bits": "fraction_component_bit_cap",
        "max_certificate_record_bytes": "certificate_record_byte_cap",
        "max_candidate_metadata_bytes": "candidate_metadata_byte_cap",
        "max_candidate_vector_elements": "candidate_vector_cap",
        "max_hash_bytes": "hash_byte_cap",
    }
    if cert_record_bytes > MAX_RECORD_JSON_BYTES:
        _error(work, "budget_exceeded", "certificate_record_byte_cap")
    if certificate_ops > MAX_EXACT_RATIONAL_OPERATIONS:
        _error(work, "budget_exceeded", "certificate_operation_cap")
    required["max_candidate_metadata_bytes"] = _candidate_metadata_upper_bound(
        problem, k, limits, work, required, cert_input_bytes,
        hash_events_reserved, hash_bytes)
    work.values["serialization_bytes_reserved"] = (
        2 * cert_input_bytes + cert_record_bytes
        + 5 * required["max_candidate_metadata_bytes"])
    work.values["candidate_metadata_bytes_reserved"] = required["max_candidate_metadata_bytes"]
    work.values["certificate_input_bytes_reserved"] = cert_input_bytes
    work.values["hash_events_reserved"] = hash_events_reserved
    work.values["hash_bytes_reserved"] = hash_bytes
    for key in _BUDGET_KEYS:
        if limits[key] < required[key]:
            _error(work, "budget_exceeded", reasons[key])
    work.values["fraction_ops_reserved"] = fraction_ops
    work.values["fraction_ops_certificate_reserved"] = certificate_ops
    work.values["fraction_conversions_reserved"] = conversions
    work.values["preflight"] = required
    work.values["effective_budget"] = dict(limits)
    return required


class P1PoissonHatWeakLearner:
    """Return the exact best positive-gain inactive interior P1 hat proposal."""

    family = "p1-poisson-hat"

    def propose(self, problem: object, state: object, residual: object,
                budget: object = None) -> Candidate:
        work = _Work()
        raw_budget = _parse_budget_shape(budget, work)
        if type(problem) is not P1Poisson1DRitzProblem:
            _error(work, "problem_state_mismatch", "wrong_problem_type")
        n = problem.number_of_cells
        work.values["n_cells"] = n
        try:
            problem.assert_integrity()
        except Exception:
            _error(work, "problem_state_mismatch", "problem_data_changed")

        build_audit = problem.claim_build_audit()
        work.values["adapter_cell_visits"] = int(build_audit["adapter_cell_visits"])
        work.values["stiffness_contribution_updates"] = int(build_audit["stiffness_contribution_updates"])
        work.values["load_contribution_updates"] = int(build_audit["load_contribution_updates"])
        work.values["adapter_operator_rows"] = int(build_audit["adapter_operator_rows"])
        work.values["binary64_scalar_updates"] = int(build_audit["binary64_scalar_updates"])
        work.values["operator_row_visits"] = int(build_audit["adapter_operator_rows"])
        work.values["hash_events"] = int(build_audit["hash_events"])
        work.values["hash_bytes"] = int(build_audit["hash_bytes"])

        snapshot = _state_snapshot(state, max(n - 1, 0), work)
        active = _validate_active_ids(state, problem, work)
        k = max(n - 1, 0) - len(active)
        work.values["inactive_hats"] = k
        limits = _effective_budget(raw_budget, n, k, work)

        if n == 1:
            _error(work, "empty_dictionary", "no_interior_hat")
        if k == 0:
            _error(work, "dictionary_exhausted", "all_hats_active")

        state_fingerprint, state_hash_events, state_hash_bytes = _state_fingerprint(
            problem, snapshot)
        work.values["hash_events"] += state_hash_events
        work.values["hash_bytes"] += state_hash_bytes
        cert_input = _certificate_input_json(problem, snapshot)
        work.values["certificate_input_bytes"] = len(cert_input)
        work.values["certificate_input_serializations"] = 1
        work.values["serialization_bytes"] += len(cert_input)
        required = _limits_preflight(
            limits, problem, k, work, len(cert_input), state_hash_bytes,
            state_hash_events, build_audit)
        if work.values["hash_bytes"] > limits["max_hash_bytes"]:
            _error(work, "budget_exceeded", "hash_byte_cap")

        if (type(residual) is not np.ndarray or residual.dtype != np.dtype(np.float64)
                or residual.ndim != 1 or residual.shape != (n - 1,)):
            _error(work, "residual_mismatch", "residual_type_or_shape")
        if not np.all(np.isfinite(residual)):
            _error(work, "residual_mismatch", "residual_nonfinite")
        try:
            fresh = problem.residual(snapshot)
        except Exception:
            _error(work, "residual_mismatch", "residual_recompute_failed")
        work.values["residual_operator_rows"] = n - 1
        work.values["operator_row_visits"] += n - 1
        if residual.tobytes(order="C") != fresh.tobytes(order="C"):
            _error(work, "residual_mismatch", "residual_bits_differ")

        n_cells = n
        full_state = np.zeros(n_cells + 1, dtype=np.float64)
        full_state[1:-1] = snapshot.values
        ops = _ExactOps(work, limits["max_fraction_component_bits"],
                        limits["max_fraction_ops_total"])
        nodes = tuple(ops.from_float(value) for value in problem.nodes)
        source = tuple(ops.from_float(value) for value in problem.cell_source)
        state_full = tuple(ops.from_float(value) for value in full_state)
        half = Fraction(1, 2)
        one = Fraction(1)
        widths: list[Fraction] = []
        half_loads: list[Fraction] = []
        slopes: list[Fraction] = []
        inverse_widths: list[Fraction] = []
        for cell in range(n_cells):
            width = ops.sub(nodes[cell + 1], nodes[cell])
            if width <= 0:
                _error(work, "problem_state_mismatch", "problem_integrity_failure")
            widths.append(width)
            half_loads.append(ops.mul(ops.mul(source[cell], width), half))
            slope_numerator = ops.sub(state_full[cell + 1], state_full[cell])
            slopes.append(ops.div(slope_numerator, width))
            inverse_widths.append(ops.div(one, width))

        best: tuple[int, str, Fraction, Fraction, Fraction] | None = None
        for node in range(1, n_cells):
            element_id = (f"p1hat-v1:{problem.problem_fingerprint}:"
                          f"{problem.mesh_sha256}:node:{node}")
            if element_id in active:
                continue
            c = ops.add(half_loads[node - 1], half_loads[node])
            c = ops.sub(c, slopes[node - 1])
            c = ops.add(c, slopes[node])
            q = ops.add(inverse_widths[node - 1], inverse_widths[node])
            if q <= 0:
                _error(work, "problem_state_mismatch", "problem_integrity_failure")
            score_numerator = ops.mul(c, c)
            work.values["hat_evaluations"] += 1
            work.values["local_cell_visits"] += 2
            if best is None:
                best = (node, element_id, c, q, score_numerator)
                continue
            left = ops.mul(score_numerator, best[3])
            right = ops.mul(best[4], q)
            work.values["fraction_comparisons"] += 1
            if left > right or (left == right and element_id < best[1]):
                best = (node, element_id, c, q, score_numerator)

        if best is None:
            _error(work, "problem_state_mismatch", "problem_integrity_failure")
        node, element_id, local_c, local_q, score_numerator = best
        if score_numerator == 0:
            work.values["fraction_ops_actual_total"] = work.values["fraction_ops_local_actual"]
            if not _state_is_unchanged(state, state_fingerprint, problem):
                _error(work, "problem_state_mismatch", "state_changed_during_proposal")
            _error(work, "no_positive_gain", "all_gains_zero")

        direction = np.zeros(n_cells + 1, dtype=np.float64)
        direction[node] = 1.0
        certificate_audit: dict[str, int] = {}
        certificate = certify_p1_poisson_step(
            problem.certificate_data(), full_state, direction, _audit=certificate_audit)
        work.values["certificate_replays"] = 1
        work.values["fraction_ops_certificate_actual"] = certificate_audit.get(
            "exact_rational_operations", 0)
        work.values["fraction_conversions_certificate_actual"] = certificate_audit.get(
            "fraction_conversions", 0)
        work.values["certificate_input_bytes"] = certificate_audit.get("input_payload_bytes", 0)
        work.values["certificate_input_serializations"] += 1
        work.values["serialization_bytes"] += certificate_audit.get("input_payload_bytes", 0)
        work.values["hash_events"] += certificate_audit.get("hash_events", 0)
        work.values["hash_bytes"] += certificate_audit.get("hash_bytes", 0)
        work.values["fraction_ops_actual_total"] = (
            work.values["fraction_ops_local_actual"]
            + work.values["fraction_ops_certificate_actual"])
        if (work.values["fraction_ops_certificate_actual"] > limits["max_certificate_ops"]
                or work.values["fraction_ops_actual_total"] > limits["max_fraction_ops_total"]
                or work.values["fraction_conversions_local_actual"]
                + work.values["fraction_conversions_certificate_actual"]
                > limits["max_fraction_conversions"]
                or work.values["hash_bytes"] > limits["max_hash_bytes"]):
            _error(work, "budget_exceeded", "live_counter_exceeded")
        if certificate.status == "unsupported_resource_limit":
            _error(work, "budget_exceeded", "certificate_resource_limit")
        if certificate.status != "verified_positive_gain":
            _error(work, "budget_exceeded", "certificate_invalid")
        if certificate.c_exact != local_c or certificate.q_exact != local_q:
            _error(work, "budget_exceeded", "certificate_identity_mismatch")
        if certificate.exact_rational_operations != work.values["fraction_ops_certificate_actual"]:
            _error(work, "budget_exceeded", "internal_work_bound_exceeded")

        certificate_json = certificate.to_json()
        certificate_bytes = certificate_json.encode("utf-8")
        work.values["certificate_record_bytes"] = len(certificate_bytes)
        work.values["serialization_bytes"] += len(certificate_bytes)
        if len(certificate_bytes) > limits["max_certificate_record_bytes"]:
            _error(work, "budget_exceeded", "certificate_record_byte_cap")
        record_digest = hashlib.sha256(certificate_bytes).hexdigest()
        work.values["hash_events"] += 1
        work.values["hash_bytes"] += len(certificate_bytes)
        if work.values["hash_bytes"] > limits["max_hash_bytes"]:
            _error(work, "budget_exceeded", "hash_byte_cap")

        work.values["candidate_vector_elements"] = n_cells - 1
        work.values["candidate_vector_copy_elements"] = n_cells - 1
        correction = np.zeros(n_cells - 1, dtype=np.float64)
        correction[node - 1] = 1.0
        work_plain = _plain(work.snapshot())
        support = {"node_index": node, "cell_indices": [node - 1, node]}
        parameters = {"basis": "interior_p1_hat", "node_index": node}
        provenance = {
            "learner": "P1PoissonHatWeakLearner",
            "problem_fingerprint": problem.problem_fingerprint,
            "data_fingerprint": problem.data_sha256,
            "mesh_sha256": problem.mesh_sha256,
            "state_fingerprint": state_fingerprint,
            "dictionary_size": k,
            "tie_rule": "ascending_stable_element_id",
            "weakness_tau": 1.0,
            "certificate_status": certificate.status,
            "certificate_sha256": record_digest,
            "certificate_record_json": certificate_json,
            "work_audit": work_plain,
        }
        metadata_for_size = {
            "element_id": element_id,
            "family": self.family,
            "support": support,
            "parameters": parameters,
            "seed": None,
            "provenance": {key: value for key, value in provenance.items()
                           if key != "certificate_record_json"},
            "proposal_cost": {"fraction_ops": float(work.values["fraction_ops_actual_total"]),
                              "hat_evaluations": float(work.values["hat_evaluations"])},
        }
        predicted_size = max(1, int(work.values["candidate_metadata_bytes"]))
        metadata_stable = False
        for _ in range(4):
            # Reserve the current serialization in the snapshot before encoding.
            # If the predicted length is wrong, reconcile it below and retry;
            # the next snapshot then carries exact cumulative counts.
            work.values["candidate_metadata_serializations"] += 1
            work.values["candidate_metadata_serialization_bytes"] += predicted_size
            work.values["serialization_bytes"] += predicted_size
            work.values["candidate_metadata_bytes"] = predicted_size
            provenance["work_audit"] = _plain(work.snapshot())
            metadata_for_size["provenance"] = {
                key: value for key, value in provenance.items()
                if key != "certificate_record_json"
            }
            encoded_metadata = json.dumps(metadata_for_size, sort_keys=True,
                                          separators=(",", ":"), ensure_ascii=True,
                                          allow_nan=False).encode("utf-8")
            actual_size = len(encoded_metadata)
            delta = actual_size - predicted_size
            work.values["candidate_metadata_serialization_bytes"] += delta
            work.values["serialization_bytes"] += delta
            work.values["candidate_metadata_bytes"] = actual_size
            if delta == 0:
                metadata_stable = True
                break
            predicted_size = actual_size
        if not metadata_stable:
            _error(work, "budget_exceeded", "internal_work_bound_exceeded")
        if work.values["candidate_metadata_bytes"] > limits["max_candidate_metadata_bytes"]:
            _error(work, "budget_exceeded", "candidate_metadata_byte_cap")

        if not _state_is_unchanged(state, state_fingerprint, problem):
            _error(work, "problem_state_mismatch", "state_changed_during_proposal")
        return Candidate(
            element_id=element_id,
            family=self.family,
            correction=correction,
            operator_image=None,
            support=support,
            parameters=parameters,
            seed=None,
            provenance=provenance,
            proposal_cost={"fraction_ops": float(work.values["fraction_ops_actual_total"]),
                           "hat_evaluations": float(work.values["hat_evaluations"])},
        )


__all__ = ["P1HatProposalError", "P1PoissonHatWeakLearner"]
