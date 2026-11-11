"""Exact directional-gain oracle for a narrowly defined 1D P1 Poisson model.

Input binary64 values are interpreted as their exact dyadic rationals. This
module derives the continuous weak-form quantities from source/mesh/nodal data;
it is not a general PDE estimator or a solver acceptance policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re
import struct
from typing import Any

import numpy as np

from estimators_1d import P1Poisson1DData


P1_POISSON_SCHEMA = "cem.p1_poisson_1d_step_certificate.v1"
P1_POISSON_ALGORITHM = "exact-dyadic-piecewise-linear-cell-gain.v1"
_SUPPORTED_MODEL = "poisson_1d_zero_dirichlet_cellwise_constant_p1_binary64_exact"
_SIGN_CONVENTION = "c=integral(f*phi)-integral(u_prime*phi_prime);alpha=c/q;signed"
_SUCCESS_STATUSES = {"verified_positive_gain", "verified_zero_gain"}
_FAILURE_STATUSES = {"invalid_input", "unsupported_resource_limit"}
_ALL_STATUSES = _SUCCESS_STATUSES | _FAILURE_STATUSES

MAX_P1_CELLS = 4_096
MAX_EXACT_RATIONAL_OPERATIONS = 200_000
MAX_FRACTION_COMPONENT_BITS = 16_384
MAX_RECORD_JSON_BYTES = 4_000_000
MAX_RECORD_JSON_DEPTH = 64
MAX_REASON_CHARS = 256

# Counting the checked operations in _assemble_c_q (14/cell), the two
# _direct_energy calls (20/cell + 4 fixed), exact trial construction (2/cell +
# 2 fixed), alpha/gain (4 fixed), and final difference (1 fixed) gives exactly
# 36*cells + 11 operations. Preflight and the live counter both enforce caps.
_MAX_RATIONAL_OPS_PER_CELL_PREFLIGHT = 36
_MAX_RATIONAL_OPS_FIXED_PREFLIGHT = 11
_FLOAT_HEX_RE = re.compile(r"^[0-9a-f]{16}$")
_DECIMAL_INT_RE = re.compile(r"^(?:0|-?[1-9][0-9]*)$")


class _InvalidInput(ValueError):
    pass


class _ResourceLimit(ValueError):
    pass


def _bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


class _Budget:
    def __init__(self) -> None:
        self.operations = 0

    def _apply(self, left: Fraction, right: Fraction, operation: str) -> Fraction:
        self.operations += 1
        if self.operations > MAX_EXACT_RATIONAL_OPERATIONS:
            raise _ResourceLimit("exact_rational_operation_limit_exceeded")
        if _bits(left) > MAX_FRACTION_COMPONENT_BITS or _bits(right) > MAX_FRACTION_COMPONENT_BITS:
            raise _ResourceLimit("fraction_operand_bit_limit_exceeded")
        if operation == "add":
            value = left + right
        elif operation == "sub":
            value = left - right
        elif operation == "mul":
            value = left * right
        elif operation == "div":
            if right == 0:
                raise _InvalidInput("division_by_zero")
            value = left / right
        else:  # pragma: no cover - private invariant
            raise AssertionError("unknown rational operation")
        if _bits(value) > MAX_FRACTION_COMPONENT_BITS:
            raise _ResourceLimit("fraction_component_bit_limit_exceeded")
        return value

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "add")

    def sub(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "sub")

    def mul(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "mul")

    def div(self, left: Fraction, right: Fraction) -> Fraction:
        return self._apply(left, right, "div")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _float_hex(value: np.float64) -> str:
    return struct.pack(">d", float(value)).hex()


def _fraction_pair(value: Fraction | None) -> list[str] | None:
    if value is None:
        return None
    return [_int_to_decimal(value.numerator), _int_to_decimal(value.denominator)]


def _int_to_decimal(value: int) -> str:
    """Convert a bounded integer without relying on Python's decimal digit cap."""
    if value == 0:
        return "0"
    negative = value < 0
    remaining = -value if negative else value
    chunks: list[int] = []
    while remaining:
        remaining, chunk = divmod(remaining, 1_000_000_000)
        chunks.append(chunk)
    result = str(chunks[-1]) + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))
    return "-" + result if negative else result


def _parse_decimal_integer(value: object, *, positive: bool = False) -> int:
    if not isinstance(value, str) or not _DECIMAL_INT_RE.fullmatch(value):
        raise ValueError("invalid canonical rational integer")
    digits = value[1:] if value.startswith("-") else value
    # 16,384 bits require at most 4,933 decimal digits. The byte cap is an
    # earlier bound; this digit cap avoids unbounded integer conversion.
    if len(digits) > 4_933:
        raise ValueError("rational integer exceeds parser limit")
    negative = value.startswith("-")
    result = 0
    # Chunked conversion avoids Python's global decimal-int digit-limit setting.
    first = len(digits) % 9 or 9
    result = int(digits[:first])
    for offset in range(first, len(digits), 9):
        result = result * 1_000_000_000 + int(digits[offset:offset + 9])
    if negative:
        result = -result
    if positive and result <= 0:
        raise ValueError("rational denominator must be positive")
    if max(abs(result).bit_length(), 1) > MAX_FRACTION_COMPONENT_BITS:
        raise ValueError("rational integer exceeds component bit limit")
    return result


def _fraction_from_pair(value: object) -> Fraction | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("rational value must be a two-item list")
    numerator = _parse_decimal_integer(value[0])
    denominator = _parse_decimal_integer(value[1], positive=True)
    result = Fraction(numerator, denominator)
    if [_int_to_decimal(result.numerator), _int_to_decimal(result.denominator)] != value:
        raise ValueError("rational pair is not reduced and canonical")
    return result


def _object_no_duplicates(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _json_depth(payload: str) -> int:
    depth = maximum = 0
    in_string = escaped = False
    for char in payload:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            maximum = max(maximum, depth)
            if maximum > MAX_RECORD_JSON_DEPTH:
                raise ValueError("record_json_depth_limit_exceeded")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("malformed_json_nesting")
    if in_string or depth != 0:
        raise ValueError("malformed_json_nesting")
    return maximum


def _parse_float_hex_array(value: object, expected_length: int | None, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name}_must_be_a_list")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"{name}_length_mismatch")
    if len(value) > MAX_P1_CELLS + 1:
        raise ValueError("mesh_cell_limit_exceeded")
    result: list[float] = []
    for item in value:
        if not isinstance(item, str) or not _FLOAT_HEX_RE.fullmatch(item):
            raise ValueError(f"{name}_contains_invalid_float_hex")
        number = struct.unpack(">d", bytes.fromhex(item))[0]
        if not math.isfinite(number):
            raise ValueError(f"{name}_contains_nonfinite_value")
        result.append(number)
    return tuple(result)


def _validate_input_payload(payload: object) -> dict[str, object]:
    expected = {"schema", "algorithm", "model", "sign_convention", "nodes_hex",
                "cell_source_hex", "state_hex", "direction_hex"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("invalid_certificate_input_fields")
    if (payload["schema"] != P1_POISSON_SCHEMA
            or payload["algorithm"] != P1_POISSON_ALGORITHM
            or payload["model"] != _SUPPORTED_MODEL
            or payload["sign_convention"] != _SIGN_CONVENTION):
        raise ValueError("unsupported_certificate_input_protocol")
    nodes = _parse_float_hex_array(payload["nodes_hex"], None, "nodes_hex")
    if len(nodes) < 2:
        raise ValueError("mesh_requires_at_least_two_nodes")
    cells = len(nodes) - 1
    if cells > MAX_P1_CELLS:
        raise ValueError("mesh_cell_limit_exceeded")
    source = _parse_float_hex_array(payload["cell_source_hex"], cells, "cell_source_hex")
    state = _parse_float_hex_array(payload["state_hex"], cells + 1, "state_hex")
    direction = _parse_float_hex_array(payload["direction_hex"], cells + 1, "direction_hex")
    if nodes[0] != 0.0 or nodes[-1] != 1.0 or any(
            right <= left for left, right in zip(nodes, nodes[1:])):
        raise ValueError("nodes_must_increase_strictly_from_zero_to_one")
    if state[0] != 0.0 or state[-1] != 0.0:
        raise ValueError("state_must_satisfy_zero_dirichlet_values")
    if direction[0] != 0.0 or direction[-1] != 0.0:
        raise ValueError("direction_must_satisfy_zero_dirichlet_values")
    for name in ("nodes_hex", "cell_source_hex", "state_hex", "direction_hex"):
        if not isinstance(payload[name], list):
            raise ValueError(f"{name}_must_be_a_list")
    canonical = {
        "schema": P1_POISSON_SCHEMA,
        "algorithm": P1_POISSON_ALGORITHM,
        "model": _SUPPORTED_MODEL,
        "sign_convention": _SIGN_CONVENTION,
        "nodes_hex": list(payload["nodes_hex"]),
        "cell_source_hex": list(payload["cell_source_hex"]),
        "state_hex": list(payload["state_hex"]),
        "direction_hex": list(payload["direction_hex"]),
    }
    return canonical


@dataclass(frozen=True)
class P1PoissonStepCertificate:
    """Immutable exact record for one P1 Poisson direction and line step."""

    status: str
    reason: str
    input_sha256: str | None = None
    input_payload_json: str | None = None
    c_exact: Fraction | None = None
    q_exact: Fraction | None = None
    alpha: Fraction | None = None
    gain_exact: Fraction | None = None
    exact_rational_operations: int | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.status, str) or self.status not in _ALL_STATUSES
                or not isinstance(self.reason, str) or len(self.reason) > MAX_REASON_CHARS):
            raise ValueError("invalid_certificate_status_or_reason")
        fractions = (self.c_exact, self.q_exact, self.alpha, self.gain_exact)
        if any(value is not None and not isinstance(value, Fraction) for value in fractions):
            raise ValueError("certificate_values_must_be_exact_fractions")
        if self.status in _SUCCESS_STATUSES:
            if (self.reason or self.input_payload_json is None
                    or self.input_sha256 is None
                    or not re.fullmatch(r"[0-9a-f]{64}", self.input_sha256)
                    or type(self.exact_rational_operations) is not int
                    or not 0 <= self.exact_rational_operations <= MAX_EXACT_RATIONAL_OPERATIONS
                    or any(value is None for value in fractions)):
                raise ValueError("successful_certificate_record_is_incomplete")
            if any(_bits(value) > MAX_FRACTION_COMPONENT_BITS for value in fractions if value is not None):
                raise ValueError("certificate_fraction_component_limit_exceeded")
            if not isinstance(self.input_payload_json, str):
                raise ValueError("certificate_input_payload_must_be_text")
            if (len(self.input_payload_json) > MAX_RECORD_JSON_BYTES
                    or len(self.input_payload_json.encode("utf-8")) > MAX_RECORD_JSON_BYTES):
                raise ValueError("certificate_input_payload_size_limit_exceeded")
            _json_depth(self.input_payload_json)
            parsed_input = json.loads(
                self.input_payload_json, object_pairs_hook=_object_no_duplicates,
                parse_int=lambda _: (_ for _ in ()).throw(ValueError("JSON numbers are not canonical fields")),
                parse_float=lambda _: (_ for _ in ()).throw(ValueError("JSON numbers are not canonical fields")),
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")),
            )
            canonical_input = _canonical_json(_validate_input_payload(parsed_input))
            if canonical_input != self.input_payload_json:
                raise ValueError("noncanonical_certificate_input")
            if hashlib.sha256(canonical_input.encode("utf-8")).hexdigest() != self.input_sha256:
                raise ValueError("certificate_input_hash_mismatch")
            if self.q_exact <= 0:
                raise ValueError("successful_certificate_requires_positive_q")
            if self.status == "verified_positive_gain" and (
                    self.c_exact == 0 or self.alpha == 0 or self.gain_exact <= 0):
                raise ValueError("positive_gain_status_requires_nonzero_directional_work")
            if self.status == "verified_zero_gain" and (
                    self.c_exact != 0 or self.alpha != 0 or self.gain_exact != 0):
                raise ValueError("zero_gain_status_requires_zero_directional_work")
        elif (not self.reason or self.input_sha256 is not None or self.input_payload_json is not None
              or self.exact_rational_operations is not None
              or any(value is not None for value in fractions)):
            raise ValueError("failed_certificate_must_not_carry_derived_values")

    def to_dict(self) -> dict[str, object]:
        input_payload: object = None
        if self.input_payload_json is not None:
            input_payload = json.loads(self.input_payload_json, object_pairs_hook=_object_no_duplicates)
        return {
            "schema": P1_POISSON_SCHEMA,
            "algorithm": P1_POISSON_ALGORITHM,
            "status": self.status,
            "reason": self.reason,
            "input_sha256": self.input_sha256,
            "input": input_payload,
            "c_exact": _fraction_pair(self.c_exact),
            "q_exact": _fraction_pair(self.q_exact),
            "alpha": _fraction_pair(self.alpha),
            "gain_exact": _fraction_pair(self.gain_exact),
            "exact_rational_operations": (None if self.exact_rational_operations is None
                                          else str(self.exact_rational_operations)),
        }

    def to_json(self) -> str:
        encoded = _canonical_json(self.to_dict())
        if len(encoded.encode("utf-8")) > MAX_RECORD_JSON_BYTES:
            raise ValueError("record_json_size_limit_exceeded")
        return encoded

    @classmethod
    def from_json(cls, payload: str) -> "P1PoissonStepCertificate":
        try:
            return cls._from_json(payload)
        except ValueError:
            raise
        except MemoryError as exc:
            raise ValueError("record_parser_resource_failure") from exc
        except Exception as exc:
            raise ValueError("invalid_certificate_json") from exc

    @classmethod
    def _from_json(cls, payload: str) -> "P1PoissonStepCertificate":
        if not isinstance(payload, str):
            raise ValueError("certificate_json_must_be_text")
        if len(payload) > MAX_RECORD_JSON_BYTES:
            raise ValueError("record_json_size_limit_exceeded")
        if len(payload.encode("utf-8")) > MAX_RECORD_JSON_BYTES:
            raise ValueError("record_json_size_limit_exceeded")
        _json_depth(payload)
        data = json.loads(
            payload,
            object_pairs_hook=_object_no_duplicates,
            parse_int=lambda _: (_ for _ in ()).throw(ValueError("JSON numbers are not canonical fields")),
            parse_float=lambda _: (_ for _ in ()).throw(ValueError("JSON numbers are not canonical fields")),
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON number")),
        )
        expected = {"schema", "algorithm", "status", "reason", "input_sha256", "input",
                    "c_exact", "q_exact", "alpha", "gain_exact", "exact_rational_operations"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("invalid_certificate_record_fields")
        if data["schema"] != P1_POISSON_SCHEMA or data["algorithm"] != P1_POISSON_ALGORITHM:
            raise ValueError("unsupported_certificate_record_protocol")
        status = data["status"]
        if not isinstance(status, str) or status not in _ALL_STATUSES or not isinstance(data["reason"], str):
            raise ValueError("invalid_certificate_record_status")
        input_payload_json = None
        input_sha256 = data["input_sha256"]
        if status in _SUCCESS_STATUSES:
            canonical_input = _validate_input_payload(data["input"])
            input_payload_json = _canonical_json(canonical_input)
            if input_payload_json != _canonical_json(data["input"]):
                raise ValueError("noncanonical_certificate_input")
            if not isinstance(input_sha256, str) or hashlib.sha256(
                    input_payload_json.encode("utf-8")).hexdigest() != input_sha256:
                raise ValueError("certificate_input_hash_mismatch")
            operations_field = _parse_decimal_integer(data["exact_rational_operations"], positive=False)
            if operations_field < 0:
                raise ValueError("negative_operation_count")
        elif data["input"] is not None or input_sha256 is not None:
            raise ValueError("failed_certificate_must_not_include_input_binding")
        else:
            operations_field = None
        result = cls(
            status=status,
            reason=data["reason"],
            input_sha256=input_sha256,
            input_payload_json=input_payload_json,
            c_exact=_fraction_from_pair(data["c_exact"]),
            q_exact=_fraction_from_pair(data["q_exact"]),
            alpha=_fraction_from_pair(data["alpha"]),
            gain_exact=_fraction_from_pair(data["gain_exact"]),
            exact_rational_operations=operations_field,
        )
        if result.to_json() != payload:
            raise ValueError("noncanonical_certificate_json")
        return result


def _failure(status: str, reason: str) -> P1PoissonStepCertificate:
    return P1PoissonStepCertificate(status=status, reason=reason[:MAX_REASON_CHARS])


def _input_payload(nodes: np.ndarray, source: np.ndarray, state: np.ndarray,
                   direction: np.ndarray) -> dict[str, object]:
    return {
        "schema": P1_POISSON_SCHEMA,
        "algorithm": P1_POISSON_ALGORITHM,
        "model": _SUPPORTED_MODEL,
        "sign_convention": _SIGN_CONVENTION,
        "nodes_hex": [_float_hex(value) for value in nodes],
        "cell_source_hex": [_float_hex(value) for value in source],
        "state_hex": [_float_hex(value) for value in state],
        "direction_hex": [_float_hex(value) for value in direction],
    }


def _validate_live_inputs(data: object, state: object, direction: object) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if type(data) is not P1Poisson1DData:
        raise _InvalidInput("expected_p1_poisson_1d_data")
    nodes = data.nodes
    source = data.cell_source
    if type(nodes) is not np.ndarray or nodes.dtype != np.dtype(np.float64) or nodes.ndim != 1:
        raise _InvalidInput("nodes_must_be_binary64_vector")
    if type(source) is not np.ndarray or source.dtype != np.dtype(np.float64) or source.ndim != 1:
        raise _InvalidInput("cell_source_must_be_binary64_vector")
    if nodes.size < 2:
        raise _InvalidInput("mesh_requires_at_least_two_nodes")
    cells = nodes.size - 1
    if cells > MAX_P1_CELLS:
        raise _ResourceLimit("mesh_cell_limit_exceeded")
    if (type(state) is not np.ndarray or state.dtype != np.dtype(np.float64)
            or state.ndim != 1):
        raise _InvalidInput("state_must_be_binary64_nodal_vector")
    if (type(direction) is not np.ndarray or direction.dtype != np.dtype(np.float64)
            or direction.ndim != 1):
        raise _InvalidInput("direction_must_be_binary64_nodal_vector")
    if max(source.size, state.size, direction.size) > MAX_P1_CELLS + 1:
        raise _ResourceLimit("input_vector_size_limit_exceeded")
    # Copy only after structural/resource preflight. All value validation,
    # hashing, and arithmetic below use these same snapshots.
    nodes = nodes.copy()
    source = source.copy()
    state = state.copy()
    direction = direction.copy()
    if source.size != cells:
        raise _InvalidInput("cell_source_length_mismatch")
    if state.size != cells + 1:
        raise _InvalidInput("state_length_mismatch")
    if direction.size != cells + 1:
        raise _InvalidInput("direction_length_mismatch")
    boundary = data.dirichlet_values
    if (type(boundary) is not tuple or len(boundary) != 2
            or any(type(value) is not float for value in boundary)
            or boundary != (0.0, 0.0)):
        raise _InvalidInput("only_zero_dirichlet_data_supported")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(source)):
        raise _InvalidInput("nonfinite_mesh_or_source")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(direction)):
        raise _InvalidInput("nonfinite_state_or_direction")
    if nodes[0] != 0.0 or nodes[-1] != 1.0 or np.any(nodes[1:] <= nodes[:-1]):
        raise _InvalidInput("nodes_must_increase_strictly_from_zero_to_one")
    if state[0] != 0.0 or state[-1] != 0.0:
        raise _InvalidInput("state_must_satisfy_zero_dirichlet_values")
    if direction[0] != 0.0 or direction[-1] != 0.0:
        raise _InvalidInput("direction_must_satisfy_zero_dirichlet_values")
    if cells * _MAX_RATIONAL_OPS_PER_CELL_PREFLIGHT + _MAX_RATIONAL_OPS_FIXED_PREFLIGHT > MAX_EXACT_RATIONAL_OPERATIONS:
        raise _ResourceLimit("exact_rational_operation_preflight_exceeded")
    return nodes, source, state, direction


def _exact_binary64(values: np.ndarray) -> tuple[Fraction, ...]:
    return tuple(Fraction.from_float(float(value)) for value in values)


def _assemble_c_q(nodes: tuple[Fraction, ...], source: tuple[Fraction, ...],
                  state: tuple[Fraction, ...], direction: tuple[Fraction, ...],
                  budget: _Budget) -> tuple[Fraction, Fraction]:
    c = Fraction(0)
    q = Fraction(0)
    half = Fraction(1, 2)
    for index in range(len(source)):
        h = budget.sub(nodes[index + 1], nodes[index])
        delta_u = budget.sub(state[index + 1], state[index])
        delta_phi = budget.sub(direction[index + 1], direction[index])
        phi_sum = budget.add(direction[index], direction[index + 1])
        phi_average = budget.mul(phi_sum, half)
        cell_load = budget.mul(budget.mul(source[index], h), phi_average)
        q_cell = budget.div(budget.mul(delta_phi, delta_phi), h)
        stiffness = budget.div(budget.mul(delta_u, delta_phi), h)
        c_cell = budget.sub(cell_load, stiffness)
        c = budget.add(c, c_cell)
        q = budget.add(q, q_cell)
    return c, q


def _direct_energy(state: tuple[Fraction, ...], nodes: tuple[Fraction, ...],
                   source: tuple[Fraction, ...], budget: _Budget) -> Fraction:
    quadratic = Fraction(0)
    load = Fraction(0)
    half = Fraction(1, 2)
    for index in range(len(source)):
        h = budget.sub(nodes[index + 1], nodes[index])
        delta = budget.sub(state[index + 1], state[index])
        cell_quadratic = budget.div(budget.mul(delta, delta), h)
        quadratic = budget.add(quadratic, cell_quadratic)
        endpoint_sum = budget.add(state[index], state[index + 1])
        cell_average = budget.mul(endpoint_sum, half)
        cell_load = budget.mul(budget.mul(source[index], h), cell_average)
        load = budget.add(load, cell_load)
    return budget.sub(budget.mul(quadratic, half), load)


def certify_p1_poisson_step(
    data: P1Poisson1DData,
    state_values: object,
    direction_values: object,
    *,
    _audit: dict[str, int] | None = None,
) -> P1PoissonStepCertificate:
    """Derive exact continuous ``c/q`` and line gain for the supported P1 case.

    Arrays must be finite native binary64 vectors. Their bit patterns are
    interpreted as exact dyadic rationals. The function is pure and never
    updates a solver state.
    """
    budget: _Budget | None = None
    conversions = 0
    payload_bytes = 0
    hash_events = 0
    hash_bytes = 0
    try:
        if _audit is not None and type(_audit) is not dict:
            raise TypeError("private certificate audit must be a dict")
        nodes_values, source_values, state_values, direction_values = _validate_live_inputs(
            data, state_values, direction_values)
        payload = _input_payload(nodes_values, source_values, state_values, direction_values)
        payload_json = _canonical_json(payload)
        payload_encoded = payload_json.encode("utf-8")
        if len(payload_encoded) > MAX_RECORD_JSON_BYTES:
            raise _ResourceLimit("input_record_size_limit_exceeded")
        payload_bytes = len(payload_encoded)
        input_hasher = hashlib.sha256()
        input_hasher.update(payload_encoded)
        hash_events += 1
        hash_bytes += payload_bytes
        input_digest = input_hasher.hexdigest()
        conversions = (nodes_values.size + source_values.size + state_values.size
                       + direction_values.size)
        nodes = _exact_binary64(nodes_values)
        source = _exact_binary64(source_values)
        state = _exact_binary64(state_values)
        direction = _exact_binary64(direction_values)
        budget = _Budget()
        c, q = _assemble_c_q(nodes, source, state, direction, budget)
        if q <= 0:
            raise _InvalidInput("zero_direction_or_nonpositive_curvature")
        alpha = budget.div(c, q)
        gain = budget.div(budget.mul(c, c), budget.mul(Fraction(2), q))
        trial = tuple(budget.add(value, budget.mul(alpha, correction))
                      for value, correction in zip(state, direction))
        direct_gain = budget.sub(
            _direct_energy(state, nodes, source, budget),
            _direct_energy(trial, nodes, source, budget),
        )
        if direct_gain != gain:
            raise _InvalidInput("direct_energy_difference_mismatch")
        status = "verified_zero_gain" if c == 0 else "verified_positive_gain"
        return P1PoissonStepCertificate(
            status=status,
            reason="",
            input_sha256=input_digest,
            input_payload_json=payload_json,
            c_exact=c,
            q_exact=q,
            alpha=alpha,
            gain_exact=gain,
            exact_rational_operations=budget.operations,
        )
    except _ResourceLimit as exc:
        return _failure("unsupported_resource_limit", str(exc))
    except (_InvalidInput, TypeError, ValueError, OverflowError, FloatingPointError,
            ArithmeticError, struct.error, AttributeError) as exc:
        return _failure("invalid_input", str(exc) or type(exc).__name__)
    except MemoryError:
        return _failure("unsupported_resource_limit", "memory_resource_limit_exceeded")
    except Exception:
        return _failure("invalid_input", "internal_validation_failure")
    finally:
        if _audit is not None and type(_audit) is dict:
            _audit.clear()
            _audit.update({
                "exact_rational_operations": 0 if budget is None else budget.operations,
                "fraction_conversions": conversions,
                "input_payload_bytes": payload_bytes,
                "hash_events": hash_events,
                "hash_bytes": hash_bytes,
            })


def verify_p1_poisson_step_certificate(
    certificate: P1PoissonStepCertificate | str,
    data: P1Poisson1DData,
    state_values: object,
    direction_values: object,
) -> bool:
    """Recompute a successful record from fresh inputs; serialized status is not proof."""
    try:
        if isinstance(certificate, str):
            certificate = P1PoissonStepCertificate.from_json(certificate)
        if (not isinstance(certificate, P1PoissonStepCertificate)
                or certificate.status not in _SUCCESS_STATUSES):
            return False
        fresh = certify_p1_poisson_step(data, state_values, direction_values)
        return fresh.status in _SUCCESS_STATUSES and fresh == certificate
    except (TypeError, ValueError, OverflowError, ArithmeticError, RecursionError,
            AttributeError, UnicodeError, MemoryError):
        return False


__all__ = [
    "P1PoissonStepCertificate",
    "certify_p1_poisson_step",
    "verify_p1_poisson_step_certificate",
]
