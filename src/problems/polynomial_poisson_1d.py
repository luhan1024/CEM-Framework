"""Replayable rational enclosures for a bounded 1D polynomial Poisson family.

This module is deliberately narrow: rational polynomial data on (0, 1),
homogeneous Dirichlet trial bases, and a float64 discrete Ritz problem assembled
by rounding the exact rational weak-form matrix and load. It is not a general
quadrature or PDE certification package.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import struct
from typing import Sequence

import numpy as np

from .problem import DiscreteRitzProblem
from .ritz_certificate import (
    DiscreteRitzStepCertificate,
    MAX_CERTIFICATE_DIMENSION,
    MAX_FRACTION_COMPONENT_BITS,
    MAX_LDL_SUM_PRODUCTS,
    certify_discrete_ritz_step,
    verify_discrete_ritz_certificate,
)


POLYNOMIAL_POISSON_SCHEMA = "cem.polynomial_poisson_1d_enclosure.v1"
POLYNOMIAL_POISSON_ALGORITHM = "exact-rational-polynomial-assembly.v1"
BINARY64_ROUNDING_ALGORITHM = "integer-ratio-ieee754-binary64-rne.v1"
MAX_POLYNOMIAL_DIMENSION = 16
MAX_POLYNOMIAL_DEGREE = 12
MAX_INPUT_COEFFICIENT_BITS = 256
MAX_INTERMEDIATE_FRACTION_BITS = 16_384
MAX_EXACT_RATIONAL_OPERATIONS = 200_000
MAX_RECORD_JSON_BYTES = 4_000_000
MAX_RECORD_JSON_DEPTH = 64


class PolynomialPoisson1DError(ValueError):
    """Invalid or unsupported input to the bounded polynomial family path."""


class _LimitExceeded(PolynomialPoisson1DError):
    pass


def _component_bits(value: Fraction) -> int:
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


class _Budget:
    def __init__(self) -> None:
        self.operations = 0
        self.max_component_bits = 0

    def check(self, value: Fraction) -> Fraction:
        size = _component_bits(value)
        self.max_component_bits = max(self.max_component_bits, size)
        if size > MAX_INTERMEDIATE_FRACTION_BITS:
            raise _LimitExceeded("intermediate_fraction_bit_limit_exceeded")
        return value

    def _tick(self) -> None:
        self.operations += 1
        if self.operations > MAX_EXACT_RATIONAL_OPERATIONS:
            raise _LimitExceeded("exact_rational_operation_limit_exceeded")

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        self._tick()
        return self.check(left + right)

    def sub(self, left: Fraction, right: Fraction) -> Fraction:
        self._tick()
        return self.check(left - right)

    def mul(self, left: Fraction, right: Fraction) -> Fraction:
        self._tick()
        return self.check(left * right)

    def div(self, left: Fraction, right: Fraction) -> Fraction:
        if right == 0:
            raise PolynomialPoisson1DError("division_by_zero_in_exact_arithmetic")
        self._tick()
        return self.check(left / right)


def _round_integer_ratio_ties_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient & 1):
        quotient += 1
    return quotient


def _floor_log2_ratio(numerator: int, denominator: int) -> int:
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif (numerator << -exponent) < denominator:
        exponent -= 1
    return exponent


def _float_from_bits(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, byteorder="big", signed=False))[0]


def _bits_from_float(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), byteorder="big", signed=False)


def _round_fraction_binary64(value: Fraction) -> float:
    """Correctly round an exact rational to IEEE-754 binary64, ties-to-even.

    The conversion is built from integer quotient/remainder operations rather
    than delegated to the host's Fraction-to-float implementation. Nonzero
    negative underflow preserves the sign bit on zero; exact rational zero is
    canonical positive zero. Overflow is rejected instead of returning inf.
    """
    if value == 0:
        return 0.0
    sign = value < 0
    magnitude = -value if sign else value
    numerator, denominator = magnitude.numerator, magnitude.denominator
    exponent = _floor_log2_ratio(numerator, denominator)

    if exponent >= -1022:
        if exponent > 1023:
            raise OverflowError("exact rational overflows finite binary64")
        shift = 52 - exponent
        if shift >= 0:
            scaled_numerator = numerator << shift
            scaled_denominator = denominator
        else:
            scaled_numerator = numerator
            scaled_denominator = denominator << -shift
        significand = _round_integer_ratio_ties_even(
            scaled_numerator, scaled_denominator)
        if significand == (1 << 53):
            significand >>= 1
            exponent += 1
        if exponent > 1023:
            raise OverflowError("exact rational rounds beyond finite binary64")
        if significand < (1 << 52):
            raise AssertionError("normal binary64 significand underflow")
        exponent_bits = exponent + 1023
        fraction_bits = significand - (1 << 52)
    else:
        significand = _round_integer_ratio_ties_even(numerator << 1074, denominator)
        if significand == 0:
            return _float_from_bits((int(sign) << 63))
        if significand >= (1 << 52):
            # Rounding at the top of the subnormal range reaches min-normal.
            exponent_bits = 1
            fraction_bits = 0
        else:
            exponent_bits = 0
            fraction_bits = significand

    bits = (int(sign) << 63) | (exponent_bits << 52) | fraction_bits
    return _float_from_bits(bits)


def _ceil_fraction_binary64(value: Fraction) -> float:
    """Return the least finite nonnegative binary64 not below value."""
    if value < 0:
        raise PolynomialPoisson1DError("negative_exact_radius")
    if value == 0:
        return 0.0
    nearest = _round_fraction_binary64(value)
    if Fraction.from_float(nearest) < value:
        nearest = math.nextafter(nearest, math.inf)
    if not math.isfinite(nearest) or Fraction.from_float(nearest) < value:
        raise OverflowError("exact radius has no finite binary64 upper bound")
    return nearest


def _float_hex(value: float) -> str:
    return float(value).hex()


def _fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _pairs(values: Sequence[Fraction]) -> list[list[int]]:
    return [_fraction_pair(value) for value in values]


def _matrix_pairs(values: Sequence[Sequence[Fraction]]) -> list[list[list[int]]]:
    return [_pairs(row) for row in values]


def _input_fraction(value: object) -> Fraction:
    if isinstance(value, bool):
        raise PolynomialPoisson1DError("polynomial_coefficients_must_be_exact_rationals")
    if isinstance(value, Fraction):
        result = value
    elif type(value) is int:
        result = Fraction(value)
    else:
        raise PolynomialPoisson1DError("polynomial_coefficients_must_be_int_or_fraction")
    if (_component_bits(result) > MAX_INPUT_COEFFICIENT_BITS):
        raise _LimitExceeded("input_coefficient_bit_limit_exceeded")
    return result


def _canonical_polynomial(values: object) -> tuple[Fraction, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise PolynomialPoisson1DError("polynomial_coefficients_must_be_nonempty_sequence")
    if len(values) > MAX_POLYNOMIAL_DEGREE + 1:
        raise _LimitExceeded("polynomial_degree_limit_exceeded")
    coefficients = [_input_fraction(value) for value in values]
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    if len(coefficients) - 1 > MAX_POLYNOMIAL_DEGREE:
        raise _LimitExceeded("polynomial_degree_limit_exceeded")
    return tuple(coefficients)


def _canonical_inputs(
    forcing: object, basis: object, u: object, phi: object,
) -> dict[str, object]:
    f = _canonical_polynomial(forcing)
    if not isinstance(basis, (tuple, list)) or not basis:
        raise PolynomialPoisson1DError("basis_must_be_nonempty_sequence")
    if len(basis) > MAX_POLYNOMIAL_DIMENSION:
        raise _LimitExceeded("polynomial_dimension_limit_exceeded")
    basis_polys = tuple(_canonical_polynomial(poly) for poly in basis)
    u_array, u_dtype, u_hex = _canonical_float_vector("u", u, len(basis_polys))
    phi_array, phi_dtype, phi_hex = _canonical_float_vector("phi", phi, len(basis_polys))
    del u_array, phi_array
    return {
        "forcing": _pairs(f),
        "basis": [_pairs(poly) for poly in basis_polys],
        "u": {"input_dtype": u_dtype, "float64_hex": u_hex},
        "phi": {"input_dtype": phi_dtype, "float64_hex": phi_hex},
    }


def _canonical_float_vector(name: str, values: object, dimension: int):
    try:
        array = np.asarray(values)
    except Exception as exc:
        raise PolynomialPoisson1DError(f"invalid_{name}_vector") from exc
    if array.ndim != 1 or array.shape != (dimension,):
        raise PolynomialPoisson1DError(f"{name}_vector_shape_mismatch")
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise PolynomialPoisson1DError(f"{name}_must_be_float32_or_float64")
    if not np.all(np.isfinite(array)):
        raise PolynomialPoisson1DError(f"{name}_must_be_finite")
    dtype_name = "float32" if array.dtype.itemsize == 4 else "float64"
    promoted = np.asarray(array, dtype=np.float64)
    return promoted, dtype_name, [_float_hex(x) for x in promoted]


def _read_pair(value: object, *, max_bits: int) -> Fraction:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(part) is not int for part in value) or value[1] <= 0):
        raise PolynomialPoisson1DError("invalid_canonical_fraction_pair")
    if max(abs(value[0]).bit_length(), value[1].bit_length()) > max_bits:
        raise _LimitExceeded("input_coefficient_bit_limit_exceeded")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise PolynomialPoisson1DError("fraction_pair_not_reduced_canonical")
    return result


def _read_poly(value: object) -> tuple[Fraction, ...]:
    if not isinstance(value, list) or not value:
        raise PolynomialPoisson1DError("invalid_polynomial_payload")
    if len(value) > MAX_POLYNOMIAL_DEGREE + 1:
        raise _LimitExceeded("polynomial_degree_limit_exceeded")
    result = tuple(_read_pair(pair, max_bits=MAX_INPUT_COEFFICIENT_BITS) for pair in value)
    if len(result) > 1 and result[-1] == 0:
        raise PolynomialPoisson1DError("polynomial_payload_has_trailing_zero")
    if len(result) - 1 > MAX_POLYNOMIAL_DEGREE:
        raise _LimitExceeded("polynomial_degree_limit_exceeded")
    return result


def _read_float_vector(value: object, dimension: int, name: str):
    if not isinstance(value, dict) or set(value) != {"input_dtype", "float64_hex"}:
        raise PolynomialPoisson1DError(f"invalid_{name}_payload")
    dtype_name = value["input_dtype"]
    if dtype_name not in ("float32", "float64"):
        raise PolynomialPoisson1DError(f"invalid_{name}_input_dtype")
    values = value["float64_hex"]
    if not isinstance(values, list) or len(values) != dimension:
        raise PolynomialPoisson1DError(f"invalid_{name}_vector_payload")
    converted = []
    for encoded in values:
        if not isinstance(encoded, str):
            raise PolynomialPoisson1DError(f"invalid_{name}_float_encoding")
        try:
            item = float.fromhex(encoded)
        except ValueError as exc:
            raise PolynomialPoisson1DError(f"invalid_{name}_float_encoding") from exc
        if not math.isfinite(item) or item.hex() != encoded:
            raise PolynomialPoisson1DError(f"noncanonical_{name}_float_encoding")
        if dtype_name == "float32":
            promoted = float(np.float32(item))
            if struct.pack(">d", promoted) != struct.pack(">d", item):
                raise PolynomialPoisson1DError("float32_value_not_exactly_promotable")
        converted.append(item)
    return np.asarray(converted, dtype=np.float64)


class _DuplicateKey(ValueError):
    pass


def _object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate JSON key")
        result[key] = value
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _check_json_nesting_depth(payload: str) -> None:
    """Reject excessive container nesting before invoking the recursive decoder."""
    depth = 0
    in_string = False
    escaped = False
    for character in payload:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_RECORD_JSON_DEPTH:
                raise ValueError("polynomial enclosure JSON nesting depth limit exceeded")
        elif character in "]}" and depth:
            depth -= 1


@dataclass(frozen=True)
class PolynomialPoissonEnclosure1D:
    """Immutable canonical JSON record; validity requires full replay."""

    _payload: str

    def to_json(self) -> str:
        return self._payload

    @classmethod
    def from_json(cls, payload: str) -> "PolynomialPoissonEnclosure1D":
        if not isinstance(payload, str):
            raise TypeError("enclosure record must be JSON text")
        try:
            encoded_size = len(payload.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("polynomial enclosure JSON must be valid Unicode text") from exc
        if encoded_size > MAX_RECORD_JSON_BYTES:
            raise ValueError("polynomial enclosure JSON exceeds record size limit")
        _check_json_nesting_depth(payload)
        try:
            data = json.loads(payload, object_pairs_hook=_object_no_duplicates)
        except (json.JSONDecodeError, _DuplicateKey, RecursionError) as exc:
            raise ValueError("invalid polynomial enclosure JSON") from exc
        if not isinstance(data, dict) or data.get("schema") != POLYNOMIAL_POISSON_SCHEMA:
            raise ValueError("unsupported polynomial enclosure schema")
        if _canonical_json(data) != payload:
            raise ValueError("polynomial enclosure JSON is not canonical")
        return cls(payload)


def _preflight_operation_bound(dimension: int, forcing_degree: int,
                               basis_degree: int) -> int:
    derivative_terms = max(1, basis_degree)
    basis_terms = basis_degree + 1
    forcing_terms = forcing_degree + 1
    return (
        2 * dimension * dimension * derivative_terms * derivative_terms
        + 2 * dimension * basis_terms * forcing_terms
        + 8 * dimension * dimension
        + 16 * dimension * basis_terms
        + 2 * dimension * dimension * dimension
        + 128
    )


def _poly_add(left, right, budget: _Budget):
    size = max(len(left), len(right))
    values = []
    for index in range(size):
        a = left[index] if index < len(left) else Fraction(0)
        b = right[index] if index < len(right) else Fraction(0)
        values.append(budget.add(a, b))
    return _trim(values)


def _trim(values):
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values) if values else (Fraction(0),)


def _poly_scale(poly, scalar: Fraction, budget: _Budget):
    return _trim([budget.mul(value, scalar) for value in poly])


def _poly_derivative(poly, budget: _Budget):
    if len(poly) <= 1:
        return (Fraction(0),)
    return _trim([budget.mul(poly[index], Fraction(index))
                  for index in range(1, len(poly))])


def _poly_multiply(left, right, budget: _Budget):
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            product = budget.mul(left_value, right_value)
            result[i + j] = budget.add(result[i + j], product)
    return _trim(result)


def _poly_integral_0_1(poly, budget: _Budget):
    result = Fraction(0)
    for power, coefficient in enumerate(poly):
        term = budget.div(coefficient, Fraction(power + 1))
        result = budget.add(result, term)
    return result


def _linear_combination(polys, coefficients, budget: _Budget):
    result = (Fraction(0),)
    for poly, coefficient in zip(polys, coefficients, strict=True):
        result = _poly_add(result, _poly_scale(poly, coefficient, budget), budget)
    return result


def _exact_spd(matrix, budget: _Budget) -> bool:
    n = len(matrix)
    lower = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    diagonal = [Fraction(0) for _ in range(n)]
    for i in range(n):
        lower[i][i] = Fraction(1)
        total = Fraction(0)
        for k in range(i):
            term = budget.mul(budget.mul(lower[i][k], lower[i][k]), diagonal[k])
            total = budget.add(total, term)
        pivot = budget.sub(matrix[i][i], total)
        if pivot <= 0:
            return False
        diagonal[i] = pivot
        for j in range(i + 1, n):
            total = Fraction(0)
            for k in range(i):
                term = budget.mul(
                    budget.mul(lower[j][k], lower[i][k]), diagonal[k])
                total = budget.add(total, term)
            numerator = budget.sub(matrix[j][i], total)
            lower[j][i] = budget.div(numerator, pivot)
    return True


def _dot(left, right, budget: _Budget) -> Fraction:
    result = Fraction(0)
    for a, b in zip(left, right, strict=True):
        result = budget.add(result, budget.mul(a, b))
    return result


def _quadratic(vector, matrix, budget: _Budget) -> Fraction:
    result = Fraction(0)
    for i, left in enumerate(vector):
        for j, right in enumerate(vector):
            result = budget.add(result, budget.mul(
                budget.mul(left, matrix[i][j]), right))
    return result


def _exact_energy(vector, matrix, load, budget: _Budget) -> Fraction:
    quadratic = _quadratic(vector, matrix, budget)
    linear = _dot(load, vector, budget)
    return budget.sub(budget.div(quadratic, Fraction(2)), linear)


def _derive(inputs: dict[str, object]):
    if set(inputs) != {"forcing", "basis", "u", "phi"}:
        raise PolynomialPoisson1DError("invalid_polynomial_enclosure_inputs")
    basis_payload = inputs["basis"]
    if not isinstance(basis_payload, list) or not basis_payload:
        raise PolynomialPoisson1DError("basis_must_be_nonempty_sequence")
    dimension = len(basis_payload)
    if dimension > MAX_POLYNOMIAL_DIMENSION:
        raise _LimitExceeded("polynomial_dimension_limit_exceeded")
    forcing = _read_poly(inputs["forcing"])
    basis = tuple(_read_poly(poly) for poly in basis_payload)
    max_degree = max(len(poly) - 1 for poly in basis)
    forcing_degree = len(forcing) - 1
    if max_degree > MAX_POLYNOMIAL_DEGREE or forcing_degree > MAX_POLYNOMIAL_DEGREE:
        raise _LimitExceeded("polynomial_degree_limit_exceeded")
    u_array = _read_float_vector(inputs["u"], dimension, "u")
    phi_array = _read_float_vector(inputs["phi"], dimension, "phi")
    u_fraction = tuple(Fraction.from_float(float(value)) for value in u_array)
    phi_fraction = tuple(Fraction.from_float(float(value)) for value in phi_array)

    # Inputs and a conservative operation-count upper bound are checked before
    # any polynomial products or matrix allocation.
    predicted_ops = _preflight_operation_bound(dimension, forcing_degree, max_degree)
    if predicted_ops > MAX_EXACT_RATIONAL_OPERATIONS:
        raise _LimitExceeded("exact_rational_operation_preflight_exceeded")
    for polynomial in (forcing, *basis):
        if any(_component_bits(coefficient) > MAX_INPUT_COEFFICIENT_BITS
               for coefficient in polynomial):
            raise _LimitExceeded("input_coefficient_bit_limit_exceeded")

    budget = _Budget()
    for polynomial in basis:
        at_zero = polynomial[0]
        at_one = Fraction(0)
        for coefficient in polynomial:
            at_one = budget.add(at_one, coefficient)
        if at_zero != 0 or at_one != 0:
            raise PolynomialPoisson1DError("basis_violates_homogeneous_dirichlet_boundary")

    derivatives = tuple(_poly_derivative(poly, budget) for poly in basis)
    stiffness_exact = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            row.append(_poly_integral_0_1(
                _poly_multiply(derivatives[i], derivatives[j], budget), budget))
        stiffness_exact.append(tuple(row))
    stiffness_exact = tuple(stiffness_exact)
    load_exact = tuple(_poly_integral_0_1(
        _poly_multiply(forcing, basis[i], budget), budget) for i in range(dimension))

    if not _exact_spd(stiffness_exact, budget):
        raise PolynomialPoisson1DError("exact_polynomial_basis_not_linearly_independent")

    underflow_paths = []

    def rounded(value: Fraction, path: str) -> float:
        result = _round_fraction_binary64(value)
        if value != 0 and result == 0.0:
            underflow_paths.append(path)
        return result

    stiffness_float = tuple(tuple(rounded(value, f"K[{i},{j}]")
                                  for j, value in enumerate(row))
                            for i, row in enumerate(stiffness_exact))
    load_float = tuple(rounded(value, f"F[{i}]") for i, value in enumerate(load_exact))
    stiffness_dyad = tuple(tuple(Fraction.from_float(value) for value in row)
                           for row in stiffness_float)
    load_dyad = tuple(Fraction.from_float(value) for value in load_float)
    if not _exact_spd(stiffness_dyad, budget):
        raise PolynomialPoisson1DError("rounded_polynomial_stiffness_not_exactly_spd")

    try:
        problem = DiscreteRitzProblem(
            np.asarray(stiffness_float, dtype=np.float64),
            np.asarray(load_float, dtype=np.float64),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise PolynomialPoisson1DError("rounded_polynomial_problem_rejected") from exc

    u_poly = _linear_combination(basis, u_fraction, budget)
    phi_poly = _linear_combination(basis, phi_fraction, budget)
    u_derivative = _poly_derivative(u_poly, budget)
    phi_derivative = _poly_derivative(phi_poly, budget)
    load_directional = _poly_integral_0_1(
        _poly_multiply(forcing, phi_poly, budget), budget)
    stiffness_directional_state = _poly_integral_0_1(
        _poly_multiply(u_derivative, phi_derivative, budget), budget)
    c_cont = budget.sub(load_directional, stiffness_directional_state)
    q_cont = _poly_integral_0_1(
        _poly_multiply(phi_derivative, phi_derivative, budget), budget)
    if q_cont <= 0:
        raise PolynomialPoisson1DError("zero_or_nonpositive_continuous_direction")

    c_dyad = Fraction(0)
    for i in range(dimension):
        residual_i = load_dyad[i]
        for j in range(dimension):
            residual_i = budget.sub(
                residual_i, budget.mul(stiffness_dyad[i][j], u_fraction[j]))
        c_dyad = budget.add(c_dyad, budget.mul(residual_i, phi_fraction[i]))
    q_dyad = _quadratic(phi_fraction, stiffness_dyad, budget)
    if q_dyad <= 0:
        raise PolynomialPoisson1DError("zero_or_nonpositive_dyadic_direction")

    c_hat_float = rounded(c_cont, "c_hat")
    q_hat_float = rounded(q_cont, "q_hat")
    c_hat = Fraction.from_float(c_hat_float)
    q_hat = Fraction.from_float(q_hat_float)
    delta_exact = max(abs(c_cont - c_hat), abs(c_dyad - c_hat))
    epsilon_exact = max(abs(q_cont - q_hat), abs(q_dyad - q_hat))
    delta_float = _ceil_fraction_binary64(delta_exact)
    epsilon_float = _ceil_fraction_binary64(epsilon_exact)
    delta = Fraction.from_float(delta_float)
    epsilon = Fraction.from_float(epsilon_float)

    certificate = certify_discrete_ritz_step(
        problem, u_array, phi_array, c_hat_float, q_hat_float,
        delta_float, epsilon_float,
    )
    if certificate.status not in ("verified_discrete_lower_bound", "zero_certified_gain"):
        raise PolynomialPoisson1DError(
            f"discrete_ritz_certificate_rejected:{certificate.reason}")
    if certificate.trial_exact is None or certificate.alpha is None:
        raise PolynomialPoisson1DError("discrete_ritz_certificate_incomplete")

    alpha = certificate.alpha
    trial = tuple(budget.add(u_fraction[i], budget.mul(alpha, phi_fraction[i]))
                  for i in range(dimension))
    if trial != certificate.trial_exact:
        raise PolynomialPoisson1DError("certificate_trial_mismatch")
    continuous_gain = budget.sub(
        _exact_energy(u_fraction, stiffness_exact, load_exact, budget),
        _exact_energy(trial, stiffness_exact, load_exact, budget),
    )
    discrete_gain = budget.sub(
        _exact_energy(u_fraction, stiffness_dyad, load_dyad, budget),
        _exact_energy(trial, stiffness_dyad, load_dyad, budget),
    )
    if (continuous_gain < certificate.lower_bound
            or discrete_gain != certificate.realized_gain
            or discrete_gain < certificate.lower_bound):
        raise PolynomialPoisson1DError("exact_gain_replay_failed")
    if budget.operations > predicted_ops:
        raise PolynomialPoisson1DError("operation_preflight_bound_underestimated")

    limits = {
        "dimension": MAX_POLYNOMIAL_DIMENSION,
        "degree": MAX_POLYNOMIAL_DEGREE,
        "input_coefficient_bits": MAX_INPUT_COEFFICIENT_BITS,
        "intermediate_fraction_bits": MAX_INTERMEDIATE_FRACTION_BITS,
        "rational_operations": MAX_EXACT_RATIONAL_OPERATIONS,
        "record_json_bytes": MAX_RECORD_JSON_BYTES,
        "certificate_dimension": MAX_CERTIFICATE_DIMENSION,
        "certificate_ldl_sum_products": MAX_LDL_SUM_PRODUCTS,
        "certificate_fraction_bits": MAX_FRACTION_COMPONENT_BITS,
    }
    counts = {
        "dimension": dimension,
        "forcing_degree": forcing_degree,
        "max_basis_degree": max_degree,
        "preflight_operation_bound": predicted_ops,
        "actual_rational_operations": budget.operations,
        "max_intermediate_fraction_bits": budget.max_component_bits,
    }
    derived = {
        "stiffness_exact": _matrix_pairs(stiffness_exact),
        "load_exact": _pairs(load_exact),
        "stiffness_float64_hex": [[_float_hex(value) for value in row]
                                   for row in stiffness_float],
        "load_float64_hex": [_float_hex(value) for value in load_float],
        "c_cont": _fraction_pair(c_cont),
        "q_cont": _fraction_pair(q_cont),
        "c_dyad": _fraction_pair(c_dyad),
        "q_dyad": _fraction_pair(q_dyad),
        "c_hat_hex": _float_hex(c_hat_float),
        "q_hat_hex": _float_hex(q_hat_float),
        "delta_hex": _float_hex(delta_float),
        "epsilon_hex": _float_hex(epsilon_float),
        "continuous_gain": _fraction_pair(continuous_gain),
        "discrete_gain": _fraction_pair(discrete_gain),
        "underflow_paths": underflow_paths,
        "limits": limits,
        "counts": counts,
    }
    payload = {
        "schema": POLYNOMIAL_POISSON_SCHEMA,
        "algorithm": POLYNOMIAL_POISSON_ALGORITHM,
        "rounding": BINARY64_ROUNDING_ALGORITHM,
        "inputs": inputs,
        "derived": derived,
        "certificate": certificate.to_dict(),
    }
    encoded_payload = _canonical_json(payload)
    if len(encoded_payload.encode("utf-8")) > MAX_RECORD_JSON_BYTES:
        raise _LimitExceeded("polynomial_enclosure_record_size_limit_exceeded")
    return problem, payload, certificate, u_array, phi_array, (
        c_hat_float, q_hat_float, delta_float, epsilon_float)


def build_polynomial_poisson_1d_enclosure(
    forcing: Sequence[int | Fraction],
    basis: Sequence[Sequence[int | Fraction]],
    u: object,
    phi: object,
) -> tuple[DiscreteRitzProblem, PolynomialPoissonEnclosure1D]:
    """Build the rounded Ritz problem and replayable dual-target enclosure.

    Polynomial coefficients are exact integers/Fractions in ascending powers.
    u and phi must be one-dimensional float32/float64 arrays; float32 values
    are promoted exactly to binary64 before their dyadic interpretation.
    """
    inputs = _canonical_inputs(forcing, basis, u, phi)
    problem, payload, _, _, _, _ = _derive(inputs)
    return problem, PolynomialPoissonEnclosure1D(_canonical_json(payload))


def _parse_record_payload(record: PolynomialPoissonEnclosure1D) -> dict[str, object]:
    if not isinstance(record, PolynomialPoissonEnclosure1D):
        raise PolynomialPoisson1DError("expected_polynomial_poisson_enclosure_record")
    serialized = record.to_json()
    if not isinstance(serialized, str):
        raise PolynomialPoisson1DError("polynomial_enclosure_record_payload_must_be_text")
    if len(serialized.encode("utf-8")) > MAX_RECORD_JSON_BYTES:
        raise _LimitExceeded("polynomial_enclosure_record_size_limit_exceeded")
    _check_json_nesting_depth(serialized)
    try:
        payload = json.loads(serialized, object_pairs_hook=_object_no_duplicates)
    except (json.JSONDecodeError, _DuplicateKey, RecursionError) as exc:
        raise PolynomialPoisson1DError("invalid_polynomial_enclosure_record_json") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != serialized:
        raise PolynomialPoisson1DError("noncanonical_polynomial_enclosure_record")
    if set(payload) != {"schema", "algorithm", "rounding", "inputs", "derived", "certificate"}:
        raise PolynomialPoisson1DError("invalid_polynomial_enclosure_record_fields")
    if (payload["schema"] != POLYNOMIAL_POISSON_SCHEMA
            or payload["algorithm"] != POLYNOMIAL_POISSON_ALGORITHM
            or payload["rounding"] != BINARY64_ROUNDING_ALGORITHM):
        raise PolynomialPoisson1DError("unsupported_polynomial_enclosure_algorithm")
    return payload


def _validate_canonical_inputs(inputs: object) -> dict[str, object]:
    if not isinstance(inputs, dict) or set(inputs) != {"forcing", "basis", "u", "phi"}:
        raise PolynomialPoisson1DError("invalid_polynomial_enclosure_inputs")
    forcing = _read_poly(inputs["forcing"])
    basis_payload = inputs["basis"]
    if not isinstance(basis_payload, list) or not basis_payload:
        raise PolynomialPoisson1DError("basis_must_be_nonempty_sequence")
    basis = tuple(_read_poly(poly) for poly in basis_payload)
    u = _read_float_vector(inputs["u"], len(basis), "u")
    phi = _read_float_vector(inputs["phi"], len(basis), "phi")
    canonical = {
        "forcing": _pairs(forcing),
        "basis": [_pairs(poly) for poly in basis],
        "u": {"input_dtype": inputs["u"]["input_dtype"],
              "float64_hex": [_float_hex(value) for value in u]},
        "phi": {"input_dtype": inputs["phi"]["input_dtype"],
                "float64_hex": [_float_hex(value) for value in phi]},
    }
    if canonical != inputs:
        raise PolynomialPoisson1DError("noncanonical_polynomial_enclosure_inputs")
    return canonical


def verify_polynomial_poisson_1d_enclosure(
    record: PolynomialPoissonEnclosure1D,
) -> bool:
    """Replay the full polynomial assembly, enclosure, and both exact gains."""
    try:
        payload = _parse_record_payload(record)
        inputs = _validate_canonical_inputs(payload["inputs"])
        problem, expected, certificate, u, phi, values = _derive(inputs)
        if expected != payload:
            return False
        embedded = payload["certificate"]
        if not isinstance(embedded, dict):
            return False
        parsed_certificate = DiscreteRitzStepCertificate.from_json(_canonical_json(embedded))
        if parsed_certificate.to_dict() != embedded:
            return False
        c_hat, q_hat, delta, epsilon = values
        if parsed_certificate != certificate:
            return False
        if not verify_discrete_ritz_certificate(
                parsed_certificate, problem, u, phi, c_hat, q_hat, delta, epsilon):
            return False
        return True
    except (PolynomialPoisson1DError, OverflowError, TypeError, ValueError, RecursionError,
            IndexError, ArithmeticError, np.linalg.LinAlgError):
        return False
