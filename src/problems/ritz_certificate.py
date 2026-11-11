"""Exact-rational checks for a declared finite-dimensional Ritz step.

This module verifies only the supplied finite IEEE-754 values, interpreted as
exact dyadic rationals, and the caller's declared c/q enclosures. It does not
construct or validate the provenance of those enclosures and does not certify
a continuous PDE or perform a solver update.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import numpy as np
from .problem import DiscreteRitzProblem


_STATUSES = {"verified_discrete_lower_bound", "zero_certified_gain",
             "invalid_input", "unsupported_resource_limit"}
MAX_CERTIFICATE_DIMENSION = 32
MAX_LDL_SUM_PRODUCTS = 6_000
MAX_FRACTION_COMPONENT_BITS = 16_384
_FRACTION_FIELDS = (
    "c_exact", "q_exact", "c_hat", "q_hat", "delta", "epsilon",
    "alpha", "lower_bound", "realized_gain",
)


def _fraction_payload(value: Fraction | None) -> list[int] | None:
    return None if value is None else [value.numerator, value.denominator]


def _fraction_from_payload(value: object) -> Fraction | None:
    if value is None:
        return None
    if (not isinstance(value, list) or len(value) != 2
            or any(type(part) is not int for part in value) or value[1] <= 0):
        raise ValueError("invalid rational certificate field")
    result = Fraction(value[0], value[1])
    if [result.numerator, result.denominator] != value:
        raise ValueError("rational certificate field is not canonical")
    return result


@dataclass(frozen=True)
class DiscreteRitzStepCertificate:
    """Immutable result for one exact-rational trial along a fixed direction."""

    status: str
    reason: str
    input_sha256: str | None = None
    c_exact: Fraction | None = None
    q_exact: Fraction | None = None
    c_hat: Fraction | None = None
    q_hat: Fraction | None = None
    delta: Fraction | None = None
    epsilon: Fraction | None = None
    alpha: Fraction | None = None
    lower_bound: Fraction | None = None
    realized_gain: Fraction | None = None
    trial_exact: tuple[Fraction, ...] | None = None
    feasibility_basis: str = "not_verified"

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ValueError("unknown certificate status")
        if not isinstance(self.reason, str):
            raise ValueError("certificate reason must be text")
        if self.feasibility_basis not in (
                "not_verified", "unconstrained_domain", "unsupported_callback"):
            raise ValueError("unknown feasibility basis")
        if any(value is not None and not isinstance(value, Fraction)
               for value in (getattr(self, name) for name in _FRACTION_FIELDS)):
            raise ValueError("certificate numeric fields must be exact Fractions")
        if self.trial_exact is not None and (
                not isinstance(self.trial_exact, tuple)
                or any(not isinstance(value, Fraction) for value in self.trial_exact)):
            raise ValueError("exact trial must be a tuple of Fractions")
        if self.input_sha256 is not None and (
                not isinstance(self.input_sha256, str) or len(self.input_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.input_sha256)):
            raise ValueError("invalid input SHA-256")
        if self.status in ("verified_discrete_lower_bound", "zero_certified_gain"):
            if (self.reason or self.input_sha256 is None or len(self.input_sha256) != 64
                    or self.feasibility_basis != "unconstrained_domain"
                    or any(getattr(self, name) is None for name in _FRACTION_FIELDS)
                    or self.trial_exact is None):
                raise ValueError("successful certificate record is incomplete")
            if self.status == "verified_discrete_lower_bound" and self.lower_bound <= 0:
                raise ValueError("positive certificate requires positive lower bound")
            if self.status == "zero_certified_gain" and self.lower_bound != 0:
                raise ValueError("zero-gain status requires zero lower bound")
        elif not self.reason:
            raise ValueError("rejected certificate record requires a reason")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "cem.discrete_ritz_step_certificate.v1",
            "status": self.status,
            "reason": self.reason,
            "input_sha256": self.input_sha256,
            "trial_exact": (None if self.trial_exact is None else
                            [_fraction_payload(x) for x in self.trial_exact]),
            "feasibility_basis": self.feasibility_basis,
        }
        result.update({name: _fraction_payload(getattr(self, name))
                       for name in _FRACTION_FIELDS})
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "DiscreteRitzStepCertificate":
        data = json.loads(payload)
        expected = {"schema", "status", "reason", "input_sha256", "trial_exact",
                    "feasibility_basis", *_FRACTION_FIELDS}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("invalid certificate record fields")
        if data["schema"] != "cem.discrete_ritz_step_certificate.v1":
            raise ValueError("unsupported certificate schema")
        trial_raw = data["trial_exact"]
        if trial_raw is None:
            trial = None
        elif isinstance(trial_raw, list):
            values = [_fraction_from_payload(x) for x in trial_raw]
            if any(x is None for x in values):
                raise ValueError("invalid exact trial record")
            trial = tuple(values)  # type: ignore[arg-type]
        else:
            raise ValueError("invalid exact trial record")
        if data["feasibility_basis"] not in (
                "not_verified", "unconstrained_domain", "unsupported_callback"):
            raise ValueError("invalid feasibility basis")
        if (not isinstance(data["status"], str) or data["status"] not in _STATUSES
                or not isinstance(data["reason"], str)
                or (data["input_sha256"] is not None and
                    (not isinstance(data["input_sha256"], str)
                     or len(data["input_sha256"]) != 64
                     or any(c not in "0123456789abcdef" for c in data["input_sha256"])) )):
            raise ValueError("invalid certificate identity/status fields")
        kwargs = {name: _fraction_from_payload(data[name]) for name in _FRACTION_FIELDS}
        return cls(
            status=data["status"], reason=data["reason"],
            input_sha256=data["input_sha256"], trial_exact=trial,
            feasibility_basis=data["feasibility_basis"], **kwargs,
        )


class _InvalidInput(Exception):
    pass


class _ResourceLimit(_InvalidInput):
    pass


def _check_fraction_budget(value: Fraction) -> Fraction:
    if (abs(value.numerator).bit_length() > MAX_FRACTION_COMPONENT_BITS
            or value.denominator.bit_length() > MAX_FRACTION_COMPONENT_BITS):
        raise _ResourceLimit("exact_fraction_bit_budget_exceeded")
    return value


def _as_f64_array(name: str, value: object, ndim: int) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if ndim == 1 and len(value) > MAX_CERTIFICATE_DIMENSION:
            raise _ResourceLimit("dimension_exceeds_exact_certificate_limit")
        if ndim == 2 and (len(value) > MAX_CERTIFICATE_DIMENSION or
                          any(not isinstance(row, (list, tuple, np.ndarray))
                              or len(row) > MAX_CERTIFICATE_DIMENSION for row in value)):
            raise _ResourceLimit("dimension_exceeds_exact_certificate_limit")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise _InvalidInput(f"invalid_{name}_array") from exc
    if array.dtype.kind != "f" or array.dtype.itemsize not in (4, 8):
        raise _InvalidInput(f"{name}_must_contain_float32_or_float64")
    if array.ndim != ndim:
        raise _InvalidInput(f"invalid_{name}_shape_or_nonfinite")
    if ndim == 2:
        if array.shape[0] == 0 or array.shape[0] != array.shape[1]:
            raise _InvalidInput("K_must_be_nonempty_square")
        if array.shape[0] > MAX_CERTIFICATE_DIMENSION:
            raise _ResourceLimit("dimension_exceeds_exact_certificate_limit")
    elif array.shape[0] > MAX_CERTIFICATE_DIMENSION:
        raise _ResourceLimit("dimension_exceeds_exact_certificate_limit")
    if not np.all(np.isfinite(array)):
        raise _InvalidInput(f"invalid_{name}_shape_or_nonfinite")
    # float32 values are represented exactly in binary64; wider dtypes are rejected.
    return np.asarray(array, dtype=np.float64)


def _as_f64_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating)):
        raise _InvalidInput(f"{name}_must_be_float32_or_float64")
    if np.asarray(value).dtype.itemsize not in (4, 8):
        raise _InvalidInput(f"{name}_must_be_float32_or_float64")
    result = float(value)
    if not np.isfinite(result):
        raise _InvalidInput(f"{name}_must_be_finite")
    return result


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(value)


def _input_hash(K: np.ndarray, f: np.ndarray, u: np.ndarray, phi: np.ndarray,
                scalars: tuple[float, ...], feasibility_mode: str) -> str:
    arrays = {}
    for name, array in (("K", K), ("f", f), ("u", u), ("phi", phi)):
        canonical = np.asarray(array, dtype="<f8", order="C")
        arrays[name] = {"shape": list(canonical.shape), "hex": canonical.tobytes().hex()}
    payload = json.dumps({"arrays": arrays,
                          "scalars_hex": [np.asarray(x, dtype="<f8").tobytes().hex()
                                          for x in scalars],
                          "feasibility_mode": feasibility_mode},
                         sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_exact_spd(matrix: tuple[tuple[Fraction, ...], ...]) -> bool:
    """Exact no-pivot LDL^T test for a symmetric rational matrix."""
    n = len(matrix)
    lower = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    diagonal = [Fraction(0) for _ in range(n)]
    for i in range(n):
        lower[i][i] = Fraction(1)
        total = Fraction(0)
        for k in range(i):
            term = _check_fraction_budget(lower[i][k] ** 2 * diagonal[k])
            total = _check_fraction_budget(total + term)
        pivot = _check_fraction_budget(matrix[i][i] - total)
        if pivot <= 0:
            return False
        diagonal[i] = pivot
        for j in range(i + 1, n):
            total = Fraction(0)
            for k in range(i):
                term = _check_fraction_budget(lower[j][k] * lower[i][k] * diagonal[k])
                total = _check_fraction_budget(total + term)
            numerator = _check_fraction_budget(matrix[j][i] - total)
            lower[j][i] = _check_fraction_budget(numerator / pivot)
    return True


def certify_discrete_ritz_step(
    problem: DiscreteRitzProblem,
    u: object,
    phi: object,
    c_hat: object,
    q_hat: object,
    delta: object,
    epsilon: object,
) -> DiscreteRitzStepCertificate:
    """Verify declared c/q enclosures and an exact finite-dimensional step.

    The problem's stored K/source, ``u``, ``phi``, and enclosure declarations
    must be finite float32/float64 arrays/scalars. Their numerical values are
    converted exactly to rational numbers. The first version supports only
    an unconstrained discrete domain (``problem.feasible is None``); arbitrary Python
    callbacks are not proof protocols and are rejected, even if they return
    true. The actual ``DiscreteRitzProblem.feasible`` property is inspected;
    callers cannot bypass the constraint gate by omitting it. Enclosure
    validity is checked, but how the caller obtained the
    enclosure radii is outside this verifier's scope.
    """
    digest: str | None = None
    base: dict[str, object] = {}
    try:
        if not isinstance(problem, DiscreteRitzProblem):
            raise _InvalidInput("expected_discrete_ritz_problem")
        if problem.feasible is not None:
            raise _InvalidInput("unsupported_feasibility_callback")
        matrix = _as_f64_array("K", problem.K, 2)
        source = _as_f64_array("f", problem.source, 1)
        state = _as_f64_array("u", u, 1)
        direction = _as_f64_array("phi", phi, 1)
        scalars_f = tuple(_as_f64_scalar(name, x) for name, x in (
            ("c_hat", c_hat), ("q_hat", q_hat), ("delta", delta), ("epsilon", epsilon)))
        c_hat_f, q_hat_f, delta_f, epsilon_f = scalars_f
        if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
            raise _InvalidInput("K_must_be_nonempty_square")
        n = matrix.shape[0]
        if n > MAX_CERTIFICATE_DIMENSION:
            raise _ResourceLimit("dimension_exceeds_exact_certificate_limit")
        ldl_sum_products = sum(i * (n - i) for i in range(n))
        if ldl_sum_products > MAX_LDL_SUM_PRODUCTS:
            raise _ResourceLimit("exact_ldl_operation_budget_exceeded")
        if source.shape != (n,) or state.shape != (n,) or direction.shape != (n,):
            raise _InvalidInput("incompatible_vector_shape")
        digest = _input_hash(matrix, source, state, direction, scalars_f, "unconstrained")
        if delta_f < 0 or epsilon_f < 0:
            raise _InvalidInput("negative_enclosure_radius")
        Kq = tuple(tuple(_fraction(float(matrix[i, j])) for j in range(n)) for i in range(n))
        fq = tuple(_fraction(float(x)) for x in source)
        uq = tuple(_fraction(float(x)) for x in state)
        phiq = tuple(_fraction(float(x)) for x in direction)
        chatq, qhatq, deltaq, epsilonq = map(_fraction, scalars_f)
        base = {"input_sha256": digest, "c_hat": chatq, "q_hat": qhatq,
                "delta": deltaq, "epsilon": epsilonq}
        if any(Kq[i][j] != Kq[j][i] for i in range(n) for j in range(i + 1, n)):
            raise _InvalidInput("K_not_exactly_symmetric")
        if qhatq <= epsilonq:
            raise _InvalidInput("q_hat_not_above_epsilon")
        residual = tuple(fq[i] - sum((Kq[i][j] * uq[j] for j in range(n)), Fraction(0))
                         for i in range(n))
        c = sum((residual[i] * phiq[i] for i in range(n)), Fraction(0))
        q = sum((phiq[i] * Kq[i][j] * phiq[j]
                 for i in range(n) for j in range(n)), Fraction(0))
        base.update({"c_exact": c, "q_exact": q})
        if q <= 0:
            raise _InvalidInput("nonpositive_directional_curvature")
        if not _is_exact_spd(Kq):
            raise _InvalidInput("K_not_exactly_positive_definite")
        if abs(c - chatq) > deltaq:
            raise _InvalidInput("c_enclosure_failed")
        if abs(q - qhatq) > epsilonq:
            raise _InvalidInput("q_enclosure_failed")
        s = max(abs(chatq) - deltaq, Fraction(0))
        sign = 1 if chatq > 0 else (-1 if chatq < 0 else 0)
        alpha = Fraction(sign) * s / (qhatq + epsilonq)
        trial = tuple(uq[i] + alpha * phiq[i] for i in range(n))
        lower = s * s / (2 * (qhatq + epsilonq))
        gain = alpha * c - alpha * alpha * q / 2
        if gain < lower:
            raise _InvalidInput("exact_gain_below_claimed_lower_bound")
        base.update({"alpha": alpha, "lower_bound": lower, "realized_gain": gain,
                     "trial_exact": trial, "feasibility_basis": "unconstrained_domain"})
        status = "verified_discrete_lower_bound" if lower > 0 else "zero_certified_gain"
        return DiscreteRitzStepCertificate(status=status, reason="", **base)
    except _InvalidInput as exc:
        status = "unsupported_resource_limit" if isinstance(exc, _ResourceLimit) else "invalid_input"
        return DiscreteRitzStepCertificate(status=status, reason=str(exc), **base)


def verify_discrete_ritz_certificate(
    certificate: DiscreteRitzStepCertificate,
    problem: DiscreteRitzProblem,
    u: object,
    phi: object,
    c_hat: object,
    q_hat: object,
    delta: object,
    epsilon: object,
) -> bool:
    """Recompute a record and independently check its exact objective difference.

    A serialized status is not proof. Verification binds the record to fresh
    inputs, recomputes every certificate field, and checks ``J(u)-J(trial)``
    from the original quadratic objective rather than from the line-step
    identity used to form ``realized_gain``.
    """
    if not isinstance(certificate, DiscreteRitzStepCertificate):
        return False
    try:
        fresh = certify_discrete_ritz_step(problem, u, phi, c_hat, q_hat, delta, epsilon)
        if fresh.status not in ("verified_discrete_lower_bound", "zero_certified_gain"):
            return False
        if fresh != certificate:
            return False
        if not isinstance(problem, DiscreteRitzProblem) or problem.feasible is not None:
            return False
        matrix = _as_f64_array("K", problem.K, 2)
        source = _as_f64_array("f", problem.source, 1)
        state = _as_f64_array("u", u, 1)
        if matrix.shape[0] > MAX_CERTIFICATE_DIMENSION:
            return False
        n = matrix.shape[0]
        Kq = tuple(tuple(_fraction(float(matrix[i, j])) for j in range(n)) for i in range(n))
        fq = tuple(_fraction(float(x)) for x in source)
        uq = tuple(_fraction(float(x)) for x in state)
        trial = certificate.trial_exact
        if trial is None or len(trial) != n:
            return False

        def exact_energy(values: tuple[Fraction, ...]) -> Fraction:
            quadratic = sum((values[i] * Kq[i][j] * values[j]
                             for i in range(n) for j in range(n)), Fraction(0))
            linear = sum((fq[i] * values[i] for i in range(n)), Fraction(0))
            return quadratic / 2 - linear

        direct_gain = exact_energy(uq) - exact_energy(trial)
        return (direct_gain == certificate.realized_gain
                and direct_gain >= certificate.lower_bound)
    except (TypeError, ValueError, IndexError, _InvalidInput):
        return False
