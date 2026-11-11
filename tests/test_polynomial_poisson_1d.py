from __future__ import annotations

from fractions import Fraction
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

import problems.polynomial_poisson_1d as polynomial_path
from problems import (
    DiscreteRitzStepCertificate,
    PolynomialPoisson1DError,
    PolynomialPoissonEnclosure1D,
    build_polynomial_poisson_1d_enclosure,
    certify_discrete_ritz_step,
    verify_polynomial_poisson_1d_enclosure,
)


ZERO = Fraction(0)
ONE = Fraction(1)
B1 = (Fraction(0), Fraction(1), Fraction(-1))
B2 = (Fraction(0), Fraction(0), Fraction(1), Fraction(-1))


def _poly_add(a, b):
    out = [ZERO] * max(len(a), len(b))
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _poly_scale(poly, scalar):
    out = [value * scalar for value in poly]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _poly_mul(a, b):
    out = [ZERO] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def _poly_derivative(poly):
    if len(poly) == 1:
        return (ZERO,)
    return tuple(Fraction(i) * poly[i] for i in range(1, len(poly)))


def _poly_integral(poly):
    return sum((coefficient / (power + 1)
                for power, coefficient in enumerate(poly)), ZERO)


def _linear_combination(basis, vector):
    result = (ZERO,)
    for poly, coefficient in zip(basis, vector, strict=True):
        result = _poly_add(result, _poly_scale(poly, coefficient))
    return result


def _poly_eval(poly, x):
    value = ZERO
    for coefficient in reversed(poly):
        value = value * x + coefficient
    return value


def _composite_simpson(poly, panels=16):
    if panels <= 0 or panels % 2:
        raise ValueError("Simpson panel count must be positive and even")
    total = ZERO
    for index in range(panels + 1):
        weight = 1 if index in (0, panels) else (4 if index % 2 else 2)
        total += weight * _poly_eval(poly, Fraction(index, panels))
    return total / (3 * panels)


def _fourth_derivative(poly):
    result = poly
    for _ in range(4):
        result = _poly_derivative(result)
    return result


def _simpson_error_bound(poly, panels=16):
    derivative = _fourth_derivative(poly)
    # For x in [0,1], |sum a_k x^k| <= sum |a_k|.
    sup_bound = sum((abs(value) for value in derivative), ZERO)
    return sup_bound / (180 * panels**4)


def _fraction_field(record, field):
    pair = json.loads(record.to_json())["derived"][field]
    return Fraction(pair[0], pair[1])


def _tamper_record(record, *, field=None, replacement=None, certificate=None):
    payload = json.loads(record.to_json())
    if field is not None:
        payload["derived"][field] = replacement
    if certificate is not None:
        payload["certificate"] = certificate.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return PolynomialPoissonEnclosure1D.from_json(encoded)


class PolynomialPoisson1DTests(unittest.TestCase):
    def _build(self, forcing, basis, u, phi):
        return build_polynomial_poisson_1d_enclosure(forcing, basis, u, phi)

    def _assert_independent_simpson(self, forcing, basis, u, phi, record, panels=16):
        forcing_poly = tuple(Fraction(value) for value in forcing)
        basis_polys = tuple(tuple(Fraction(value) for value in poly) for poly in basis)
        u_exact = tuple(Fraction.from_float(float(value)) for value in u)
        phi_exact = tuple(Fraction.from_float(float(value)) for value in phi)
        u_poly = _linear_combination(basis_polys, u_exact)
        phi_poly = _linear_combination(basis_polys, phi_exact)
        f_phi = _poly_mul(forcing_poly, phi_poly)
        du_dphi = _poly_mul(_poly_derivative(u_poly), _poly_derivative(phi_poly))
        dphi_squared = _poly_mul(_poly_derivative(phi_poly), _poly_derivative(phi_poly))
        exact_c = _poly_integral(f_phi) - _poly_integral(du_dphi)
        exact_q = _poly_integral(dphi_squared)
        self.assertEqual(_fraction_field(record, "c_cont"), exact_c)
        self.assertEqual(_fraction_field(record, "q_cont"), exact_q)
        for integrand, exact_value in (
            (f_phi, _poly_integral(f_phi)),
            (du_dphi, _poly_integral(du_dphi)),
            (dphi_squared, exact_q),
        ):
            self.assertLessEqual(
                abs(_composite_simpson(integrand, panels) - exact_value),
                _simpson_error_bound(integrand, panels),
            )

    def test_continuous_active_single_basis_replay_and_simpson_oracle(self):
        u = np.array([0.0], dtype=np.float64)
        phi = np.array([1.0], dtype=np.float64)
        problem, record = self._build((1,), (B1,), u, phi)
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
        self.assertEqual(PolynomialPoissonEnclosure1D.from_json(record.to_json()), record)
        payload = json.loads(record.to_json())
        derived = payload["derived"]
        self.assertEqual(Fraction(*derived["stiffness_exact"][0][0]), Fraction(1, 3))
        self.assertEqual(Fraction(*derived["load_exact"][0]), Fraction(1, 6))
        self.assertEqual(_fraction_field(record, "c_cont"), Fraction(1, 6))
        self.assertEqual(_fraction_field(record, "q_cont"), Fraction(1, 3))
        self._assert_independent_simpson((1,), (B1,), u, phi, record)

        chat = Fraction.from_float(float.fromhex(derived["c_hat_hex"]))
        qhat = Fraction.from_float(float.fromhex(derived["q_hat_hex"]))
        c_cont = _fraction_field(record, "c_cont")
        q_cont = _fraction_field(record, "q_cont")
        c_dyad = _fraction_field(record, "c_dyad")
        q_dyad = _fraction_field(record, "q_dyad")
        self.assertGreater(abs(c_cont - chat), abs(c_dyad - chat))
        self.assertGreater(abs(q_cont - qhat), abs(q_dyad - qhat))
        for exact_radius, encoded in (
            (max(abs(c_cont - chat), abs(c_dyad - chat)), derived["delta_hex"]),
            (max(abs(q_cont - qhat), abs(q_dyad - qhat)), derived["epsilon_hex"]),
        ):
            radius = float.fromhex(encoded)
            self.assertGreaterEqual(Fraction.from_float(radius), exact_radius)
            self.assertLess(Fraction.from_float(math.nextafter(radius, 0.0)), exact_radius)

        phi_poly = B1
        c_integrand = _poly_mul((Fraction(1),), phi_poly)
        q_integrand = _poly_mul(_poly_derivative(phi_poly), _poly_derivative(phi_poly))
        self.assertLessEqual(abs(_composite_simpson(c_integrand) - c_cont),
                             _simpson_error_bound(c_integrand))
        self.assertLessEqual(abs(_composite_simpson(q_integrand) - q_cont),
                             _simpson_error_bound(q_integrand))
        self.assertIsNone(problem.work_ledger)

        # The discrete-only certificate accepts these one-ULP-short radii,
        # because its dyadic targets coincide with their centers. Full replay
        # rejects them because they omit the continuous analytic targets.
        cert = DiscreteRitzStepCertificate.from_json(
            json.dumps(payload["certificate"], sort_keys=True, separators=(",", ":")))
        for field, target, center_field, radius_field in (
            ("c_cont", c_cont, "c_hat_hex", "delta_hex"),
            ("q_cont", q_cont, "q_hat_hex", "epsilon_hex"),
        ):
            center = Fraction.from_float(float.fromhex(derived[center_field]))
            full_radius = float.fromhex(derived[radius_field])
            short_radius = float(np.nextafter(full_radius, 0.0))
            self.assertLess(Fraction.from_float(short_radius), abs(target - center))
            short_delta = short_radius if radius_field == "delta_hex" else float.fromhex(derived["delta_hex"])
            short_epsilon = short_radius if radius_field == "epsilon_hex" else float.fromhex(derived["epsilon_hex"])
            discrete_only = certify_discrete_ritz_step(
                problem, u, phi, float.fromhex(derived["c_hat_hex"]),
                float.fromhex(derived["q_hat_hex"]), short_delta, short_epsilon,
            )
            self.assertIn(discrete_only.status,
                          ("verified_discrete_lower_bound", "zero_certified_gain"))
            forged = _tamper_record(record, field=radius_field, replacement=short_radius.hex(),
                                    certificate=discrete_only)
            self.assertFalse(verify_polynomial_poisson_1d_enclosure(forged))

    def test_discrete_active_c_and_q_have_independent_one_ulp_controls(self):
        forcing_c = (Fraction(6, 2**50), Fraction(3, 2))
        u_c = np.array([0.25, 0.25], dtype=np.float64)
        phi_c = np.array([1.0, 0.0], dtype=np.float64)
        problem_c, record_c = self._build(forcing_c, (B1, B2), u_c, phi_c)
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(record_c))
        self._assert_independent_simpson(forcing_c, (B1, B2), u_c, phi_c, record_c)
        c_cont = _fraction_field(record_c, "c_cont")
        c_dyad = _fraction_field(record_c, "c_dyad")
        self.assertEqual(c_cont, Fraction(1, 2**50))
        self.assertEqual(c_dyad - c_cont, Fraction(1, 2**57))
        data_c = json.loads(record_c.to_json())["derived"]
        c_center = Fraction.from_float(float.fromhex(data_c["c_hat_hex"]))
        self.assertGreater(abs(c_dyad - c_center), abs(c_cont - c_center))
        dyadic_c_radius = abs(c_dyad - c_center)
        too_small_c = float(np.nextafter(float(dyadic_c_radius), 0.0))
        rejected_c = certify_discrete_ritz_step(
            problem_c, u_c, phi_c, float.fromhex(data_c["c_hat_hex"]),
            float.fromhex(data_c["q_hat_hex"]), too_small_c,
            float.fromhex(data_c["epsilon_hex"]),
        )
        self.assertEqual(rejected_c.reason, "c_enclosure_failed")

        forcing_q = (Fraction(1),)
        u_q = np.zeros(2, dtype=np.float64)
        phi_q = np.array([1.0, 1.0], dtype=np.float64)
        problem_q, record_q = self._build(forcing_q, (B1, B2), u_q, phi_q)
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(record_q))
        self._assert_independent_simpson(forcing_q, (B1, B2), u_q, phi_q, record_q)
        q_cont = _fraction_field(record_q, "q_cont")
        q_dyad = _fraction_field(record_q, "q_dyad")
        self.assertEqual(q_cont, Fraction(4, 5))
        data_q = json.loads(record_q.to_json())["derived"]
        q_center = Fraction.from_float(float.fromhex(data_q["q_hat_hex"]))
        self.assertGreater(abs(q_dyad - q_center), abs(q_cont - q_center))
        too_small_q = float(np.nextafter(float(abs(q_dyad - q_center)), 0.0))
        rejected_q = certify_discrete_ritz_step(
            problem_q, u_q, phi_q, float.fromhex(data_q["c_hat_hex"]),
            float.fromhex(data_q["q_hat_hex"]), float.fromhex(data_q["delta_hex"]),
            too_small_q,
        )
        self.assertEqual(rejected_q.reason, "q_enclosure_failed")

    def test_float32_promotion_preserves_values_and_signed_zero(self):
        _, record = self._build(
            (1,), (B1,), np.array([-0.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        payload = json.loads(record.to_json())
        self.assertEqual(payload["inputs"]["u"]["input_dtype"], "float32")
        self.assertEqual(payload["inputs"]["u"]["float64_hex"], ["-0x0.0p+0"])
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
        with self.assertRaisesRegex(PolynomialPoisson1DError, "float32_or_float64"):
            self._build((1,), (B1,), np.array([0.0], dtype=np.float16), np.array([1.0]))
        with self.assertRaisesRegex(PolynomialPoisson1DError, "float32_or_float64"):
            self._build((1,), (B1,), np.array([0], dtype=np.int64), np.array([1.0]))

    def test_nearest_even_rounding_normal_subnormal_signed_zero_and_overflow(self):
        rounder = polynomial_path._round_fraction_binary64
        vectors = (
            (Fraction(1, 3), 0x3FD5555555555555),
            (Fraction(1, 2**1022), 0x0010000000000000),
            (Fraction(1, 2**1022) - Fraction(1, 2**1074), 0x000FFFFFFFFFFFFF),
            (Fraction(1, 2**1074), 0x0000000000000001),
            (Fraction(1, 2**1075), 0x0000000000000000),
            (Fraction(3, 2**1075), 0x0000000000000002),
            (Fraction(1) + Fraction(1, 2**53), 0x3FF0000000000000),
            (Fraction(1) + Fraction(3, 2**53), 0x3FF0000000000002),
            (Fraction(1, 2**1022) - Fraction(1, 2**1075), 0x0010000000000000),
            (-Fraction(1, 2**1075), 0x8000000000000000),
            (-Fraction(1, 2**1076), 0x8000000000000000),
        )
        for rational, expected_bits in vectors:
            with self.subTest(rational=rational):
                self.assertEqual(polynomial_path._bits_from_float(rounder(rational)), expected_bits)
        max_finite = Fraction.from_float(float.fromhex("0x1.fffffffffffffp+1023"))
        self.assertEqual(polynomial_path._bits_from_float(rounder(max_finite)),
                         0x7FEFFFFFFFFFFFFF)
        with self.assertRaises(OverflowError):
            rounder(max_finite + Fraction(2**970))
        self.assertEqual(polynomial_path._bits_from_float(rounder(Fraction(0))), 0)
        self.assertEqual(polynomial_path._bits_from_float(
            polynomial_path._ceil_fraction_binary64(Fraction(1, 2**1075))), 1)
        with self.assertRaises(OverflowError):
            polynomial_path._ceil_fraction_binary64(max_finite + 1)
        with self.assertRaisesRegex(PolynomialPoisson1DError, "negative_exact_radius"):
            polynomial_path._ceil_fraction_binary64(Fraction(-1, 3))

    def test_invalid_boundary_dependent_basis_zero_direction_and_tampering(self):
        with self.assertRaisesRegex(PolynomialPoisson1DError, "homogeneous_dirichlet"):
            self._build((1,), ((Fraction(0), Fraction(1)),),
                        np.array([0.0]), np.array([1.0]))
        with self.assertRaisesRegex(PolynomialPoisson1DError, "linearly_independent"):
            self._build((1,), (B1, _poly_scale(B1, Fraction(2))),
                        np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        with self.assertRaisesRegex(PolynomialPoisson1DError, "zero_or_nonpositive_continuous_direction"):
            self._build((1,), (B1,), np.array([0.0]), np.array([0.0]))

        _, record = self._build((1,), (B1,), np.array([0.0]), np.array([1.0]))
        payload = json.loads(record.to_json())
        payload["derived"]["c_cont"] = [0, 1]
        forged = PolynomialPoissonEnclosure1D.from_json(
            json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self.assertFalse(verify_polynomial_poisson_1d_enclosure(forged))
        negative_radius = _tamper_record(
            record, field="delta_hex", replacement=(-0.1).hex())
        self.assertFalse(verify_polynomial_poisson_1d_enclosure(negative_radius))
        with self.assertRaisesRegex(ValueError, "canonical"):
            PolynomialPoissonEnclosure1D.from_json(record.to_json() + " ")
        with self.assertRaises(ValueError):
            PolynomialPoissonEnclosure1D.from_json('{"schema":"x","schema":"y"}')
        deeply_nested = "[" * 1500 + "0" + "]" * 1500
        with self.assertRaisesRegex(ValueError, "nesting depth limit"):
            PolynomialPoissonEnclosure1D.from_json(deeply_nested)
        self.assertFalse(verify_polynomial_poisson_1d_enclosure(
            PolynomialPoissonEnclosure1D(deeply_nested)))
        with self.assertRaisesRegex(ValueError, "valid Unicode text"):
            PolynomialPoissonEnclosure1D.from_json("\ud800")
        self.assertFalse(verify_polynomial_poisson_1d_enclosure(
            PolynomialPoissonEnclosure1D(None)))
        self.assertFalse(verify_polynomial_poisson_1d_enclosure(
            PolynomialPoissonEnclosure1D(123)))
        with patch.object(polynomial_path, "MAX_RECORD_JSON_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "size limit"):
                PolynomialPoissonEnclosure1D.from_json(" " * 11)
            with self.assertRaisesRegex(PolynomialPoisson1DError, "record_size_limit"):
                self._build((1,), (B1,), np.array([0.0]), np.array([1.0]))

    def test_input_dimension_degree_and_coefficient_caps_fail_closed(self):
        with patch.object(polynomial_path, "MAX_POLYNOMIAL_DIMENSION", 1):
            _, record = self._build((1,), (B1,), np.array([0.0]), np.array([1.0]))
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
            with self.assertRaisesRegex(PolynomialPoisson1DError, "dimension_limit"):
                self._build((1,), (B1, B2), np.zeros(2), np.array([1.0, 0.0]))

        with patch.object(polynomial_path, "MAX_POLYNOMIAL_DEGREE", 2):
            _, record = self._build((1,), (B1,), np.array([0.0]), np.array([1.0]))
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
            with self.assertRaisesRegex(PolynomialPoisson1DError, "degree_limit"):
                self._build((1,), (B2,), np.array([0.0]), np.array([1.0]))

        with patch.object(polynomial_path, "MAX_INPUT_COEFFICIENT_BITS", 4):
            _, record = self._build((8,), (B1,), np.array([0.0]), np.array([1.0]))
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
            with patch.object(polynomial_path, "_poly_multiply") as multiply:
                with self.assertRaisesRegex(PolynomialPoisson1DError, "coefficient_bit_limit"):
                    self._build((16,), (B1,), np.array([0.0]), np.array([1.0]))
                multiply.assert_not_called()

    def test_intermediate_bits_and_operation_caps_fail_closed(self):
        _, baseline_record = self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(baseline_record))
        with patch.object(polynomial_path, "MAX_INTERMEDIATE_FRACTION_BITS", 55):
            _, record = self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
        with patch.object(polynomial_path, "MAX_INTERMEDIATE_FRACTION_BITS", 54):
            with self.assertRaisesRegex(PolynomialPoisson1DError, "intermediate_fraction_bit_limit"):
                self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))

        predicted = polynomial_path._preflight_operation_bound(1, 0, 2)
        with patch.object(polynomial_path, "MAX_EXACT_RATIONAL_OPERATIONS", predicted):
            _, record = self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
        with patch.object(polynomial_path, "MAX_EXACT_RATIONAL_OPERATIONS", predicted - 1):
            with self.assertRaisesRegex(PolynomialPoisson1DError, "operation_preflight_exceeded"):
                self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))
        with (patch.object(polynomial_path, "MAX_EXACT_RATIONAL_OPERATIONS", 1),
              patch.object(polynomial_path, "_preflight_operation_bound", return_value=1)):
            with self.assertRaisesRegex(PolynomialPoisson1DError, "operation_limit_exceeded"):
                self._build((0,), (B1,), np.array([0.0]), np.array([1.0]))

    def test_nonzero_state_multiple_directions_and_read_only_replay(self):
        forcing = (Fraction(1),)
        basis = (B1, B2)
        u = np.array([0.25, 0.125], dtype=np.float64)
        input_before = (u.tobytes(),)
        for phi in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, -0.5])):
            phi_before = phi.tobytes()
            problem, record = self._build(forcing, basis, u, phi)
            self._assert_independent_simpson(forcing, basis, u, phi, record)
            k_before, f_before = problem.K.tobytes(), problem.source.tobytes()
            self.assertFalse(problem.K.flags.writeable)
            self.assertFalse(problem.source.flags.writeable)
            self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
            self.assertEqual((u.tobytes(),), input_before)
            self.assertEqual(phi.tobytes(), phi_before)
            self.assertEqual((problem.K.tobytes(), problem.source.tobytes()),
                             (k_before, f_before))

    def test_nonzero_negative_exact_center_underflows_to_recorded_negative_zero(self):
        min_subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
        _, record = self._build(
            (0,), (B1,), np.array([min_subnormal]), np.array([1.0]))
        payload = json.loads(record.to_json())
        self.assertEqual(payload["derived"]["c_hat_hex"], "-0x0.0p+0")
        self.assertIn("c_hat", payload["derived"]["underflow_paths"])
        self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))

    def test_independent_energy_gains_match_record_for_all_active_side_fixtures(self):
        fixtures = (
            ((1,), (B1,), np.array([0.0]), np.array([1.0])),
            ((Fraction(6, 2**50), Fraction(3, 2)), (B1, B2),
             np.array([0.25, 0.25]), np.array([1.0, 0.0])),
            ((1,), (B1, B2), np.zeros(2), np.array([1.0, 1.0])),
        )
        for forcing, basis, u, phi in fixtures:
            with self.subTest(dimension=len(basis)):
                problem, record = self._build(forcing, basis, u, phi)
                self.assertTrue(verify_polynomial_poisson_1d_enclosure(record))
                payload = json.loads(record.to_json())
                basis_polys = tuple(tuple(Fraction(value) for value in poly)
                                    for poly in basis)
                forcing_poly = tuple(Fraction(value) for value in forcing)
                derivatives = tuple(_poly_derivative(poly) for poly in basis_polys)
                K_cont = tuple(tuple(_poly_integral(_poly_mul(derivatives[i], derivatives[j]))
                                     for j in range(len(basis_polys)))
                               for i in range(len(basis_polys)))
                F_cont = tuple(_poly_integral(_poly_mul(forcing_poly, poly))
                               for poly in basis_polys)
                u_exact = tuple(Fraction.from_float(float(value)) for value in u)
                phi_exact = tuple(Fraction.from_float(float(value)) for value in phi)
                alpha_pair = payload["certificate"]["alpha"]
                alpha = Fraction(alpha_pair[0], alpha_pair[1])
                trial = tuple(u_exact[i] + alpha * phi_exact[i]
                              for i in range(len(u_exact)))

                def exact_energy(matrix, load, values):
                    quadratic = sum((values[i] * matrix[i][j] * values[j]
                                     for i in range(len(values))
                                     for j in range(len(values))), ZERO)
                    linear = sum((load[i] * values[i]
                                  for i in range(len(values))), ZERO)
                    return quadratic / 2 - linear

                K_discrete = tuple(tuple(Fraction.from_float(float(value)) for value in row)
                                   for row in problem.K)
                F_discrete = tuple(Fraction.from_float(float(value))
                                   for value in problem.source)
                gain_cont = (exact_energy(K_cont, F_cont, u_exact)
                             - exact_energy(K_cont, F_cont, trial))
                gain_discrete = (exact_energy(K_discrete, F_discrete, u_exact)
                                 - exact_energy(K_discrete, F_discrete, trial))
                lower = Fraction(*payload["certificate"]["lower_bound"])
                self.assertEqual(gain_cont, _fraction_field(record, "continuous_gain"))
                self.assertEqual(gain_discrete, _fraction_field(record, "discrete_gain"))
                self.assertGreaterEqual(gain_cont, lower)
                self.assertGreaterEqual(gain_discrete, lower)


if __name__ == "__main__":
    unittest.main()
