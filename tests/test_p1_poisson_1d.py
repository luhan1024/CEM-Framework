from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np

from estimators_1d import EquilibratedFlux1DEstimator, P1Poisson1DData
from problems import (
    P1PoissonStepCertificate,
    certify_p1_poisson_step,
    verify_p1_poisson_step_certificate,
)
from problems.p1_poisson_1d import (
    MAX_P1_CELLS,
    MAX_FRACTION_COMPONENT_BITS,
    _fraction_from_pair,
    _fraction_pair,
)


def F(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def independent_oracle(data: P1Poisson1DData, u_values: np.ndarray,
                       phi_values: np.ndarray):
    """Independent test oracle: local exact form and direct objective difference."""
    x = [F(value) for value in data.nodes]
    f = [F(value) for value in data.cell_source]
    u = [F(value) for value in u_values]
    phi = [F(value) for value in phi_values]
    c = Fraction(0)
    q = Fraction(0)
    for i in range(len(f)):
        h = x[i + 1] - x[i]
        c += f[i] * h * (phi[i] + phi[i + 1]) / 2
        c -= (u[i + 1] - u[i]) * (phi[i + 1] - phi[i]) / h
        q += (phi[i + 1] - phi[i]) ** 2 / h
    if q <= 0:
        return c, q, None, None, None
    alpha = c / q
    trial = [u[i] + alpha * phi[i] for i in range(len(u))]

    def energy(values):
        quadratic = sum(
            ((values[i + 1] - values[i]) ** 2 / (x[i + 1] - x[i]) for i in range(len(f))),
            Fraction(0),
        )
        load = sum(
            (f[i] * (x[i + 1] - x[i]) * (values[i] + values[i + 1]) / 2
             for i in range(len(f))),
            Fraction(0),
        )
        return quadratic / 2 - load

    direct_gain = energy(u) - energy(trial)
    return c, q, alpha, c * c / (2 * q), direct_gain


def data_for(nodes, source):
    return P1Poisson1DData(np.asarray(nodes, dtype=np.float64),
                           np.asarray(source, dtype=np.float64))


class P1PoissonStepCertificateTests(unittest.TestCase):
    def test_nonuniform_mesh_matches_independent_fraction_and_energy_oracle(self):
        data = data_for([0.0, 0.25, 0.625, 1.0], [1.0, -0.5, 2.0])
        u = np.asarray([0.0, 0.25, -0.125, 0.0], dtype=np.float64)
        phi = np.asarray([0.0, 0.5, -0.25, 0.0], dtype=np.float64)
        oracle = independent_oracle(data, u, phi)
        cert = certify_p1_poisson_step(data, u, phi)
        self.assertEqual(cert.status, "verified_positive_gain")
        self.assertEqual((cert.c_exact, cert.q_exact, cert.alpha, cert.gain_exact), oracle[:4])
        self.assertEqual(oracle[3], oracle[4])
        self.assertEqual(cert.exact_rational_operations, 36 * 3 + 11)
        self.assertTrue(verify_p1_poisson_step_certificate(cert, data, u, phi))

    def test_negative_c_uses_signed_negative_step_and_still_decreases_energy(self):
        data = data_for([0.0, 0.25, 0.625, 1.0], [1.0, -0.5, 2.0])
        u = np.asarray([0.0, 0.25, -0.125, 0.0], dtype=np.float64)
        positive_phi = np.asarray([0.0, 0.5, -0.25, 0.0], dtype=np.float64)
        negative = certify_p1_poisson_step(data, u, positive_phi)
        positive = certify_p1_poisson_step(data, u, -positive_phi)
        self.assertLess(negative.c_exact, 0)
        self.assertGreater(positive.c_exact, 0)
        self.assertLess(negative.alpha, 0)
        self.assertGreater(negative.gain_exact, 0)
        self.assertEqual(positive.gain_exact, negative.gain_exact)
        self.assertEqual(negative.gain_exact, independent_oracle(data, u, positive_phi)[4])

    def test_zero_c_has_verified_zero_gain(self):
        data = data_for([0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0, 0.0])
        u = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        phi = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cert = certify_p1_poisson_step(data, u, phi)
        self.assertEqual(cert.status, "verified_zero_gain")
        self.assertEqual(cert.c_exact, 0)
        self.assertGreater(cert.q_exact, 0)
        self.assertEqual(cert.alpha, 0)
        self.assertEqual(cert.gain_exact, 0)
        self.assertEqual(independent_oracle(data, u, phi)[4], 0)

    def test_zero_direction_and_invalid_domain_fail_closed_without_values(self):
        data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        zeros = np.zeros(3, dtype=np.float64)
        zero_direction = certify_p1_poisson_step(data, zeros, zeros)
        self.assertEqual(zero_direction.status, "invalid_input")
        self.assertEqual(zero_direction.reason, "zero_direction_or_nonpositive_curvature")
        self.assertIsNone(zero_direction.gain_exact)
        bad_state = np.asarray([0.0, 0.25, 1.0], dtype=np.float64)
        invalid = certify_p1_poisson_step(data, bad_state, np.asarray([0.0, 1.0, 0.0]))
        self.assertEqual(invalid.status, "invalid_input")
        self.assertIsNone(invalid.input_sha256)
        self.assertIsNone(invalid.c_exact)

    def test_shape_dtype_nonfinite_and_boundary_rejections(self):
        data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        zero = np.zeros(3, dtype=np.float64)
        self.assertEqual(certify_p1_poisson_step(data, zero, [0.0, 1.0, 0.0]).status,
                         "invalid_input")
        self.assertEqual(certify_p1_poisson_step(data, zero, np.asarray([0, 1, 0], dtype=np.int64)).status,
                         "invalid_input")
        self.assertEqual(certify_p1_poisson_step(
            data, zero, np.asarray([0.0, np.nan, 0.0], dtype=np.float64)).status,
            "invalid_input")
        self.assertEqual(certify_p1_poisson_step(
            data, zero, np.asarray([1.0, 1.0, 0.0], dtype=np.float64)).status,
            "invalid_input")

    def test_binary64_subnormal_input_is_handled_as_exact_dyadic(self):
        tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
        data = data_for([0.0, tiny, 1.0], [1.0, 0.0])
        u = np.zeros(3, dtype=np.float64)
        phi = np.asarray([0.0, tiny, 0.0], dtype=np.float64)
        cert = certify_p1_poisson_step(data, u, phi)
        oracle = independent_oracle(data, u, phi)
        self.assertEqual(cert.status, "verified_positive_gain")
        self.assertEqual((cert.c_exact, cert.q_exact, cert.alpha, cert.gain_exact), oracle[:4])
        self.assertEqual(oracle[3], oracle[4])

    def test_largest_finite_binary64_input_matches_independent_fraction_oracle(self):
        largest = np.finfo(np.float64).max
        data = data_for([0.0, 0.5, 1.0], [largest, largest])
        u = np.zeros(3, dtype=np.float64)
        phi = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        oracle = independent_oracle(data, u, phi)
        cert = certify_p1_poisson_step(data, u, phi)

        self.assertEqual(cert.status, "verified_positive_gain")
        self.assertEqual((cert.c_exact, cert.q_exact, cert.alpha, cert.gain_exact),
                         oracle[:4])
        self.assertEqual(oracle[3], oracle[4])
        self.assertGreater(cert.c_exact, 0)
        self.assertGreater(cert.alpha, 0)
        self.assertGreater(cert.gain_exact, 0)
        self.assertTrue(verify_p1_poisson_step_certificate(cert, data, u, phi))

    def test_signed_zero_endpoint_has_exact_zero_semantics_and_raw_bit_identity(self):
        positive_zero_data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        negative_zero_data = data_for([-0.0, 0.5, 1.0], [1.0, 1.0])
        u = np.zeros(3, dtype=np.float64)
        phi = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        positive_zero = certify_p1_poisson_step(positive_zero_data, u, phi)
        negative_zero = certify_p1_poisson_step(negative_zero_data, u, phi)

        self.assertEqual(positive_zero.status, "verified_positive_gain")
        self.assertEqual(negative_zero.status, "verified_positive_gain")
        self.assertEqual(
            (positive_zero.c_exact, positive_zero.q_exact,
             positive_zero.alpha, positive_zero.gain_exact),
            (negative_zero.c_exact, negative_zero.q_exact,
             negative_zero.alpha, negative_zero.gain_exact),
        )
        self.assertNotEqual(positive_zero.input_sha256, negative_zero.input_sha256)
        self.assertIn('"nodes_hex":["8000000000000000"',
                      negative_zero.input_payload_json)
        self.assertTrue(verify_p1_poisson_step_certificate(
            negative_zero, negative_zero_data, u, phi))
        self.assertTrue(verify_p1_poisson_step_certificate(
            negative_zero.to_json(), negative_zero_data, u, phi))

    def test_float64_equilibrated_flux_crosscheck_on_analytic_p1_error(self):
        # For f=1 on (0,1), u_exact=x(1-x)/2. Its nodal P1 interpolant on
        # {0,1/2,1} has exact squared H1-seminorm error 1/48.
        data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        p1_values = np.asarray([0.0, 0.125, 0.0], dtype=np.float64)
        estimator_record = EquilibratedFlux1DEstimator().estimate(data, p1_values)
        self.assertEqual(estimator_record.status, "numerical_bound")
        self.assertAlmostEqual(estimator_record.eta_eq**2, 1.0 / 48.0, places=14)
        direction = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        cert = certify_p1_poisson_step(data, p1_values, direction)
        oracle = independent_oracle(data, p1_values, direction)
        self.assertEqual((cert.c_exact, cert.q_exact, cert.gain_exact),
                         (oracle[0], oracle[1], oracle[3]))
        self.assertEqual(cert.status, "verified_zero_gain")

    def test_record_roundtrip_tampering_and_fresh_input_binding(self):
        data = data_for([0.0, 0.5, 1.0], [1.0, 2.0])
        u = np.asarray([0.0, 0.125, 0.0], dtype=np.float64)
        phi = np.asarray([0.0, 0.5, 0.0], dtype=np.float64)
        cert = certify_p1_poisson_step(data, u, phi)
        encoded = cert.to_json()
        decoded = P1PoissonStepCertificate.from_json(encoded)
        self.assertEqual(decoded, cert)
        self.assertTrue(verify_p1_poisson_step_certificate(encoded, data, u, phi))
        self.assertFalse(verify_p1_poisson_step_certificate(cert, data, u + np.asarray([0, 0.01, 0]), phi))
        with self.assertRaisesRegex(ValueError, "certificate_input_hash_mismatch"):
            replace(cert, input_sha256="0" * 64)
        forged = object.__new__(P1PoissonStepCertificate)
        for field in cert.__dataclass_fields__:
            object.__setattr__(forged, field,
                               "0" * 64 if field == "input_sha256" else getattr(cert, field))
        self.assertFalse(verify_p1_poisson_step_certificate(forged, data, u, phi))
        forged_gain = replace(cert, gain_exact=cert.gain_exact + 1)
        self.assertFalse(verify_p1_poisson_step_certificate(forged_gain, data, u, phi))
        self.assertFalse(verify_p1_poisson_step_certificate(
            encoded.replace('"gain_exact":', '"gain_exact":["0","1"],"gain_exact":', 1),
            data, u, phi))

    def test_parser_rejects_duplicate_keys_noncanonical_numbers_depth_and_size(self):
        duplicate = '{"schema":"x","schema":"y"}'
        with self.assertRaises(ValueError):
            P1PoissonStepCertificate.from_json(duplicate)
        with self.assertRaises(ValueError):
            P1PoissonStepCertificate.from_json('{"a":1}')
        nested = "[" * 65 + "0" + "]" * 65
        with self.assertRaises(ValueError):
            P1PoissonStepCertificate.from_json(nested)
        with self.assertRaises(ValueError):
            P1PoissonStepCertificate.from_json(" " * 4_000_001)

    def test_parser_and_builder_resource_exceptions_fail_closed(self):
        data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        u = np.zeros(3, dtype=np.float64)
        phi = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        with patch("problems.p1_poisson_1d.json.loads", side_effect=MemoryError):
            with self.assertRaisesRegex(ValueError, "record_parser_resource_failure"):
                P1PoissonStepCertificate.from_json("{}")
            self.assertFalse(verify_p1_poisson_step_certificate("{}", data, u, phi))
        with patch("problems.p1_poisson_1d._exact_binary64", side_effect=MemoryError):
            failed = certify_p1_poisson_step(data, u, phi)
        self.assertEqual(failed.status, "unsupported_resource_limit")
        self.assertEqual(failed.reason, "memory_resource_limit_exceeded")
        self.assertIsNone(failed.c_exact)

    def test_canonical_fraction_encoding_supports_full_bit_cap(self):
        value = Fraction(1, 1 << (MAX_FRACTION_COMPONENT_BITS - 1))
        encoded = _fraction_pair(value)
        self.assertEqual(_fraction_from_pair(encoded), value)
        with self.assertRaises(ValueError):
            _fraction_from_pair(["2", "4"])

    def test_unhashable_status_is_a_defined_parser_error_and_verification_failure(self):
        with self.assertRaisesRegex(ValueError, "invalid_certificate_record_status"):
            bad_status_json = (
                '{"algorithm":"exact-dyadic-piecewise-linear-cell-gain.v1",'
                '"alpha":null,"c_exact":null,"gain_exact":null,'
                '"exact_rational_operations":null,"input":null,'
                '"input_sha256":null,"q_exact":null,"reason":"bad",'
                '"schema":"cem.p1_poisson_1d_step_certificate.v1",'
                '"status":[]}')
            P1PoissonStepCertificate.from_json(bad_status_json)
        with self.assertRaises(ValueError):
            P1PoissonStepCertificate(status=[], reason="bad")
        data = data_for([0.0, 0.5, 1.0], [1.0, 1.0])
        cert = certify_p1_poisson_step(
            data, np.zeros(3, dtype=np.float64), np.asarray([0.0, 1.0, 0.0]))
        forged = object.__new__(P1PoissonStepCertificate)
        object.__setattr__(forged, "status", [])
        self.assertFalse(verify_p1_poisson_step_certificate(
            forged, data, np.zeros(3, dtype=np.float64), np.asarray([0.0, 1.0, 0.0])))
        self.assertFalse(verify_p1_poisson_step_certificate(
            bad_status_json, data, np.zeros(3, dtype=np.float64), np.asarray([0.0, 1.0, 0.0])))
        self.assertEqual(cert.status, "verified_positive_gain")

    def test_certificate_uses_snapshots_and_never_mutates_callers(self):
        data = data_for([0.0, 0.25, 1.0], [1.0, -2.0])
        data.nodes.setflags(write=True)
        data.cell_source.setflags(write=True)
        u = np.asarray([0.0, 0.25, 0.0], dtype=np.float64)
        phi = np.asarray([0.0, 0.5, 0.0], dtype=np.float64)
        before = (data.nodes.copy(), data.cell_source.copy(), u.copy(), phi.copy())
        cert = certify_p1_poisson_step(data, u, phi)
        self.assertEqual(cert.status, "verified_positive_gain")
        np.testing.assert_array_equal(data.nodes, before[0])
        np.testing.assert_array_equal(data.cell_source, before[1])
        np.testing.assert_array_equal(u, before[2])
        np.testing.assert_array_equal(phi, before[3])

    def test_mesh_cell_cap_and_full_supported_mesh_fail_closed_or_pass(self):
        too_many_cells = MAX_P1_CELLS + 1
        nodes = np.linspace(0.0, 1.0, too_many_cells + 1, dtype=np.float64)
        too_large = data_for(nodes, np.zeros(too_many_cells, dtype=np.float64))
        zeros = np.zeros(too_many_cells + 1, dtype=np.float64)
        phi = zeros.copy()
        phi[1:-1] = 1.0
        rejected = certify_p1_poisson_step(too_large, zeros, phi)
        self.assertEqual(rejected.status, "unsupported_resource_limit")
        self.assertIsNone(rejected.gain_exact)

        supported_nodes = np.linspace(0.0, 1.0, MAX_P1_CELLS + 1, dtype=np.float64)
        supported = data_for(supported_nodes, np.zeros(MAX_P1_CELLS, dtype=np.float64))
        supported_u = np.zeros(MAX_P1_CELLS + 1, dtype=np.float64)
        supported_phi = np.zeros(MAX_P1_CELLS + 1, dtype=np.float64)
        supported_phi[1:-1] = 1.0
        max_mesh = certify_p1_poisson_step(supported, supported_u, supported_phi)
        self.assertEqual(max_mesh.status, "verified_zero_gain")
        self.assertLessEqual(max_mesh.exact_rational_operations,
                             48 * MAX_P1_CELLS + 32)
        self.assertEqual(max_mesh.exact_rational_operations,
                         36 * MAX_P1_CELLS + 11)
        self.assertLessEqual(max_mesh.exact_rational_operations,
                             200_000)


if __name__ == "__main__":
    unittest.main()
