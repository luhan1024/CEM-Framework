from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from problems import (
    DiscreteRitzProblem,
    DiscreteRitzStepCertificate,
    certify_discrete_ritz_step,
    verify_discrete_ritz_certificate,
)


class DiscreteRitzCertificateTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
        self.f = np.array([1.0, 0.2], dtype=np.float64)
        self.u = np.zeros(2, dtype=np.float64)
        self.phi = np.array([1.0, 0.0], dtype=np.float64)
        self.problem = DiscreteRitzProblem(self.K, self.f)

    def certify(self, **overrides):
        args = dict(K=self.K, f=self.f, u=self.u, phi=self.phi,
                    c_hat=0.875, q_hat=2.0, delta=0.125, epsilon=0.0)
        args.update(overrides)
        problem = args.pop("problem", None)
        if problem is None:
            matrix = args.pop("K")
            source = args.pop("f")
            problem = DiscreteRitzProblem(self.K, self.f)
            problem.K = matrix
            problem.source = source
        else:
            args.pop("K", None)
            args.pop("f", None)
        return certify_discrete_ritz_step(problem=problem, **args)

    def test_exact_enclosures_alpha_and_gain_lower_bound(self):
        cert = self.certify()
        self.assertEqual(cert.status, "verified_discrete_lower_bound")
        self.assertEqual(cert.c_exact, Fraction(1))
        self.assertEqual(cert.q_exact, Fraction(2))
        self.assertEqual(cert.alpha, Fraction(3, 8))
        self.assertEqual(cert.lower_bound, Fraction(9, 64))
        self.assertGreaterEqual(cert.realized_gain, cert.lower_bound)
        # Independent oracle: evaluate J at the original state and exact trial.
        Kq = [[Fraction.from_float(float(x)) for x in row] for row in self.K]
        fq = [Fraction.from_float(float(x)) for x in self.f]
        uq = [Fraction.from_float(float(x)) for x in self.u]
        tq = cert.trial_exact
        energy = lambda x: (
            sum((x[i] * Kq[i][j] * x[j] for i in range(2) for j in range(2)), Fraction(0)) / 2
            - sum((fq[i] * x[i] for i in range(2)), Fraction(0))
        )
        direct_gain = energy(tuple(uq)) - energy(tq)
        self.assertEqual(direct_gain, cert.realized_gain)
        self.assertGreaterEqual(direct_gain, cert.lower_bound)
        self.assertTrue(verify_discrete_ritz_certificate(
            cert, self.problem, self.u, self.phi, 0.875, 2.0, 0.125, 0.0))
        self.assertEqual(cert.feasibility_basis, "unconstrained_domain")

    def test_enclosure_endpoints_are_inclusive_and_wrong_bounds_reject(self):
        boundary = self.certify(c_hat=0.75, delta=0.25)
        self.assertEqual(boundary.status, "verified_discrete_lower_bound")
        self.assertEqual(boundary.reason, "")
        wrong_c = self.certify(c_hat=0.5, delta=0.1)
        wrong_q = self.certify(q_hat=2.5, epsilon=0.4)
        self.assertEqual(wrong_c.reason, "c_enclosure_failed")
        self.assertEqual(wrong_q.reason, "q_enclosure_failed")

    def test_zero_certified_gain(self):
        result = certify_discrete_ritz_step(
            DiscreteRitzProblem(np.eye(1), np.zeros(1)), np.zeros(1), np.ones(1),
            0.0, 1.0, 0.0, 0.0,
        )
        self.assertEqual(result.status, "zero_certified_gain")
        self.assertEqual(result.alpha, Fraction(0))
        self.assertEqual(result.lower_bound, Fraction(0))
        self.assertEqual(result.realized_gain, Fraction(0))

    def test_nonexact_symmetry_and_nonpositive_direction_are_rejected(self):
        nonsymmetric = self.K.copy()
        nonsymmetric[1, 0] = np.nextafter(nonsymmetric[1, 0], 1.0)
        self.assertEqual(self.certify(K=nonsymmetric).reason, "K_not_exactly_symmetric")
        zero_direction = self.certify(phi=np.zeros(2), c_hat=0.0)
        self.assertEqual(zero_direction.reason, "nonpositive_directional_curvature")
        indefinite = np.diag([-1.0, 1.0])
        corrupted = DiscreteRitzProblem(self.K, self.f)
        corrupted.K = indefinite
        negative_q = certify_discrete_ritz_step(
            corrupted, np.zeros(2), np.array([1.0, 0.0]),
            -1.0, 1.0, 0.0, 0.0,
        )
        self.assertEqual(negative_q.reason, "nonpositive_directional_curvature")
        non_spd_positive_direction = certify_discrete_ritz_step(
            corrupted, np.zeros(2), np.array([0.0, 1.0]),
            1.0, 1.0, 0.0, 0.0,
        )
        self.assertEqual(non_spd_positive_direction.reason, "K_not_exactly_positive_definite")
        with self.assertRaises(ValueError):
            DiscreteRitzProblem(indefinite, np.array([0.0, 1.0]))

    def test_invalid_radii_and_qhat_gate(self):
        for args, reason in (
            ({"delta": -0.1}, "negative_enclosure_radius"),
            ({"epsilon": -0.1}, "negative_enclosure_radius"),
            ({"q_hat": 0.0, "epsilon": 0.0}, "q_hat_not_above_epsilon"),
            ({"q_hat": 0.25, "epsilon": 0.25}, "q_hat_not_above_epsilon"),
        ):
            with self.subTest(args=args):
                self.assertEqual(self.certify(**args).reason, reason)

    def test_shapes_nonfinite_and_non_float_inputs_reject(self):
        cases = (
            ({"K": np.eye(3)}, "incompatible_vector_shape"),
            ({"phi": np.array([[1.0], [0.0]])}, "invalid_phi_shape_or_nonfinite"),
            ({"f": np.array([np.nan, 0.0])}, "invalid_f_shape_or_nonfinite"),
            ({"K": np.eye(2, dtype=np.int64)}, "K_must_contain_float32_or_float64"),
            ({"c_hat": 1}, "c_hat_must_be_float32_or_float64"),
        )
        for args, reason in cases:
            with self.subTest(args=args):
                self.assertEqual(self.certify(**args).reason, reason)

    def test_arbitrary_feasibility_callback_is_not_treated_as_proof(self):
        constrained = DiscreteRitzProblem(self.K, self.f, feasible=lambda trial: True)
        result = self.certify(problem=constrained)
        self.assertEqual(result.status, "invalid_input")
        self.assertEqual(result.reason, "unsupported_feasibility_callback")

    def test_json_roundtrip_is_canonical_and_keeps_input_hash(self):
        cert = self.certify()
        encoded = cert.to_json()
        reread = DiscreteRitzStepCertificate.from_json(encoded)
        self.assertEqual(reread, cert)
        self.assertEqual(reread.to_json(), encoded)
        self.assertEqual(len(cert.input_sha256), 64)
        self.assertEqual(hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                         hashlib.sha256(reread.to_json().encode("utf-8")).hexdigest())
        self.assertTrue(verify_discrete_ritz_certificate(
            reread, self.problem, self.u, self.phi, 0.875, 2.0, 0.125, 0.0))
        forged = json.loads(encoded)
        forged["realized_gain"] = [999, 1]
        forged_cert = DiscreteRitzStepCertificate.from_json(json.dumps(forged))
        self.assertFalse(verify_discrete_ritz_certificate(
            forged_cert, self.problem, self.u, self.phi, 0.875, 2.0, 0.125, 0.0))
        damaged = json.loads(encoded)
        damaged["lower_bound"] = [1, 0]
        with self.assertRaises(ValueError):
            DiscreteRitzStepCertificate.from_json(json.dumps(damaged))

    def test_exact_arithmetic_dimension_limit_fails_closed(self):
        n = 32
        boundary = certify_discrete_ritz_step(
            DiscreteRitzProblem(np.eye(n), np.ones(n)), np.zeros(n), np.eye(n)[0],
            1.0, 1.0, 0.0, 0.0,
        )
        self.assertEqual(boundary.status, "verified_discrete_lower_bound")
        n = 33
        with patch("problems.ritz_certificate._input_hash") as input_hash, \
                patch("problems.ritz_certificate._is_exact_spd") as exact_spd:
            result = certify_discrete_ritz_step(
                DiscreteRitzProblem(np.eye(n), np.ones(n)), np.zeros(n), np.ones(n),
                33.0, 33.0, 0.0, 0.0,
            )
        input_hash.assert_not_called()
        exact_spd.assert_not_called()
        self.assertEqual(result.status, "unsupported_resource_limit")
        self.assertEqual(result.reason, "dimension_exceeds_exact_certificate_limit")

    def test_exact_trial_is_not_silently_substituted_by_rounded_float_trial(self):
        cert = certify_discrete_ritz_step(
            DiscreteRitzProblem(np.ones((1, 1)), np.ones(1)), np.zeros(1), np.ones(1),
            1.0, 3.0, 0.0, 2.0,
        )
        exact_component = cert.trial_exact[0]
        rounded_float_component = Fraction.from_float(float(exact_component))
        self.assertNotEqual(rounded_float_component, exact_component)
        self.assertGreaterEqual(cert.realized_gain, cert.lower_bound)
        exact_energy_gain = Fraction(0) - (exact_component * exact_component / 2 - exact_component)
        self.assertEqual(exact_energy_gain, cert.realized_gain)
        self.assertTrue(verify_discrete_ritz_certificate(
            cert, DiscreteRitzProblem(np.ones((1, 1)), np.ones(1)),
            np.zeros(1), np.ones(1), 1.0, 3.0, 0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
