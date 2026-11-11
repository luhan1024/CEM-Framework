from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from decimal import Decimal, localcontext
import unittest
from unittest.mock import patch

import numpy as np

from problems import (
    DiscreteRitzProblem,
    DiscreteRitzStepCertificate,
    certify_discrete_ritz_step,
    verify_discrete_ritz_certificate,
)
from core import WorkLedger


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

    def test_analytic_poisson_fixture_checks_declared_enclosure_and_discrete_gain(self):
        # Continuous problem: -u''=1 on (0,1), u(0)=u(1)=0, with
        # psi=x(1-x). Exact integrals are c=1/6 and q=1/3. The certificate
        # below concerns only the discrete float64 Ritz quadratic, not the
        # continuous PDE energy.
        continuous_c = Fraction(1, 6)
        continuous_q = Fraction(1, 3)

        # Independent high-precision Simpson path. Its integrands psi and
        # (psi')^2 are degree-two polynomials, so Simpson is exact before
        # Decimal rounding; this path is diagnostic, not the enclosure proof.
        with localcontext() as context:
            context.prec = 80
            zero = Decimal(0)
            half = Decimal(1) / Decimal(2)
            one = Decimal(1)

            def psi(x):
                return x * (one - x)

            def dpsi(x):
                return one - Decimal(2) * x

            c_simpson = (psi(zero) + Decimal(4) * psi(half) + psi(one)) / Decimal(6)
            q_simpson = (dpsi(zero) ** 2 + Decimal(4) * dpsi(half) ** 2
                         + dpsi(one) ** 2) / Decimal(6)
        self.assertLess(abs(Fraction(c_simpson) - continuous_c), Fraction(1, 10**78))
        self.assertLess(abs(Fraction(q_simpson) - continuous_q), Fraction(1, 10**78))

        # Independently coded exact-rational two-subinterval midpoint rule.
        # Its error is computed exactly here only for these stated polynomial
        # integrands and this partition; no general quadrature claim is made.
        midpoint_nodes = (Fraction(1, 4), Fraction(3, 4))
        midpoint_c = sum((x * (1 - x) for x in midpoint_nodes), Fraction(0)) / 2
        midpoint_q = sum(((1 - 2 * x) ** 2 for x in midpoint_nodes), Fraction(0)) / 2
        self.assertEqual(midpoint_c, Fraction(3, 16))
        self.assertEqual(midpoint_q, Fraction(1, 4))
        self.assertEqual(abs(midpoint_c - continuous_c), Fraction(1, 48))
        self.assertEqual(abs(midpoint_q - continuous_q), Fraction(1, 12))
        self.assertLess(
            abs((Fraction(c_simpson) - midpoint_c) - (continuous_c - midpoint_c)),
            Fraction(1, 10**78),
        )
        self.assertLess(
            abs((Fraction(q_simpson) - midpoint_q) - (continuous_q - midpoint_q)),
            Fraction(1, 10**78),
        )

        K = np.array([[float(continuous_q)]], dtype=np.float64)
        source = np.array([float(continuous_c)], dtype=np.float64)
        u = np.array([0.0], dtype=np.float64)
        phi = np.array([1.0], dtype=np.float64)
        problem = DiscreteRitzProblem(K, source)
        midpoint_c_hat = float(midpoint_c)
        midpoint_q_hat = float(midpoint_q)

        # Independent exact-rational oracle for the verifier's stored dyadic
        # values and the midpoint centers. Outward conversion is checked back
        # in Fraction arithmetic, including float64 coefficient error.
        c_dyad = Fraction.from_float(float(problem.source[0]))
        q_dyad = Fraction.from_float(float(problem.K[0, 0]))

        def outward_radius(exact_radius):
            rounded = float(exact_radius)
            if Fraction.from_float(rounded) < exact_radius:
                rounded = float(np.nextafter(rounded, np.inf))
            self.assertGreaterEqual(Fraction.from_float(rounded), exact_radius)
            return rounded

        delta_exact = max(abs(c_dyad - midpoint_c),
                          abs(continuous_c - midpoint_c))
        epsilon_exact = max(abs(q_dyad - midpoint_q),
                            abs(continuous_q - midpoint_q))
        delta = outward_radius(delta_exact)
        epsilon = outward_radius(epsilon_exact)
        self.assertGreater(midpoint_q_hat, epsilon)

        exact_fraction = lambda values: tuple(Fraction.from_float(float(x)) for x in values)
        K_exact = tuple(tuple(Fraction.from_float(float(x)) for x in row) for row in problem.K)
        source_exact = exact_fraction(problem.source)
        u_exact = exact_fraction(u)
        phi_exact = exact_fraction(phi)
        chat_exact = Fraction.from_float(midpoint_c_hat)
        qhat_exact = Fraction.from_float(midpoint_q_hat)
        delta_dyad = Fraction.from_float(delta)
        epsilon_dyad = Fraction.from_float(epsilon)
        c_oracle = sum(((source_exact[i] - sum(
            (K_exact[i][j] * u_exact[j] for j in range(len(u_exact))), Fraction(0)))
            * phi_exact[i] for i in range(len(u_exact))), Fraction(0))
        q_oracle = sum((phi_exact[i] * K_exact[i][j] * phi_exact[j]
                        for i in range(len(phi_exact))
                        for j in range(len(phi_exact))), Fraction(0))
        self.assertEqual(c_oracle, c_dyad)
        self.assertEqual(q_oracle, q_dyad)
        self.assertLessEqual(abs(c_oracle - chat_exact), delta_dyad)
        self.assertLessEqual(abs(q_oracle - qhat_exact), epsilon_dyad)
        self.assertLessEqual(abs(continuous_c - chat_exact), delta_dyad)
        self.assertLessEqual(abs(continuous_q - qhat_exact), epsilon_dyad)

        # One-ULP-understated radii relative to the verifier's dyadic targets
        # fail independently at their own gate (the main radii also cover the
        # continuous analytic targets, which may require slightly more).
        problem_for_c = DiscreteRitzProblem(K, source)
        c_dyad_radius = outward_radius(abs(c_dyad - chat_exact))
        c_too_small = float(np.nextafter(c_dyad_radius, 0.0))
        c_rejected = certify_discrete_ritz_step(
            problem_for_c, u.copy(), phi.copy(), midpoint_c_hat,
            midpoint_q_hat, c_too_small, epsilon,
        )
        self.assertEqual(c_rejected.reason, "c_enclosure_failed")
        problem_for_q = DiscreteRitzProblem(K, source)
        q_dyad_radius = outward_radius(abs(q_dyad - qhat_exact))
        self.assertLess(Fraction.from_float(q_dyad_radius),
                        abs(continuous_q - qhat_exact))
        q_too_small = float(np.nextafter(q_dyad_radius, 0.0))
        q_rejected = certify_discrete_ritz_step(
            problem_for_q, u.copy(), phi.copy(), midpoint_c_hat,
            midpoint_q_hat, delta, q_too_small,
        )
        self.assertEqual(q_rejected.reason, "q_enclosure_failed")

        # The certificate API has no solver/state-update handle. Attach the
        # problem's actual work ledger and check every input/problem object it
        # can reach across both certification and independent verification.
        work_ledger = WorkLedger()
        problem.work_ledger = work_ledger
        arrays_before = tuple(array.tobytes() for array in (K, source, u, phi, problem.K, problem.source))
        work_before = work_ledger.to_dicts()
        problem_before = (problem.feasible, problem.spd_tol, problem.work_ledger,
                          problem.K.flags.writeable, problem.source.flags.writeable)

        cert = certify_discrete_ritz_step(
            problem, u, phi, midpoint_c_hat, midpoint_q_hat, delta, epsilon,
        )
        self.assertEqual(cert.status, "verified_discrete_lower_bound")
        self.assertTrue(verify_discrete_ritz_certificate(
            cert, problem, u, phi, midpoint_c_hat, midpoint_q_hat, delta, epsilon))

        # Independent exact discrete objective calculation; do not call the
        # verifier's helper or DiscreteRitzProblem.objective.
        s = max(abs(chat_exact) - delta_dyad, Fraction(0))
        sign = 1 if chat_exact > 0 else (-1 if chat_exact < 0 else 0)
        alpha_oracle = Fraction(sign) * s / (qhat_exact + epsilon_dyad)
        trial_oracle = tuple(u_exact[i] + alpha_oracle * phi_exact[i]
                             for i in range(len(u_exact)))

        def exact_energy(values):
            quadratic = sum((values[i] * K_exact[i][j] * values[j]
                             for i in range(len(values))
                             for j in range(len(values))), Fraction(0))
            linear = sum((source_exact[i] * values[i]
                          for i in range(len(values))), Fraction(0))
            return quadratic / 2 - linear

        direct_gain = exact_energy(u_exact) - exact_energy(trial_oracle)
        lower_oracle = s * s / (2 * (qhat_exact + epsilon_dyad))
        continuous_gain = alpha_oracle * continuous_c - (
            alpha_oracle * alpha_oracle * continuous_q / 2)
        self.assertEqual(alpha_oracle, cert.alpha)
        self.assertEqual(direct_gain, cert.realized_gain)
        self.assertEqual(lower_oracle, cert.lower_bound)
        self.assertGreater(direct_gain, 0)
        self.assertLessEqual(lower_oracle, direct_gain)
        self.assertGreater(continuous_gain, 0)
        self.assertLessEqual(lower_oracle, continuous_gain)

        self.assertEqual(arrays_before, tuple(
            array.tobytes() for array in (K, source, u, phi, problem.K, problem.source)))
        self.assertEqual(work_ledger.to_dicts(), work_before)
        self.assertEqual(problem_before, (
            problem.feasible, problem.spd_tol, problem.work_ledger,
            problem.K.flags.writeable, problem.source.flags.writeable))


if __name__ == "__main__":
    unittest.main()
