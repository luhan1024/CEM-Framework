import json
import math
import unittest
from fractions import Fraction
from dataclasses import replace

import numpy as np

from estimators_1d import EquilibratedFlux1DEstimator, P1Poisson1DData


class EquilibratedFlux1DTests(unittest.TestCase):
    def fixture(self):
        data = P1Poisson1DData(
            nodes=np.array([0.0, 1.0 / 3.0, 1.0]),
            cell_source=np.array([1.0, 2.0]),
        )
        # Exact nodal values of the solution of -u''=f with zero endpoints.
        values = np.array([0.0, 5.0 / 27.0, 0.0])
        return data, values

    def test_rational_fixture_matches_independently_derived_exact_energy_error(self):
        data, values = self.fixture()
        record = EquilibratedFlux1DEstimator().estimate(data, values)

        # Independent hand derivation with Fractions, not estimator helpers:
        # exact u' - P1 slope is (1/6-x) on [0,1/3] and (2/3-2s)
        # on [0,2/3], so squared energy error is 1/324 + 32/324 = 11/108.
        exact_squared_error = Fraction(1, 324) + Fraction(32, 324)
        exact_error = math.sqrt(float(exact_squared_error))
        self.assertEqual(record.status, "numerical_bound")
        self.assertAlmostEqual(record.eta_eq, exact_error, places=14)
        self.assertAlmostEqual(record.eta_eq**2, float(Fraction(11, 108)), places=14)
        self.assertEqual(len(record.flux_node_values), 3)
        self.assertTrue(np.allclose(record.flux_node_values,
                                    [13.0 / 18.0, 7.0 / 18.0, -17.0 / 18.0],
                                    rtol=0.0, atol=1e-15))
        self.assertLessEqual(record.equilibrium_defect, 1e-15)
        self.assertLessEqual(record.equilibrium_defect, record.equilibrium_tolerance)
        self.assertEqual(record.integration_rule,
                         "exact_piecewise_constant_cell_integrals_float64")
        self.assertEqual(len(record.data_hash), 64)
        self.assertEqual(len(record.solution_hash), 64)

    def test_input_arrays_and_record_exports_are_detached_snapshots(self):
        nodes = np.array([0.0, 0.5, 1.0])
        source = np.array([1.0, 1.0])
        data = P1Poisson1DData(nodes, source)
        nodes[1] = 0.25
        source[0] = 9.0
        self.assertEqual(data.nodes[1], 0.5)
        self.assertEqual(data.cell_source[0], 1.0)
        with self.assertRaises(ValueError):
            data.nodes[0] = 2.0

        result = EquilibratedFlux1DEstimator().estimate(data, np.zeros(3))
        exported = result.to_dict()
        exported["assumptions"].append("changed")
        exported["flux_node_values"][0] = 100.0
        self.assertNotIn("changed", result.assumptions)
        self.assertNotEqual(exported["flux_node_values"][0], result.flux_node_values[0])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_invalid_problem_data_fail_closed_at_construction(self):
        invalid = (
            (np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0]), (0.0, 0.0)),
            (np.array([0.0, 0.7, 0.5, 1.0]), np.ones(3), (0.0, 0.0)),
            (np.array([0.0, 1.0]), np.array([np.nan]), (0.0, 0.0)),
            (np.array([0.0, 1.0]), np.array([np.inf]), (0.0, 0.0)),
            (np.array([0.0, 0.5, 1.0]), np.array([1.0]), (0.0, 0.0)),
            (np.array([0.0, 1.0]), np.array([1.0]), (0.0, 1.0)),
        )
        for nodes, source, boundary in invalid:
            with self.subTest(nodes=nodes, source=source, boundary=boundary):
                with self.assertRaises(ValueError):
                    P1Poisson1DData(nodes, source, boundary)

    def test_invalid_solution_returns_failed_record_without_eta(self):
        data, _ = self.fixture()
        estimator = EquilibratedFlux1DEstimator()
        for invalid in (
            np.array([0.0, np.nan, 0.0]),
            np.array([0.0, np.inf, 0.0]),
            np.array([0.0, 0.1]),
            np.array([0.0, 0.1, 1.0]),
        ):
            with self.subTest(invalid=invalid):
                record = estimator.estimate(data, invalid)
                self.assertEqual(record.status, "failed")
                self.assertIsNone(record.eta_eq)
                self.assertIsNotNone(record.failure_reason)
                self.assertEqual(record.data_hash, data.data_hash)
                self.assertEqual(record.flux_node_values, ())

    def test_wrong_data_type_is_failed_without_fabricated_bound(self):
        record = EquilibratedFlux1DEstimator().estimate(object(), np.zeros(2))
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.eta_eq)
        self.assertIsNone(record.data_hash)
        self.assertIn("P1Poisson1DData", record.failure_reason)

    def test_equilibrium_tolerance_failure_has_no_eta_or_flux(self):
        nodes = np.array([0.0, 0.08564916714362436, 0.2368105065960997,
                          0.5821620360643678, 0.8012744652063969, 1.0])
        source = np.array([-4.5264929211044583e-95, -2.155971630897659e-95,
                           -2.019986129147251e-94, -2.3193237764418946e-95,
                           -8.652130762749416e-95])
        data = P1Poisson1DData(nodes, source)
        record = EquilibratedFlux1DEstimator(equilibrium_rtol=0.0).estimate(
            data, np.zeros(nodes.size)
        )
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.eta_eq)
        self.assertEqual(record.flux_node_values, ())
        self.assertIn("exceeds tolerance", record.failure_reason)

    def test_default_equilibrium_tolerance_scales_with_cell_count(self):
        short_nodes = np.linspace(0.0, 1.0, 5)
        long_nodes = np.linspace(0.0, 1.0, 41)
        short_data = P1Poisson1DData(short_nodes, np.ones(short_nodes.size - 1))
        long_data = P1Poisson1DData(long_nodes, np.ones(long_nodes.size - 1))
        estimator = EquilibratedFlux1DEstimator()
        short_record = estimator.estimate(short_data, np.zeros(short_nodes.size))
        long_record = estimator.estimate(long_data, np.zeros(long_nodes.size))
        self.assertEqual(short_record.status, "numerical_bound")
        self.assertEqual(long_record.status, "numerical_bound")
        self.assertGreater(long_record.equilibrium_tolerance, short_record.equilibrium_tolerance)

    def test_record_rejects_defect_above_tolerance(self):
        data, values = self.fixture()
        record = EquilibratedFlux1DEstimator().estimate(data, values)
        with self.assertRaisesRegex(ValueError, "equilibrium_defect <= equilibrium_tolerance"):
            replace(record, equilibrium_defect=record.equilibrium_tolerance + 1.0)

    def test_overflow_from_finite_inputs_fails_without_eta(self):
        data = P1Poisson1DData(np.array([0.0, 0.5, 1.0]), np.array([1e308, 1e308]))
        with np.errstate(all="raise"):
            record = EquilibratedFlux1DEstimator().estimate(data, np.zeros(3))
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.eta_eq)
        self.assertIn("FloatingPointError", record.failure_reason)

    def test_record_normalizes_lists_and_rejects_incomplete_or_nonfinite_success(self):
        data, values = self.fixture()
        result = EquilibratedFlux1DEstimator().estimate(data, values)
        normalized = replace(result, assumptions=list(result.assumptions),
                             flux_node_values=list(result.flux_node_values))
        self.assertIsInstance(normalized.assumptions, tuple)
        self.assertIsInstance(normalized.flux_node_values, tuple)
        with self.assertRaises(ValueError):
            replace(result, data_hash=None)
        with self.assertRaises(ValueError):
            replace(result, equilibrium_defect=float("nan"))
        with self.assertRaises(ValueError):
            replace(result, equilibrium_tolerance=float("inf"))
        with self.assertRaises(ValueError):
            replace(result, flux_node_values=[float("inf")])


if __name__ == "__main__":
    unittest.main()
