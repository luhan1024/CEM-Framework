from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from core import SolverState
from p1_poisson_hat import (
    P1HatProposalError,
    P1PoissonHatWeakLearner,
    _state_fingerprint,
    _state_is_unchanged,
)
from estimators_1d import P1Poisson1DData
from problems import (
    P1Poisson1DRitzProblem,
    verify_p1_poisson_step_certificate,
)
from problems.p1_poisson_1d import (
    MAX_EXACT_RATIONAL_OPERATIONS,
    MAX_FRACTION_COMPONENT_BITS,
    MAX_P1_CELLS,
    MAX_RECORD_JSON_BYTES,
)


def _data(nodes, source):
    return P1Poisson1DData(np.asarray(nodes, dtype=np.float64),
                           np.asarray(source, dtype=np.float64))


def _fraction(value):
    return Fraction.from_float(float(value))


def _oracle_hat(data, interior_state, node):
    x = [_fraction(v) for v in data.nodes]
    f = [_fraction(v) for v in data.cell_source]
    u = [Fraction(0), *(_fraction(v) for v in interior_state), Fraction(0)]
    phi = [Fraction(0)] * (len(x) - 1) + [Fraction(0)]
    phi[node] = Fraction(1)
    c = Fraction(0)
    q = Fraction(0)
    for cell in range(len(f)):
        h = x[cell + 1] - x[cell]
        c += f[cell] * h * (phi[cell] + phi[cell + 1]) / 2
        c -= (u[cell + 1] - u[cell]) * (phi[cell + 1] - phi[cell]) / h
        q += (phi[cell + 1] - phi[cell]) ** 2 / h
    alpha = c / q

    def energy(values):
        quadratic = sum((values[i + 1] - values[i]) ** 2 /
                        (x[i + 1] - x[i]) for i in range(len(f)))
        load = sum(f[i] * (x[i + 1] - x[i]) *
                   (values[i] + values[i + 1]) / 2 for i in range(len(f)))
        return quadratic / 2 - load

    trial = [u[i] + alpha * phi[i] for i in range(len(u))]
    direct_gain = energy(u) - energy(trial)
    return c, q, alpha, c * c / (2 * q), direct_gain


def _independent_problem_hash_audit(problem):
    nodes = np.asarray(problem.nodes, dtype="<f8", order="C")
    source = np.asarray(problem.cell_source, dtype="<f8", order="C")

    def tagged_array_hash(array):
        meta = json.dumps({"shape": list(array.shape), "dtype": "binary64-le"},
                          sort_keys=True, separators=(",", ":")).encode("ascii")
        raw = array.tobytes(order="C")
        return hashlib.sha256(meta + raw).hexdigest(), len(meta) + len(raw)

    def p1_array_hash(array):
        meta = json.dumps({"shape": array.shape}, separators=(",", ":")).encode("utf-8")
        raw = array.tobytes(order="C")
        return hashlib.sha256(meta + raw).hexdigest(), len(meta) + len(raw)

    mesh_hash, mesh_bytes = tagged_array_hash(nodes)
    source_hash, source_bytes = tagged_array_hash(source)
    p1_nodes_hash, p1_nodes_bytes = p1_array_hash(nodes)
    p1_source_hash, p1_source_bytes = p1_array_hash(source)
    data_payload = {
        "schema_version": "1",
        "nodes_sha256": p1_nodes_hash,
        "cell_source_sha256": p1_source_hash,
        "dirichlet_values": (0.0, 0.0),
        "integration_rule": "exact_piecewise_constant_cell_integrals_float64",
    }
    data_json = json.dumps(data_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data_hash = hashlib.sha256(data_json).hexdigest()
    if problem.number_of_cells == 1:
        assembly_payload = b"p1-empty-operator-v1"
    else:
        assembly_payload = b"".join(
            np.asarray(array, dtype="<f8", order="C").tobytes(order="C")
            for array in (problem.diagonal, problem.upper, problem.lower, problem.load_full)
        )
    assembly_hash = hashlib.sha256(assembly_payload).hexdigest()
    identity = {
        "schema": "cem.p1_poisson_1d_ritz_problem.v1",
        "assembly_id": "p1-tridiagonal-local-assembly.v1",
        "data_sha256": data_hash,
        "mesh_sha256": mesh_hash,
        "source_sha256": source_hash,
        "cells": problem.number_of_cells,
        "dimension": max(problem.number_of_cells - 1, 0),
        "assembly_sha256": assembly_hash,
        "boundary": ["0x0.0p+0", "0x0.0p+0"],
    }
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode("ascii")
    total_bytes = (mesh_bytes + source_bytes + p1_nodes_bytes + p1_source_bytes
                   + len(data_json) + len(assembly_payload) + len(identity_json))
    events = 2 + 2 + 2 + 2 + 1 + 1 + 1
    if problem.number_of_cells > 1:
        events += 3
    return hashlib.sha256(identity_json).hexdigest(), total_bytes, events


class P1PoissonRitzAdapterTests(unittest.TestCase):
    def test_matrix_free_operator_and_load_match_independent_local_assembly(self):
        data = _data([0.0, 0.2, 0.65, 1.0], [1.25, -0.5, 2.0])
        problem = P1Poisson1DRitzProblem(data)
        dense = np.zeros((4, 4), dtype=np.float64)
        load = np.zeros(4, dtype=np.float64)
        for cell, source in enumerate(data.cell_source):
            h = data.nodes[cell + 1] - data.nodes[cell]
            local = (1.0 / h) * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
            dense[cell:cell + 2, cell:cell + 2] += local
            load[cell:cell + 2] += source * h * 0.5
        values = np.asarray([0.125, -0.25, 0.375], dtype=np.float64)
        self.assertEqual(problem.shape, (2, 2))
        values = np.asarray([0.125, -0.25], dtype=np.float64)
        np.testing.assert_allclose(problem.apply_operator(values), dense[1:-1, 1:-1] @ values,
                                   rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(problem.source, load[1:-1], rtol=0.0, atol=1e-14)
        self.assertEqual(problem.data_sha256, data.data_hash)

    def test_hash_input_ledger_matches_independent_digest_updates_n1_and_multicell(self):
        for data in (_data([0.0, 1.0], [2.0]),
                     _data([0.0, 0.125, 0.5, 1.0], [1.0, -2.0, 0.25])):
            with self.subTest(cells=data.cell_source.size):
                problem = P1Poisson1DRitzProblem(data)
                expected_fingerprint, expected_bytes, expected_events = (
                    _independent_problem_hash_audit(problem))
                self.assertEqual(problem.problem_fingerprint, expected_fingerprint)
                self.assertEqual(problem.build_audit["hash_bytes"], expected_bytes)
                self.assertEqual(problem.build_audit["hash_events"], expected_events)
                if data.cell_source.size == 1:
                    self.assertEqual(problem.build_audit["adapter_cell_visits"], 0)
                    self.assertEqual(problem.build_audit["adapter_operator_rows"], 0)
                    self.assertGreater(problem.build_audit["hash_bytes"], 0)

    def test_problem_detects_mutated_caller_data(self):
        data = _data([0.0, 0.5, 1.0], [1.0, 1.0])
        problem = P1Poisson1DRitzProblem(data)
        data.cell_source.setflags(write=True)
        data.cell_source[0] = 2.0
        with self.assertRaisesRegex(ValueError, "problem_data_fingerprint_mismatch"):
            problem.assert_integrity()


class P1PoissonHatWeakLearnerTests(unittest.TestCase):
    def setUp(self):
        self.learner = P1PoissonHatWeakLearner()

    def test_state_fingerprint_distinguishes_none_and_empty_outer_coefficients(self):
        problem = P1Poisson1DRitzProblem(
            _data([0.0, 0.5, 1.0], [1.0, 1.0]))
        none_state = SolverState(np.zeros(1, dtype=np.float64),
                                 outer_coefficients=None)
        empty_state = SolverState(np.zeros(1, dtype=np.float64),
                                  outer_coefficients=np.zeros(0, dtype=np.float64))
        none_fingerprint, none_events, none_bytes = _state_fingerprint(
            problem, none_state.snapshot())
        empty_fingerprint, empty_events, empty_bytes = _state_fingerprint(
            problem, empty_state.snapshot())

        self.assertNotEqual(none_fingerprint, empty_fingerprint)
        self.assertEqual(empty_events, none_events + 1)
        self.assertGreater(empty_bytes, none_bytes)
        self.assertFalse(_state_is_unchanged(empty_state, none_fingerprint, problem))
        self.assertFalse(_state_is_unchanged(none_state, empty_fingerprint, problem))

    def _proposal(self, data, state_values, budget=None, problem=None):
        problem = problem or P1Poisson1DRitzProblem(data)
        state = SolverState(np.asarray(state_values, dtype=np.float64))
        residual = problem.residual(state)
        return problem, state, self.learner.propose(problem, state, residual, budget)

    def test_nonuniform_all_hat_local_results_match_fraction_oracle_and_full_replay(self):
        data = _data([0.0, 0.2, 0.65, 1.0], [1.25, -0.5, 2.0])
        values = np.asarray([0.125, -0.25], dtype=np.float64)
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(values)
        candidate = self.learner.propose(problem, state, problem.residual(state), None)
        oracle = {j: _oracle_hat(data, values, j) for j in range(1, 4)}
        best_gain = max(item[3] for item in oracle.values())
        best_nodes = [j for j, item in oracle.items() if item[3] == best_gain]
        expected = min(best_nodes, key=lambda j:
                       f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:{j}")
        self.assertEqual(candidate.element_id,
                         f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:{expected}")
        np.testing.assert_array_equal(candidate.correction,
                                      np.eye(2, dtype=np.float64)[expected - 1])
        self.assertEqual(candidate.support["cell_indices"], (expected - 1, expected))
        self.assertEqual(candidate.provenance["weakness_tau"], 1.0)
        certificate_json = candidate.provenance["certificate_record_json"]
        self.assertEqual(hashlib.sha256(certificate_json.encode("utf-8")).hexdigest(),
                         candidate.provenance["certificate_sha256"])
        from problems import P1PoissonStepCertificate
        certificate = P1PoissonStepCertificate.from_json(certificate_json)
        full_state = np.concatenate(([0.0], values, [0.0])).astype(np.float64)
        direction = np.zeros(4, dtype=np.float64)
        direction[expected] = 1.0
        expected_c, expected_q, expected_alpha, expected_gain, direct_gain = oracle[expected]
        self.assertEqual((certificate.c_exact, certificate.q_exact,
                          certificate.alpha, certificate.gain_exact),
                         (expected_c, expected_q, expected_alpha, expected_gain))
        self.assertEqual(expected_gain, direct_gain)
        self.assertTrue(verify_p1_poisson_step_certificate(certificate, data,
                                                           full_state, direction))
        audit = candidate.provenance["work_audit"]
        n = data.cell_source.size
        k = n - 1
        self.assertEqual(audit["hat_evaluations"], k)
        self.assertEqual(audit["local_cell_visits"], 2 * k)
        self.assertEqual(audit["fraction_ops_reserved"], 6 * n + 7 * k + 36 * n + 11)
        self.assertEqual(audit["fraction_ops_certificate_actual"], 36 * n + 11)
        self.assertEqual(audit["certificate_replays"], 1)
        self.assertLessEqual(audit["fraction_ops_actual_total"],
                             audit["fraction_ops_reserved"])
        self.assertEqual(audit["fraction_conversions_local_actual"], 3 * n + 2)
        self.assertEqual(audit["fraction_conversions_certificate_actual"], 4 * n + 3)
        self.assertEqual(audit["operator_row_visits"], 2 * (n - 1))
        self.assertEqual(audit["candidate_metadata_bytes"],
                         len(json.dumps({
                             "element_id": candidate.element_id,
                             "family": candidate.family,
                             "support": dict(candidate.support),
                             "parameters": dict(candidate.parameters),
                             "seed": candidate.seed,
                             "provenance": {key: _plain_metadata(value)
                                            for key, value in candidate.provenance.items()
                                            if key != "certificate_record_json"},
                             "proposal_cost": dict(candidate.proposal_cost),
                         }, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                            allow_nan=False).encode("utf-8")))
        self.assertLessEqual(audit["candidate_metadata_bytes"],
                             audit["candidate_metadata_bytes_reserved"])
        self.assertLessEqual(audit["candidate_metadata_serializations"], 4)
        self.assertEqual(
            audit["serialization_bytes"],
            2 * audit["certificate_input_bytes"]
            + audit["certificate_record_bytes"]
            + audit["preflight_metadata_bytes_serialized"]
            + audit["candidate_metadata_serialization_bytes"],
        )
        self.assertLessEqual(audit["serialization_bytes"],
                             audit["serialization_bytes_reserved"])
        self.assertLessEqual(audit["hash_bytes"], audit["hash_bytes_reserved"])
        self.assertLessEqual(audit["hash_events"], audit["hash_events_reserved"])

    def test_uniform_tie_uses_ascending_stable_id(self):
        data = _data([0.0, 0.25, 0.5, 0.75, 1.0],
                     [1.0, 1.0, 1.0, 1.0])
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(3, dtype=np.float64))
        candidate = self.learner.propose(problem, state, problem.residual(state), None)
        self.assertTrue(candidate.element_id.endswith(":node:1"))

    def test_each_singleton_hat_matches_independent_fraction_and_full_certificate(self):
        data = _data([0.0, 0.2, 0.65, 1.0], [1.25, -0.5, 2.0])
        values = np.asarray([0.125, -0.25], dtype=np.float64)
        problem = P1Poisson1DRitzProblem(data)
        for selected in (1, 2):
            active_ids = tuple(
                f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:{node}"
                for node in (1, 2) if node != selected
            )
            state = SolverState(values, active_ids)
            candidate = self.learner.propose(problem, state, problem.residual(state), None)
            self.assertTrue(candidate.element_id.endswith(f":node:{selected}"))
            from problems import P1PoissonStepCertificate
            certificate = P1PoissonStepCertificate.from_json(
                candidate.provenance["certificate_record_json"])
            expected = _oracle_hat(data, values, selected)
            self.assertEqual((certificate.c_exact, certificate.q_exact,
                              certificate.alpha, certificate.gain_exact), expected[:4])
            self.assertEqual(expected[3], expected[4])

    def test_active_hat_is_skipped_and_invalid_ids_are_rejected(self):
        data = _data([0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 0.5, 2.0, -1.0])
        problem = P1Poisson1DRitzProblem(data)
        node_two = f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:2"
        state = SolverState(np.zeros(3, dtype=np.float64), (node_two,))
        candidate = self.learner.propose(problem, state, problem.residual(state), None)
        self.assertNotEqual(candidate.element_id, node_two)
        for ids, reason in (((node_two, node_two), "duplicate_active_id"),
                            (("foreign-id",), "malformed_active_id"),
                            ((f"p1hat-v1:{'0' * 64}:{problem.mesh_sha256}:node:1",),
                             "foreign_active_id")):
            bad = SolverState(np.zeros(3, dtype=np.float64), ids)
            with self.subTest(reason=reason), self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, bad, problem.residual(bad), None)
            self.assertEqual(caught.exception.status, "active_set_invalid")
            self.assertEqual(caught.exception.reason_code, reason)

    def test_n1_returns_empty_with_zero_adapter_residual_candidate_and_certificate_work(self):
        data = _data([0.0, 1.0], [1.0])
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(0, dtype=np.float64))
        with patch.object(P1Poisson1DRitzProblem, "residual",
                          side_effect=AssertionError("residual must not run")):
            with self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, state, np.zeros(0, dtype=np.float64), None)
        error = caught.exception
        self.assertEqual((error.status, error.reason_code),
                         ("empty_dictionary", "no_interior_hat"))
        audit = error.work_audit
        for key in ("adapter_cell_visits", "adapter_operator_rows", "residual_operator_rows",
                    "candidate_vector_elements", "certificate_replays",
                    "fraction_ops_actual_total", "hat_evaluations"):
            self.assertEqual(audit[key], 0, key)
        self.assertGreater(audit["hash_bytes"], 0)
        with self.assertRaises(TypeError):
            audit["hat_evaluations"] = 1

    def test_all_active_returns_exhausted_without_residual_or_numeric_budget_failure(self):
        data = _data([0.0, 0.5, 1.0], [1.0, 1.0])
        problem = P1Poisson1DRitzProblem(data)
        ids = tuple(f"p1hat-v1:{problem.problem_fingerprint}:{problem.mesh_sha256}:node:{j}"
                    for j in range(1, 2))
        state = SolverState(np.zeros(1, dtype=np.float64), ids)
        zero_budget = {key: 0 for key in (
            "max_cells", "max_hat_evaluations", "max_local_cell_visits",
            "max_adapter_cell_visits", "max_adapter_operator_rows",
            "max_residual_operator_rows", "max_operator_row_visits",
            "max_fraction_ops_total", "max_certificate_ops",
            "max_fraction_conversions", "max_fraction_component_bits",
            "max_certificate_record_bytes", "max_candidate_metadata_bytes",
            "max_candidate_vector_elements", "max_hash_bytes",
        )}
        with patch.object(P1Poisson1DRitzProblem, "residual",
                          side_effect=AssertionError("residual must not run")):
            with self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, state, np.zeros(1, dtype=np.float64), zero_budget)
        error = caught.exception
        self.assertEqual((error.status, error.reason_code),
                         ("dictionary_exhausted", "all_hats_active"))
        self.assertEqual(error.work_audit["fraction_ops_reserved"], 0)
        self.assertEqual(error.work_audit["fraction_ops_actual_total"], 0)
        self.assertEqual(error.work_audit["fraction_conversions_local_actual"], 0)
        self.assertEqual(error.work_audit["hat_evaluations"], 0)
        self.assertEqual(error.work_audit["certificate_replays"], 0)

    def test_all_zero_gain_has_undefined_tau_and_no_certificate_replay(self):
        data = _data([0.0, 0.5, 1.0], [0.0, 0.0])
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(1, dtype=np.float64))
        with self.assertRaises(P1HatProposalError) as caught:
            self.learner.propose(problem, state, problem.residual(state), None)
        error = caught.exception
        self.assertEqual((error.status, error.reason_code),
                         ("no_positive_gain", "all_gains_zero"))
        self.assertEqual(error.work_audit["hat_evaluations"], 1)
        self.assertEqual(error.work_audit["fraction_ops_certificate_reserved"], 36 * 2 + 11)
        self.assertEqual(error.work_audit["fraction_ops_certificate_actual"], 0)
        self.assertEqual(error.work_audit["certificate_replays"], 0)
        self.assertNotIn("weakness_tau", error.work_audit)

    def test_budget_shape_unknown_values_and_preflight_reject_before_scan(self):
        data = _data([0.0, 0.5, 1.0], [1.0, 1.0])
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(1, dtype=np.float64))
        residual = problem.residual(state)
        cases = [
            ([], "invalid_budget_shape"),
            ({"unknown": 1}, "unknown_budget_key"),
            ({"max_cells": True}, "invalid_budget_value"),
            ({"max_cells": 1.5}, "invalid_budget_value"),
            ({"max_cells": -1}, "invalid_budget_value"),
            ({"max_fraction_ops_total": 250_001}, "budget_limit_increased"),
        ]
        for budget, reason in cases:
            with self.subTest(reason=reason), self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, state, residual, budget)
            self.assertEqual(caught.exception.status, "budget_exceeded")
            self.assertEqual(caught.exception.reason_code, reason)
        with patch.object(P1Poisson1DRitzProblem, "residual",
                          side_effect=AssertionError("residual must not run")):
            with self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, state, residual,
                                     {"max_fraction_ops_total": 1})
        self.assertEqual(caught.exception.reason_code, "fraction_operation_cap")
        self.assertEqual(caught.exception.work_audit["hat_evaluations"], 0)

        metadata_problem = P1Poisson1DRitzProblem(data)
        metadata_state = SolverState(np.zeros(1, dtype=np.float64))
        metadata_residual = metadata_problem.residual(metadata_state)
        with self.assertRaises(P1HatProposalError) as caught:
            self.learner.propose(metadata_problem, metadata_state, metadata_residual,
                                 {"max_candidate_metadata_bytes": 1})
        error = caught.exception
        self.assertEqual(error.reason_code, "candidate_metadata_byte_cap")
        self.assertEqual(error.work_audit["hat_evaluations"], 0)
        self.assertGreater(error.work_audit["candidate_metadata_bytes_reserved"], 1)
        self.assertGreater(error.work_audit["hash_bytes"], 0)
        self.assertGreater(error.work_audit["certificate_input_bytes"], 0)
        self.assertGreater(error.work_audit["hash_bytes_reserved"],
                           error.work_audit["hash_bytes"])
        self.assertEqual(
            error.work_audit["serialization_bytes"],
            error.work_audit["certificate_input_bytes"]
            + error.work_audit["preflight_metadata_bytes_serialized"],
        )
        self.assertGreater(error.work_audit["serialization_bytes_reserved"],
                           error.work_audit["serialization_bytes"])
        self.assertEqual(error.work_audit["candidate_metadata_serializations"], 0)
        self.assertEqual(error.work_audit["candidate_metadata_serialization_bytes"], 0)
        self.assertEqual(error.work_audit["hat_evaluations"], 0)

    def test_residual_is_checked_by_exact_shape_dtype_and_bytes(self):
        data = _data([0.0, 0.5, 1.0], [1.0, 1.0])
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(1, dtype=np.float64))
        good = problem.residual(state)
        variants = [np.asarray(good, dtype=np.float32), np.asarray([good[0], 1.0]),
                    np.asarray([np.nextafter(good[0], np.inf)], dtype=np.float64)]
        for residual in variants:
            with self.subTest(dtype=residual.dtype, shape=residual.shape), \
                    self.assertRaises(P1HatProposalError) as caught:
                self.learner.propose(problem, state, residual, None)
            self.assertEqual(caught.exception.status, "residual_mismatch")
        zero_data = _data([0.0, 0.5, 1.0], [0.0, 0.0])
        zero_problem = P1Poisson1DRitzProblem(zero_data)
        zero_state = SolverState(np.zeros(1, dtype=np.float64))
        signed_zero = np.asarray([-0.0], dtype=np.float64)
        with self.assertRaises(P1HatProposalError) as caught:
            self.learner.propose(zero_problem, zero_state, signed_zero, None)
        self.assertEqual(caught.exception.reason_code, "residual_bits_differ")

    def test_wrong_source_mesh_and_stale_state_are_rejected(self):
        data_a = _data([0.0, 0.5, 1.0], [1.0, 1.0])
        data_b = _data([0.0, 0.25, 1.0], [1.0, -0.5])
        problem_a = P1Poisson1DRitzProblem(data_a)
        problem_b = P1Poisson1DRitzProblem(data_b)
        state = SolverState(np.zeros(1, dtype=np.float64))
        foreign_residual = problem_b.residual(state)
        with self.assertRaises(P1HatProposalError) as caught:
            self.learner.propose(problem_a, state, foreign_residual, None)
        self.assertEqual(caught.exception.status, "residual_mismatch")
        stale = problem_a.residual(state)
        changed_state = SolverState(np.asarray([0.25], dtype=np.float64))
        with self.assertRaises(P1HatProposalError) as caught:
            self.learner.propose(problem_a, changed_state, stale, None)
        self.assertEqual(caught.exception.status, "residual_mismatch")

    def test_maximum_grid_is_exhaustive_and_within_exact_preflight(self):
        n = MAX_P1_CELLS
        data = _data(np.linspace(0.0, 1.0, n + 1, dtype=np.float64),
                     np.ones(n, dtype=np.float64))
        problem = P1Poisson1DRitzProblem(data)
        state = SolverState(np.zeros(n - 1, dtype=np.float64))
        candidate = self.learner.propose(problem, state, problem.residual(state), None)
        audit = candidate.provenance["work_audit"]
        self.assertTrue(candidate.element_id.endswith(":node:1"))
        self.assertEqual(audit["hat_evaluations"], n - 1)
        self.assertEqual(audit["local_cell_visits"], 2 * (n - 1))
        self.assertEqual(audit["fraction_ops_reserved"], 200_708)
        self.assertEqual(audit["fraction_ops_certificate_actual"], 36 * n + 11)
        self.assertEqual(audit["certificate_replays"], 1)
        self.assertLessEqual(audit["fraction_ops_actual_total"], 200_708)
        self.assertEqual(audit["fraction_conversions_reserved"], 28_677)
        self.assertLessEqual(audit["hash_bytes"], audit["effective_budget"]["max_hash_bytes"])
        self.assertLessEqual(audit["hash_bytes"], audit["hash_bytes_reserved"])
        self.assertLessEqual(audit["serialization_bytes"],
                             audit["serialization_bytes_reserved"])


def _plain_metadata(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_metadata(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
