from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from core import CorrectionLedger, PoolAuditRecord, SolverState
from core.ledger import canonical_json_bytes, verify_proposal_audit
from core.state import (STATE_HASH_ALGORITHM_VERSION, immutable_snapshot,
                        snapshots_equal)
from estimators_1d import P1Poisson1DData
from enrichment import Candidate, ProposalFailure
from p1_poisson_hat import P1HatProposalError, P1PoissonHatWeakLearner
from problems import (P1Poisson1DRitzProblem, RitzEnergyProblem,
                      verify_p1_poisson_step_certificate)
from problems.p1_poisson_1d import (P1PoissonStepCertificate,
                                    certify_p1_poisson_step)
from solver import CEMSolver
from strategies import (FullyCorrectivePolicy, RitzCandidateScorer,
                        StagewisePolicy)
from strategies.scorer import Score


def _data(nodes=(0.0, 0.25, 0.5, 0.75, 1.0),
          source=(1.0, 0.5, 1.5, 1.0)):
    return P1Poisson1DData(np.asarray(nodes, dtype=np.float64),
                           np.asarray(source, dtype=np.float64))


def _solver(policy=None, data=None):
    problem = P1Poisson1DRitzProblem(_data() if data is None else data)
    state = SolverState(np.zeros(problem.dimension, dtype=np.float64))
    return CEMSolver(problem, (P1PoissonHatWeakLearner(),),
                     RitzCandidateScorer(), policy or StagewisePolicy(), state)


def _record_as_json(solver):
    return json.dumps(solver.ledger.to_dicts(), sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _fraction_ritz_data(problem):
    """Rebuild the finite Ritz quadratic from stored binary64 K and f."""
    size = problem.number_of_cells + 1
    zero = Fraction(0)
    matrix = [[zero for _ in range(size)] for _ in range(size)]
    for index, value in enumerate(problem.diagonal):
        matrix[index][index] = Fraction.from_float(float(value))
    for index, value in enumerate(problem.upper):
        matrix[index][index + 1] = Fraction.from_float(float(value))
    for index, value in enumerate(problem.lower):
        matrix[index + 1][index] = Fraction.from_float(float(value))
    interior = tuple(tuple(matrix[row][column]
                            for column in range(1, size - 1))
                     for row in range(1, size - 1))
    load = tuple(Fraction.from_float(float(value)) for value in problem.source)
    return interior, load


def _fraction_ritz_objective(matrix, load, values):
    vector = tuple(Fraction.from_float(float(value)) for value in values)
    quadratic = sum(vector[i] * matrix[i][j] * vector[j]
                    for i in range(len(vector))
                    for j in range(len(vector)))
    linear = sum(load[i] * vector[i] for i in range(len(vector)))
    return quadratic / 2 - linear


class P1SolverLifecycleIntegrationTests(unittest.TestCase):
    def _assert_persisted_p1_audit(self, solver, record, before, outcome):
        serialized = _record_as_json(solver)
        raw_records = json.loads(serialized)
        persisted = raw_records[-1]
        roundtrip = CorrectionLedger.from_dicts([persisted]).to_dicts()[0]
        self.assertEqual(roundtrip, persisted)
        self.assertEqual(roundtrip["record_schema_version"], 2)
        self.assertEqual(roundtrip["proposal_audit_json"],
                         record.proposal_audit_json)
        self.assertEqual(roundtrip["proposal_audit_sha256"],
                         record.proposal_audit_sha256)

        envelope_text = roundtrip["proposal_audit_json"]
        envelope_bytes = envelope_text.encode("utf-8")
        envelope = json.loads(envelope_text)
        self.assertEqual(canonical_json_bytes(envelope), envelope_bytes)
        self.assertEqual(hashlib.sha256(envelope_bytes).hexdigest(),
                         roundtrip["proposal_audit_sha256"])
        self.assertTrue(verify_proposal_audit(
            envelope_text, roundtrip["proposal_audit_sha256"],
            problem=solver.problem, state=before))
        self.assertEqual(envelope["schema"], "cem.p1_proposal_audit.v1")
        self.assertEqual(envelope["outcome"], outcome)
        self.assertLessEqual(len(envelope_bytes), 8_000_000)

        events = {event.event_id: event for event in solver.work_ledger.events}
        event_ids = tuple(roundtrip["work_event_ids"])
        self.assertTrue(event_ids)
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertTrue(set(event_ids).issubset(events))
        for event_id in event_ids:
            self.assertEqual(events[event_id].solver_step_id,
                             events[event_ids[0]].solver_step_id)

        if outcome == "typed_failure":
            self.assertIsNone(roundtrip["element_id"])
            self.assertEqual(envelope["element_id"], None)
            self.assertEqual(envelope["problem_fingerprint"],
                             solver.problem.problem_fingerprint)
            self.assertEqual(envelope["proposal_status"],
                             roundtrip["proposal_status"])
            self.assertEqual(envelope["reason_code"], roundtrip["reason_code"])
            self.assertIsInstance(envelope["work_audit"], dict)
            for forbidden in ("candidate_fingerprint", "support", "parameters",
                              "certificate_record_json", "certificate_sha256"):
                self.assertNotIn(forbidden, envelope)
            return envelope

        self.assertEqual(envelope["problem_fingerprint"],
                         solver.problem.problem_fingerprint)
        self.assertEqual(envelope["element_id"], roundtrip["element_id"])
        self.assertEqual(envelope["support"], roundtrip["support"])
        self.assertEqual(envelope["parameters"], roundtrip["parameters"])
        self.assertIsInstance(envelope["work_audit"], dict)
        certificate_json = envelope["certificate_record_json"]
        certificate_bytes = certificate_json.encode("utf-8")
        self.assertEqual(hashlib.sha256(certificate_bytes).hexdigest(),
                         envelope["certificate_sha256"])
        certificate = P1PoissonStepCertificate.from_json(certificate_json)
        self.assertEqual(certificate.to_json(), certificate_json)

        node = envelope["support"]["node_index"]
        full_state = np.zeros(solver.problem.number_of_cells + 1,
                              dtype=np.float64)
        full_state[1:-1] = before.values
        direction = np.zeros_like(full_state)
        direction[node] = 1.0
        candidate_fingerprint = hashlib.sha256(
            direction[1:-1].tobytes(order="C")).hexdigest()
        expected_element_id = (
            f"p1hat-v1:{solver.problem.problem_fingerprint}:"
            f"{solver.problem.mesh_sha256}:node:{node}")
        self.assertEqual(envelope["element_id"], expected_element_id)
        self.assertEqual(envelope["candidate_fingerprint"],
                         candidate_fingerprint)
        self.assertTrue(verify_p1_poisson_step_certificate(
            certificate, solver.problem.certificate_data(), full_state,
            direction))
        return envelope

    def _assert_p1_candidate_mutation_rejected(self, policy, mutate):
        class WrappedLearner:
            family = "p1-poisson-hat"
            provenance = {"learner": "P1PoissonHatWeakLearner"}

            def __init__(self):
                self.inner = P1PoissonHatWeakLearner()

            def propose(inner_self, problem, state, residual, budget):
                candidate = inner_self.inner.propose(
                    problem, state, residual, budget)
                return mutate(problem, state, candidate)

        problem = P1Poisson1DRitzProblem(_data())
        state = SolverState(np.zeros(problem.dimension, dtype=np.float64))
        solver = CEMSolver(problem, (WrappedLearner(),), RitzCandidateScorer(),
                           policy, state)
        sentinel = object()
        solver._active_candidates["sentinel"] = sentinel
        before = immutable_snapshot(state.snapshot())
        before_hash = state.hash()
        registry_before = dict(solver._active_candidates)

        self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertFalse(record.accepted)
        self.assertEqual(record.status, "proposal_failed")
        self.assertEqual(record.proposal_status, "invalid_proposal")
        self.assertEqual(record.reason_code,
                         "p1_candidate_certificate_binding_mismatch")
        self.assertIsNone(record.element_id)
        self.assertTrue(snapshots_equal(before, state.snapshot()))
        self.assertEqual(state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry_before.keys())
        self.assertTrue(all(solver._active_candidates[key] is value
                            for key, value in registry_before.items()))
        events = {event.event_id: event for event in solver.work_ledger.events}
        self.assertTrue(record.work_event_ids)
        self.assertEqual(len(record.work_event_ids), len(set(record.work_event_ids)))
        self.assertTrue(set(record.work_event_ids).issubset(events))
        self.assertFalse(any(events[event_id].phase == "score"
                             for event_id in record.work_event_ids))
        envelope = self._assert_persisted_p1_audit(
            solver, record, before, "typed_failure")
        self.assertEqual(envelope["proposal_status"], "invalid_proposal")

    def test_p1_direction_tampering_rejected_before_stagewise_or_fc_scoring(self):
        def replace_direction(problem, state, candidate):
            return replace(candidate,
                           correction=np.ones(problem.dimension, dtype=np.float64))

        for policy in (StagewisePolicy(), FullyCorrectivePolicy()):
            with self.subTest(policy=policy.mode):
                self._assert_p1_candidate_mutation_rejected(
                    policy, replace_direction)

    def test_wrong_shape_p1_correction_fails_before_scoring(self):
        def oversized_shape(problem, state, candidate):
            return replace(candidate, correction=np.ones(
                problem.dimension + 17, dtype=np.float64))

        self._assert_p1_candidate_mutation_rejected(
            StagewisePolicy(), oversized_shape)

    def test_oversized_correction_is_rejected_before_fingerprint_copy(self):
        problem = P1Poisson1DRitzProblem(_data())
        state = SolverState(np.zeros(problem.dimension, dtype=np.float64))
        candidate = P1PoissonHatWeakLearner().propose(
            problem, state.snapshot(), problem.residual(state), {})
        oversized = replace(candidate, correction=np.ones(250_000,
                                                          dtype=np.float64))
        with patch("problems.p1_poisson_ritz.hashlib.sha256",
                   side_effect=AssertionError("must reject before hashing correction")):
            self.assertFalse(problem.validate_p1_candidate_binding(
                oversized, state.snapshot()))

    def test_p1_proposer_identity_cannot_be_removed_to_skip_binding(self):
        def remove_marker_and_replace_direction(problem, state, candidate):
            provenance = dict(candidate.provenance)
            provenance.pop("learner")
            return replace(candidate,
                           correction=np.ones(problem.dimension, dtype=np.float64),
                           provenance=provenance)

        self._assert_p1_candidate_mutation_rejected(
            StagewisePolicy(), remove_marker_and_replace_direction)

    def test_p1_support_element_and_certificate_tampering_fail_closed(self):
        def other_node(problem, candidate):
            current = candidate.support["node_index"]
            return 1 if current != 1 else 2

        def mutate_support(problem, state, candidate):
            support = dict(candidate.support)
            node = other_node(problem, candidate)
            support["node_index"] = node
            support["cell_indices"] = (node - 1, node)
            return replace(candidate, support=support)

        def mutate_element_id(problem, state, candidate):
            node = other_node(problem, candidate)
            element_id = candidate.element_id.rsplit(":", 1)[0] + f":node:{node}"
            return replace(candidate, element_id=element_id)

        def mutate_certificate(problem, state, candidate):
            node = other_node(problem, candidate)
            full_state = np.zeros(problem.number_of_cells + 1,
                                  dtype=np.float64)
            full_state[1:-1] = state.values
            direction = np.zeros_like(full_state)
            direction[node] = 1.0
            certificate = certify_p1_poisson_step(
                problem.certificate_data(), full_state, direction)
            provenance = dict(candidate.provenance)
            provenance["certificate_record_json"] = certificate.to_json()
            provenance["certificate_sha256"] = hashlib.sha256(
                certificate.to_json().encode("utf-8")).hexdigest()
            provenance["certificate_status"] = certificate.status
            return replace(candidate, provenance=provenance)

        for name, mutate in (("support", mutate_support),
                             ("element_id", mutate_element_id),
                             ("certificate", mutate_certificate)):
            with self.subTest(tamper=name):
                self._assert_p1_candidate_mutation_rejected(
                    StagewisePolicy(), mutate)

    def test_rehashed_persisted_envelope_tampering_is_rejected(self):
        solver = _solver()
        before = immutable_snapshot(solver.state.snapshot())
        self.assertIsNotNone(solver.step())
        persisted = solver.ledger.to_dicts()[-1]
        original = json.loads(persisted["proposal_audit_json"])
        node = original["support"]["node_index"]
        other = 1 if node != 1 else 2

        changed_support = json.loads(json.dumps(original))
        changed_support["support"]["node_index"] = other
        changed_support["support"]["cell_indices"] = [other - 1, other]

        changed_id = json.loads(json.dumps(original))
        changed_id["element_id"] = (
            changed_id["element_id"].rsplit(":", 1)[0] + f":node:{other}")

        changed_certificate = json.loads(json.dumps(original))
        full_state = np.zeros(solver.problem.number_of_cells + 1,
                              dtype=np.float64)
        full_state[1:-1] = before.values
        direction = np.zeros_like(full_state)
        direction[other] = 1.0
        other_certificate = certify_p1_poisson_step(
            solver.problem.certificate_data(), full_state, direction)
        changed_certificate["certificate_record_json"] = other_certificate.to_json()
        changed_certificate["certificate_sha256"] = hashlib.sha256(
            other_certificate.to_json().encode("utf-8")).hexdigest()
        changed_certificate["certificate_status"] = other_certificate.status

        for changed in (changed_support, changed_id, changed_certificate):
            with self.subTest(tampered_fields=changed):
                encoded = canonical_json_bytes(changed).decode("utf-8")
                digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
                self.assertFalse(verify_proposal_audit(encoded, digest))
                self.assertFalse(verify_proposal_audit(
                    encoded, digest, problem=solver.problem, state=before))

    def test_private_rollback_snapshot_is_bytes_backed_and_unwritable(self):
        source = SolverState(np.array([0.0, 1.0]), (),
                             np.array([0.5], dtype=np.float64)).snapshot()
        baseline = immutable_snapshot(source)
        self.assertTrue(snapshots_equal(source, baseline))
        for array in (baseline.values, baseline.outer_coefficients):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertTrue(snapshots_equal(source, baseline))

    def test_mutated_proposal_snapshot_cannot_corrupt_rollback_baseline(self):
        class MutatingLearner:
            family = "p1-poisson-hat"

            def __init__(self, should_raise):
                self.should_raise = should_raise
                self.solver = None
                self.did_mutate = False

            def propose(self, problem, state, residual, budget):
                state.values.setflags(write=True)
                state.values[0] = 42.0
                self.did_mutate = True
                self.solver._active_candidates["poisoned"] = object()
                if self.should_raise:
                    raise ValueError("injected_after_snapshot_mutation")
                return object()

        for should_raise in (False, True):
            with self.subTest(should_raise=should_raise):
                learner = MutatingLearner(should_raise)
                problem = P1Poisson1DRitzProblem(_data())
                state = SolverState(np.zeros(problem.dimension, dtype=np.float64))
                solver = CEMSolver(problem, (learner,), RitzCandidateScorer(),
                                   StagewisePolicy(), state)
                learner.solver = solver
                sentinel = object()
                solver._active_candidates["sentinel"] = sentinel
                before = immutable_snapshot(state.snapshot())
                before_hash = state.hash()
                registry_before = dict(solver._active_candidates)

                self.assertIsNone(solver.step())
                record = solver.ledger.records[-1]
                self.assertTrue(learner.did_mutate)
                self.assertFalse(record.accepted)
                self.assertTrue(snapshots_equal(before, state.snapshot()))
                self.assertEqual(state.hash(), before_hash)
                self.assertEqual(solver._active_candidates.keys(),
                                 registry_before.keys())
                self.assertIs(solver._active_candidates["sentinel"], sentinel)
                self.assertNotIn("poisoned", solver._active_candidates)
                self.assertEqual(record.reason_code, "proposal_snapshot_mutated")
                if should_raise:
                    self.assertEqual(record.status,
                                     "exception:ValueError:injected_after_snapshot_mutation")
                else:
                    self.assertEqual(record.status,
                                     "exception:RuntimeError:proposal_snapshot_mutated")
                known_ids = {event.event_id for event in solver.work_ledger.events}
                self.assertTrue(record.work_event_ids)
                self.assertEqual(len(record.work_event_ids),
                                 len(set(record.work_event_ids)))
                self.assertTrue(set(record.work_event_ids).issubset(known_ids))

    def test_state_hash_v2_separates_optional_empty_coefficients_both_ways(self):
        self.assertEqual(STATE_HASH_ALGORITHM_VERSION, 2)
        no_coeff = SolverState(np.array([1.0]), (), None)
        empty_coeff = SolverState(np.array([1.0]), (), np.zeros(0, dtype=np.float64))
        self.assertNotEqual(no_coeff.hash(), empty_coeff.hash())
        self.assertFalse(snapshots_equal(no_coeff.snapshot(), empty_coeff.snapshot()))

        before = no_coeff.snapshot()
        no_coeff.outer_coefficients = np.zeros(0, dtype=np.float64)
        self.assertNotEqual(no_coeff.hash(), _hash_snapshot(before))
        self.assertFalse(snapshots_equal(before, no_coeff.snapshot()))

        before = empty_coeff.snapshot()
        empty_coeff.outer_coefficients = None
        self.assertNotEqual(empty_coeff.hash(), _hash_snapshot(before))
        self.assertFalse(snapshots_equal(before, empty_coeff.snapshot()))

    def test_stagewise_accept_persists_replayable_certificate_and_events(self):
        solver = _solver()
        before = solver.state.snapshot()
        score = solver.step()
        self.assertIsNotNone(score)
        record = solver.ledger.records[-1]
        payload = solver.ledger.to_dicts()[-1]
        self.assertTrue(record.accepted)
        self.assertEqual(record.record_schema_version, 2)
        self.assertEqual(record.state_hash_algorithm_version, 2)
        self.assertEqual(record.element_id, solver.state.active_element_ids[-1])
        self.assertIsNotNone(record.proposal_audit_json)
        self.assertEqual(hashlib.sha256(record.proposal_audit_json.encode("utf-8")).hexdigest(),
                         record.proposal_audit_sha256)
        audit = json.loads(record.proposal_audit_json)
        self.assertEqual(audit["element_id"], record.element_id)
        self.assertEqual(audit["problem_fingerprint"], solver.problem.problem_fingerprint)
        cert = P1PoissonStepCertificate.from_json(audit["certificate_record_json"])
        node = audit["parameters"]["node_index"]
        full_state = np.zeros(solver.problem.number_of_cells + 1, dtype=np.float64)
        full_state[1:-1] = before.values
        direction = np.zeros(solver.problem.number_of_cells + 1, dtype=np.float64)
        direction[node] = 1.0
        candidate_fingerprint = hashlib.sha256(
            direction[1:-1].tobytes(order="C")).hexdigest()
        self.assertEqual(audit["candidate_fingerprint"], candidate_fingerprint)
        self.assertTrue(verify_p1_poisson_step_certificate(
            cert, solver.problem.certificate_data(), full_state, direction))
        expected_values = before.values.copy()
        expected_values[node - 1] += score.alpha
        np.testing.assert_array_equal(solver.state.values, expected_values)
        self.assertIn("certificate_record_json", audit)
        event_ids = record.work_event_ids
        known_ids = {event.event_id for event in solver.work_ledger.events}
        self.assertTrue(event_ids)
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertTrue(set(event_ids).issubset(known_ids))
        self.assertTrue(np.array_equal(solver.state.active_element_ids,
                                       (record.element_id,)))
        self.assertNotEqual(before.values.tobytes(), solver.state.values.tobytes())
        self.assertEqual(payload["element_id"], record.element_id)
        self.assertNotIn("proposal-error", _record_as_json(solver))
        matrix, load = _fraction_ritz_data(solver.problem)
        before_objective = _fraction_ritz_objective(
            matrix, load, before.values)
        after_objective = _fraction_ritz_objective(
            matrix, load, solver.state.values)
        self.assertLess(after_objective, before_objective)

        node = json.loads(record.proposal_audit_json)["support"]["node_index"]
        direction = tuple(Fraction(1) if index == node - 1 else Fraction(0)
                          for index in range(len(before.values)))
        old_values = tuple(Fraction.from_float(float(value))
                           for value in before.values)
        delta = tuple(Fraction.from_float(float(new))
                      - Fraction.from_float(float(old))
                      for new, old in zip(solver.state.values, before.values))
        action = tuple(sum(matrix[i][j] * old_values[j]
                           for j in range(len(old_values)))
                       for i in range(len(old_values)))
        correlation = sum(direction[i] * (load[i] - action[i])
                          for i in range(len(direction)))
        curvature = sum(direction[i] * matrix[i][j] * direction[j]
                        for i in range(len(direction))
                        for j in range(len(direction)))
        realized_drop_from_step = (
            sum(correlation * delta[i] for i in range(len(delta)))
            - sum(delta[i] * matrix[i][j] * delta[j]
                  for i in range(len(delta))
                  for j in range(len(delta))) / 2)
        self.assertEqual(before_objective - after_objective,
                         realized_drop_from_step)
        self.assertGreater(realized_drop_from_step, 0)
        exact_line_gain = correlation * correlation / (2 * curvature)
        np.testing.assert_allclose(score.predicted_reduction,
                                   float(exact_line_gain), rtol=1e-15,
                                   atol=1e-15)
        np.testing.assert_allclose(record.before, float(before_objective),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(record.after, float(after_objective),
                                   rtol=0, atol=1e-15)
        # This oracle uses exact dyadic representations of the stored matrix,
        # load, and committed float64 state; it is separate from the continuous
        # P1 directional certificate replay above.
        self.assertTrue(verify_proposal_audit(record.proposal_audit_json,
                                              record.proposal_audit_sha256))
        self.assertFalse(verify_proposal_audit(record.proposal_audit_json + " ",
                                               record.proposal_audit_sha256))
        v2_roundtrip = CorrectionLedger.from_dicts([payload]).to_dicts()[0]
        self.assertEqual(v2_roundtrip["record_schema_version"], 2)
        self.assertEqual(v2_roundtrip["proposal_audit_json"],
                         record.proposal_audit_json)
        self._assert_persisted_p1_audit(
            solver, record, before, "verified_candidate")

    def test_stagewise_policy_rejection_is_transactional_and_audited(self):
        class RejectPolicy(StagewisePolicy):
            def decide(self, score):
                return False

        solver = _solver(RejectPolicy())
        before = solver.state.snapshot()
        before_hash = solver.state.hash()
        registry = dict(solver._active_candidates)
        self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertFalse(record.accepted)
        self.assertEqual(record.status, "rejected")
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry.keys())
        self.assertEqual(record.element_id, json.loads(record.proposal_audit_json)["element_id"])
        self.assertTrue(record.proposal_audit_sha256)
        self._assert_persisted_p1_audit(
            solver, record, before, "verified_candidate")
        known_ids = {event.event_id for event in solver.work_ledger.events}
        self.assertTrue(set(record.work_event_ids).issubset(known_ids))

    def test_scorer_rejection_persists_replayable_p1_audit(self):
        class RejectingScorer:
            objective_geometry = "ritz_energy"

            def __init__(self):
                self.delegate = RitzCandidateScorer()

            def score(self, problem, state, candidate):
                score = self.delegate.score(problem, state, candidate)
                return replace(score, finite=False,
                               reason="injected_scorer_rejection")

        problem = P1Poisson1DRitzProblem(_data())
        state = SolverState(np.zeros(problem.dimension, dtype=np.float64))
        solver = CEMSolver(problem, (P1PoissonHatWeakLearner(),),
                           RejectingScorer(), StagewisePolicy(), state)
        sentinel = object()
        solver._active_candidates["sentinel"] = sentinel
        before = immutable_snapshot(state.snapshot())
        before_hash = state.hash()

        self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertFalse(record.accepted)
        self.assertEqual(record.status, "injected_scorer_rejection")
        self.assertIsNotNone(record.element_id)
        self.assertTrue(snapshots_equal(before, state.snapshot()))
        self.assertEqual(state.hash(), before_hash)
        self.assertEqual(tuple(solver._active_candidates), ("sentinel",))
        self.assertIs(solver._active_candidates["sentinel"], sentinel)
        self._assert_persisted_p1_audit(
            solver, record, before, "verified_candidate")

    def test_solver_records_dictionary_exhaustion_after_active_set_progression(self):
        solver = _solver()
        accepted_ids = []
        for _ in range(solver.problem.number_of_cells - 1):
            active_before = tuple(solver.state.active_element_ids)
            score = solver.step()
            self.assertIsNotNone(score)
            record = solver.ledger.records[-1]
            self.assertTrue(record.accepted)
            self.assertNotIn(record.element_id, active_before)
            accepted_ids.append(record.element_id)
            self.assertEqual(tuple(solver.state.active_element_ids),
                             tuple(accepted_ids))

        before = immutable_snapshot(solver.state.snapshot())
        before_hash = solver.state.hash()
        registry_before = dict(solver._active_candidates)
        self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertEqual(record.status, "proposal_failed")
        self.assertEqual(record.proposal_status, "dictionary_exhausted")
        self.assertEqual(record.reason_code, "all_hats_active")
        self.assertIsNone(record.element_id)
        self.assertFalse(record.accepted)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry_before.keys())
        self.assertTrue(all(solver._active_candidates[key] is value
                            for key, value in registry_before.items()))
        envelope = self._assert_persisted_p1_audit(
            solver, record, before, "typed_failure")
        self.assertEqual(envelope["work_audit"]["inactive_hats"], 0)
        self.assertEqual(envelope["work_audit"]["hat_evaluations"], 0)
        self.assertEqual(envelope["work_audit"]["certificate_replays"], 0)

    def test_typed_empty_dictionary_failure_is_structured_and_has_no_candidate(self):
        data = _data(nodes=(0.0, 1.0), source=(1.0,))
        solver = _solver(data=data)
        before = solver.state.snapshot()
        before_hash = solver.state.hash()
        self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        encoded = _record_as_json(solver)
        self.assertIsNone(record.element_id)
        self.assertFalse(record.accepted)
        self.assertEqual(record.status, "proposal_failed")
        self.assertEqual(record.proposal_status, "empty_dictionary")
        self.assertEqual(record.reason_code, "no_interior_hat")
        self.assertTrue(record.proposal_audit_json)
        audit = json.loads(record.proposal_audit_json)
        self.assertEqual(audit["outcome"], "typed_failure")
        self.assertIsNone(audit["element_id"])
        self.assertNotIn("certificate_record_json", audit)
        self.assertTrue(verify_proposal_audit(record.proposal_audit_json,
                                              record.proposal_audit_sha256))
        self.assertNotIn("proposal-error", encoded)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(before_hash, solver.state.hash())
        self._assert_persisted_p1_audit(
            solver, record, before, "typed_failure")
        known_ids = {event.event_id for event in solver.work_ledger.events}
        self.assertTrue(set(record.work_event_ids).issubset(known_ids))

    def test_fully_corrective_two_step_p1_refit_and_rollback(self):
        solver = _solver(FullyCorrectivePolicy())
        first = solver.step()
        self.assertIsNotNone(first)
        solver.learners = (P1PoissonHatWeakLearner(),)
        second = solver.step()
        self.assertIsNotNone(second)
        self.assertEqual(len(solver.state.active_element_ids), 2)
        self.assertTrue(np.all(np.isfinite(solver.state.outer_coefficients)))
        self.assertLessEqual(solver.problem.objective(solver.state), 0.0)
        nodes = solver.problem.nodes
        cell_source = solver.problem.cell_source
        stiffness = np.zeros((nodes.size, nodes.size), dtype=np.float64)
        load = np.zeros(nodes.size, dtype=np.float64)
        for cell, source in enumerate(cell_source):
            width = nodes[cell + 1] - nodes[cell]
            stiffness[cell, cell] += 1.0 / width
            stiffness[cell, cell + 1] -= 1.0 / width
            stiffness[cell + 1, cell] -= 1.0 / width
            stiffness[cell + 1, cell + 1] += 1.0 / width
            load[cell] += source * width / 2.0
            load[cell + 1] += source * width / 2.0
        accepted_records = [record for record in solver.ledger.records if record.accepted]
        self.assertEqual(len(accepted_records), 2)
        for record in accepted_records:
            self.assertTrue(record.proposal_audit_json)
            self.assertTrue(record.proposal_audit_sha256)
        active_nodes = [json.loads(record.proposal_audit_json)["support"]["node_index"]
                        for record in accepted_records]
        basis = np.zeros((solver.problem.dimension, len(active_nodes)))
        for column, node in enumerate(active_nodes):
            basis[node - 1, column] = 1.0
        interior_stiffness = stiffness[1:-1, 1:-1]
        interior_load = load[1:-1]
        gram = basis.T @ interior_stiffness @ basis
        rhs = basis.T @ interior_load
        expected_coefficients = np.linalg.solve(gram, rhs)
        np.testing.assert_allclose(solver.state.outer_coefficients,
                                   expected_coefficients, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(solver.state.values, basis @ expected_coefficients,
                                   rtol=1e-12, atol=1e-14)

        solver = _solver(FullyCorrectivePolicy())
        before = solver.state.snapshot()
        before_hash = solver.state.hash()
        registry = dict(solver._active_candidates)
        with patch.object(solver, "_fc_trial", side_effect=ValueError("injected_refit_failure")):
            self.assertIsNone(solver.step())
        failed = solver.ledger.records[-1]
        self.assertFalse(failed.accepted)
        self.assertIn("injected_refit_failure", failed.status)
        self.assertTrue(failed.proposal_audit_json)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry.keys())
        known_ids = {event.event_id for event in solver.work_ledger.events}
        self.assertTrue(set(failed.work_event_ids).issubset(known_ids))

    def test_fully_corrective_rank_deficient_trial_rolls_back_completely(self):
        solver = _solver(FullyCorrectivePolicy())
        self.assertIsNotNone(solver.step())
        solver.learners = (P1PoissonHatWeakLearner(),)
        before = immutable_snapshot(solver.state.snapshot())
        before_hash = solver.state.hash()
        registry_before = dict(solver._active_candidates)

        original_fc_trial = solver._fc_trial

        def rank_deficient_trial(active_map):
            trial_map = dict(active_map)
            active_ids = tuple(trial_map)
            self.assertEqual(len(active_ids), 2)
            old_id, new_id = active_ids
            trial_map[old_id] = replace(
                trial_map[old_id], correction=trial_map[new_id].correction)
            return original_fc_trial(trial_map)

        with patch.object(solver, "_fc_trial", side_effect=rank_deficient_trial):
            self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertFalse(record.accepted)
        self.assertIn("rank_deficient_or_ill_conditioned_ritz_active_set",
                      record.status)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry_before.keys())
        self.assertTrue(all(solver._active_candidates[key] is value
                            for key, value in registry_before.items()))
        events = {event.event_id: event for event in solver.work_ledger.events}
        self.assertTrue(record.work_event_ids)
        self.assertEqual(len(record.work_event_ids),
                         len(set(record.work_event_ids)))
        self.assertTrue(set(record.work_event_ids).issubset(events))
        failed_refits = [events[event_id] for event_id in record.work_event_ids
                         if events[event_id].phase == "refit"]
        self.assertEqual(len(failed_refits), 1)
        self.assertEqual(failed_refits[0].status, "failed")

    def test_oversize_audit_uses_bounded_terminal_failure_record(self):
        solver = _solver()
        before = solver.state.snapshot()
        before_hash = solver.state.hash()
        with patch("solver.MAX_PROPOSAL_AUDIT_BYTES", 32):
            self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertEqual(record.status, "proposal_failed")
        self.assertEqual(record.reason_code, "proposal_audit_oversize")
        self.assertIsNone(record.element_id)
        self.assertIsNone(record.proposal_audit_json)
        self.assertIsNone(record.proposal_audit_sha256)
        self.assertIsNone(record.proposal_status)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        encoded = json.dumps(solver.ledger.to_dicts()[-1], sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(encoded), 16_384)
        self.assertLessEqual(len(record.work_event_ids), 32)
        self.assertTrue(all(len(item.encode("utf-8")) <= 128
                            for item in record.work_event_ids))
        known_ids = {event.event_id for event in solver.work_ledger.events}
        self.assertTrue(set(record.work_event_ids).issubset(known_ids))

    def test_oversize_certificate_uses_bounded_terminal_failure_record(self):
        solver = _solver()
        before = immutable_snapshot(solver.state.snapshot())
        before_hash = solver.state.hash()
        sentinel = object()
        solver._active_candidates["sentinel"] = sentinel
        registry_before = dict(solver._active_candidates)

        with patch("solver.MAX_PROPOSAL_CERTIFICATE_BYTES", 1):
            self.assertIsNone(solver.step())

        record = solver.ledger.records[-1]
        self.assertEqual(record.status, "proposal_failed")
        self.assertEqual(record.reason_code, "proposal_audit_oversize")
        self.assertIsNone(record.element_id)
        self.assertIsNone(record.proposal_audit_json)
        self.assertIsNone(record.proposal_audit_sha256)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertEqual(solver.state.hash(), before_hash)
        self.assertEqual(solver._active_candidates.keys(), registry_before.keys())
        self.assertTrue(all(solver._active_candidates[key] is value
                            for key, value in registry_before.items()))
        encoded = canonical_json_bytes(solver.ledger.to_dicts()[-1])
        self.assertLessEqual(len(encoded), 16_384)
        self.assertLessEqual(len(record.work_event_ids), 32)
        self.assertTrue(all(len(item.encode("utf-8")) <= 128
                            for item in record.work_event_ids))
        events = {event.event_id: event for event in solver.work_ledger.events}
        self.assertEqual(len(record.work_event_ids), len(set(record.work_event_ids)))
        self.assertTrue(set(record.work_event_ids).issubset(events))
        self.assertFalse(any(events[event_id].phase == "score"
                             for event_id in record.work_event_ids))

    def test_typed_failure_audit_overflow_uses_same_bounded_fallback(self):
        data = _data(nodes=(0.0, 1.0), source=(1.0,))
        solver = _solver(data=data)
        before = solver.state.snapshot()
        with patch("solver.MAX_PROPOSAL_AUDIT_BYTES", 1):
            self.assertIsNone(solver.step())
        record = solver.ledger.records[-1]
        self.assertEqual(record.reason_code, "proposal_audit_oversize")
        self.assertIsNone(record.proposal_status)
        self.assertIsNone(record.element_id)
        self.assertIsNone(record.proposal_audit_json)
        self.assertIsNone(record.proposal_audit_sha256)
        self.assertTrue(snapshots_equal(before, solver.state.snapshot()))
        self.assertLessEqual(len(json.dumps(solver.ledger.to_dicts()[-1],
                                           sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")), 16_384)

    def test_ledger_v1_record_round_trips_with_nullable_v2_fields(self):
        solver = _solver()
        self.assertIsNotNone(solver.step())
        current = solver.ledger.to_dicts()[-1]
        legacy_keys = {
            "element_id", "family", "support", "parameters", "proposal_cost",
            "scoring_cost", "predicted_reduction", "before", "after",
            "accepted", "status", "state_hash_before", "state_hash_after",
            "touched_element_ids", "frozen_element_ids", "work_event_ids",
        }
        legacy = {key: current[key] for key in legacy_keys}
        loaded = CorrectionLedger.from_dicts([legacy])
        restored = loaded.to_dicts()[0]
        self.assertEqual(loaded.records[0].record_schema_version, 1)
        self.assertEqual(loaded.records[0].state_hash_algorithm_version, 1)
        self.assertEqual(loaded.records[0].proposal_audit_json, None)
        self.assertEqual(restored["element_id"], legacy["element_id"])
        self.assertEqual(restored["state_hash_before"], legacy["state_hash_before"])
        self.assertEqual(restored["proposal_audit_json"], None)

    def test_ledger_v1_pool_audit_round_trips_with_legacy_fields(self):
        original = PoolAuditRecord(
            pool_id="pool-v1-fixture",
            rule="BestPredictedGainSelector",
            pool_size=2,
            candidates=({"candidate_hash": "candidate-a", "status": "scored"},
                        {"candidate_hash": "candidate-b", "status": "scored"}),
            pool_max_gain=1.25,
            selected_gain=1.0,
            finite_pool_realized_ratio=0.8,
            ratio_reason="finite_pool",
            accepted=True,
            status="accepted",
            proposal_costs=({"fraction_ops": 12}, {"fraction_ops": 15}),
            scoring_statuses=({"status": "scored_valid"},
                              {"status": "scored_valid"}),
            before=-1.0,
            after=-2.0,
            state_hash_before="legacy-before-hash",
            state_hash_after="legacy-after-hash",
            work_event_ids=("event:1", "event:2"),
            generation_metadata={"source": "legacy_fixture"},
            budget={"max_candidates": 2},
            selector_identity="tests.LegacySelector",
            selector_rule="stable_best_gain",
        )
        source = CorrectionLedger()
        source.append(original)
        legacy = source.to_dicts()[0]
        legacy.pop("record_schema_version")
        legacy.pop("state_hash_algorithm_version")

        loaded = CorrectionLedger.from_dicts([legacy])
        record = loaded.records[0]
        self.assertIsInstance(record, PoolAuditRecord)
        self.assertEqual(record.record_schema_version, 1)
        self.assertEqual(record.state_hash_algorithm_version, 1)
        restored = loaded.to_dicts()[0]
        expected = dict(legacy)
        expected["record_schema_version"] = 1
        expected["state_hash_algorithm_version"] = 1
        self.assertEqual(restored, expected)
        for key, value in legacy.items():
            self.assertEqual(restored[key], value)
        self.assertEqual(CorrectionLedger.from_dicts([restored]).to_dicts(),
                         [restored])

        malformed_v2 = dict(restored)
        malformed_v2["record_schema_version"] = 2
        malformed_v2["state_hash_algorithm_version"] = 99
        with self.assertRaisesRegex(ValueError, "invalid state hash version"):
            CorrectionLedger.from_dicts([malformed_v2])

    def test_only_nominal_and_complete_ritz_adapters_pass_type_gate(self):
        class IncompleteRitz(RitzEnergyProblem):
            objective_geometry = "ritz_energy"

        with self.assertRaises(TypeError):
            CEMSolver(IncompleteRitz(), (), RitzCandidateScorer(),
                      StagewisePolicy(), SolverState(np.zeros(1)))
        class StructuralImpostor:
            objective_geometry = "ritz_energy"
            shape = (1, 1)
            source = np.ones(1)
            operator = None

        with self.assertRaises(TypeError):
            CEMSolver(StructuralImpostor(), (), RitzCandidateScorer(),
                      StagewisePolicy(), SolverState(np.zeros(1)))


def _hash_snapshot(snapshot):
    state = SolverState(snapshot.values, snapshot.active_element_ids,
                        snapshot.outer_coefficients)
    return state.hash()


if __name__ == "__main__":
    unittest.main()
