from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
import numpy as np
from core.state import (SolverState, StateSnapshot, StateTransaction,
                        STATE_HASH_ALGORITHM_VERSION, immutable_snapshot,
                        snapshots_equal)
from core.ledger import (CorrectionLedger, CorrectionRecord, PoolAuditRecord,
                         MAX_PROPOSAL_CERTIFICATE_BYTES, MAX_PROPOSAL_AUDIT_BYTES,
                         MAX_TERMINAL_AUDIT_BYTES, MAX_ATTEMPT_WORK_EVENT_IDS,
                         MAX_WORK_EVENT_ID_BYTES, ProposalAuditTooLarge,
                         canonical_json_bytes, correction_record_dict,
                         verify_proposal_audit)
from problems.problem import LinearPDEProblem, RitzEnergyProblem
from enrichment.pool import CandidatePool, PoolCandidateResult, candidate_hash, safe_gain, freeze_metadata
from enrichment.learner import ProposalFailure
from core.cost import WorkLedger


def _jsonable(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def _proposal_audit_envelope(candidate=None, failure=None, problem=None,
                             state=None):
    if candidate is not None:
        provenance = getattr(candidate, "provenance", {})
        if provenance.get("learner") != "P1PoissonHatWeakLearner":
            return None
        validator = getattr(problem, "validate_p1_candidate_binding", None)
        if (state is None or not callable(validator)
                or not validator(candidate, state)):
            raise ValueError("p1_candidate_certificate_binding_invalid")
        certificate_json = provenance.get("certificate_record_json")
        certificate_sha256 = provenance.get("certificate_sha256")
        if not isinstance(certificate_json, str) or not isinstance(certificate_sha256, str):
            raise ValueError("P1 candidate lacks replayable certificate provenance")
        certificate_bytes = certificate_json.encode("utf-8")
        if len(certificate_bytes) > MAX_PROPOSAL_CERTIFICATE_BYTES:
            raise ProposalAuditTooLarge(len(certificate_bytes))
        actual_certificate_sha256 = hashlib.sha256(certificate_bytes).hexdigest()
        if actual_certificate_sha256 != certificate_sha256:
            raise ValueError("P1 certificate digest mismatch")
        envelope = {
            "schema": "cem.p1_proposal_audit.v1",
            "outcome": "verified_candidate",
            "element_id": candidate.element_id,
            "problem_fingerprint": provenance.get("problem_fingerprint"),
            "candidate_fingerprint": candidate.fingerprint(),
            "support": _jsonable(candidate.support),
            "parameters": _jsonable(candidate.parameters),
            "certificate_status": provenance.get("certificate_status"),
            "certificate_record_json": certificate_json,
            "certificate_sha256": certificate_sha256,
            "work_audit": _jsonable(provenance.get("work_audit")),
        }
    elif failure is not None:
        envelope = {
            "schema": "cem.p1_proposal_audit.v1",
            "outcome": "typed_failure",
            "element_id": None,
            "problem_fingerprint": getattr(problem, "problem_fingerprint", None),
            "proposal_status": failure.status,
            "reason_code": failure.reason_code,
            "work_audit": _jsonable(failure.work_audit),
        }
    else:
        return None
    encoded = canonical_json_bytes(envelope)
    if len(encoded) > MAX_PROPOSAL_AUDIT_BYTES:
        raise ProposalAuditTooLarge(len(encoded))
    serialized = encoded.decode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if not verify_proposal_audit(serialized, digest, problem=problem,
                                 state=state if candidate is not None else None):
        raise ValueError("proposal audit failed canonical replay verification")
    return serialized, digest


def _snapshot_values_equal(left: StateSnapshot, right: StateSnapshot) -> bool:
    return snapshots_equal(left, right)

@dataclass
class CEMSolver:
    problem: object
    learners: tuple
    scorer: object
    policy: object
    state: SolverState
    ledger: CorrectionLedger | None = None
    work_ledger: WorkLedger | None = None
    def __post_init__(self):
        self.learners = tuple(self.learners); self.ledger = self.ledger or CorrectionLedger(); self.work_ledger = self.work_ledger or WorkLedger()
        problem_geometry=getattr(self.problem, "objective_geometry", None); scorer_geometry=getattr(self.scorer, "objective_geometry", None)
        if problem_geometry == "ritz_energy":
            if scorer_geometry != "ritz_energy" or not isinstance(self.problem, RitzEnergyProblem):
                raise TypeError("RitzEnergyProblem requires RitzCandidateScorer and nominal RitzEnergyProblem inheritance")
            self.problem.validate_contract()
        if scorer_geometry == "ritz_energy" and not isinstance(self.problem, RitzEnergyProblem):
            raise TypeError("RitzCandidateScorer requires a nominal RitzEnergyProblem")
        if callable(getattr(self.problem, "attach_work_ledger", None)):
            self.problem.attach_work_ledger(self.work_ledger)
        elif hasattr(self.problem, "work_ledger"):
            try: self.problem.work_ledger = self.work_ledger
            except (AttributeError, TypeError): pass
        mode = getattr(self.policy, "mode", "stagewise")
        if mode not in ("stagewise", "fully_corrective"):
            raise ValueError("policy.mode must be 'stagewise' or 'fully_corrective'")
        if hasattr(self.problem, "validate_state"): self.problem.validate_state(self.state)
        if mode == "fully_corrective" and not isinstance(self.problem, (LinearPDEProblem, RitzEnergyProblem)):
            raise TypeError("FullyCorrectivePolicy requires a supported linear or Ritz problem")
        if mode in ("stagewise", "fully_corrective"):
            coeff = self.state.outer_coefficients
            if self.state.active_element_ids or (coeff is not None and np.asarray(coeff).size):
                raise ValueError("solver restart requires active_element_ids=() and outer_coefficients=None or empty; checkpoint registry restore is not implemented")
        self._base_values = self.state.values.copy(); self._active_candidates = {}

    def _record(self, candidate, before, score, accepted, status, touched=(), frozen=(), work_event_ids=(), *,
                proposal_status=None, reason_code=None, proposal_audit_json=None,
                proposal_audit_sha256=None):
        eid = None if candidate is None else getattr(candidate, "element_id", None)
        family = None if candidate is None else getattr(candidate, "family", None)
        support = None if candidate is None else getattr(candidate, "support", {})
        params = None if candidate is None else getattr(candidate, "parameters", {})
        cost = None if candidate is None else getattr(candidate, "proposal_cost", {})
        scoring_cost = None if candidate is None else (
            {"estimated_operator_evaluations": 1,
             "estimated_weighted_inner_products": 3}
            if np.isfinite(getattr(score, "q", np.nan))
            else {"estimated_operator_evaluations": 0,
                  "estimated_weighted_inner_products": 0})
        record = CorrectionRecord(
            eid, family, support, params, cost, scoring_cost,
            getattr(score, "predicted_reduction", np.nan),
            getattr(score, "before", np.nan), getattr(score, "after", np.nan),
            accepted, status, before, self.state.hash(),
            tuple(touched) if candidate is not None else (),
            tuple(frozen) if candidate is not None else (),
            tuple(dict.fromkeys(work_event_ids)),
            record_schema_version=2,
            state_hash_algorithm_version=STATE_HASH_ALGORITHM_VERSION,
            proposal_status=proposal_status,
            reason_code=reason_code,
            proposal_audit_json=proposal_audit_json,
            proposal_audit_sha256=proposal_audit_sha256,
        )
        if proposal_audit_json is not None and not verify_proposal_audit(
                proposal_audit_json, proposal_audit_sha256):
            raise ValueError("refusing to append an unverifiable proposal audit")
        if reason_code == "proposal_audit_oversize":
            event_ids = record.work_event_ids
            if (len(event_ids) > MAX_ATTEMPT_WORK_EVENT_IDS
                    or any(len(item.encode("utf-8")) > MAX_WORK_EVENT_ID_BYTES
                           for item in event_ids)):
                raise RuntimeError("oversize fallback event-reference bound exceeded")
            if len(canonical_json_bytes(correction_record_dict(record))) > MAX_TERMINAL_AUDIT_BYTES:
                raise RuntimeError("oversize fallback record bound exceeded")
        self.ledger.append(record)

    def _state_matches_snapshot(self, snapshot: StateSnapshot,
                                expected_hash: str) -> bool:
        try:
            return (self.state.hash() == expected_hash
                    and snapshots_equal(self.state.snapshot(), snapshot))
        except Exception:
            return False

    def _restore_pre_step_state(self, snapshot: StateSnapshot,
                                expected_hash: str,
                                registry: dict[str, object]) -> bool:
        unchanged = self._state_matches_snapshot(snapshot, expected_hash)
        if not unchanged:
            self.state.restore(snapshot)
        registry_unchanged = (self._active_candidates.keys() == registry.keys()
                              and all(self._active_candidates[key] is registry[key]
                                      for key in registry))
        if not registry_unchanged:
            self._active_candidates = dict(registry)
            unchanged = False
        return unchanged

    def _fc_trial(self, active_map):
        ids = tuple(active_map)
        if isinstance(self.problem, RitzEnergyProblem):
            corrections = np.column_stack([active_map[i].correction for i in ids]) if ids else np.zeros((self.state.values.size, 0))
            try:
                actions=np.column_stack([self.problem.apply_operator(corrections[:,j]) for j in range(corrections.shape[1])]) if ids else np.zeros_like(corrections)
                base_action=self.problem.apply_operator(self._base_values)
                G=corrections.T @ actions; rhs=corrections.T @ (self.problem.source-base_action)
                if ids:
                    eig=np.linalg.eigvalsh(G); scale=max(1.0, float(np.max(eig)))
                    if not np.all(np.isfinite(G)) or not np.allclose(G,G.T,rtol=1e-10,atol=1e-12) or not np.all(np.isfinite(eig)) or float(np.min(eig)) <= 1e-12*scale: raise ValueError("rank_deficient_or_ill_conditioned_ritz_active_set")
                    coeff=np.linalg.solve(G, rhs)
                else: coeff=np.zeros(0)
                values=self._base_values+corrections@coeff
                if not np.all(np.isfinite(values)) or not self.problem.admissible(values): raise ValueError("invalid fully-corrective Ritz trial state")
            except Exception:
                self._last_refit_event_id=self.work_ledger.record_call("refit", "refit_calls", solver_step_id=getattr(self, "_current_step_id", None), candidate_attempt_id=getattr(self, "_current_attempt_id", None), status="failed"); raise
            self._last_refit_event_id=self.work_ledger.record_call("refit", "refit_calls", solver_step_id=getattr(self, "_current_step_id", None), candidate_attempt_id=getattr(self, "_current_attempt_id", None)); return values, coeff, ids
        images = [active_map[i].operator_image for i in ids]
        if any(x is None for x in images): raise ValueError("fully corrective candidates require operator images")
        try:
            B = np.column_stack(images); rhs = self.problem.source - self.problem.apply_operator(self._base_values)
            W = np.sqrt(self.problem.weights)[:, None]; coeff = np.linalg.lstsq(W * B, np.sqrt(self.problem.weights) * rhs, rcond=None)[0]
            corrections = np.column_stack([active_map[i].correction for i in ids])
            values = self._base_values + corrections @ coeff
            if not np.all(np.isfinite(values)) or not self.problem.admissible(values): raise ValueError("invalid fully-corrective trial state")
        except Exception:
            self._last_refit_event_id=self.work_ledger.record_call("refit", "refit_calls", solver_step_id=getattr(self, "_current_step_id", None), candidate_attempt_id=getattr(self, "_current_attempt_id", None), status="failed")
            raise
        self._last_refit_event_id=self.work_ledger.record_call("refit", "refit_calls", solver_step_id=getattr(self, "_current_step_id", None), candidate_attempt_id=getattr(self, "_current_attempt_id", None))
        return values, coeff, ids

    def step(self, budget=None):
        budget = {} if budget is None else dict(budget)
        step_id = self.work_ledger.new_step()
        self._current_step_id = step_id
        step_work_ids = []
        step_snapshot = immutable_snapshot(self.state.snapshot())
        step_hash = self.state.hash()
        try:
            with self.work_ledger.span("residual", solver_step_id=step_id,
                                       event_ids=step_work_ids):
                with self.work_ledger.context(solver_step_id=step_id,
                                              event_ids=step_work_ids):
                    residual = np.asarray(self.problem.residual(self.state), dtype=float)
            if residual.ndim != 1 or not np.all(np.isfinite(residual)):
                raise ValueError("problem residual must be a finite one-dimensional array")
            if not self._state_matches_snapshot(step_snapshot, step_hash):
                self.state.restore(step_snapshot)
                raise RuntimeError("state_changed_during_residual")
        except Exception as exc:
            if not self._state_matches_snapshot(step_snapshot, step_hash):
                self.state.restore(step_snapshot)
            failure = type("Failure", (), {"q": np.nan, "predicted_reduction": np.nan,
                                           "before": np.nan, "after": np.nan})()
            self._record(None, step_hash, failure, False,
                         f"invalid_residual:{type(exc).__name__}:{exc}", (),
                         self.state.active_element_ids, step_work_ids)
            return None

        for learner in self.learners:
            # Keep the private rollback baseline out of learner reach.  The
            # proposal receives a distinct, ordinary snapshot copy so even a
            # hostile setflags(write=True) can only alter its own input.
            before_snapshot = immutable_snapshot(self.state.snapshot())
            before_hash = self.state.hash()
            proposal_snapshot = self.state.snapshot()
            candidate = None
            audit_json = audit_sha256 = None
            proposal_status = reason_code = None
            registry_before = dict(self._active_candidates)
            attempt_id = self.work_ledger.new_attempt()
            self._current_attempt_id = attempt_id
            p1_lifecycle = getattr(learner, "family", None) == "p1-poisson-hat"
            work_ids = list(step_work_ids) if p1_lifecycle else []
            wall_id = None
            try:
                try:
                    with self.work_ledger.span(
                            "proposal", solver_step_id=step_id,
                            candidate_attempt_id=attempt_id,
                            event_ids=work_ids) as wall_id:
                        with self.work_ledger.context(solver_step_id=step_id,
                                                      candidate_attempt_id=attempt_id,
                                                      event_ids=work_ids):
                            candidate = learner.propose(
                                self.problem, proposal_snapshot, residual.copy(), budget)
                    proposal_event = self.work_ledger.record_call(
                        "proposal", "proposal_calls", solver_step_id=step_id,
                        candidate_id=getattr(candidate, "element_id", None),
                        candidate_attempt_id=attempt_id)
                    work_ids.extend((wall_id, proposal_event))
                except Exception:
                    failed_event = self.work_ledger.record_call(
                        "proposal", "proposal_calls", solver_step_id=step_id,
                        candidate_id=getattr(candidate, "element_id", None),
                        candidate_attempt_id=attempt_id, status="failed")
                    work_ids.extend(((wall_id,) if wall_id is not None else ()) +
                                    (failed_event,))
                    raise

                if not self._state_matches_snapshot(before_snapshot, before_hash):
                    self.state.restore(before_snapshot)
                    raise RuntimeError("state_changed_during_proposal")
                if not snapshots_equal(proposal_snapshot, before_snapshot):
                    raise RuntimeError("proposal_snapshot_mutated")
                if candidate.element_id in self.state.active_element_ids:
                    raise ValueError("duplicate element_id")
                provenance = getattr(candidate, "provenance", {})
                candidate_claims_p1 = (isinstance(provenance, Mapping)
                                       and provenance.get("learner")
                                       == "P1PoissonHatWeakLearner")
                if p1_lifecycle or candidate_claims_p1:
                    validator = getattr(self.problem,
                                        "validate_p1_candidate_binding", None)
                    if (not p1_lifecycle
                            or not candidate_claims_p1
                            or not callable(validator)
                            or not validator(candidate, before_snapshot)):
                        raise ProposalFailure(
                            "invalid_proposal",
                            "p1_candidate_certificate_binding_mismatch",
                            {"validation": "candidate_certificate_binding"})
                audit = _proposal_audit_envelope(
                    candidate=candidate, problem=self.problem,
                    state=before_snapshot)
                if audit is not None:
                    audit_json, audit_sha256 = audit
                    proposal_status = candidate.provenance.get("certificate_status")

                try:
                    with self.work_ledger.span(
                            "score", solver_step_id=step_id,
                            candidate_id=candidate.element_id,
                            candidate_attempt_id=attempt_id,
                            event_ids=work_ids) as wall_id:
                        with self.work_ledger.context(solver_step_id=step_id,
                                                      candidate_id=candidate.element_id,
                                                      candidate_attempt_id=attempt_id,
                                                      event_ids=work_ids):
                            score = self.scorer.score(self.problem, self.state, candidate)
                    score_event = self.work_ledger.record_call(
                        "score", "score_eval_calls", solver_step_id=step_id,
                        candidate_id=candidate.element_id,
                        candidate_attempt_id=attempt_id)
                    work_ids.extend((wall_id, score_event))
                except Exception:
                    failed_event = self.work_ledger.record_call(
                        "score", "score_eval_calls", solver_step_id=step_id,
                        candidate_id=candidate.element_id,
                        candidate_attempt_id=attempt_id, status="failed")
                    work_ids.extend(((wall_id,) if wall_id is not None else ()) +
                                    (failed_event,))
                    raise

                if not self._state_matches_snapshot(before_snapshot, before_hash):
                    self.state.restore(before_snapshot)
                    raise RuntimeError("state_changed_during_scoring")
                accepted = bool(self.policy.decide(score))
                if not self._state_matches_snapshot(before_snapshot, before_hash):
                    self.state.restore(before_snapshot)
                    raise RuntimeError("state_changed_during_policy")
                if accepted:
                    with StateTransaction(self.state) as tx:
                        if getattr(self.policy, "mode", "stagewise") == "fully_corrective":
                            if not hasattr(self.problem, "operator"):
                                raise TypeError("FullyCorrectivePolicy requires a linear/Ritz operator")
                            trial_map = dict(self._active_candidates)
                            trial_map[candidate.element_id] = candidate
                            self._last_refit_event_id = None
                            with self.work_ledger.context(solver_step_id=step_id,
                                                          candidate_attempt_id=attempt_id,
                                                          event_ids=work_ids):
                                values, coeff, ids = self._fc_trial(trial_map)
                            work_ids.append(self._last_refit_event_id)
                            with self.work_ledger.context(solver_step_id=step_id,
                                                          candidate_attempt_id=attempt_id,
                                                          event_ids=work_ids):
                                before_obj = self.problem.objective(self.state)
                                trial_state = SolverState(values, ids, coeff)
                                trial_obj = self.problem.objective(trial_state)
                            if trial_obj > before_obj + 1e-12:
                                raise ValueError("fully-corrective trial did not decrease objective")
                            self.state.values = values
                            self.state.active_element_ids = ids
                            self.state.outer_coefficients = coeff
                            self._active_candidates = trial_map
                        else:
                            self.state.values = self.state.values + score.alpha * candidate.correction
                            self.state.active_element_ids = (*self.state.active_element_ids,
                                                             candidate.element_id)
                            self.state.outer_coefficients = np.asarray(
                                (*(() if self.state.outer_coefficients is None
                                   else self.state.outer_coefficients), score.alpha), float)
                        tx.commit()
                else:
                    if not self._state_matches_snapshot(before_snapshot, before_hash):
                        self.state.restore(before_snapshot)
                        raise RuntimeError("state_changed_on_rejection")

                status = "accepted" if accepted else score.reason or "rejected"
                self._record(
                    candidate, before_hash, score, accepted, status,
                    (candidate.element_id,) if accepted else (),
                    self.state.active_element_ids[:-1] if accepted
                    and getattr(self.policy, "mode", "stagewise") != "fully_corrective"
                    else tuple(item for item in self.state.active_element_ids
                               if item != candidate.element_id),
                    work_ids, proposal_status=proposal_status,
                    reason_code=(score.reason or None),
                    proposal_audit_json=audit_json,
                    proposal_audit_sha256=audit_sha256)
                if accepted:
                    return score
            except ProposalAuditTooLarge:
                self._restore_pre_step_state(before_snapshot, before_hash,
                                             registry_before)
                failure = type("Failure", (), {"q": np.nan,
                                               "predicted_reduction": np.nan,
                                               "before": np.nan,
                                               "after": np.nan})()
                self._record(None, before_hash, failure, False, "proposal_failed",
                             (), self.state.active_element_ids, work_ids,
                             reason_code="proposal_audit_oversize")
                return None
            except Exception as exc:
                if getattr(self, "_last_refit_event_id", None) and self._last_refit_event_id not in work_ids:
                    work_ids.append(self._last_refit_event_id)
                state_unchanged = self._restore_pre_step_state(
                    before_snapshot, before_hash, registry_before)
                proposal_snapshot_unchanged = snapshots_equal(
                    proposal_snapshot, before_snapshot)
                failure = type("Failure", (), {"q": np.nan,
                                               "predicted_reduction": np.nan,
                                               "before": np.nan,
                                               "after": np.nan})()
                if isinstance(exc, ProposalFailure):
                    try:
                        audit = _proposal_audit_envelope(failure=exc,
                                                         problem=self.problem)
                    except ProposalAuditTooLarge:
                        self._record(None, before_hash, failure, False,
                                     "proposal_failed", (),
                                     self.state.active_element_ids, work_ids,
                                     reason_code="proposal_audit_oversize")
                        return None
                    audit_json, audit_sha256 = audit
                    self._record(
                        None, before_hash, failure, False, "proposal_failed", (),
                        self.state.active_element_ids, work_ids,
                        proposal_status=exc.status,
                        reason_code=(
                            exc.reason_code
                            if state_unchanged and proposal_snapshot_unchanged
                            else ("proposal_snapshot_mutated"
                                  if not proposal_snapshot_unchanged
                                  else "state_changed_during_proposal")),
                        proposal_audit_json=audit_json,
                        proposal_audit_sha256=audit_sha256)
                else:
                    self._record(
                        candidate, before_hash, failure, False,
                        f"exception:{type(exc).__name__}:{exc}", (),
                        self.state.active_element_ids, work_ids,
                        proposal_status=proposal_status,
                        reason_code=(
                            None if state_unchanged and proposal_snapshot_unchanged
                            else ("proposal_snapshot_mutated"
                                  if not proposal_snapshot_unchanged
                                  else "state_changed_during_attempt")),
                        proposal_audit_json=audit_json,
                        proposal_audit_sha256=audit_sha256)
        return None

    def step_pool(self, selector, pool=None, budget=None):
        """Opt-in finite-pool path; the default ``step`` remains ordered-first-accept."""
        if selector is None or not hasattr(selector, "select"): raise TypeError("selector must expose select(pool, scored_candidates)")
        budget = {} if budget is None else dict(budget); budget_snapshot=freeze_metadata(budget, "budget"); before_hash = self.state.hash(); step_id=self.work_ledger.new_step(); selection_id=self.work_ledger.new_selection(); self._current_step_id=step_id; work_ids=[]
        selector_identity=f"{type(selector).__module__}.{type(selector).__qualname__}"; selector_rule=str(getattr(selector, "name", type(selector).__name__))
        before_obj = np.nan
        try:
            with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, event_ids=work_ids): before_obj = self.problem.objective(self.state)
        except Exception as exc:
            empty=CandidatePool((), {"source":"objective_failure", "state_hash": before_hash}, (), self.state.active_element_ids)
            self.ledger.append(PoolAuditRecord(empty.pool_id, "none", 0, (), np.nan, np.nan, None, "objective_failed", False, f"objective_failed:{type(exc).__name__}:{exc}", (), (), before_obj, before_obj, before_hash, self.state.hash(), tuple(dict.fromkeys(work_ids)), empty.generation_metadata, budget_snapshot, selector_identity, selector_rule))
            return None
        try:
            with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, event_ids=work_ids): residual = np.asarray(self.problem.residual(self.state), dtype=float)
        except Exception as exc: residual = np.array([np.nan]); residual_error=f"{type(exc).__name__}:{exc}"
        else: residual_error="" if residual.ndim == 1 and np.all(np.isfinite(residual)) else "problem residual must be a finite one-dimensional array"
        if residual_error:
            empty=CandidatePool((), {"source":"residual_failure", "state_hash": before_hash}, ({"status":"invalid_residual", "error":residual_error},), self.state.active_element_ids)
            self.ledger.append(PoolAuditRecord(empty.pool_id, "none", 0, (), np.nan, np.nan, None, "invalid_residual", False, "invalid_residual", (), (), before_obj, before_obj, before_hash, self.state.hash(), tuple(dict.fromkeys(work_ids)), empty.generation_metadata, budget_snapshot, selector_identity, selector_rule))
            return None
        proposal_costs=[]; proposal_failures=[]; attempt_ids={}
        if pool is None:
            snap=self.state.snapshot(); candidates=[]
            for index, learner in enumerate(self.learners):
                candidate=None; proposal_attempt=self.work_ledger.new_attempt(); proposal_succeeded=False; wall_id=None
                try:
                    with self.work_ledger.span("proposal", solver_step_id=step_id, selection_id=selection_id, candidate_attempt_id=proposal_attempt, event_ids=work_ids) as wall_id:
                        with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, candidate_attempt_id=proposal_attempt, event_ids=work_ids): candidate=learner.propose(self.problem, snap, residual.copy(), budget)
                    proposal_succeeded=True
                    proposal_event = self.work_ledger.record_call("proposal", "proposal_calls", solver_step_id=step_id, selection_id=selection_id, candidate_id=getattr(candidate, "element_id", None), candidate_attempt_id=proposal_attempt)
                    work_ids.extend((wall_id, proposal_event))
                    if candidate.element_id in self.state.active_element_ids: raise ValueError("duplicate element_id")
                    candidate_hash(candidate)
                    attempt_ids[candidate.element_id]=proposal_attempt
                    candidates.append(candidate); proposal_costs.append(getattr(candidate, "proposal_cost", {}))
                except Exception as exc:
                    if not proposal_succeeded:
                        work_ids.extend(((wall_id,) if wall_id is not None else ()) + (self.work_ledger.record_call("proposal", "proposal_calls", solver_step_id=step_id, selection_id=selection_id, candidate_id=getattr(candidate, "element_id", None), candidate_attempt_id=proposal_attempt, status="failed"),))
                    proposal_failures.append({"learner_index": index, "status": "candidate_validation_failed" if candidate is not None else "proposal_failed", "error": f"{type(exc).__name__}:{exc}"})
            pool=CandidatePool(tuple(candidates), {"source":"auto_learners", "learner_count": len(self.learners), "state_hash": before_hash}, tuple(proposal_failures), self.state.active_element_ids)
        else:
            if set(pool.active_element_ids) != set(self.state.active_element_ids): raise ValueError("pool active_element_ids do not match solver state")
            bound_hash=pool.generation_metadata.get("state_hash") if hasattr(pool.generation_metadata, "get") else None
            if bound_hash is not None and bound_hash != before_hash: raise ValueError("pool is bound to a different solver state hash")
            proposal_costs=[getattr(c, "proposal_cost", {}) for c in pool.candidates]; attempt_ids={c.element_id:self.work_ledger.new_attempt() for c in pool.candidates}
        results=[PoolCandidateResult(None, None, x.get("status", "proposal_failed"), "not_scored", x.get("error", "")) for x in pool.proposal_failures]
        for candidate in pool.candidates:
            try:
                with self.work_ledger.span("score", solver_step_id=step_id, selection_id=selection_id, candidate_id=candidate.element_id, candidate_attempt_id=attempt_ids.get(candidate.element_id), event_ids=work_ids) as wall_id:
                    with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, candidate_id=candidate.element_id, candidate_attempt_id=attempt_ids.get(candidate.element_id), event_ids=work_ids): score=self.scorer.score(self.problem, self.state, candidate)
                score_event=self.work_ledger.record_call("score", "score_eval_calls", solver_step_id=step_id, selection_id=selection_id, candidate_id=candidate.element_id, candidate_attempt_id=attempt_ids.get(candidate.element_id)); work_ids.extend((wall_id, score_event))
                gain=safe_gain(score); malformed=(not np.isfinite(gain)) and not hasattr(score, "predicted_reduction")
                valid=bool(getattr(score, "finite", False)) and bool(getattr(score, "accepted_shape", False)) and getattr(score, "reason", "") == "" and np.isfinite(gain)
                results.append(PoolCandidateResult(candidate, score, "proposed", "scored_valid" if valid else "scored_invalid", getattr(score, "reason", "") or ("malformed_predicted_reduction" if malformed else ("invalid_score" if not valid else "")), pool.hash_for(candidate)))
            except Exception as exc:
                score_event=self.work_ledger.record_call("score", "score_eval_calls", solver_step_id=step_id, selection_id=selection_id, candidate_id=candidate.element_id, candidate_attempt_id=attempt_ids.get(candidate.element_id), status="failed"); work_ids.extend((wall_id, score_event))
                results.append(PoolCandidateResult(candidate, None, "proposed", "score_failed", f"{type(exc).__name__}:{exc}", pool.hash_for(candidate)))
        scored_results=tuple(results)
        try: selection=selector.select(pool, scored_results)
        except Exception as exc:
            self.ledger.append(PoolAuditRecord(pool.pool_id, type(selector).__name__, len(pool), tuple(item.as_dict() for item in scored_results), np.nan, np.nan, None, "selector_exception", False, f"selector_exception:{type(exc).__name__}:{exc}", tuple(proposal_costs), tuple({"status": item.scoring_status} for item in scored_results), before_obj, before_obj, before_hash, self.state.hash(), tuple(dict.fromkeys(work_ids)), pool.generation_metadata, budget_snapshot, selector_identity, selector_rule))
            return None
        audit_results=scored_results
        selected = getattr(selection, "selected", None); selected_gain=np.nan if selected is None or selected.score is None else safe_gain(selected.score)
        accepted=False; status="selector_rejected" if selected is None else "rejected"; after_obj=before_obj
        if selected is not None:
            candidate=selected.candidate; score=selected.score
            registry_before=dict(self._active_candidates)
            result_by_hash={(None if x.candidate is None else x.candidate_hash_value): x for x in scored_results}
            selected_hash=selected.candidate_hash_value if candidate is not None else None
            linked=result_by_hash.get(selected_hash)
            selection_results=getattr(selection, "results", ())
            results_match=(tuple(sorted(id(x) for x in selection_results)) == tuple(sorted(id(x) for x in scored_results)))
            valid_selection=(results_match and linked is selected and linked is not None and linked.score is score and bool(getattr(score, "finite", False)) and bool(getattr(score, "accepted_shape", False)) and getattr(score, "reason", "") == "" and np.isfinite(getattr(score, "predicted_reduction", np.nan)) and getattr(score, "predicted_reduction", 0) > 0)
            if not valid_selection:
                accepted=False; status="selector_invalid_selection"; candidate=None; score=None
            else:
                candidate=selection.selected.candidate; score=selection.selected.score
            try:
                if valid_selection:
                    accepted=bool(self.policy.decide(score))
                if valid_selection and accepted:
                    with StateTransaction(self.state) as tx:
                        if getattr(self.policy, "mode", "stagewise") == "fully_corrective":
                            trial_map=dict(self._active_candidates); trial_map[candidate.element_id]=candidate
                            self._last_refit_event_id = None
                            with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, candidate_attempt_id=attempt_ids.get(candidate.element_id), event_ids=work_ids): values, coeff, ids=self._fc_trial(trial_map)
                            work_ids.append(self._last_refit_event_id); trial_state=SolverState(values, ids, coeff)
                            with self.work_ledger.context(solver_step_id=step_id, selection_id=selection_id, candidate_attempt_id=attempt_ids.get(candidate.element_id), event_ids=work_ids): after_obj=self.problem.objective(trial_state)
                            if after_obj > before_obj + 1e-12: raise ValueError("fully-corrective trial did not decrease objective")
                            self.state.values=values; self.state.active_element_ids=ids; self.state.outer_coefficients=coeff; self._active_candidates=trial_map
                        else:
                            trial_values=self.state.values + score.alpha*candidate.correction
                            trial_ids=(*self.state.active_element_ids, candidate.element_id)
                            trial_coeff=np.asarray((*(() if self.state.outer_coefficients is None else self.state.outer_coefficients), score.alpha), float)
                            trial_state=SolverState(trial_values, trial_ids, trial_coeff); after_obj=self.problem.objective(trial_state)
                            self.state.values=trial_values; self.state.active_element_ids=trial_ids; self.state.outer_coefficients=trial_coeff
                        tx.commit()
                    status="accepted"
                elif valid_selection: status=getattr(score, "reason", "rejected") or "rejected"
            except Exception as exc:
                accepted=False; self._active_candidates=registry_before
                if getattr(self, "_last_refit_event_id", None) and self._last_refit_event_id not in work_ids: work_ids.append(self._last_refit_event_id)
                status=f"exception:{type(exc).__name__}:{exc}"
        rule=getattr(selection, "rule", type(selector).__name__); pool_max=getattr(selection, "pool_max_gain", np.nan); ratio=getattr(selection, "ratio", None); ratio_reason=getattr(selection, "ratio_reason", "unavailable")
        self.ledger.append(PoolAuditRecord(pool.pool_id, rule, len(pool), tuple(item.as_dict() for item in audit_results), pool_max, selected_gain, ratio, ratio_reason, accepted, status, tuple(proposal_costs), tuple({"status": item.scoring_status} for item in audit_results), before_obj, after_obj, before_hash, self.state.hash(), tuple(dict.fromkeys(work_ids)), pool.generation_metadata, budget_snapshot, selector_identity, selector_rule))
        return selection.selected.score if accepted else None

    def patch(self, append=(), replace=None, remove=()):
        replace = {} if replace is None else dict(replace); append = tuple(append); remove = tuple(remove)
        old = dict(self._active_candidates); old_state = self.state.snapshot(); before_hash = self.state.hash()
        touched = tuple(dict.fromkeys([*(c.element_id for c in append), *replace.keys(), *remove]))
        patch_id = "patch:" + ",".join(touched or ("empty",)); patch_step_id = self.work_ledger.new_step(); patch_attempt_id = self.work_ledger.new_attempt()
        self._current_step_id = patch_step_id; self._current_attempt_id = patch_attempt_id; work_ids=[]; before_obj=np.nan; after_obj=np.nan
        patch_support = {"append": tuple(c.element_id for c in append), "replace": tuple(replace), "remove": tuple(remove)}
        patch_cost = {"append": len(append), "replace": len(replace), "remove": len(remove)}
        try:
            with self.work_ledger.span("patch", solver_step_id=patch_step_id, candidate_attempt_id=patch_attempt_id, event_ids=work_ids):
                with self.work_ledger.context(solver_step_id=patch_step_id, candidate_attempt_id=patch_attempt_id, event_ids=work_ids):
                    if len({c.element_id for c in append}) != len(append): raise ValueError("duplicate patch element_id")
                    if set(replace) & set(remove): raise ValueError("cannot replace and remove same element")
                    before_obj = self.problem.objective(self.state)
                    for eid in remove:
                        if eid not in self._active_candidates: raise ValueError("unknown remove element_id")
                        self._active_candidates.pop(eid)
                    for eid, cand in replace.items():
                        if eid not in self._active_candidates or cand.element_id != eid: raise ValueError("replace requires existing stable element_id")
                        if hasattr(self.problem, "operator") and (cand.operator_image is None or not np.allclose(cand.operator_image, self.problem.apply_operator(cand.correction), rtol=1e-10, atol=1e-12)): raise ValueError("replacement operator image is stale or missing")
                        self._active_candidates[eid] = cand
                    for cand in append:
                        if cand.element_id in self._active_candidates: raise ValueError("patch element_id already active")
                        if hasattr(self.problem, "operator") and (cand.operator_image is None or not np.allclose(cand.operator_image, self.problem.apply_operator(cand.correction), rtol=1e-10, atol=1e-12)): raise ValueError("patch operator image is stale or missing")
                        self._active_candidates[cand.element_id] = cand
                    if getattr(self.policy, "mode", "stagewise") != "fully_corrective": raise ValueError("patch requires FullyCorrectivePolicy")
                    if self._active_candidates:
                        values, coeff, ids = self._fc_trial(self._active_candidates)
                        self.state.values = values; self.state.active_element_ids=ids; self.state.outer_coefficients=coeff
                    else: self.state.values = self._base_values.copy(); self.state.active_element_ids=(); self.state.outer_coefficients=np.zeros(0)
                    after_obj = self.problem.objective(self.state)
                    if after_obj > before_obj + 1e-12: raise ValueError("patch trial did not decrease objective")
            self.ledger.append(CorrectionRecord(patch_id, "patch", patch_support, {"active_element_ids": tuple(self.state.active_element_ids)}, patch_cost, {"estimated_operator_evaluations": len(self._active_candidates)}, before_obj - after_obj, before_obj, after_obj, True, "patch_accepted", before_hash, self.state.hash(), touched, tuple(eid for eid in old if eid not in touched), tuple(dict.fromkeys(work_ids))))
        except Exception as exc:
            self._active_candidates = old; self.state.restore(old_state)
            self.ledger.append(CorrectionRecord(patch_id, "patch", patch_support, {}, patch_cost, {"estimated_operator_evaluations": 0}, np.nan, before_obj, before_obj, False, f"patch_rejected:{type(exc).__name__}:{exc}", before_hash, self.state.hash(), touched, tuple(old), tuple(dict.fromkeys(work_ids))))
            raise
