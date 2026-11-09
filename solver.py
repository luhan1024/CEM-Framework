from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .state import SolverState, StateTransaction
from .ledger import CorrectionLedger, CorrectionRecord

@dataclass
class CEMSolver:
    problem: object
    learners: tuple
    scorer: object
    policy: object
    state: SolverState
    ledger: CorrectionLedger | None = None
    def __post_init__(self):
        self.learners = tuple(self.learners); self.ledger = self.ledger or CorrectionLedger()
        if hasattr(self.problem, "validate_state"): self.problem.validate_state(self.state)
        if getattr(self.policy, "mode", "stagewise") == "fully_corrective" and not hasattr(self.problem, "operator"):
            raise TypeError("FullyCorrectivePolicy requires LinearPDEProblem")
        self._base_values = self.state.values.copy(); self._active_candidates = {}

    def _record(self, candidate, before, score, accepted, status, touched=(), frozen=()):
        eid = getattr(candidate, "element_id", "proposal-error")
        family = getattr(candidate, "family", "unknown")
        support = getattr(candidate, "support", {}); params = getattr(candidate, "parameters", {})
        cost = getattr(candidate, "proposal_cost", {})
        scoring_cost = {"estimated_operator_evaluations": 1, "estimated_weighted_inner_products": 3} if np.isfinite(getattr(score, "q", np.nan)) else {"estimated_operator_evaluations": 0, "estimated_weighted_inner_products": 0}
        self.ledger.append(CorrectionRecord(eid, family, support, params, cost, scoring_cost, getattr(score,"predicted_reduction",np.nan), getattr(score,"before",np.nan), getattr(score,"after",np.nan), accepted, status, before, self.state.hash(), tuple(touched), tuple(frozen)))

    def _fc_trial(self, active_map):
        ids = tuple(active_map)
        images = [active_map[i].operator_image for i in ids]
        if any(x is None for x in images): raise ValueError("fully corrective candidates require operator images")
        B = np.column_stack(images); rhs = self.problem.source - self.problem.apply_operator(self._base_values)
        W = np.sqrt(self.problem.weights)[:, None]; coeff = np.linalg.lstsq(W * B, np.sqrt(self.problem.weights) * rhs, rcond=None)[0]
        corrections = np.column_stack([active_map[i].correction for i in ids])
        values = self._base_values + corrections @ coeff
        if not np.all(np.isfinite(values)) or not self.problem.admissible(values): raise ValueError("invalid fully-corrective trial state")
        return values, coeff, ids

    def step(self, budget=None):
        budget = {} if budget is None else dict(budget); residual = self.problem.residual(self.state)
        for learner in self.learners:
            before_hash = self.state.hash(); candidate = None; registry_before = dict(self._active_candidates)
            try:
                snap = self.state.snapshot(); candidate = learner.propose(self.problem, snap, residual.copy(), budget)
                if candidate.element_id in self.state.active_element_ids: raise ValueError("duplicate element_id")
                score = self.scorer.score(self.problem, self.state, candidate)
                accepted = bool(self.policy.decide(score))
                if accepted:
                    with StateTransaction(self.state) as tx:
                        if getattr(self.policy, "mode", "stagewise") == "fully_corrective":
                            if not hasattr(self.problem, "operator"): raise TypeError("FullyCorrectivePolicy requires LinearPDEProblem")
                            trial_map = dict(self._active_candidates); trial_map[candidate.element_id] = candidate
                            values, coeff, ids = self._fc_trial(trial_map)
                            before_obj = self.problem.objective(self.state); trial_state = SolverState(values, ids, coeff)
                            if self.problem.objective(trial_state) > before_obj + 1e-12: raise ValueError("fully-corrective trial did not decrease objective")
                            self.state.values = values; self.state.active_element_ids = ids; self.state.outer_coefficients = coeff; self._active_candidates = trial_map
                        else:
                            self.state.values = self.state.values + score.alpha * candidate.correction
                            self.state.active_element_ids = (*self.state.active_element_ids, candidate.element_id)
                            self.state.outer_coefficients = np.asarray((*(() if self.state.outer_coefficients is None else self.state.outer_coefficients), score.alpha), float)
                        tx.commit()
                status = "accepted" if accepted else score.reason or "rejected"
                self._record(candidate, before_hash, score, accepted, status, (candidate.element_id,) if accepted else (), self.state.active_element_ids[:-1] if accepted else self.state.active_element_ids)
                if accepted: return score
            except Exception as exc:
                self._active_candidates = registry_before
                self._record(candidate, before_hash, type("Failure", (), {"q":np.nan,"predicted_reduction":np.nan,"before":np.nan,"after":np.nan})(), False, f"exception:{type(exc).__name__}:{exc}", (), self.state.active_element_ids)
        return None

    def patch(self, append=(), replace=None, remove=()):
        replace = {} if replace is None else dict(replace); append = tuple(append); remove = tuple(remove)
        if len({c.element_id for c in append}) != len(append): raise ValueError("duplicate patch element_id")
        if set(replace) & set(remove): raise ValueError("cannot replace and remove same element")
        old = dict(self._active_candidates); old_state = self.state.snapshot()
        before_hash = self.state.hash(); before_obj = self.problem.objective(self.state)
        touched = tuple(dict.fromkeys([*(c.element_id for c in append), *replace.keys(), *remove]))
        patch_id = "patch:" + ",".join(touched or ("empty",))
        patch_support = {"append": tuple(c.element_id for c in append),
                         "replace": tuple(replace), "remove": tuple(remove)}
        patch_cost = {"append": len(append), "replace": len(replace), "remove": len(remove)}
        try:
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
            if after_obj > before_obj + 1e-12:
                raise ValueError("patch trial did not decrease objective")
            self.ledger.append(CorrectionRecord(
                patch_id, "patch", patch_support,
                {"active_element_ids": tuple(self.state.active_element_ids)},
                patch_cost, {"estimated_operator_evaluations": len(self._active_candidates)},
                before_obj - after_obj, before_obj, after_obj, True, "patch_accepted",
                before_hash, self.state.hash(), touched,
                tuple(eid for eid in old if eid not in touched)))
        except Exception as exc:
            self._active_candidates = old; self.state.restore(old_state)
            self.ledger.append(CorrectionRecord(
                patch_id, "patch", patch_support, {}, patch_cost,
                {"estimated_operator_evaluations": 0}, np.nan, before_obj, before_obj,
                False, f"patch_rejected:{type(exc).__name__}:{exc}",
                before_hash, self.state.hash(), touched,
                tuple(old)))
            raise
