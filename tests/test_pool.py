import unittest
import json
import numpy as np
from cem_framework import *

class PoolTests(unittest.TestCase):
    def setup_solver(self, learners, policy=None):
        p=LinearPDEProblem(np.eye(2), np.array([1., 1.])); return p, CEMSolver(p, tuple(learners), LinearCandidateScorer(), policy or StagewisePolicy(), SolverState(np.zeros(2)))
    def test_best_selector_is_order_invariant_and_tie_stable(self):
        p,s=self.setup_solver((AnalyticLearner("b",np.array([0.,1.]),"b"), AnalyticLearner("a",np.array([1.,0.]),"a")))
        pool=CandidatePool(tuple(l.propose(p,s.state,p.residual(s.state),{}) for l in s.learners), {"seed": 4}); scores=tuple(PoolCandidateResult(c,s.scorer.score(p,s.state,c),"proposed","scored_valid","") for c in pool.candidates); sel=BestPredictedGainSelector().select(pool,scores); self.assertEqual(sel.selected.candidate.element_id,"a"); pool2=CandidatePool(tuple(reversed(pool.candidates)), {"seed":4}); scores2=tuple(PoolCandidateResult(c,s.scorer.score(p,s.state,c),"proposed","scored_valid","") for c in pool2.candidates); self.assertEqual(BestPredictedGainSelector().select(pool2,scores2).selected.candidate.element_id,"a"); self.assertEqual(pool.pool_id,pool2.pool_id); self.assertNotEqual(pool.pool_id,CandidatePool(pool.candidates,{"seed":5}).pool_id)
    def test_invalid_nonpositive_and_empty_pool(self):
        p,s=self.setup_solver(()); selector=BestPredictedGainSelector()
        def fake(gain, reason=""):
            return PoolCandidateResult(Candidate(str(gain),"x",np.array([1.,0.]),np.array([1.,0.]),{}, {},None,{}), type("S",(),{"predicted_reduction":gain,"reason":reason,"finite":True,"accepted_shape":True})(), "proposed","scored_valid",reason)
        pool=CandidatePool((fake(1).candidate,fake(0).candidate,fake(-1).candidate)); result=selector.select(pool,(fake(1),fake(0),fake(-1))); self.assertEqual(result.selected.candidate.element_id,"1"); self.assertAlmostEqual(result.ratio,1.0); empty=selector.select(CandidatePool(()),()); self.assertIsNone(empty.selected); self.assertEqual(empty.ratio_reason,"no_valid_positive_gain")
        for finite, shape in ((False, True), (True, False)):
            score=type("S",(),{"predicted_reduction":1.,"reason":"","finite":finite,"accepted_shape":shape})(); item=PoolCandidateResult(fake(2).candidate, score, "proposed", "scored_invalid", "invalid_score"); self.assertIsNone(selector.select(CandidatePool((item.candidate,)),(item,)).selected)
    def test_proposal_and_score_failures_are_audited(self):
        p,s=self.setup_solver(())
        class BadProposal:
            family="bad"
            def propose(self,*args): raise RuntimeError("proposal")
        class BadScore:
            def score(self,*args): raise RuntimeError("score")
        good=AnalyticLearner("good",np.array([1.,0.]),"good"); s.learners=(BadProposal(),good); s.scorer=BadScore(); before=s.state.hash(); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); self.assertEqual(before,s.state.hash()); audit=s.ledger.records[-1]; self.assertEqual(audit.status,"selector_rejected"); self.assertEqual(len(audit.candidates),2)
    def test_acceptance_rejection_does_not_retry(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),AnalyticLearner("b",np.array([0.,1.]),"b")), policy=MeritDecreasePolicy()); s.policy=type("Reject",(),{"mode":"stagewise","decide":lambda self,score: False})(); before=s.state.hash(); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); self.assertEqual(before,s.state.hash()); self.assertEqual(len(s.state.active_element_ids),0); self.assertEqual(s.ledger.records[-1].status,"rejected")
    def test_fc_pool_path_and_ledger_export(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),AnalyticLearner("b",np.array([0.,1.]),"b")), policy=FullyCorrectivePolicy()); self.assertIsNotNone(s.step_pool(BestPredictedGainSelector())); self.assertEqual(s.state.active_element_ids,("a",)); exported=s.ledger.to_dicts(); self.assertEqual(exported[-1]["rule"],"best_predicted_gain"); self.assertEqual(exported[-1]["finite_pool_realized_ratio"],1.0); refs=s.ledger.records[-1].work_event_ids; self.assertEqual(len(refs),len(set(refs))); self.assertTrue(set(refs).issubset({e.event_id for e in s.work_ledger.events}))
    def test_pool_duplicate_and_active_ids_rejected(self):
        c=Candidate("a","x",np.array([1.]),np.array([1.]),{}, {},None,{}); self.assertRaises(ValueError, lambda: CandidatePool((c,c))); self.assertRaises(ValueError, lambda: CandidatePool((c,), active_element_ids=("a",)))
        self.assertNotEqual(CandidatePool((c,), {"scale":np.float64(1.0)}).pool_id, CandidatePool((c,), {"scale":np.float64(2.0)}).pool_id)
        variants=(Candidate("a","x",np.array([1.]),np.array([2.]),{}, {},None,{}), Candidate("a","x",np.array([1.]),np.array([1.]),{"s":1}, {},None,{}), Candidate("a","x",np.array([1.]),np.array([1.]),{}, {"p":1},None,{}), Candidate("a","x",np.array([1.]),np.array([1.]),{}, {},None,{}, version="2")); self.assertEqual(len({CandidatePool((v,)).pool_id for v in (c,)+variants}),5); self.assertNotEqual(CandidatePool((c,), active_element_ids=()).pool_id, CandidatePool((c,), active_element_ids=("other",)).pool_id)

    def test_pool_provenance_rejects_cycles_and_nonstring_failure_keys(self):
        c=Candidate("a","x",np.array([1.]),np.array([1.]),{}, {},None,{})
        cyclic={}; cyclic["self"]=cyclic
        with self.assertRaises(ValueError): CandidatePool((c,), generation_metadata=cyclic)
        with self.assertRaises(TypeError): CandidatePool((c,), proposal_failures=({1: "bad"},))
        failures={"nested": []}; pool=CandidatePool((c,), proposal_failures=(failures,)); failures["nested"].append(9)
        self.assertEqual(pool.proposal_failures[0]["nested"], ())
        list_cycle=[]; list_cycle.append(list_cycle)
        with self.assertRaises(ValueError): CandidatePool((c,), generation_metadata={"cycle": list_cycle})
        with self.assertRaises(ValueError): CandidatePool((c,), proposal_failures=({"cycle": list_cycle},))
        array_cycle=np.empty(1, dtype=object); array_cycle[0]=array_cycle
        with self.assertRaises(ValueError): CandidatePool((c,), generation_metadata={"cycle": array_cycle})
        with self.assertRaises(ValueError): CandidatePool((c,), proposal_failures=({"cycle": array_cycle},))
        ordered=CandidatePool((c,), active_element_ids=("other", "third")); permuted=CandidatePool((c,), active_element_ids=("third", "other")); self.assertEqual(ordered.pool_id, permuted.pool_id)

    def test_numpy_integer_seed_is_normalized_for_hash_and_json(self):
        c=Candidate("seeded","x",np.array([1.]),np.array([1.]),{}, {},np.int64(4),{})
        self.assertIs(type(c.seed), int)
        pool=CandidatePool((c,)); self.assertIsInstance(pool.pool_id, str); json.dumps(pool.pool_id, allow_nan=False)
        with self.assertRaises(TypeError): Candidate("bool-seed","x",np.array([1.]),None,{}, {},True,{})
    def test_pool_audit_exposes_invalid_score_fields(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),))
        class BadScorer:
            def score(self,*args): return type("S",(),{"predicted_reduction":1.,"reason":"","finite":False,"accepted_shape":True})()
        s.scorer=BadScorer(); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); item=s.ledger.records[-1].candidates[0]; self.assertFalse(item["finite"]); self.assertEqual(item["scoring_status"],"scored_invalid")
    def test_selector_cannot_inject_foreign_candidate_or_score(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),)); before=s.state.hash(); foreign=AnalyticLearner("foreign",np.array([0.,1.]),"foreign").propose(p,s.state,p.residual(s.state),{})
        class Forged:
            rule="forged"
            def select(self, pool, results):
                score=results[0].score
                return SelectionResult(PoolCandidateResult(foreign, score, "proposed", "scored", ""), results, score.predicted_reduction, 1.0, "ok", "forged")
        self.assertIsNone(s.step_pool(Forged())); self.assertEqual(before,s.state.hash()); self.assertEqual(s.ledger.records[-1].status,"selector_invalid_selection")
    def test_selector_exception_and_invalid_residual_are_audited(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),)); before=s.state.hash()
        class Exploding:
            def select(self,*args): raise RuntimeError("selector")
        self.assertIsNone(s.step_pool(Exploding())); self.assertEqual(before,s.state.hash()); self.assertTrue(s.ledger.records[-1].status.startswith("selector_exception:")); self.assertEqual(s.ledger.records[-1].candidates[0]["scoring_status"],"scored_valid")
        class BadProblem(LinearPDEProblem):
            def residual(self, state): return np.array([np.nan, 0.])
        sb=CEMSolver(BadProblem(np.eye(2),np.ones(2)),(),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(sb.step_pool(BestPredictedGainSelector())); self.assertEqual(sb.ledger.records[-1].status,"invalid_residual"); self.assertEqual(sb.state.hash(),SolverState(np.zeros(2)).hash()); json.dumps(sb.ledger.to_dicts(), allow_nan=False)
    def test_bound_and_unbound_pool_context(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),)); c=s.learners[0].propose(p,s.state,p.residual(s.state),{}); unbound=CandidatePool((c,)); self.assertIsNotNone(s.step_pool(BestPredictedGainSelector(), unbound)); s2=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),))[1]; c2=s2.learners[0].propose(p,s2.state,p.residual(s2.state),{}); bound=CandidatePool((c2,), {"state_hash":"wrong"}); self.assertRaises(ValueError, lambda: s2.step_pool(BestPredictedGainSelector(), bound))

    def test_pool_audit_freezes_provenance_and_selector_identity(self):
        p,s=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),)); c=s.learners[0].propose(p,s.state,p.residual(s.state),{}); metadata={"scope":{"nodes":np.array([1,2])}}; budget={"seed":np.int64(4),"limits":[np.float64(2.0)]}; pool=CandidatePool((c,), metadata); self.assertIsNotNone(s.step_pool(BestPredictedGainSelector(), pool, budget)); metadata["scope"]["nodes"][0]=99; budget["limits"][0]=8.0; audit=s.ledger.records[-1]; self.assertEqual(audit.generation_metadata["scope"]["nodes"], (1,2)); self.assertEqual(audit.budget["seed"], 4); self.assertEqual(audit.selector_rule, "best_predicted_gain"); self.assertTrue(audit.selector_identity.endswith("BestPredictedGainSelector")); exported=s.ledger.to_dicts()[-1]; import json; json.dumps(exported, allow_nan=False)

    def test_pool_provenance_rejects_nonfinite_or_unsupported_values(self):
        p,s=self.setup_solver(()); c=Candidate("x","x",np.array([1.,0.]),np.array([1.,0.]),{}, {},None,{}); self.assertRaises(ValueError, lambda: CandidatePool((c,), {"bad":np.nan})); self.assertRaises(TypeError, lambda: CandidatePool((c,), {"bad":object()})); self.assertRaises(ValueError, lambda: s.step_pool(BestPredictedGainSelector(), CandidatePool((c,)), {"bad":np.inf}))
    def test_all_nan_audit_paths_export_strict_json(self):
        p,s=self.setup_solver(()); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); json.dumps(s.ledger.to_dicts(), allow_nan=False)
        class BadProblem(LinearPDEProblem):
            def objective(self, state): raise RuntimeError("objective")
        sb=CEMSolver(BadProblem(np.eye(2),np.ones(2)),(),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(sb.step_pool(BestPredictedGainSelector())); json.dumps(sb.ledger.to_dicts(), allow_nan=False)
        class NanScorer:
            def score(self,*args): return type("S",(),{"predicted_reduction":np.nan,"reason":"","finite":False,"accepted_shape":True})()
        sn=self.setup_solver((AnalyticLearner("a",np.array([1.,0.]),"a"),))[1]; sn.scorer=NanScorer(); self.assertIsNone(sn.step_pool(BestPredictedGainSelector())); json.dumps(sn.ledger.to_dicts(), allow_nan=False)
    def test_invalid_candidate_hash_fails_before_commit(self):
        p,s=self.setup_solver(()); good=Candidate("good","x",np.array([1.,0.]),np.array([1.,0.]),{}, {},None,{}); bad=Candidate("bad","x",np.array([0.,1.]),np.array([np.nan,np.nan]),{}, {},None,{}); before=s.state.hash(); self.assertRaises(ValueError, lambda: s.step_pool(BestPredictedGainSelector(), CandidatePool((good,bad)))); self.assertEqual(before,s.state.hash()); self.assertEqual(s.ledger.records,[])
    def test_malformed_score_audit_cannot_break_valid_commit(self):
        p,s=self.setup_solver((AnalyticLearner("good",np.array([1.,0.]),"good"),AnalyticLearner("bad",np.array([0.,1.]),"bad")))
        class MixedScorer:
            def score(self, problem, state, candidate):
                if candidate.element_id == "bad": return type("Malformed",(),{"finite":False,"accepted_shape":True,"reason":""})()
                return LinearCandidateScorer().score(problem,state,candidate)
        s.scorer=MixedScorer(); self.assertIsNotNone(s.step_pool(BestPredictedGainSelector())); audit=s.ledger.records[-1]; bad=next(x for x in audit.candidates if x["element_id"]=="bad"); self.assertEqual(bad["score_reason"],"malformed_predicted_reduction")
    def test_element_id_must_be_string(self):
        c=Candidate(1,"x",np.array([1.]),np.array([1.]),{}, {},None,{}); self.assertRaises(TypeError, lambda: CandidatePool((c,)))
    def test_internal_candidate_validation_failure_is_audited(self):
        p,s=self.setup_solver(())
        class BadLearner:
            family="bad"
            def propose(self,*args): return Candidate("bad","x",np.array([1.,0.]),np.array([np.nan,np.nan]),{"bad":object()}, {},None,{})
        s.learners=(BadLearner(),); before=s.state.hash(); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); self.assertEqual(before,s.state.hash()); self.assertEqual(s.ledger.records[-1].status,"selector_rejected"); self.assertEqual(s.ledger.records[-1].candidates[0]["proposal_status"],"proposal_failed")
    def test_post_trial_objective_failure_is_atomic(self):
        class PostFailProblem(LinearPDEProblem):
            def objective(self, state):
                if np.any(state.values != 0): raise RuntimeError("post objective")
                return super().objective(state)
        p=PostFailProblem(np.eye(2),np.array([1.,0.])); s=CEMSolver(p,(AnalyticLearner("a",np.array([1.,0.]),"a"),),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); before=s.state.hash(); self.assertIsNone(s.step_pool(BestPredictedGainSelector())); self.assertEqual(before,s.state.hash()); self.assertEqual(s._active_candidates,{})

if __name__ == "__main__": unittest.main()
