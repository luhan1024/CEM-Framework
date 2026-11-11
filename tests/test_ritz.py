import unittest
from unittest.mock import patch
import numpy as np
from core import *
from enrichment import *
from problems import *
from solver import *
from strategies import *

class RitzTests(unittest.TestCase):
    def test_matrix_free_large_operator_no_densification_and_ledger(self):
        class Diag:
            shape=(10000,10000)
            def __init__(self): self.calls=0
            def apply(self, values): self.calls+=1; return 2.0*np.asarray(values)
        op=Diag(); f=np.zeros(10000); f[:2]=[2.,4.]; p=MatrixFreeRitzProblem(op,f); ls=(AnalyticLearner("a",np.r_[1.,np.zeros(9999)],"a"),AnalyticLearner("b",np.r_[0.,1.,np.zeros(9998)],"b")); state=SolverState(np.zeros(10000)); s=CEMSolver(p,ls,RitzCandidateScorer(),FullyCorrectivePolicy(),state)
        def reject_square_alloc(factory):
            def guarded(shape, *args, **kwargs):
                dims=tuple(shape) if isinstance(shape,(tuple,list)) else ()
                if dims == (10000,10000): raise AssertionError("matrix-free path allocated an n x n dense array")
                return factory(shape,*args,**kwargs)
            return guarded
        with patch.object(np,"zeros",reject_square_alloc(np.zeros)), patch.object(np,"empty",reject_square_alloc(np.empty)), patch.object(np,"full",reject_square_alloc(np.full)), patch.object(np,"eye",reject_square_alloc(np.eye)):
            self.assertIsNotNone(s.step()); s.learners=(ls[1],); self.assertIsNotNone(s.step())
        self.assertGreater(op.calls,0); self.assertTrue(all(e.quantity in ("operator_apply_calls","objective_eval_calls","proposal_calls","score_eval_calls","refit_calls","wall_time") for e in s.work_ledger.events)); self.assertTrue(s.ledger.records[-1].work_event_ids); self.assertLess(op.calls,30)

    def test_matrix_free_fully_corrective_matches_dense_record(self):
        K=np.array([[2.,.5],[.5,1.]]); f=np.array([1.,.2]); ls=(AnalyticLearner("a",np.array([1.,0.]),"a"),AnalyticLearner("b",np.array([0.,1.]),"b")); pd=DiscreteRitzProblem(K,f); pm=MatrixFreeRitzProblem(type("Op",(),{"shape":K.shape,"apply":lambda self,x: K@x})(),f); sd=CEMSolver(pd,ls,RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); sm=CEMSolver(pm,ls,RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2)))
        for solver in (sd,sm): self.assertIsNotNone(solver.step()); solver.learners=(ls[1],); self.assertIsNotNone(solver.step())
        self.assertTrue(np.allclose(sd.state.values,sm.state.values)); self.assertTrue(np.allclose(sd.state.outer_coefficients,sm.state.outer_coefficients)); self.assertEqual(sd.state.active_element_ids,sm.state.active_element_ids); self.assertAlmostEqual(pd.objective(sd.state),pm.objective(sm.state)); refs=sm.ledger.records[-1].work_event_ids; events=[e for e in sm.work_ledger.events if e.event_id in refs]; self.assertEqual(len(refs),len(set(refs))); self.assertEqual(len(events),len(refs)); self.assertEqual({e.quantity:sum(x.quantity==e.quantity for x in events) for e in events},{"operator_apply_calls":10,"objective_eval_calls":4,"proposal_calls":1,"score_eval_calls":1,"refit_calls":1,"wall_time":2}); self.assertEqual({e.phase for e in events if e.quantity=="operator_apply_calls"},{"matrix_free_ritz_operator_apply"}); self.assertEqual({e.phase for e in events if e.quantity=="objective_eval_calls"},{"matrix_free_ritz_objective"}); self.assertEqual({e.phase for e in events if e.quantity=="refit_calls"},{"refit"}); step_ids={e.solver_step_id for e in events}; attempt_ids={e.candidate_attempt_id for e in events}; self.assertEqual(len(step_ids),1); self.assertEqual(len(attempt_ids),1); self.assertTrue(next(iter(step_ids))); self.assertTrue(next(iter(attempt_ids)))
        dense_refs=sd.ledger.records[-1].work_event_ids; dense_events=[e for e in sd.work_ledger.events if e.event_id in dense_refs]; self.assertEqual(len(dense_refs),len(dense_events)); self.assertEqual(len(dense_refs),len(set(dense_refs))); self.assertEqual({e.quantity:sum(x.quantity==e.quantity for x in dense_events) for e in dense_events},{"operator_apply_calls":10,"objective_eval_calls":4,"proposal_calls":1,"score_eval_calls":1,"refit_calls":1,"wall_time":2}); self.assertEqual({e.phase for e in dense_events if e.quantity=="operator_apply_calls"},{"ritz_operator_apply"}); self.assertEqual({e.phase for e in dense_events if e.quantity=="objective_eval_calls"},{"ritz_objective"}); self.assertEqual({e.phase for e in dense_events if e.quantity=="refit_calls"},{"refit"}); step_ids={e.solver_step_id for e in dense_events}; attempt_ids={e.candidate_attempt_id for e in dense_events}; self.assertEqual(len(step_ids),1); self.assertEqual(len(attempt_ids),1); self.assertTrue(next(iter(step_ids))); self.assertTrue(next(iter(attempt_ids)))

    def test_matrix_free_matches_dense_and_non_symmetric_identity_fails(self):
        K=np.array([[2.,.5],[.5,1.]]); f=np.array([1.,.2]); pd=DiscreteRitzProblem(K,f); pm=MatrixFreeRitzProblem(type("Op",(),{"shape":K.shape,"apply":lambda self,x: K@x})(),f); cd=AnalyticLearner("a",np.array([1.,0.]),"a"); sd=CEMSolver(pd,(cd,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); sm=CEMSolver(pm,(cd,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertTrue(np.allclose(sd.step().after,sm.step().after)); self.assertTrue(np.allclose(sd.state.values,sm.state.values))
        A=np.array([[1.,2.],[0.,1.]]); pn=MatrixFreeRitzProblem(type("NOp",(),{"shape":A.shape,"apply":lambda self,x: A@x})(),np.array([1.,1.])); state=SolverState(np.array([.2,.3])); c=AnalyticLearner("n",np.array([0.,1.]),"n").propose(pn,state,pn.residual(state),{}); score=RitzCandidateScorer().score(pn,state,c); self.assertEqual(score.reason,"ritz_identity_mismatch")
    def test_geometry_pairing_fails_at_construction(self):
        p=DiscreteRitzProblem(np.eye(2),np.ones(2)); state=SolverState(np.zeros(2)); learner=AnalyticLearner("a",np.array([1.,0.]),"a")
        with self.assertRaises(TypeError): CEMSolver(p,(learner,),LinearCandidateScorer(),StagewisePolicy(),state)
        with self.assertRaises(TypeError): CEMSolver(LinearPDEProblem(np.eye(2),np.ones(2)),(learner,),RitzCandidateScorer(),StagewisePolicy(),state)
        with self.assertRaises(ValueError): MatrixFreeRitzProblem(type("Op",(),{"shape":(2,2),"apply":lambda self,x:x})(),np.ones(2),spd_assumption="assumed")
        with self.assertRaises(TypeError): MatrixFreeRitzProblem(type("Op",(),{"shape":(2,2),"apply":lambda self,x:x})(),np.ones(2),transpose_operator=object())
        class BadOp:
            shape=(2,2)
            def apply(self, values): return np.array([np.nan,0.])
        pb=MatrixFreeRitzProblem(BadOp(),np.ones(2)); self.assertRaises(ValueError, lambda: pb.apply_operator(np.ones(2)))
    def test_stagewise_identity_and_pool(self):
        K=np.array([[2., .5],[.5, 1.]]); f=np.array([1., .2]); p=DiscreteRitzProblem(K,f); learner=AnalyticLearner("a",np.array([1.,0.]),"a"); s=CEMSolver(p,(learner,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); score=s.step(); self.assertIsNotNone(score); self.assertAlmostEqual(score.before-score.after,score.predicted_reduction); self.assertAlmostEqual(score.q,2.); self.assertAlmostEqual(score.correlation,1.); self.assertTrue(np.allclose(s.state.values,[.5,0.])); s2=CEMSolver(p,(AnalyticLearner("b",np.array([0.,1.]),"b"),),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNotNone(s2.step_pool(BestPredictedGainSelector()))
    def test_fully_corrective_ritz_projection(self):
        K=np.array([[2., .5],[.5, 1.]]); f=np.array([1., .2]); p=DiscreteRitzProblem(K,f); ls=(AnalyticLearner("a",np.array([1.,0.]),"a"),AnalyticLearner("b",np.array([0.,1.]),"b")); s=CEMSolver(p,ls,RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); self.assertIsNotNone(s.step()); s.learners=(ls[1],); self.assertIsNotNone(s.step()); self.assertTrue(np.allclose(s.state.values,np.linalg.solve(K,f))); self.assertLessEqual(p.objective(s.state),0.0)
    def test_public_fc_step_rolls_back_ill_conditioned_ritz_set(self):
        K=np.diag([0.1,1.]); f=np.array([0.1,1e5]); p=DiscreteRitzProblem(K,f); first=AnalyticLearner("first",np.array([1.,0.]),"first"); near=AnalyticLearner("near",np.array([1.,1e-7]),"near"); s=CEMSolver(p,(first,),RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); self.assertIsNotNone(s.step()); before=s.state.hash(); s.learners=(near,); self.assertIsNone(s.step()); self.assertEqual(s.state.hash(),before); self.assertIn("rank_deficient_or_ill_conditioned_ritz_active_set",s.ledger.records[-1].status); refs=s.ledger.records[-1].work_event_ids; self.assertEqual(len(refs),len(set(refs))); self.assertTrue(set(refs).issubset({e.event_id for e in s.work_ledger.events}))
    def test_validation_zero_stale_and_rank_deficient(self):
        with self.assertRaises(ValueError): DiscreteRitzProblem(np.zeros((0,0)),np.zeros(0))
        with self.assertRaises(ValueError): DiscreteRitzProblem([[np.nan]], [1.])
        with self.assertRaises(ValueError): DiscreteRitzProblem([[1.]], [np.inf])
        with self.assertRaises(ValueError): DiscreteRitzProblem(np.eye(2), [1.])
        with self.assertRaises(ValueError): DiscreteRitzProblem([[1.,2.],[0.,1.]], [1.,0.])
        with self.assertRaises(ValueError): DiscreteRitzProblem([[1.,0.],[0.,0.]], [1.,0.])
        p=DiscreteRitzProblem(np.eye(2),np.ones(2)); zero=AnalyticLearner("z",np.zeros(2),"z"); s=CEMSolver(p,(zero,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(s.step()); self.assertEqual(s.ledger.records[-1].status,"zero_or_nonfinite_ritz_curvature"); self.assertTrue(np.allclose(s.state.values,0))
        wrong=type("Wrong",(),{"family":"wrong","propose":lambda self,*args: Candidate("wrong","wrong",np.array([1.]),None,{}, {},None,{})})(); sw=CEMSolver(p,(wrong,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(sw.step()); self.assertEqual(sw.ledger.records[-1].status,"shape_mismatch"); self.assertTrue(np.allclose(sw.state.values,0))
        feasible=DiscreteRitzProblem(np.eye(2),np.ones(2),feasible=lambda x: np.all(x<=0)); sf=CEMSolver(feasible,(AnalyticLearner("f",np.array([1.,0.]),"f"),),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(sf.step()); self.assertEqual(sf.ledger.records[-1].status,"inadmissible_trial"); self.assertTrue(np.allclose(sf.state.values,0))
        bad=Candidate("bad","x",np.array([1.,0.]),np.array([9.,9.]),{}, {},None,{}); class_learner=type("L",(),{"family":"x","propose":lambda self,*args: bad})(); ss=CEMSolver(p,(class_learner,),RitzCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); self.assertIsNone(ss.step()); self.assertEqual(ss.ledger.records[-1].status,"stale_or_inconsistent_operator_image")
        a=AnalyticLearner("a",np.array([1.,0.]),"a"); b=AnalyticLearner("b",np.array([2.,0.]),"b"); sf=CEMSolver(p,(a,),RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); ca=a.propose(p,sf.state,p.residual(sf.state),{}); cb=b.propose(p,sf.state,p.residual(sf.state),{}); sf._active_candidates={"a":ca,"b":cb}; sf._current_step_id="test"; sf._current_attempt_id="test"; self.assertRaises(ValueError, lambda: sf._fc_trial(sf._active_candidates))
        near_a=AnalyticLearner("na",np.array([1.,0.]),"na"); near_b=AnalyticLearner("nb",np.array([1.,1e-7]),"nb"); sn=CEMSolver(p,(near_a,),RitzCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); nca=near_a.propose(p,sn.state,p.residual(sn.state),{}); ncb=near_b.propose(p,sn.state,p.residual(sn.state),{}); sn._active_candidates={"na":nca,"nb":ncb}; sn._current_step_id="test"; sn._current_attempt_id="test"; self.assertRaises(ValueError, lambda: sn._fc_trial(sn._active_candidates)); self.assertTrue(np.allclose(sn.state.values,0))

if __name__ == "__main__": unittest.main()
