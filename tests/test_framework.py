import unittest
import numpy as np
from cem_framework import *

class FrameworkTests(unittest.TestCase):
    def problem(self): return LinearPDEProblem(np.eye(3), np.array([1.,0.,0.]))
    def test_projection_and_identity(self):
        p=self.problem(); c=AnalyticLearner("a",np.array([1.,0.,0.]),"a"); s=CEMSolver(p,(c,),LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.zeros(3))); out=s.step(); self.assertEqual(out.reason,""); self.assertAlmostEqual(p.objective(s.state),0.0)
    def test_rejection_preserves_state_but_logs(self):
        p=self.problem(); c=AnalyticLearner("bad",np.array([0.,1.,0.]),"bad"); st=SolverState(np.array([1.,0.,0.])); before=st.hash(); s=CEMSolver(p,(c,),LinearCandidateScorer(),MeritDecreasePolicy(),st); self.assertIsNone(s.step()); self.assertEqual(before,st.hash()); self.assertEqual(len(s.ledger.records),1); self.assertFalse(s.ledger.records[0].accepted); exported=s.ledger.to_dicts(); self.assertEqual(exported[0]["element_id"],"bad"); self.assertIsInstance(exported[0]["support"],dict)
    def test_boundary_and_shape_failures(self):
        p=LinearPDEProblem(np.eye(2),np.ones(2),feasible=lambda x: np.all(x>=0)); self.assertRaises(ValueError, lambda: CEMSolver(p,(),LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.array([-1.,0.])))); bad=AnalyticLearner("bad",np.array([1.,0.]),"b"); s=CEMSolver(p,(bad,),LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.zeros(2))); self.assertIsNotNone(s.step()); self.assertRaises(ValueError, lambda: Candidate("x","x",np.ones((2,1)),None,{}, {},None,{}))
    def test_two_families_patch_and_modes(self):
        p=self.problem(); ls=(AnalyticLearner("analytic",np.array([1.,0.,0.]),"a"),LocalBasisLearner("local",np.array([0.,1.,0.]),"l",{"nodes":[1]})); s=CEMSolver(p,ls,LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.zeros(3))); s.step(); self.assertEqual(s.state.active_element_ids,("a",)); self.assertEqual(StagewisePolicy().mode,"stagewise"); self.assertEqual(FullyCorrectivePolicy().mode,"fully_corrective")
    def test_fully_corrective_refits_outer_coefficients(self):
        A=np.array([[1.,.5],[.5,1.]]); p=LinearPDEProblem(A,np.array([1.,0.])); a=AnalyticLearner("a",np.array([1.,0.]),"a"); b=AnalyticLearner("b",np.array([0.,1.]),"b"); s1=CEMSolver(p,(a,b),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2))); s1.step(); s1.learners=(b,); s1.step(); s2=CEMSolver(p,(a,b),LinearCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))); s2.step(); s2.learners=(b,); s2.step(); self.assertFalse(np.allclose(s1.state.values,s2.state.values)); self.assertTrue(np.allclose(s2.state.outer_coefficients,[4/3,-2/3]))
    def test_mutating_learner_and_stale_image_are_rejected(self):
        p=self.problem(); st=SolverState(np.zeros(3)); before=st.hash()
        class Mutator:
            family="bad"
            def propose(self, problem, state, residual, budget): state.values[0]=9; raise RuntimeError("boom")
        s=CEMSolver(p,(Mutator(),),LinearCandidateScorer(),MeritDecreasePolicy(),st); self.assertIsNone(s.step()); self.assertEqual(before,st.hash()); self.assertTrue(s.ledger.records[0].status.startswith("exception:"))
        stale=Candidate("stale","x",np.array([1.,0.,0.]),np.array([0.,0.,0.]),{}, {},None,{})
        class L:
            family="x"
            def propose(self,*args): return stale
        s=CEMSolver(p,(L(),),LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.zeros(3))); self.assertIsNone(s.step()); self.assertEqual(s.ledger.records[0].status,"stale_or_inconsistent_operator_image")
    def test_fc_commit_failure_restores_registry_and_state(self):
        p=self.problem(); a=AnalyticLearner("a",np.array([1.,0.,0.]),"a"); s=CEMSolver(p,(a,),LinearCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(3))); before=s.state.hash(); old=dict(s._active_candidates)
        original=s._fc_trial
        def fail(active): s._active_candidates["ghost"] = a.propose(p,s.state,p.residual(s.state),{}); raise RuntimeError("injected FC failure")
        s._fc_trial=fail
        self.assertIsNone(s.step()); self.assertEqual(before,s.state.hash()); self.assertEqual(s._active_candidates,old); self.assertTrue(s.ledger.records[-1].status.startswith("exception:")); s._fc_trial=original
    def test_candidate_roundtrip(self):
        c=Candidate("e","family",np.array([1.,2.]),np.array([3.,4.]),{"support":"x"},{"width":.2},7,{"source":"test"},{"proposal":1.})
        self.assertTrue(np.array_equal(candidate_from_json(candidate_to_json(c)).correction,c.correction)); self.assertRaises((TypeError, AttributeError), lambda: c.support.__setitem__("x", "changed")); self.assertFalse(c.correction.flags.writeable)
        self.assertIsInstance(CorrectionLedger().to_dicts(), list)
    def test_patch_replace_remove_is_atomic(self):
        p=self.problem(); a=AnalyticLearner("a",np.array([1.,0.,0.]),"a"); b=AnalyticLearner("b",np.array([0.,1.,0.]),"b"); s=CEMSolver(p,(a,b),LinearCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(3))); s.step(); cb=b.propose(p,s.state,p.residual(s.state),{}); s.patch(append=(cb,)); self.assertEqual(s.state.active_element_ids,("a","b")); s.patch(remove=("b",)); self.assertEqual(s.state.active_element_ids,("a",)); before=s.state.hash(); bad=Candidate("a","a",np.array([0.,0.,1.]),np.array([1.,0.,0.]),{}, {},None,{}); self.assertRaises(ValueError, lambda: s.patch(replace={"a":bad})); self.assertEqual(before,s.state.hash()); self.assertRaises(ValueError, lambda: s.patch(remove=("missing",))); self.assertTrue(any(r.status == "patch_accepted" for r in s.ledger.records)); self.assertTrue(any(r.status.startswith("patch_rejected") for r in s.ledger.records))
    def test_fully_corrective_requires_linear_problem(self):
        p=NonlinearPDEProblem(lambda x: x, lambda x: float(np.dot(x,x))); self.assertRaises(TypeError, lambda: CEMSolver(p,(),NonlinearCandidateScorer(),FullyCorrectivePolicy(),SolverState(np.zeros(2))))
    def test_matrix_free_callable_operator(self):
        op=CallableOperator((2,2), lambda x: np.array([x[0]+2*x[1], 3*x[0]-x[1]])); p=LinearPDEProblem(op,np.array([1.,0.])); a=AnalyticLearner("a",np.array([1.,0.]),"a"); s=CEMSolver(p,(a,),LinearCandidateScorer(),MeritDecreasePolicy(),SolverState(np.zeros(2))); self.assertIsNotNone(s.step()); self.assertEqual(s.state.active_element_ids,("a",)); self.assertTrue(np.all(np.isfinite(s.state.values)))
        class MinimalOperator:
            shape=(2,2)
            def apply(self, x): return np.array([x[0]+2*x[1], 3*x[0]-x[1]])
        p2=LinearPDEProblem(MinimalOperator(),np.array([1.,0.])); self.assertTrue(np.allclose(p2.apply_operator(np.array([1.,2.])),[5.,1.]))
        dense=NumpyMatrixOperator(np.eye(2)); self.assertFalse(dense.matrix.flags.writeable); self.assertRaises(ValueError, lambda: dense.matrix.__setitem__((0,0), 4.))

if __name__ == "__main__": unittest.main()
