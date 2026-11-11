import unittest
import json
import numpy as np
from core import *
from enrichment import *
from problems import *
from solver import *
from strategies import *

class CostTests(unittest.TestCase):
    def test_measured_calls_attach_to_step_record(self):
        p=LinearPDEProblem(np.eye(2),np.array([1.,0.])); wl=WorkLedger(run_id="run"); s=CEMSolver(p,(AnalyticLearner("a",np.array([1.,0.]),"a"),),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(2)),work_ledger=wl); self.assertIsNotNone(s.step()); quantities=[e.quantity for e in wl.events]; self.assertEqual(quantities.count("proposal_calls"),1); self.assertEqual(quantities.count("score_eval_calls"),1); self.assertGreater(quantities.count("operator_apply_calls"),0); refs=s.ledger.records[-1].work_event_ids; self.assertEqual(len(refs),len(set(refs))); self.assertTrue(refs); self.assertTrue(set(refs).issubset({e.event_id for e in wl.events}))
    def test_failed_attempt_keeps_failed_work_event(self):
        class Bad:
            family="bad"
            def propose(self,*args): raise RuntimeError("fail")
        p=LinearPDEProblem(np.eye(1),np.ones(1)); wl=WorkLedger(run_id="run"); s=CEMSolver(p,(Bad(),),LinearCandidateScorer(),StagewisePolicy(),SolverState(np.zeros(1)),work_ledger=wl); self.assertIsNone(s.step()); failed=[e for e in wl.events if e.status=="failed"]; self.assertTrue(failed); self.assertTrue(any(e.quantity=="proposal_calls" for e in failed))
    def test_span_and_summary_do_not_merge_units(self):
        wl=WorkLedger(run_id="run"); step=wl.new_step(); attempt=wl.new_attempt(); wl.record_call("proposal","proposal_calls",solver_step_id=step,candidate_attempt_id=attempt)
        with wl.span("proposal",solver_step_id=step,candidate_attempt_id=attempt): pass
        summary=wl.summary(); self.assertIn(("proposal_calls","call","completed","observed","exclusive"),summary); self.assertIn(("wall_time","second","completed","observed","exclusive"),summary); self.assertEqual(sum(v for k,v in summary.items() if k[0]=="proposal_calls"),1)
        self.assertTrue(all(e.timing_semantics in ("none","exclusive") for e in wl.events))
    def test_schema_rejects_unimplemented_shared_and_invalid_timing(self):
        wl=WorkLedger(run_id="run")
        def event(**overrides):
            values=dict(event_id="manual",run_id="run",solver_step_id=None,selection_id=None,candidate_id=None,candidate_attempt_id=None,parent_attempt_id=None,retry_index=0,phase="x",quantity="proposal_calls",value=1,unit="call",observed_state="observed",status="completed",parent_event_id=None,start_monotonic=None,end_monotonic=None,duration_seconds=None,timing_semantics="none",shared_scope="exclusive",owner_id="run",allocation_key=None,metadata={})
            values.update(overrides); return CostEvent(**values)
        with self.assertRaises(ValueError): wl.append(event(shared_scope="shared_raw"))
        with self.assertRaises(ValueError): wl.append(event(quantity="wall_time",unit="second"))
        with self.assertRaises(ValueError): wl.append(event(quantity="wall_time",unit="second",timing_semantics="exclusive",start_monotonic=1.0,end_monotonic=2.0,duration_seconds=2.0))
    def test_metadata_is_recursively_snapshotted_and_export_is_detached(self):
        wl=WorkLedger(run_id="run"); source={"nested":{"items":[1,np.int64(2)]},"array":np.array([[3,4]],dtype=np.int64)}
        event_id=wl.record_call("proposal","proposal_calls",metadata=source); event=wl.events[-1]
        source["nested"]["items"].append(9); source["array"][0,0]=99; source["nested"]["new"]="changed"
        self.assertEqual(event.event_id,event_id); self.assertEqual(event.metadata["nested"]["items"],(1,2)); self.assertEqual(event.metadata["array"],((3,4),)); self.assertNotIn("new",event.metadata["nested"])
        with self.assertRaises(TypeError): event.metadata["new"]="x"
        with self.assertRaises(TypeError): event.metadata["nested"]["new"]="x"
        with self.assertRaises(AttributeError): event.metadata["nested"]["items"].append(3)
        with self.assertRaises(TypeError): event.metadata["array"][0][0]=7
        first_export=wl.to_dicts(); first_export[-1]["metadata"]["nested"]["items"].append(77); first_export[-1]["metadata"]["array"][0][0]=88
        second_export=wl.to_dicts(); self.assertEqual(second_export[-1]["metadata"]["nested"]["items"],[1,2]); self.assertEqual(second_export[-1]["metadata"]["array"],[[3,4]])
        json.dumps(second_export,allow_nan=False)
    def test_invalid_metadata_fails_before_event_append(self):
        wl=WorkLedger(run_id="run"); cyclic={}; cyclic["self"]=cyclic; before=len(wl.events)
        with self.assertRaises(ValueError): wl.record_call("proposal","proposal_calls",metadata=cyclic)
        with self.assertRaises(TypeError): wl.record_call("proposal","proposal_calls",metadata={"bad":object()})
        with self.assertRaises(TypeError): wl.record_call("proposal","proposal_calls",metadata={1:"non-string key"})
        with self.assertRaises(ValueError): wl.record_call("proposal","proposal_calls",metadata={"bad":float("inf")})
        self.assertEqual(len(wl.events),before)

if __name__ == "__main__": unittest.main()
