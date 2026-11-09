import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class HeterogeneousRitzTutorialTests(unittest.TestCase):
    def test_root_invocation_is_deterministic_and_audited(self):
        root = Path(__file__).resolve().parents[3]
        script = root / "Code" / "cem_framework" / "examples" / "heterogeneous_ritz_pool.py"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        outputs = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])
        result = outputs[0]
        self.assertTrue(result["not_a_pde_benchmark"])
        self.assertEqual(len(result["pools"]), 2)
        self.assertEqual(result["final_active_element_ids"], ["local-e2", "analytic-e1"])
        self.assertLess(result["objective_history"][-1], result["objective_history"][0])
        top_level_event_ids = set(result["ledger_event_ids"])
        for pool in result["pools"]:
            self.assertTrue(pool["ledger_event_ids"])
            self.assertTrue(set(pool["ledger_event_ids"]).issubset(top_level_event_ids))
            self.assertEqual(pool["selection_rule"], "best_predicted_gain")
            self.assertGreater(pool["score_prediction"], 0.0)
            self.assertLess(pool["objective_after"], pool["objective_before"])
            self.assertTrue(all("family" in candidate for candidate in pool["candidates"]))
            self.assertTrue(all("support" in candidate and "provenance" in candidate
                                for candidate in pool["candidates"]))
        first_families = {candidate["family"] for candidate in result["pools"][0]["candidates"]}
        self.assertEqual(first_families, {"analytic_global", "local_numeric"})


if __name__ == "__main__":
    unittest.main()
