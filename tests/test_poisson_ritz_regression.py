import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


class PoissonRitzRegressionTests(unittest.TestCase):
    def test_fixed_grid_root_regression_is_deterministic(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "examples" / "poisson_ritz_regression.py"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        outputs = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, str(script)], cwd=root, env=env,
                check=True, capture_output=True, text=True,
            )
            outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])
        result = outputs[0]
        self.assertTrue(np.allclose(
            result["oracle"], [3.0 / 32.0, 1.0 / 8.0, 3.0 / 32.0],
            rtol=0.0, atol=1e-15,
        ))
        self.assertTrue(np.allclose(result["final_values"], result["oracle"],
                                    rtol=0.0, atol=1e-12))
        self.assertLessEqual(result["residual_inf_norm"], 1e-12)
        self.assertEqual(len(result["pool_records"]), 3)
        self.assertTrue(all(
            set(record["work_event_ids"]).issubset(set(result["ledger_event_ids"]))
            for record in result["pool_records"]
        ))
        history = result["objective_history"]
        self.assertTrue(all(history[index + 1] <= history[index] + 1e-13
                            for index in range(len(history) - 1)))


if __name__ == "__main__":
    unittest.main()
