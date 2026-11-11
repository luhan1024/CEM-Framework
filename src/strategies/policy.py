from __future__ import annotations
from dataclasses import dataclass

class AcceptancePolicy:
    def decide(self, score): raise NotImplementedError

@dataclass
class MeritDecreasePolicy(AcceptancePolicy):
    tolerance: float = 1e-12
    def decide(self, score): return bool(score.finite and score.after <= score.before - self.tolerance)

class StagewisePolicy:
    mode = "stagewise"
    tolerance = 1e-12
    def decide(self, score): return bool(score.finite and score.after <= score.before - self.tolerance)
    def update_coefficients(self, state, active, score): return score.alpha

class FullyCorrectivePolicy:
    mode = "fully_corrective"
    tolerance = 1e-12
    def decide(self, score): return bool(score.finite and score.after <= score.before - self.tolerance)
    def update_coefficients(self, state, active, score): return "refit_outer_coefficients"
