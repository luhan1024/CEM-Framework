from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Mapping, Any
import hashlib
import numpy as np
from types import MappingProxyType

def _freeze(value):
    if isinstance(value, np.ndarray):
        out = value.copy(); out.setflags(write=False); return out
    if isinstance(value, dict): return MappingProxyType({k: _freeze(v) for k,v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
    return value

@dataclass(frozen=True)
class Candidate:
    element_id: str
    family: str
    correction: np.ndarray
    operator_image: np.ndarray | None
    support: Mapping[str, Any]
    parameters: Mapping[str, Any]
    seed: int | None
    provenance: Mapping[str, Any]
    proposal_cost: Mapping[str, float] = field(default_factory=dict)
    version: str = "1"
    def __post_init__(self):
        c = np.array(self.correction, dtype=float, copy=True)
        if c.ndim != 1 or not np.all(np.isfinite(c)): raise ValueError("candidate correction must be finite 1-D")
        c.setflags(write=False); object.__setattr__(self, "correction", c)
        if self.operator_image is not None:
            image = np.array(self.operator_image, dtype=float, copy=True); image.setflags(write=False); object.__setattr__(self, "operator_image", image)
        object.__setattr__(self, "support", _freeze(dict(self.support))); object.__setattr__(self, "parameters", _freeze(dict(self.parameters))); object.__setattr__(self, "provenance", _freeze(dict(self.provenance))); object.__setattr__(self, "proposal_cost", _freeze(dict(self.proposal_cost)))
    def fingerprint(self):
        return hashlib.sha256(self.correction.tobytes()).hexdigest()

class WeakLearner(Protocol):
    family: str
    def propose(self, problem, state, residual, budget) -> Candidate: ...

@dataclass
class AnalyticLearner:
    family: str
    basis: np.ndarray
    element_id: str
    support: Mapping[str, Any] = field(default_factory=dict)
    def propose(self, problem, state, residual, budget):
        phi = np.asarray(self.basis, float)
        if phi.ndim != 1: raise ValueError("analytic basis must be 1-D")
        image = problem.apply_operator(phi) if hasattr(problem, "apply_operator") else None
        return Candidate(self.element_id, self.family, phi, image, self.support, {"kind":"analytic"}, budget.get("seed"), {"learner":"analytic"}, {"proposal":0.0})

@dataclass
class LocalBasisLearner:
    family: str
    basis: np.ndarray
    element_id: str
    support: Mapping[str, Any]
    def propose(self, problem, state, residual, budget):
        phi = np.asarray(self.basis, float)
        image = problem.apply_operator(phi) if hasattr(problem, "apply_operator") else None
        return Candidate(self.element_id, self.family, phi, image, self.support, {"kind":"local_numeric"}, budget.get("seed"), {"learner":"local_basis"}, {"proposal":0.0})
