from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Mapping, Any
from collections.abc import Mapping as ABCMapping
import hashlib
import numpy as np
from types import MappingProxyType

def _freeze(value, path="metadata", active=None):
    if active is None: active = set()
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, np.generic): return _freeze(value.item(), path, active)
    if isinstance(value, float):
        if not np.isfinite(value): raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return _freeze(value.tolist(), path, active)
        finally: active.remove(identity)
    if isinstance(value, ABCMapping):
        identity = id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try:
            frozen = {}
            for key, item in value.items():
                if not isinstance(key, str): raise TypeError(f"{path} keys must be strings")
                frozen[key] = _freeze(item, f"{path}.{key}", active)
            return MappingProxyType(frozen)
        finally: active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return tuple(_freeze(item, f"{path}[{i}]", active) for i, item in enumerate(value))
        finally: active.remove(identity)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")

def _readonly_array(value):
    storage = np.array(value, dtype=float, copy=True)
    backing = storage.tobytes(order="C")
    view = np.frombuffer(backing, dtype=storage.dtype, count=storage.size).reshape(storage.shape)
    view.setflags(write=False)
    return view

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
        if isinstance(self.seed, (bool, np.bool_)):
            raise TypeError("candidate seed must be an integer or None")
        if isinstance(self.seed, np.integer):
            object.__setattr__(self, "seed", int(self.seed))
        elif self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("candidate seed must be an integer or None")
        c = _readonly_array(self.correction)
        if c.ndim != 1 or not np.all(np.isfinite(c)): raise ValueError("candidate correction must be finite 1-D")
        object.__setattr__(self, "correction", c)
        if self.operator_image is not None:
            image = _readonly_array(self.operator_image); object.__setattr__(self, "operator_image", image)
        object.__setattr__(self, "support", _freeze(self.support, "support")); object.__setattr__(self, "parameters", _freeze(self.parameters, "parameters")); object.__setattr__(self, "provenance", _freeze(self.provenance, "provenance")); object.__setattr__(self, "proposal_cost", _freeze(self.proposal_cost, "proposal_cost"))
    def fingerprint(self):
        return hashlib.sha256(self.correction.tobytes()).hexdigest()

class WeakLearner(Protocol):
    family: str
    def propose(self, problem, state, residual, budget) -> Candidate: ...


class ProposalFailure(RuntimeError):
    """Structured no-candidate or failed-proposal outcome for solver ledgers."""

    def __init__(self, status: str, reason_code: str,
                 work_audit: Mapping[str, Any]) -> None:
        if not isinstance(status, str) or not status:
            raise ValueError("proposal failure status must be nonempty text")
        if not isinstance(reason_code, str) or not reason_code:
            raise ValueError("proposal failure reason_code must be nonempty text")
        frozen_audit = _freeze(work_audit, "work_audit")
        if not isinstance(frozen_audit, ABCMapping):
            raise TypeError("proposal failure work_audit must be a mapping")
        self.status = status
        self.reason_code = reason_code
        self.work_audit = frozen_audit
        super().__init__(f"{status}:{reason_code}")

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
