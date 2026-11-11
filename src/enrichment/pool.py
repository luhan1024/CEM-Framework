"""Opt-in finite candidate-pool selection diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
import hashlib, json
import numpy as np

def _normalize(value, path="metadata", active=None):
    if active is None: active = set()
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, np.generic): return _normalize(value.item(), path, active)
    if isinstance(value, float):
        if not np.isfinite(value): raise ValueError(f"{path} contains non-finite number")
        return value
    if isinstance(value, np.ndarray):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return _normalize(value.tolist(), path, active)
        finally: active.remove(identity)
    if isinstance(value, Mapping):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try:
            out={}
            for key, item in value.items():
                if not isinstance(key, str): raise TypeError(f"{path} keys must be strings")
                out[key]=_normalize(item, f"{path}.{key}", active)
            return out
        finally: active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return [_normalize(x, f"{path}[{i}]", active) for i,x in enumerate(value)]
        finally: active.remove(identity)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")

def _freeze(value):
    if isinstance(value, Mapping): return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(_freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        out=value.copy(); out.setflags(write=False); return out
    return value

def _plain(value):
    if isinstance(value, Mapping): return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, tuple): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value

def candidate_hash(candidate):
    payload = {
        "element_id": candidate.element_id, "family": candidate.family,
        "correction": candidate.correction.tolist(),
        "operator_image": None if candidate.operator_image is None else candidate.operator_image.tolist(),
        "support": _plain(candidate.support), "parameters": _plain(candidate.parameters),
        "seed": candidate.seed, "provenance": _plain(candidate.provenance),
        "proposal_cost": _plain(candidate.proposal_cost), "version": candidate.version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def safe_gain(score):
    try: return float(getattr(score, "predicted_reduction"))
    except (AttributeError, TypeError, ValueError): return np.nan

@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple
    generation_metadata: Mapping = None
    proposal_failures: tuple = ()
    active_element_ids: tuple = ()
    _candidate_hashes: tuple = field(init=False, repr=False)
    _pool_id: str = field(init=False, repr=False)
    def __post_init__(self):
        candidates=tuple(self.candidates); ids=tuple(c.element_id for c in candidates)
        if any(not isinstance(eid, str) for eid in ids): raise TypeError("candidate element_id must be str")
        if len(set(ids)) != len(ids): raise ValueError("candidate pool contains duplicate element_id")
        active_ids=tuple(self.active_element_ids)
        if any(not isinstance(eid, str) for eid in active_ids): raise TypeError("active_element_ids must be strings")
        if len(set(active_ids)) != len(active_ids): raise ValueError("active_element_ids contains duplicates")
        active=set(active_ids)
        if active.intersection(ids): raise ValueError("candidate pool contains active element_id")
        if any(not isinstance(item, Mapping) for item in self.proposal_failures): raise TypeError("proposal_failures must be mappings")
        object.__setattr__(self, "candidates", candidates)
        if self.generation_metadata is not None and not isinstance(self.generation_metadata, Mapping): raise TypeError("generation_metadata must be mapping-like")
        normalized_generation=_normalize({} if self.generation_metadata is None else self.generation_metadata, "generation_metadata")
        normalized_failures=tuple(_normalize(x, f"proposal_failures[{i}]") for i,x in enumerate(self.proposal_failures))
        object.__setattr__(self, "generation_metadata", _freeze(normalized_generation))
        object.__setattr__(self, "proposal_failures", tuple(_freeze(x) for x in normalized_failures))
        object.__setattr__(self, "active_element_ids", active_ids)
        hashes=tuple(candidate_hash(c) for c in candidates)
        object.__setattr__(self, "_candidate_hashes", hashes)
        canonical = {"candidates": sorted((c.element_id, h) for c,h in zip(candidates, hashes)),
                     "generation_metadata": _plain(self.generation_metadata),
                     "proposal_failures": sorted(json.dumps(_plain(x), sort_keys=True, separators=(",", ":"), allow_nan=False) for x in self.proposal_failures),
                     "active_element_ids": sorted(active_ids)}
        object.__setattr__(self, "_pool_id", hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    @property
    def pool_id(self): return self._pool_id
    def hash_for(self, candidate):
        for c,h in zip(self.candidates, self._candidate_hashes):
            if c is candidate: return h
        raise ValueError("candidate is not a member of this pool")
    def __len__(self): return len(self.candidates)

def freeze_metadata(value, name="metadata"):
    """Validate and freeze JSON-friendly caller-declared provenance."""
    return _freeze(_normalize({} if value is None else value, name))

def plain_metadata(value):
    return _plain(value)

@dataclass(frozen=True)
class PoolCandidateResult:
    candidate: object
    score: object = None
    proposal_status: str = "proposed"
    scoring_status: str = "not_scored"
    error: str = ""
    candidate_hash_value: str | None = None
    def as_dict(self):
        score=self.score; gain=safe_gain(score) if score is not None else None
        score_reason=None if score is None else getattr(score, "reason", "")
        if score is not None and not np.isfinite(gain): score_reason=score_reason or "malformed_predicted_reduction"
        return {"element_id": None if self.candidate is None else self.candidate.element_id,
                "candidate_hash": self.candidate_hash_value if self.candidate is not None and self.candidate_hash_value is not None else (None if self.candidate is None else candidate_hash(self.candidate)),
                "proposal_status": self.proposal_status, "scoring_status": self.scoring_status,
                "error": self.error, "finite": None if score is None else bool(getattr(score, "finite", False)),
                "accepted_shape": None if score is None else bool(getattr(score, "accepted_shape", False)),
                "score_reason": score_reason,
                "predicted_reduction": gain}

@dataclass(frozen=True)
class SelectionResult:
    selected: PoolCandidateResult | None
    results: tuple
    pool_max_gain: float
    ratio: float | None
    ratio_reason: str
    rule: str = "best_predicted_gain"

class BestPredictedGainSelector:
    name = "best_predicted_gain"
    def select(self, pool, scored_candidates):
        results=tuple(scored_candidates); valid=[]
        for item in results:
            score=item.score
            gain=np.nan if score is None else safe_gain(score)
            if item.candidate is not None and item.scoring_status == "scored_valid" and bool(getattr(score, "finite", False)) and bool(getattr(score, "accepted_shape", False)) and getattr(score, "reason", "") == "" and np.isfinite(gain) and gain > 0:
                valid.append((gain, item.candidate.element_id, item.candidate_hash_value or candidate_hash(item.candidate), item))
        if not valid:
            return SelectionResult(None, results, np.nan, None, "no_valid_positive_gain")
        valid.sort(key=lambda x: (-x[0], x[1], x[2])); maximum=valid[0][0]; chosen=valid[0][3]
        ratio=chosen.score.predicted_reduction / maximum
        return SelectionResult(chosen, results, maximum, float(ratio), "ok")
