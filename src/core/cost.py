"""Append-only measured work events; estimates remain outside this ledger."""
from __future__ import annotations
from dataclasses import dataclass, fields, replace
from contextlib import contextmanager
from collections import defaultdict
from collections.abc import Mapping
import time, uuid
import numpy as np
from contextvars import ContextVar
from types import MappingProxyType

@dataclass(frozen=True)
class CostEvent:
    event_id: str; run_id: str; solver_step_id: str | None; selection_id: str | None
    candidate_id: str | None; candidate_attempt_id: str | None; parent_attempt_id: str | None
    retry_index: int; phase: str; quantity: str; value: int | float; unit: str
    observed_state: str; status: str; parent_event_id: str | None
    start_monotonic: float | None; end_monotonic: float | None; duration_seconds: float | None
    timing_semantics: str; shared_scope: str; owner_id: str; allocation_key: str | None
    metadata: Mapping

def _freeze_metadata(value, path="metadata", active=None):
    """Copy JSON-compatible metadata into recursively immutable containers."""
    if active is None: active=set()
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, np.generic): return _freeze_metadata(value.item(), path, active)
    if isinstance(value, float):
        if not np.isfinite(value): raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, np.ndarray):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return _freeze_metadata(value.tolist(), path, active)
        finally: active.remove(identity)
    if isinstance(value, Mapping):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try:
            frozen={}
            for key, item in value.items():
                if not isinstance(key, str): raise TypeError(f"{path} keys must be strings")
                frozen[key]=_freeze_metadata(item, f"{path}.{key}", active)
            return MappingProxyType(frozen)
        finally: active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity=id(value)
        if identity in active: raise ValueError(f"{path} contains a cycle")
        active.add(identity)
        try: return tuple(_freeze_metadata(item, f"{path}[{i}]", active) for i,item in enumerate(value))
        finally: active.remove(identity)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")

def _plain_metadata(value):
    """Return detached JSON-compatible data from frozen metadata."""
    if isinstance(value, Mapping): return {key:_plain_metadata(item) for key,item in value.items()}
    if isinstance(value, tuple): return [_plain_metadata(item) for item in value]
    if isinstance(value, np.ndarray): return _plain_metadata(value.tolist())
    if isinstance(value, np.generic): return _plain_metadata(value.item())
    return value

class WorkLedger:
    def __init__(self, run_id=None): self.run_id=run_id or uuid.uuid4().hex; self._events=[]; self._counter=0; self._context_var=ContextVar(f"workledger:{id(self)}", default={})
    @property
    def events(self): return tuple(self._events)
    def _id(self, prefix): self._counter+=1; return f"{prefix}:{self._counter}"
    def new_step(self): return self._id("step")
    def new_selection(self): return self._id("selection")
    def new_attempt(self): return self._id("attempt")
    @contextmanager
    def context(self, **values):
        token=self._context_var.set({**self._context_var.get(), **values})
        try: yield
        finally: self._context_var.reset(token)
    def _context(self, values):
        merged=dict(self._context_var.get())
        merged.update({k:v for k,v in values.items() if v is not None})
        return merged
    def append(self, event):
        if not isinstance(event, CostEvent): raise TypeError("event must be CostEvent")
        if event.event_id in {e.event_id for e in self._events}: raise ValueError("duplicate event_id")
        if not isinstance(event.event_id, str) or not event.event_id: raise ValueError("event_id must be non-empty text")
        if isinstance(event.retry_index, (bool, np.bool_)) or not isinstance(event.retry_index, (int, np.integer)) or event.retry_index < 0: raise ValueError("retry_index must be a nonnegative integer")
        if event.observed_state not in ("observed", "estimated", "derived"): raise ValueError("invalid observed_state")
        if event.status not in ("completed", "failed", "aborted"): raise ValueError("invalid cost event status")
        allowed={"operator_apply_calls":"call","objective_eval_calls":"call","score_eval_calls":"call","proposal_calls":"call","refit_calls":"call","wall_time":"second"}
        if event.quantity not in allowed or event.unit != allowed[event.quantity]: raise ValueError("unsupported quantity/unit pair")
        try: finite_value = bool(np.isfinite(event.value))
        except (TypeError, ValueError): finite_value = False
        if not finite_value or event.value < 0: raise ValueError("cost event value must be finite and nonnegative")
        if event.quantity.endswith("_calls") and (isinstance(event.value, (bool, np.bool_)) or not isinstance(event.value, (int, np.integer))): raise ValueError("call event value must be integer")
        if event.timing_semantics not in ("none", "exclusive"): raise ValueError("unsupported timing semantics")
        if event.shared_scope != "exclusive": raise ValueError("unsupported shared_scope")
        if event.quantity == "wall_time" and event.timing_semantics != "exclusive": raise ValueError("wall_time requires exclusive timing")
        if event.timing_semantics != "none":
            try: ordered = event.start_monotonic is not None and event.end_monotonic is not None and event.end_monotonic >= event.start_monotonic
            except TypeError: ordered = False
            if not ordered: raise ValueError("invalid wall span")
        if event.timing_semantics != "none":
            vals=(event.start_monotonic,event.end_monotonic,event.duration_seconds)
            if any(x is None or not np.isfinite(x) or x < 0 for x in vals): raise ValueError("invalid wall span values")
            if not np.isclose(event.duration_seconds, event.end_monotonic - event.start_monotonic, rtol=1e-6, atol=1e-9): raise ValueError("duration_seconds must match wall span")
        if event.timing_semantics == "none" and any(x is not None for x in (event.start_monotonic, event.end_monotonic, event.duration_seconds)): raise ValueError("untimed event cannot carry timing fields")
        if event.timing_semantics != "none" and event.duration_seconds is None: raise ValueError("timed event requires duration_seconds")
        if event.metadata is not None and not isinstance(event.metadata, Mapping): raise TypeError("metadata must be mapping-like")
        metadata=_freeze_metadata({} if event.metadata is None else event.metadata)
        event=replace(event, metadata=metadata); self._events.append(event); return event.event_id
    def record_call(self, phase, quantity, *, solver_step_id=None, selection_id=None, candidate_id=None, candidate_attempt_id=None, status="completed", metadata=None):
        if quantity not in ("operator_apply_calls", "objective_eval_calls", "score_eval_calls", "proposal_calls", "refit_calls"): raise ValueError("unsupported call quantity")
        ctx=self._context({"solver_step_id":solver_step_id,"selection_id":selection_id,"candidate_id":candidate_id,"candidate_attempt_id":candidate_attempt_id}); solver_step_id=ctx.get("solver_step_id"); selection_id=ctx.get("selection_id"); candidate_id=ctx.get("candidate_id"); candidate_attempt_id=ctx.get("candidate_attempt_id")
        event=CostEvent(self._id("event"),self.run_id,solver_step_id,selection_id,candidate_id,candidate_attempt_id,None,0,phase,quantity,1,"call","observed",status,None,None,None,None,"none","exclusive",candidate_attempt_id or solver_step_id or self.run_id,None,metadata); event_id=self.append(event); collector=ctx.get("event_ids"); collector.append(event_id) if collector is not None else None; return event_id
    @contextmanager
    def span(self, phase, *, solver_step_id=None, selection_id=None, candidate_id=None, candidate_attempt_id=None, shared_scope="exclusive", owner_id=None, metadata=None, event_ids=None):
        ctx=self._context({"solver_step_id":solver_step_id,"selection_id":selection_id,"candidate_id":candidate_id,"candidate_attempt_id":candidate_attempt_id,"event_ids":event_ids}); solver_step_id=ctx.get("solver_step_id"); selection_id=ctx.get("selection_id"); candidate_id=ctx.get("candidate_id"); candidate_attempt_id=ctx.get("candidate_attempt_id"); event_id=self._id("event"); start=time.perf_counter(); status="completed"
        try: yield event_id
        except Exception:
            status="failed"; raise
        finally:
            end=time.perf_counter(); event=CostEvent(event_id,self.run_id,solver_step_id,selection_id,candidate_id,candidate_attempt_id,None,0,phase,"wall_time",end-start,"second","observed",status,None,start,end,end-start,"exclusive",shared_scope,owner_id or candidate_attempt_id or solver_step_id or self.run_id,None,metadata); self.append(event); collector=ctx.get("event_ids"); collector.append(event_id) if collector is not None else None
    def summary(self):
        out=defaultdict(float)
        for event in self.events:
            key=(event.quantity,event.unit,event.status,event.observed_state,event.shared_scope); out[key]+=event.value
        return dict(out)
    def to_dicts(self):
        return [{f.name:_plain_metadata(getattr(e,f.name)) for f in fields(e)} for e in self._events]
