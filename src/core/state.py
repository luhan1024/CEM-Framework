from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import struct
import numpy as np

STATE_HASH_ALGORITHM_VERSION = 2


def _hash_field(digest, name: bytes, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(name)))
    digest.update(name)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _array_hash_payload(array: np.ndarray) -> bytes:
    dtype = array.dtype.str.encode("ascii")
    shape = struct.pack(">I", array.ndim) + b"".join(
        struct.pack(">Q", int(size)) for size in array.shape)
    raw = array.tobytes(order="C")
    return (struct.pack(">I", len(dtype)) + dtype + shape
            + struct.pack(">Q", len(raw)) + raw)


def snapshots_equal(left: "StateSnapshot", right: "StateSnapshot") -> bool:
    """Compare state snapshots by exact field presence, metadata, and bytes."""
    if not isinstance(left, StateSnapshot) or not isinstance(right, StateSnapshot):
        return False
    if (left.values.dtype != right.values.dtype
            or left.values.shape != right.values.shape
            or left.values.tobytes(order="C") != right.values.tobytes(order="C")
            or tuple(left.active_element_ids) != tuple(right.active_element_ids)):
        return False
    left_coeff = left.outer_coefficients
    right_coeff = right.outer_coefficients
    if (left_coeff is None) != (right_coeff is None):
        return False
    if left_coeff is None:
        return True
    return (left_coeff.dtype == right_coeff.dtype
            and left_coeff.shape == right_coeff.shape
            and left_coeff.tobytes(order="C") == right_coeff.tobytes(order="C"))


def immutable_snapshot(snapshot: "StateSnapshot") -> "StateSnapshot":
    """Return an isolated snapshot backed by immutable ``bytes`` objects.

    ``ndarray.setflags(write=False)`` is reversible when an array owns its
    memory.  A bytes-backed array cannot be made writable, so this helper is
    suitable for private rollback baselines that must never be exposed to a
    learner or other callback.
    """
    if not isinstance(snapshot, StateSnapshot):
        raise TypeError("snapshot must be a StateSnapshot")

    def _bytes_backed(array):
        source = np.asarray(array)
        return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(
            source.shape)

    return StateSnapshot(
        _bytes_backed(snapshot.values),
        tuple(snapshot.active_element_ids),
        None if snapshot.outer_coefficients is None
        else _bytes_backed(snapshot.outer_coefficients),
    )

def _copy(x):
    return np.array(x, copy=True)

@dataclass
class SolverState:
    values: np.ndarray
    active_element_ids: tuple[str, ...] = ()
    outer_coefficients: np.ndarray | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float).copy()
        if self.values.ndim != 1:
            raise ValueError("SolverState.values must be one-dimensional")
        if self.outer_coefficients is not None:
            self.outer_coefficients = np.asarray(self.outer_coefficients, dtype=float).copy()

    def snapshot(self) -> "StateSnapshot":
        return StateSnapshot(_copy(self.values), tuple(self.active_element_ids), None if self.outer_coefficients is None else _copy(self.outer_coefficients))

    def restore(self, snap: "StateSnapshot") -> None:
        self.values = _copy(snap.values); self.active_element_ids = tuple(snap.active_element_ids)
        self.outer_coefficients = None if snap.outer_coefficients is None else _copy(snap.outer_coefficients)

    def hash(self) -> str:
        h = hashlib.sha256(b"cem.solver_state.v2\0")
        _hash_field(h, b"values", _array_hash_payload(self.values))
        ids = json.dumps(list(self.active_element_ids), ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
        _hash_field(h, b"active_element_ids", ids)
        if self.outer_coefficients is None:
            _hash_field(h, b"outer_coefficients", b"none")
        else:
            _hash_field(h, b"outer_coefficients", b"present\0" +
                        _array_hash_payload(self.outer_coefficients))
        return h.hexdigest()

@dataclass(frozen=True)
class StateSnapshot:
    values: np.ndarray
    active_element_ids: tuple[str, ...]
    outer_coefficients: np.ndarray | None

    def __post_init__(self):
        self.values.setflags(write=False)
        if self.outer_coefficients is not None: self.outer_coefficients.setflags(write=False)

class StateTransaction:
    def __init__(self, state: SolverState):
        self.state = state; self.before = state.snapshot(); self.committed = False
    def commit(self): self.committed = True
    def rollback(self): self.state.restore(self.before); self.committed = False
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self.committed: self.rollback()
        return False
