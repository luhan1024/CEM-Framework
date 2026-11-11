from dataclasses import dataclass, fields
from typing import Any
from collections.abc import Mapping
import hashlib
import json
import re
import struct
import numpy as np


MAX_PROPOSAL_CERTIFICATE_BYTES = 4_000_000
MAX_PROPOSAL_AUDIT_BYTES = 8_000_000
MAX_TERMINAL_AUDIT_BYTES = 16_384
MAX_ATTEMPT_WORK_EVENT_IDS = 32
MAX_WORK_EVENT_ID_BYTES = 128


class ProposalAuditTooLarge(ValueError):
    def __init__(self, attempted_bytes: int) -> None:
        self.attempted_bytes = int(attempted_bytes)
        super().__init__("proposal_audit_oversize")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _verify_p1_candidate_envelope_binding(payload: Mapping) -> bool:
    """Check that a candidate envelope is internally bound to its P1 hat cert."""
    try:
        support = payload.get("support")
        parameters = payload.get("parameters")
        if not isinstance(support, Mapping) or not isinstance(parameters, Mapping):
            return False
        node = support.get("node_index")
        cells = support.get("cell_indices")
        if (type(node) is not int or not isinstance(cells, list)
                or any(type(value) is not int for value in cells)
                or cells != [node - 1, node]
                or parameters.get("basis") != "interior_p1_hat"
                or type(parameters.get("node_index")) is not int
                or parameters.get("node_index") != node):
            return False

        certificate = json.loads(payload["certificate_record_json"])
        if canonical_json_bytes(certificate) != payload["certificate_record_json"].encode("utf-8"):
            return False
        if certificate.get("status") != payload.get("certificate_status"):
            return False
        certificate_input = certificate.get("input")
        if not isinstance(certificate_input, Mapping):
            return False
        nodes_hex = certificate_input.get("nodes_hex")
        direction_hex = certificate_input.get("direction_hex")
        if (not isinstance(nodes_hex, list) or len(nodes_hex) < 3
                or not isinstance(direction_hex, list)
                or len(direction_hex) != len(nodes_hex)):
            return False
        cells_count = len(nodes_hex) - 1
        if not 1 <= node < cells_count:
            return False

        nodes = []
        direction = []
        for value in nodes_hex:
            if not isinstance(value, str) or len(value) != 16:
                return False
            parsed = struct.unpack(">d", bytes.fromhex(value))[0]
            if not np.isfinite(parsed):
                return False
            nodes.append(parsed)
        for value in direction_hex:
            if not isinstance(value, str) or len(value) != 16:
                return False
            parsed = struct.unpack(">d", bytes.fromhex(value))[0]
            if not np.isfinite(parsed):
                return False
            direction.append(parsed)
        if any(right <= left for left, right in zip(nodes, nodes[1:])):
            return False
        expected_direction = [0.0] * len(direction)
        expected_direction[node] = 1.0
        if any(struct.pack(">d", actual) != struct.pack(">d", expected)
               for actual, expected in zip(direction, expected_direction)):
            return False

        node_array = np.asarray(nodes, dtype="<f8")
        mesh_descriptor = json.dumps(
            {"shape": list(node_array.shape), "dtype": "binary64-le"},
            sort_keys=True, separators=(",", ":")).encode("ascii")
        mesh_sha256 = hashlib.sha256(
            mesh_descriptor + node_array.tobytes(order="C")).hexdigest()
        problem_fingerprint = payload.get("problem_fingerprint")
        expected_id = (f"p1hat-v1:{problem_fingerprint}:{mesh_sha256}:node:{node}")
        if payload.get("element_id") != expected_id:
            return False

        expected_candidate = np.zeros(cells_count - 1, dtype=np.float64)
        expected_candidate[node - 1] = 1.0
        fingerprint = hashlib.sha256(
            expected_candidate.tobytes(order="C")).hexdigest()
        return payload.get("candidate_fingerprint") == fingerprint
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError,
            json.JSONDecodeError, struct.error):
        return False


def verify_proposal_audit(payload_json: str, payload_sha256: str, *,
                          problem=None, state=None) -> bool:
    if not isinstance(payload_json, str) or not isinstance(payload_sha256, str):
        return False
    try:
        encoded = payload_json.encode("utf-8")
        if len(encoded) > MAX_PROPOSAL_AUDIT_BYTES:
            return False
        payload = json.loads(payload_json)
        if canonical_json_bytes(payload) != encoded:
            return False
        if hashlib.sha256(encoded).hexdigest() != payload_sha256:
            return False
        if payload.get("schema") != "cem.p1_proposal_audit.v1":
            return False
        if payload.get("outcome") == "verified_candidate":
            certificate_json = payload.get("certificate_record_json")
            certificate_sha256 = payload.get("certificate_sha256")
            if (not isinstance(payload.get("element_id"), str)
                    or not isinstance(payload.get("support"), Mapping)
                    or not isinstance(payload.get("parameters"), Mapping)
                    or not isinstance(payload.get("work_audit"), Mapping)
                    or payload.get("certificate_status") != "verified_positive_gain"
                    or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("problem_fingerprint", "")))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("candidate_fingerprint", "")))
                    or not isinstance(certificate_json, str)
                    or not isinstance(certificate_sha256, str)):
                return False
            certificate_bytes = certificate_json.encode("utf-8")
            if (len(certificate_bytes) > MAX_PROPOSAL_CERTIFICATE_BYTES
                    or hashlib.sha256(certificate_bytes).hexdigest() != certificate_sha256
                    or not _verify_p1_candidate_envelope_binding(payload)):
                return False
            if (problem is None) != (state is None):
                return False
            if problem is not None:
                validator = getattr(problem, "validate_p1_audit_binding", None)
                if not callable(validator) or not validator(payload, state):
                    return False
            return True
        if payload.get("outcome") == "typed_failure":
            return (payload.get("element_id") is None
                    and isinstance(payload.get("proposal_status"), str)
                    and bool(payload["proposal_status"])
                    and isinstance(payload.get("reason_code"), str)
                    and bool(payload["reason_code"])
                    and isinstance(payload.get("work_audit"), Mapping)
                    and re.fullmatch(r"[0-9a-f]{64}", str(payload.get("problem_fingerprint", "")))
                    and "certificate_record_json" not in payload
                    and "certificate_sha256" not in payload)
        return False
    except (AttributeError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        return False

def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def correction_record_dict(record):
    return {item.name: _plain(getattr(record, item.name))
            for item in fields(record)}

@dataclass
class CorrectionRecord:
    element_id: str | None; family: str | None; support: Any; parameters: Any; proposal_cost: Any
    scoring_cost: Any; predicted_reduction: float; before: float; after: float
    accepted: bool; status: str; state_hash_before: str; state_hash_after: str
    touched_element_ids: tuple[str, ...] = (); frozen_element_ids: tuple[str, ...] = (); work_event_ids: tuple[str, ...] = ()
    record_schema_version: int = 2
    state_hash_algorithm_version: int = 2
    proposal_status: str | None = None
    reason_code: str | None = None
    proposal_audit_json: str | None = None
    proposal_audit_sha256: str | None = None

@dataclass
class PoolAuditRecord:
    pool_id: str; rule: str; pool_size: int; candidates: Any
    pool_max_gain: float; selected_gain: float; finite_pool_realized_ratio: float | None
    ratio_reason: str; accepted: bool; status: str; proposal_costs: Any
    scoring_statuses: Any; before: float; after: float; state_hash_before: str; state_hash_after: str; work_event_ids: tuple[str, ...] = ()
    generation_metadata: Any = None; budget: Any = None; selector_identity: str = ""; selector_rule: str = ""
    record_schema_version: int = 2
    state_hash_algorithm_version: int = 2

class CorrectionLedger:
    def __init__(self): self.records = []
    def append(self, record): self.records.append(record)
    @classmethod
    def from_dicts(cls, records):
        """Load legacy v1 and current v2 serialized correction/pool records."""
        ledger = cls()
        for index, raw in enumerate(records):
            if not isinstance(raw, Mapping):
                raise TypeError(f"ledger record {index} must be a mapping")
            data = dict(raw)
            version = data.get("record_schema_version", 1)
            if type(version) is not int or version not in (1, 2):
                raise ValueError(f"unsupported ledger record schema version at {index}")
            record_type = PoolAuditRecord if "pool_id" in data else CorrectionRecord
            known = {item.name for item in fields(record_type)}
            unknown = set(data) - known
            if unknown:
                raise ValueError(f"unknown ledger record fields at {index}: {sorted(unknown)}")
            if version == 1:
                data["record_schema_version"] = 1
                data["state_hash_algorithm_version"] = 1
                if record_type is CorrectionRecord:
                    data.setdefault("proposal_status", None)
                    data.setdefault("reason_code", None)
                    data.setdefault("proposal_audit_json", None)
                    data.setdefault("proposal_audit_sha256", None)
            else:
                hash_version = data.get("state_hash_algorithm_version")
                if (type(hash_version) is not int or hash_version not in (1, 2)):
                    raise ValueError(f"v2 ledger record has invalid state hash version at {index}")
            ledger.append(record_type(**data))
        return ledger
    def to_dicts(self):
        # Avoid dataclasses.asdict/deepcopy: Candidate metadata uses
        # MappingProxyType intentionally and is not deepcopy-able.
        return [correction_record_dict(record) for record in self.records]
