from dataclasses import dataclass, fields
from typing import Any
from collections.abc import Mapping
import numpy as np

def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

@dataclass
class CorrectionRecord:
    element_id: str; family: str; support: Any; parameters: Any; proposal_cost: Any
    scoring_cost: Any; predicted_reduction: float; before: float; after: float
    accepted: bool; status: str; state_hash_before: str; state_hash_after: str
    touched_element_ids: tuple[str, ...] = (); frozen_element_ids: tuple[str, ...] = ()

class CorrectionLedger:
    def __init__(self): self.records = []
    def append(self, record): self.records.append(record)
    def to_dicts(self):
        # Avoid dataclasses.asdict/deepcopy: Candidate metadata uses
        # MappingProxyType intentionally and is not deepcopy-able.
        return [{f.name: _plain(getattr(record, f.name)) for f in fields(record)}
                for record in self.records]
