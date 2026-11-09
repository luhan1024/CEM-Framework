import json
import numpy as np
from collections.abc import Mapping
def _plain(x):
    if isinstance(x, Mapping): return {k:_plain(v) for k,v in x.items()}
    if isinstance(x, tuple): return [_plain(v) for v in x]
    return x
from .learner import Candidate

def candidate_to_json(candidate: Candidate) -> str:
    return json.dumps({"element_id":candidate.element_id,"family":candidate.family,"correction":candidate.correction.tolist(),"operator_image":None if candidate.operator_image is None else candidate.operator_image.tolist(),"support":_plain(candidate.support),"parameters":_plain(candidate.parameters),"seed":candidate.seed,"provenance":_plain(candidate.provenance),"proposal_cost":_plain(candidate.proposal_cost),"version":candidate.version}, sort_keys=True)

def candidate_from_json(text: str) -> Candidate:
    d=json.loads(text); return Candidate(d["element_id"],d["family"],np.asarray(d["correction"],float),None if d["operator_image"] is None else np.asarray(d["operator_image"],float),d["support"],d["parameters"],d["seed"],d["provenance"],d["proposal_cost"],d["version"])
