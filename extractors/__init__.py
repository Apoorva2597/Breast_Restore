from __future__ import annotations

from typing import List

from ..models import SectionedNote, Candidate

from .bmi import extract_bmi
from .smoking import extract_smoking
from .comorbidities import extract_comorbidities
from .procedures import extract_reconstruction, extract_lymph_node_mgmt



def extract_all(sec: SectionedNote) -> List[Candidate]:
    """Run all extractors and return a flat list of candidates."""
    cands: List[Candidate] = []
    cands.extend(extract_bmi(sec))
    cands.extend(extract_smoking(sec))
    cands.extend(extract_comorbidities(sec))
    cands.extend(extract_reconstruction(sec))
    cands.extend(extract_lymph_node_mgmt(sec))
    return cands
