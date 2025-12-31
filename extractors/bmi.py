from __future__ import annotations
import re
from typing import List
from ..models import Candidate, SectionedNote
from .utils import window_around

BMI_RX = re.compile(r"\bBMI\b\s*[:=]?\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)
BMI_RX2 = re.compile(r"\bBody\s+Mass\s+Index\b\s*[:=]?\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)

def extract_bmi(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    for section, text in note.sections.items():
        for rx in (BMI_RX2, BMI_RX):
            m = rx.search(text)
            if m:
                val = float(m.group(1))
                cands.append(Candidate(
                    field="BMI",
                    value=round(val, 1),
                    status="measured",
                    evidence=window_around(text, m.start(), m.end(), 120),
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.95
                ))
                break
    return cands
