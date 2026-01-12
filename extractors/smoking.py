import re
from typing import List, Optional
from ..models import Candidate, SectionedNote
from ..config import SMOKING_NORMALIZE
from .utils import window_around

SMOKE_LINE = re.compile(r"(smoking\s+status|tobacco\s+use|tobacco)\s*[:=]?\s*([^\n]+)", re.IGNORECASE)

def _normalize(raw: str) -> Optional[str]:
    r = raw.strip().lower()
    for k, v in SMOKING_NORMALIZE.items():
        if k in r:
            return v
    if "never" in r:
        return "never"
    if "former" in r or "quit" in r:
        return "former"
    if "current" in r or "smokes" in r:
        return "current"
    return None

def extract_smoking(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    for section, text in note.sections.items():
        m = SMOKE_LINE.search(text)
        if m:
            norm = _normalize(m.group(2))
            if norm:
                cands.append(Candidate(
                    field="SmokingStatus",
                    value=norm,
                    status="history",
                    evidence=window_around(text, m.start(), m.end(), 120),
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.8
                ))
                continue
        m2 = re.search(r"denies\s+(any\s+)?(tobacco|smoking)[^\n\.]*", text, re.IGNORECASE)
        if m2:
            cands.append(Candidate(
                field="SmokingStatus",
                value="never",
                status="history",
                evidence=window_around(text, m2.start(), m2.end(), 120),
                section=section,
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.7
            ))
    return cands
