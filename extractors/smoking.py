# extractors/smoking.py
import re
from typing import List, Optional

from models import Candidate, SectionedNote
from config import SMOKING_NORMALIZE
from .utils import window_around

SMOKE_LINE = re.compile(
    r"(smoking\s+status|tobacco\s+use|tobacco|nicotine|vaping)\s*[:=]?\s*([^\n]+)",
    re.IGNORECASE
)

DENIES_TOBACCO = re.compile(r"\bdenies\s+(any\s+)?(tobacco|smoking|nicotine|vaping)\b", re.IGNORECASE)

# Prefer sections where smoking status is actually documented
PREFERRED_SECTIONS = {
    "SOCIAL HISTORY", "ANESTHESIA", "ANESTHESIA H&P", "H&P", "PAST MEDICAL HISTORY"
}
SUPPRESS_SECTIONS = {"FAMILY HISTORY", "ALLERGIES", "REVIEW OF SYSTEMS"}

# If quit < 3 months => current (per your spec)
QUIT_RECENT = re.compile(r"\bquit\b.*\b(\d+)\s*(day|days|week|weeks|month|months)\b", re.IGNORECASE)


def _normalize(raw: str) -> Optional[str]:
    r = (raw or "").strip().lower()

    # config-based normalization first
    for k, v in SMOKING_NORMALIZE.items():
        if k in r:
            return v

    # explicit phrases
    if "never" in r or "non-smoker" in r or "nonsmoker" in r:
        return "never"

    if "former" in r or "quit" in r or "stopped" in r:
        # apply 3-month rule
        m = QUIT_RECENT.search(r)
        if m:
            n = int(m.group(1))
            unit = m.group(2).lower()
            # convert to approx days
            days = n
            if "week" in unit:
                days = n * 7
            elif "month" in unit:
                days = n * 30
            if days <= 90:
                return "current"
        return "former"

    if "current" in r or "smokes" in r or "smoker" in r or "vapes" in r or "vaping" in r:
        return "current"

    return None


def extract_smoking(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    # order sections: preferred first
    section_order = []
    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            section_order.append(s)
    for s in note.sections.keys():
        if s not in section_order and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for section in section_order:
        text = note.sections.get(section, "") or ""
        if not text:
            continue

        # structured smoking status line
        m = SMOKE_LINE.search(text)
        if m:
            norm = _normalize(m.group(2))
            if norm:
                cands.append(
                    Candidate(
                        field="SmokingStatus",
                        value=norm,
                        status="history",
                        evidence=window_around(text, m.start(), m.end(), 160),
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.85 if section in PREFERRED_SECTIONS else 0.75,
                    )
                )
                # prefer first strong documentation
                return cands

        # denial logic: ONLY allow outside ROS to avoid template noise
        if section not in {"REVIEW OF SYSTEMS"}:
            m2 = DENIES_TOBACCO.search(text)
            if m2:
                cands.append(
                    Candidate(
                        field="SmokingStatus",
                        value="never",
                        status="history",
                        evidence=window_around(text, m2.start(), m2.end(), 160),
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.75 if section in PREFERRED_SECTIONS else 0.65,
                    )
                )
                return cands

    return cands
