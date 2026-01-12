from __future__ import annotations
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ---------------------------------------------
# Prior Breast Surgery (PBS)
# ---------------------------------------------

PBS_LUMP_PATTERNS = [
    r"\blumpectomy\b",
    r"\bpartial\s+mastectomy\b",
    r"\bsegmental\s+mastectomy\b",
    r"\bbreast[- ]conserving\s+surgery\b",
    r"\bwide\s+local\s+excision\b",
]

PBS_OTHER_PATTERNS = [
    r"\bbreast\s+reduction\b",
    r"\breduction\s+mammaplasty\b",
    r"\bbenign\s+excision\b",
    r"\bexcisional\s+biopsy\b",
    r"\bmastopexy\b",
]

PBS_NEGATE_PATTERNS = [
    r"no\s+prior\s+breast\s+surgery",
    r"no\s+history\s+of\s+breast\s+surgery",
    r"denies\s+prior\s+breast\s+surgery",
]


def extract_pbs(note: SectionedNote) -> List[Candidate]:
    cands = []  # type: List[Candidate]

    for section, text in note.sections.items():
        lower = text.lower()

        # Negation check (skip entire block)
        for pat in PBS_NEGATE_PATTERNS:
            if re.search(pat, lower):
                lower = ""
                break
        if not lower:
            continue

        # Lumpectomy-like prior surgery
        for pat in PBS_LUMP_PATTERNS:
            m = re.search(pat, lower)
            if m:
                ctx = window_around(text, m.start(), m.end(), 140)
                cands.append(Candidate(
                    field="PBS_Lumpectomy",
                    value=True,
                    status="history",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.75
                ))
                break

        # Other benign/prior breast surgery
        for pat in PBS_OTHER_PATTERNS:
            m = re.search(pat, lower)
            if m:
                ctx = window_around(text, m.start(), m.end(), 140)
                cands.append(Candidate(
                    field="PBS_Other",
                    value=True,
                    status="history",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.70
                ))
                break

    return cands
