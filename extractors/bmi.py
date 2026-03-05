# extractors/bmi.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

BMI_RX = re.compile(r"\bBMI\b\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\b", re.IGNORECASE)
BMI_RX2 = re.compile(r"\bBody\s+Mass\s+Index\b\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\b", re.IGNORECASE)

PREFERRED_SECTIONS = {"VITALS", "PHYSICAL EXAM", "H&P", "ANESTHESIA", "ANESTHESIA H&P"}
SUPPRESS_SECTIONS = {"FAMILY HISTORY", "REVIEW OF SYSTEMS", "ALLERGIES"}


def _valid_bmi(val: float) -> bool:
    # conservative plausible adult BMI bounds
    return 10.0 <= val <= 80.0


def extract_bmi(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    # search preferred sections first
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

        for rx in (BMI_RX2, BMI_RX):
            m = rx.search(text)
            if not m:
                continue

            raw_val = m.group(1)
            try:
                val = float(raw_val)
            except Exception:
                continue

            if not _valid_bmi(val):
                continue

            evid = window_around(text, m.start(), m.end(), 140)

            cands.append(
                Candidate(
                    field="BMI",
                    value=val,
                    status="measured",
                    evidence=evid,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.95 if section in PREFERRED_SECTIONS else 0.90,
                )
            )
            # take first best match per section to avoid duplicates
            break

        if cands:
            # stop early once found in a preferred section
            if section in PREFERRED_SECTIONS:
                break

    return cands
