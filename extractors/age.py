# extractors/age.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# Patterns for "56-year-old", "56 year old", "56 yo", "56 y.o."
AGE_PATTERNS = [
    re.compile(r"\b(\d{1,3})\s*[-]?\s*year[s]?\s*old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*[-]?\s*y\.?\s*o\.?\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*yo\b", re.IGNORECASE),
]

# Reject common false positive contexts like "3-year history"
HISTORY_FALSE_POS = re.compile(
    r"\b(year|yr)\s*(history|hx)\b|\bfor\s+\d{1,3}\s*(years|yrs)\b",
    re.IGNORECASE,
)

PERSON_CUES = re.compile(r"\b(female|male|woman|man|patient|pt)\b", re.IGNORECASE)

# Prefer these sections if present
PREFERRED_SECTIONS = {
    "HPI", "HISTORY OF PRESENT ILLNESS",
    "ASSESSMENT", "ASSESSMENT/PLAN",
    "ANESTHESIA", "ANESTHESIA H&P",
    "H&P", "HISTORY AND PHYSICAL",
}

# Suppress sections that often contain list-like / irrelevant ages
SUPPRESS_SECTIONS = {
    "FAMILY HISTORY", "SOCIAL HISTORY", "REVIEW OF SYSTEMS", "ALLERGIES"
}


def extract_age(note: SectionedNote) -> List[Candidate]:
    """
    High-precision Age mention extraction:
      - Must look like "56-year-old" / "56 yo"
      - Must be 10..100
      - Must be near a PERSON cue
      - Must NOT be part of "x-year history" context
    Returns at most one candidate per note.
    """

    # Search preferred sections first, then everything else
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

        for rx in AGE_PATTERNS:
            for m in rx.finditer(text):
                # quick bounds
                try:
                    age_val = int(m.group(1))
                except ValueError:
                    continue
                if age_val < 10 or age_val > 100:
                    continue

                ctx = window_around(text, m.start(), m.end(), 120)

                # person cue required
                if not PERSON_CUES.search(ctx):
                    continue

                # reject "x-year history" or similar nearby
                if HISTORY_FALSE_POS.search(ctx):
                    continue

                return [
                    Candidate(
                        field="Age",
                        value=age_val,
                        status="measured",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.90 if str(note.note_type).lower() in {"op note", "operation notes", "brief op notes"} else 0.80,
                    )
                ]

    return []
