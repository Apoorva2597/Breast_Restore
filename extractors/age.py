# extractors/age.py
from __future__ import annotations

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

# We only trust age mentions that look like "56-year-old female", etc.
AGE_CONTEXT_CUES = re.compile(
    r"\b(female|male|woman|man|patient|pt)\b",
    re.IGNORECASE,
)


def extract_age(note: SectionedNote) -> List[Candidate]:
    """
    Extract Age at Date of Service (Age_DOS) from the note text.

    Heuristic: take the first plausible age mention in the note that:
      - is between 10 and 100
      - appears near 'female', 'male', 'patient', etc.
    We return at most one candidate per note.
    """
    for section, text in note.sections.items():
        for rx in AGE_PATTERNS:
            m = rx.search(text)
            if not m:
                continue

            try:
                age_val = int(m.group(1))
            except ValueError:
                continue

            # sanity bounds
            if age_val < 10 or age_val > 100:
                continue

            ctx = window_around(text, m.start(), m.end(), 80)
            if not AGE_CONTEXT_CUES.search(ctx):
                # Looks more like "3-year history" etc.
                continue

            return [
                Candidate(
                    field="Age_DOS",
                    value=age_val,
                    status="measured",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.9 if note.note_type == "op_note" else 0.8,
                )
            ]

    # default: no age found
    return []
