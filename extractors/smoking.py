# extractors/smoking.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# Smoking extraction with 3-month rule
#
# Data dictionary rule:
# Patients who smoked within the past 3 months
# are considered CURRENT smokers.
#
# Handles phrases like:
#   quit smoking 2 months ago  → Current
#   quit smoking 6 weeks ago   → Current
#   quit smoking 4 months ago  → Former
#
# Output labels:
#   Current / Former / Never
#
# Python 3.6.8 compatible
# ----------------------------------------------

CURRENT_PATTERNS = [
    re.compile(r"\bcurrent smoker\b", re.IGNORECASE),
    re.compile(r"\bsmokes\b", re.IGNORECASE),
    re.compile(r"\bactive smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking currently\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*:\s*former\b", re.IGNORECASE),
    re.compile(r"\bprevious history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bhistory of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bprior tobacco use\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
]

NEVER_PATTERNS = [
    re.compile(r"\bnever smoked\b", re.IGNORECASE),
    re.compile(r"\bnever smoker\b", re.IGNORECASE),
    re.compile(r"\bnever tobacco user\b", re.IGNORECASE),
    re.compile(r"\blifetime nonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bdenies smoking\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco use\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bno tobacco use\b", re.IGNORECASE),
    re.compile(r"\bnever used tobacco\b", re.IGNORECASE),
]

# Detect quit time expressions
QUIT_TIME_PATTERN = re.compile(
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,40}?(\d+)\s*(day|days|week|weeks|month|months)",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking)\b",
    re.IGNORECASE
)

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY",
    "HISTORY",
    "FULL"
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES"
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _quit_within_three_months(match):

    number = int(match.group(3))
    unit = match.group(4).lower()

    if unit.startswith("day"):
        return number <= 90

    if unit.startswith("week"):
        return number <= 12

    if unit.startswith("month"):
        return number <= 3

    return False


def extract_smoking(note: SectionedNote) -> List[Candidate]:

    section_order = []

    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for s in note.sections.keys():
        if s not in section_order and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for section in section_order:

        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)

        # --------------------------------------------------
        # Quit expressions with time
        # --------------------------------------------------

        for m in QUIT_TIME_PATTERN.finditer(text):

            ctx = window_around(text, m.start(), m.end(), 120)

            if _quit_within_three_months(m):
                value = "Current"
            else:
                value = "Former"

            return [
                Candidate(
                    field="SmokingStatus",
                    value=value,
                    status="present",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.92,
                )
            ]

        # --------------------------------------------------
        # Generic quit statements (assume Former)
        # --------------------------------------------------

        for m in GENERIC_QUIT_PATTERN.finditer(text):

            ctx = window_around(text, m.start(), m.end(), 120)

            return [
                Candidate(
                    field="SmokingStatus",
                    value="Former",
                    status="present",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.90,
                )
            ]

        # --------------------------------------------------
        # Current smoker patterns
        # --------------------------------------------------

        for rx in CURRENT_PATTERNS:
            for m in rx.finditer(text):

                ctx = window_around(text, m.start(), m.end(), 120)

                return [
                    Candidate(
                        field="SmokingStatus",
                        value="Current",
                        status="present",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.90,
                    )
                ]

        # --------------------------------------------------
        # Former smoker patterns
        # --------------------------------------------------

        for rx in FORMER_PATTERNS:
            for m in rx.finditer(text):

                ctx = window_around(text, m.start(), m.end(), 120)

                return [
                    Candidate(
                        field="SmokingStatus",
                        value="Former",
                        status="present",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.90,
                    )
                ]

        # --------------------------------------------------
        # Never smoker patterns
        # --------------------------------------------------

        for rx in NEVER_PATTERNS:
            for m in rx.finditer(text):

                ctx = window_around(text, m.start(), m.end(), 120)

                return [
                    Candidate(
                        field="SmokingStatus",
                        value="Never",
                        status="present",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.90,
                    )
                ]

    return []
