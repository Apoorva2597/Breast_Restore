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
# Key handling:
#   quit smoking 2 months ago   -> Current
#   quit smoking 6 weeks ago    -> Current
#   quit smoking 4 months ago   -> Former
#   quit smoking (no time, recent context) -> Current
#   years since quitting / quit date years ago -> Former
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
    re.compile(r"\bsmokes\s+\d+", re.IGNORECASE),
    re.compile(r"\bsmokes\s+(?:a\s+)?(?:couple|few)\s+cig", re.IGNORECASE),
    re.compile(r"\bcigarettes?\s+per\s+(?:day|week)\b", re.IGNORECASE),
    re.compile(r"\bpacks?/?day\b", re.IGNORECASE),
    re.compile(r"\btobacco use\s*[:\-]?\s*current\b", re.IGNORECASE),
]

FORMER_PATTERNS = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*:\s*former\b", re.IGNORECASE),
    re.compile(r"\bprevious history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bhistory of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bprior tobacco use\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\byears?\s+since\s+quitting\b", re.IGNORECASE),
    re.compile(r"\bquit date\s*[:\-]?\s*(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bquit\s+(?:smoking|tobacco)[^\.]{0,40}?\d+\s*(?:year|years)\b", re.IGNORECASE),
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
    re.compile(r"\bsmokeless tobacco\s+(?:never used|none)\b", re.IGNORECASE),
    re.compile(r"\bactive tobacco use\?\s*no\b", re.IGNORECASE),
    re.compile(r"\bcurrently smoking\?\s*no\b", re.IGNORECASE),
]

# Detect quit time expressions
QUIT_TIME_PATTERN = re.compile(
    r"(quit|stopped)\s+(smoking|tobacco)[^\.]{0,60}?(\d+)\s*(day|days|week|weeks|month|months|year|years)",
    re.IGNORECASE
)

GENERIC_QUIT_PATTERN = re.compile(
    r"\b(quit smoking|quit tobacco|stopped smoking|stopped tobacco)\b",
    re.IGNORECASE
)

SINCE_LAST_VISIT_QUIT_PATTERN = re.compile(
    r"\b(?:since\s+(?:our|the)\s+last\s+visit[^\.]{0,80}?)?(?:has\s+)?(?:quit|stopped)\s+(?:smoking|tobacco)\b",
    re.IGNORECASE
)

YEARS_SINCE_QUITTING_PATTERN = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE
)

QUIT_DATE_YEAR_PATTERN = re.compile(
    r"\bquit date\s*[:\-]?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE
)

RECENT_QUIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:trying\s+to\s+quit|plans\s+to\s+quit|encouraged\s+to\s+quit|recently\s+quit|since\s+last\s+visit[^\.]{0,80}?quit)\b",
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

    if unit.startswith("year"):
        return False

    return False


def _candidate(note, section, value, text, start, end, confidence):
    ctx = window_around(text, start, end, 120)
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
            confidence=confidence,
        )
    ]


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
        # Highest-priority explicit current smoking patterns
        # --------------------------------------------------
        for rx in CURRENT_PATTERNS:
            for m in rx.finditer(text):
                return _candidate(note, section, "Current", text, m.start(), m.end(), 0.95)

        # --------------------------------------------------
        # Quit expressions with explicit time
        # --------------------------------------------------
        for m in QUIT_TIME_PATTERN.finditer(text):
            if _quit_within_three_months(m):
                value = "Current"
                conf = 0.94
            else:
                value = "Former"
                conf = 0.94
            return _candidate(note, section, value, text, m.start(), m.end(), conf)

        # --------------------------------------------------
        # Years since quitting -> Former
        # --------------------------------------------------
        for m in YEARS_SINCE_QUITTING_PATTERN.finditer(text):
            return _candidate(note, section, "Former", text, m.start(), m.end(), 0.94)

        # --------------------------------------------------
        # Quit date year -> Former
        # --------------------------------------------------
        for m in QUIT_DATE_YEAR_PATTERN.finditer(text):
            return _candidate(note, section, "Former", text, m.start(), m.end(), 0.94)

        # --------------------------------------------------
        # Generic quit statements with recent context -> Current
        # --------------------------------------------------
        for m in SINCE_LAST_VISIT_QUIT_PATTERN.finditer(text):
            return _candidate(note, section, "Current", text, m.start(), m.end(), 0.93)

        for m in RECENT_QUIT_CONTEXT_PATTERN.finditer(text):
            return _candidate(note, section, "Current", text, m.start(), m.end(), 0.92)

        # --------------------------------------------------
        # Generic quit statements without time
        # Dataset rule / QA indicates these are often recent,
        # so default to Current rather than Former.
        # --------------------------------------------------
        for m in GENERIC_QUIT_PATTERN.finditer(text):
            return _candidate(note, section, "Current", text, m.start(), m.end(), 0.90)

        # --------------------------------------------------
        # Never smoker patterns
        # --------------------------------------------------
        for rx in NEVER_PATTERNS:
            for m in rx.finditer(text):
                return _candidate(note, section, "Never", text, m.start(), m.end(), 0.90)

        # --------------------------------------------------
        # Former smoker patterns
        # These come after quit logic so explicit recent quit
        # statements are not incorrectly forced to Former.
        # --------------------------------------------------
        for rx in FORMER_PATTERNS:
            for m in rx.finditer(text):
                return _candidate(note, section, "Former", text, m.start(), m.end(), 0.89)

    return []
