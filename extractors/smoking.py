# extractors/smoking.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# UPDATE:
# Expanded smoking extraction for clinic notes.
# Handles common phrases for:
#
# Current:
#   current smoker
#   smokes
#   active smoker
#   smoking currently
#   currently smoking
#
# Former:
#   former smoker
#   quit smoking
#   quit tobacco
#   smoking status: former
#   previous history of tobacco use
#   history of tobacco use
#   prior tobacco use
#   ex-smoker
#
# Never:
#   never smoked
#   never smoker
#   never tobacco user
#   lifetime nonsmoker
#   nonsmoker / non-smoker
#   denies smoking
#   denies tobacco
#   denies tobacco use
#   no history of tobacco use
#   no tobacco use
#
# Output labels:
#   Current / Former / Never
#
# Python 3.6.8 compatible.
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
    re.compile(r"\bquit smoking\b", re.IGNORECASE),
    re.compile(r"\bquit tobacco\b", re.IGNORECASE),
    re.compile(r"\bsmoking status\s*:\s*former\b", re.IGNORECASE),
    re.compile(r"\bprevious history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bhistory of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bprior tobacco use\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bstopped smoking\b", re.IGNORECASE),
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

PREFERRED_SECTIONS = {
    "SOCIAL HISTORY", "HISTORY", "FULL"
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY", "ALLERGIES"
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_smoking(note: SectionedNote) -> List[Candidate]:
    """
    Smoking status extraction:
      - Maps note language into Current / Former / Never
      - Prefers SOCIAL HISTORY-like sections
      - Returns at most one candidate per note
    """

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

        for rx in CURRENT_PATTERNS:
            m = rx.finditer(text)
            for hit in m:
                ctx = window_around(text, hit.start(), hit.end(), 120)
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

        for rx in FORMER_PATTERNS:
            m = rx.finditer(text)
            for hit in m:
                ctx = window_around(text, hit.start(), hit.end(), 120)
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

        for rx in NEVER_PATTERNS:
            m = rx.finditer(text)
            for hit in m:
                ctx = window_around(text, hit.start(), hit.end(), 120)
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
