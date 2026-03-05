# extractors/cancer_treatment.py
import re
from typing import List

from models import Candidate, SectionedNote
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from config import (
    TREATMENT_CONSIDERATION_EXCLUDE,
    TREATMENT_NA_EXCLUDE,
    RADIATION_CONTEXT_EXCLUDE,
    NON_BREAST_CANCER_CUES,
)
from .utils import window_around, classify_status, find_first, should_skip_block, has_any

RADIATION_POS = [
    r"\bradiation\b",
    r"\bradiation\s+therapy\b",
    r"\bradiotherapy\b",
    r"\bxrt\b",
    r"\bPMRT\b",
]

CHEMO_POS = [
    r"\bchemotherapy\b",
    r"\bchemo\b",
    r"\badriamycin\b|\bdoxorubicin\b",
    r"\bcyclophosphamide\b|\bcytoxan\b",
    r"\btaxol\b|\bpaclitaxel\b|\bdocetaxel\b|\btaxotere\b",
    r"\bcarboplatin\b|\bcisplatin\b",
    r"\bherceptin\b|\btrastuzumab\b",
    r"\bperjeta\b|\bpertuzumab\b",
    r"\bTCHP?\b",
    r"\bAC\b",
    r"\bTC\b",
]

ENDOCRINE_EXCLUDE = [
    r"\btamoxifen\b",
    r"\bletrozole\b",
    r"\banastrozole\b",
    r"\bexemestane\b",
    r"\bfulvestrant\b",
    r"\barimidex\b",
    r"\bfemara\b",
    r"\baromasin\b",
]

# extra high-FP phrases to block (very common in onc templates)
HARD_BLOCK_CUES = [
    r"\bcandidate\b.*\bchemo\b",
    r"\bconsider\b.*\bchemo\b",
    r"\bdiscuss(ed|ion)\b.*\bchemo\b",
    r"\bplan(s|ned)?\b.*\bchemo\b",
    r"\bwill\s+(start|receive)\b.*\bchemo\b",
    r"\bto\s+start\b.*\bchemo\b",
]


def _extract_flag(field: str, pos_patterns: List[str], note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    for section, text in note.sections.items():
        if not text:
            continue

        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 240)
        if should_skip_block(section, evid):
            continue

        low = evid.lower()

        # n/a / none templates
        if has_any(TREATMENT_NA_EXCLUDE, low):
            continue

        # consideration/planning
        if has_any(TREATMENT_CONSIDERATION_EXCLUDE, low):
            continue
        if any(re.search(p, low) for p in HARD_BLOCK_CUES):
            continue

        # radiation context excludes
        if field == "Radiation":
            if has_any(RADIATION_CONTEXT_EXCLUDE, low):
                continue

        # endocrine-only should not count as chemo
        if field == "Chemo":
            if any(re.search(p, low) for p in ENDOCRINE_EXCLUDE):
                if not re.search(r"\bchemo\b|\bchemotherapy\b", low):
                    continue

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)
        if status in {"planned", "denied"}:
            continue
        if status == "performed":
            status = "history"

        cands.append(Candidate(
            field=field,
            value=True,
            status=status,
            evidence=evid,
            section=section,
            note_type=note.note_type,
            note_id=note.note_id,
            note_date=note.note_date,
            confidence=0.80
        ))

        # QA: possible non-breast context
        if has_any(NON_BREAST_CANCER_CUES, low):
            cands.append(Candidate(
                field=field + "_NonBreastContext",
                value=True,
                status=status,
                evidence=evid,
                section=section,
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.60
            ))

    return cands


def extract_cancer_treatment(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    cands += _extract_flag("Radiation", RADIATION_POS, note)
    cands += _extract_flag("Chemo", CHEMO_POS, note)
    return cands
