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
from .utils import window_around, classify_status, find_first, should_skip_block

# ----------------------------
# Radiation patterns
# ----------------------------
RADIATION_POS = [
    r"\bradiation\b",
    r"\bradiation\s+therapy\b",
    r"\bradiotherapy\b",
    r"\bxrt\b",
    # r"\bRT\b",   # too ambiguous (respiratory therapy / right); leave out for now
    r"\bPMRT\b",
]

# ----------------------------
# Chemo patterns
# ----------------------------
CHEMO_POS = [
    r"\bchemotherapy\b",
    r"\bchemo\b",

    # AC family
    r"\bAC\b",
    r"\badriamycin\s*/\s*cyclophosphamide\b",
    r"\badriamycin\s+\+\s+cyclophosphamide\b",
    r"\bdoxorubicin\s*/\s*cyclophosphamide\b",
    r"\bdoxorubicin\s+\+\s+cyclophosphamide\b",

    # individual agents
    r"\badriamycin\b",
    r"\bdoxorubicin\b",
    r"\bcyclophosphamide\b",
    r"\bcytoxan\b",

    # taxanes
    r"\btaxol\b",
    r"\bpaclitaxel\b",
    r"\bdocetaxel\b",

    # platinum
    r"\bcarboplatin\b",
    r"\bcisplatin\b",

    # TC regimen
    r"\bTC\b",
]

# Endocrine therapy is NOT chemotherapy (per your spec)
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

def _has_any(patterns, text_lower):
    for p in patterns:
        if re.search(p, text_lower, re.IGNORECASE):
            return True
    return False


def _extract_flag(field, pos_patterns, note, exclude_patterns=None):
    cands = []  # type: List[Candidate]
    exclude_patterns = exclude_patterns or []

    for section, text in note.sections.items():
        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 160)

        # skip family history/allergies blocks etc.
        if should_skip_block(section, evid):
            continue

        evid_lower = evid.lower()

        # Suppress template-like "Chemo: n/a" / "Radiation: n/a"
        if _has_any(TREATMENT_NA_EXCLUDE, evid_lower):
            continue

        # Suppress “consideration/planning” statements (for now we only want received therapy)
        # This directly fixes: "Systemic chemo - Candidate..." / "determine if chemo..."
        if _has_any(TREATMENT_CONSIDERATION_EXCLUDE, evid_lower):
            continue

        # Radiation-specific non-treatment context (e.g., syndromes)
        if field == "Radiation":
            if _has_any(RADIATION_CONTEXT_EXCLUDE, evid_lower):
                continue

        # endocrine exclusion for chemo: don't infer chemo from endocrine therapy language
        if field == "Chemo":
            if _has_any(ENDOCRINE_EXCLUDE, evid_lower):
                # allow if explicit chemo word is present too (mixed sentence)
                if not re.search(r"\bchemo\b|\bchemotherapy\b", evid_lower):
                    continue

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)
        if status == "performed":
            status = "history"

        value = False if status == "denied" else True

        cands.append(Candidate(
            field=field,
            value=value,
            status=status,
            evidence=evid,
            section=section,
            note_type=note.note_type,
            note_id=note.note_id,
            note_date=note.note_date,
            confidence=0.75
        ))

        # QA flag: possible non-breast cancer indication (do not exclude yet)
        if value and _has_any(NON_BREAST_CANCER_CUES, evid_lower):
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
    cands = []  # type: List[Candidate]
    cands += _extract_flag("Radiation", RADIATION_POS, note)
    cands += _extract_flag("Chemo", CHEMO_POS, note, exclude_patterns=ENDOCRINE_EXCLUDE)
    return cands
