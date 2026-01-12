from typing import List

from ..models import Candidate, SectionedNote
from ..config import (
    NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES,
    DM_POS, DM_EXCLUDE,
    HTN_POS, HTN_EXCLUDE,
    CARDIAC_POS, CARDIAC_EXCLUDE,
    VTE_POS,
    STEROID_POS, STEROID_EXCLUDE,
    CANCER_OTHER_POS,
)
from .utils import window_around, classify_status, has_any, find_first, should_skip_block

# Family history contains relatives' conditions; do not treat as patient comorbidities.
SUPPRESS_PATIENT_COMORBIDITY_SECTIONS = {"FAMILY HISTORY"}

# Allergies list is not exposure; do not infer SteroidUse from prednisone allergy.
SUPPRESS_STEROID_SECTIONS = {"ALLERGIES"}

# VTE false positive guard: prophylaxis language ≠ VTE history
VTE_PROPHYLAXIS_EXCLUDE = [
    r"\bprophylaxis\b",
    r"\bppx\b",
    r"\bdvt\s+ppx\b",
    r"\bscd(s)?\b",
    r"\bsequential\s+compression\b",
    r"\bheparin\b.*\bprophylaxis\b",
    r"\bsubcutaneous\s+heparin\b",
]

def _extract_binary(
    field: str,
    pos_patterns: List[str],
    exclude_patterns: List[str],
    note: SectionedNote
) -> List[Candidate]:
    cands: List[Candidate] = []

    for section, text in note.sections.items():

        # 1) Section suppression
        if section in SUPPRESS_PATIENT_COMORBIDITY_SECTIONS and field in {
            "DiabetesMellitus", "Hypertension", "CardiacDisease", "VTE", "SteroidUse"
        }:
            continue
        if section in SUPPRESS_STEROID_SECTIONS and field == "SteroidUse":
            continue

        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 120)
        ctx = evid.lower()

        # Eliminate Family History + Allergies content entirely (even if embedded)
        if should_skip_block(section, evid):
            continue

        # 2) Excludes
        if exclude_patterns and has_any(exclude_patterns, ctx):
            continue

        # 3) Field-specific excludes
        if field == "VTE" and find_first(VTE_PROPHYLAXIS_EXCLUDE, ctx):
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

    return cands


def extract_comorbidities(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    cands += _extract_binary("DiabetesMellitus", DM_POS, DM_EXCLUDE, note)
    cands += _extract_binary("Hypertension", HTN_POS, HTN_EXCLUDE, note)

    # Cardiac: hardened to avoid pre-op template CHF header noise
    cands += _extract_binary("CardiacDisease", CARDIAC_POS, CARDIAC_EXCLUDE, note)

    cands += _extract_binary("VTE", VTE_POS, [], note)
    cands += _extract_binary("SteroidUse", STEROID_POS, STEROID_EXCLUDE, note)

    # Other cancer history
    for section, text in note.sections.items():
        m = find_first(CANCER_OTHER_POS, text)
        if not m:
            continue

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)
        if status == "performed":
            status = "history"

        cands.append(Candidate(
            field="CancerHistoryOther",
            value=True,
            status=status,
            evidence=window_around(text, m.start(), m.end(), 140),
            section=section,
            note_type=note.note_type,
            note_id=note.note_id,
            note_date=note.note_date,
            confidence=0.7
        ))

    return cands
