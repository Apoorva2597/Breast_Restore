# extractors/comorbidities.py
import re
from typing import List

from models import Candidate, SectionedNote
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from .utils import window_around, classify_status, find_first, should_skip_block, has_any

# Sections we should NEVER use for patient comorbidities
SUPPRESS_SECTIONS = {"FAMILY HISTORY", "ALLERGIES", "REVIEW OF SYSTEMS"}

# Sections where comorbidities are most reliable
PREFERRED_SECTIONS = {"PAST MEDICAL HISTORY", "H&P", "HISTORY AND PHYSICAL", "ANESTHESIA", "ANESTHESIA H&P", "ASSESSMENT"}

# -------------------------
# Diabetes
# -------------------------
DM_POS = [
    r"\bdiabetes\b",
    r"\bdiabetes mellitus\b",
    r"\btype\s*(1|2)\b.*\bdiabetes\b",
    r"\bIDDM\b",
    r"\bNIDDM\b",
]
DM_EXCLUDE = [
    r"\bprediabet(es|ic)\b",
    r"\bearly\s+dm\b",
    r"\bborderline\b.*\bdiabet",
    r"\bimpaired glucose tolerance\b|\bIGT\b",
    r"\bgestational\b",
    r"\bdiabetes insipidus\b",
]
DM_MED_STRONG = [
    r"\binsulin\b",
    r"\blantus\b",
    r"\bhumalog\b",
    r"\bnovolog\b",
]

# -------------------------
# Hypertension
# -------------------------
HTN_POS = [r"\bhypertension\b", r"\bHTN\b"]
HTN_EXCLUDE = [
    r"\bgestational\b",
    r"\bpreeclampsia\b|\beclampsia\b",
    r"\bwhite coat\b",
    r"\bpulmonary hypertension\b",
    r"\bportal hypertension\b",
]

# -------------------------
# Cardiac disease (MI/CHF/CAD per spec)
# -------------------------
CARDIAC_POS = [
    r"\bcoronary artery disease\b|\bCAD\b",
    r"\bcongestive heart failure\b|\bCHF\b",
    r"\bmyocardial infarction\b|\bprior MI\b|\bMI\b",
]
CARDIAC_EXCLUDE = [
    r"\bmitral valve prolapse\b|\bvalvular\b",  # not cardiac disease unless symptomatic (we keep conservative)
]

# -------------------------
# VTE (DVT/PE)
# -------------------------
VTE_POS = [
    r"\bdeep vein thrombosis\b|\bDVT\b",
    r"\bpulmonary embol(ism)?\b|\bPE\b",
]
VTE_PROPHYLAXIS_EXCLUDE = [
    r"\bprophylaxis\b|\bppx\b",
    r"\bdvt\s+ppx\b",
    r"\bscd(s)?\b|\bsequential\s+compression\b",
    r"\bheparin\b.*\bprophylaxis\b|\bsubcutaneous\s+heparin\b",
]

# -------------------------
# Steroid use (systemic only)
# -------------------------
STEROID_POS = [
    r"\bprednisone\b",
    r"\bdexamethasone\b",
    r"\bmethylprednisolone\b",
    r"\bsolu[- ]medrol\b",
    r"\bhydrocortisone\b",
]
STEROID_EXCLUDE = [
    r"\binhaled\b",
    r"\btopical\b",
    r"\bcream\b",
    r"\bointment\b",
]


def _emit(field: str, value: bool, status: str, evid: str, section: str, note: SectionedNote, conf: float) -> Candidate:
    return Candidate(
        field=field,
        value=value,
        status=status,
        evidence=evid,
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=conf
    )


def _extract_binary(field: str, pos_patterns: List[str], exclude_patterns: List[str], note: SectionedNote, confidence: float) -> List[Candidate]:
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

        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        if should_skip_block(section, evid):
            continue

        low = evid.lower()

        # exclude patterns
        if exclude_patterns and has_any(exclude_patterns, low):
            continue

        # VTE prophylaxis guard
        if field == "VenousThromboembolism":
            if has_any(VTE_PROPHYLAXIS_EXCLUDE, low):
                continue

        # Steroid systemic-only guard
        if field == "Steroid":
            if has_any(STEROID_EXCLUDE, low):
                continue

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)
        if status == "performed":
            status = "history"

        value = False if status == "denied" else True
        cands.append(_emit(field, value, status, evid, section, note, confidence))

        # stop after first confident hit in preferred section
        if section in PREFERRED_SECTIONS and value is True:
            break

    return cands


def extract_comorbidities(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    # Diabetes by mention
    cands += _extract_binary("Diabetes", DM_POS, DM_EXCLUDE, note, confidence=0.85)

    # Diabetes by insulin (allowed by your protocol; strong inference)
    for section, text in note.sections.items():
        if section in SUPPRESS_SECTIONS:
            continue
        if not text:
            continue
        m = find_first(DM_MED_STRONG, text)
        if not m:
            continue
        evid = window_around(text, m.start(), m.end(), 220)
        if should_skip_block(section, evid):
            continue
        # don’t infer if explicitly excluded
        if has_any(DM_EXCLUDE, evid.lower()):
            continue
        cands.append(_emit("Diabetes", True, "history", evid, section, note, 0.80))

    cands += _extract_binary("Hypertension", HTN_POS, HTN_EXCLUDE, note, confidence=0.85)
    cands += _extract_binary("CardiacDisease", CARDIAC_POS, CARDIAC_EXCLUDE, note, confidence=0.82)
    cands += _extract_binary("VenousThromboembolism", VTE_POS, [], note, confidence=0.82)
    cands += _extract_binary("Steroid", STEROID_POS, STEROID_EXCLUDE, note, confidence=0.78)

    return cands
