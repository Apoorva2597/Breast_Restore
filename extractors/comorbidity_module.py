# extractors/comorbidity_module.py
# Python 3.6.8 compatible

import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ---------------------------------
# Section controls
# ---------------------------------

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
    "REVIEW OF SYSTEMS",
}

PREFERRED_SECTIONS = {
    "PAST MEDICAL HISTORY",
    "PMH",
    "HISTORY AND PHYSICAL",
    "H&P",
    "ASSESSMENT",
    "ASSESSMENT AND PLAN",
    "MEDICAL HISTORY",
    "PROBLEM LIST",
    "ANESTHESIA",
    "ANESTHESIA H&P",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "DIAGNOSIS",
    "IMPRESSION",
}

LOW_VALUE_SECTIONS = {
    "PAST SURGICAL HISTORY",
    "PSH",
    "SURGICAL HISTORY",
    "HISTORY",
    "GYNECOLOGIC HISTORY",
    "OB HISTORY",
}

# ---------------------------------
# Shared context patterns
# ---------------------------------

NEGATION_RX = re.compile(
    r"\b("
    r"no|not|denies|denied|without|negative\s+for|"
    r"free\s+of|absence\s+of|(-)\s*"
    r")\b",
    re.IGNORECASE
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.IGNORECASE
)

HISTORICAL_ONLY_RX = re.compile(
    r"\b(history of|hx of|h/o)\b",
    re.IGNORECASE
)

STEROID_NEG_CONTEXT_RX = re.compile(
    r"\b(no|not|denies|without)\b.{0,40}\b(steroid|prednisone|dexamethasone|medrol|methylprednisolone|hydrocortisone)\b",
    re.IGNORECASE
)

SYSTEMIC_STEROID_EXCLUDE_RX = re.compile(
    r"\b("
    r"inhaled|inhaler|intranasal|nasal|topical|cream|ointment|lotion|eye\s*drops?|otic|ear\s*drops?"
    r")\b",
    re.IGNORECASE
)

VTE_PROPHYLAXIS_RX = re.compile(
    r"\b("
    r"prophylaxis|ppx|dvt\s*ppx|vte\s*ppx|"
    r"sequential\s+compression|compression\s+device|scd|scds|"
    r"subcutaneous\s+heparin|heparin\s+prophylaxis|enoxaparin\s+prophylaxis"
    r")\b",
    re.IGNORECASE
)

# ---------------------------------
# Concept dictionaries
# ---------------------------------

CONCEPTS = {
    "Diabetes": {
        "pos": [
            r"\bdiabetes\b",
            r"\bdiabetes mellitus\b",
            r"\bdm\b",
            r"\bt1dm\b",
            r"\bt2dm\b",
            r"\btype\s*(i|ii|1|2)\s*diabetes\b",
            r"\binsulin[- ]dependent diabetes\b",
            r"\bnon[- ]insulin[- ]dependent diabetes\b",
            r"\biddm\b",
            r"\bniddm\b",
        ],
        "exclude": [
            r"\bprediabet(es|ic)\b",
            r"\bborderline\b.{0,20}\bdiabet",
            r"\bimpaired glucose tolerance\b",
            r"\bigt\b",
            r"\bgestational diabetes\b",
            r"\bdiabetes insipidus\b",
        ],
    },
    "Hypertension": {
        "pos": [
            r"\bhypertension\b",
            r"\bhtn\b",
            r"\bhigh blood pressure\b",
        ],
        "exclude": [
            r"\bpulmonary hypertension\b",
            r"\bportal hypertension\b",
            r"\bgestational hypertension\b",
            r"\bpreeclampsia\b",
            r"\beclampsia\b",
            r"\bwhite coat hypertension\b",
        ],
    },
    "CardiacDisease": {
        "pos": [
            r"\bcoronary artery disease\b",
            r"\bcad\b",
            r"\bcongestive heart failure\b",
            r"\bchf\b",
            r"\bheart failure\b",
            r"\bmyocardial infarction\b",
            r"\bmi\b",
            r"\bprior mi\b",
            r"\bischemic heart disease\b",
            r"\bcardiomyopathy\b",
            r"\batrial fibrillation\b",
            r"\bafib\b",
            r"\ba[- ]fib\b",
        ],
        "exclude": [
            r"\bmitral valve prolapse\b",
            r"\bvalvular\b",
            r"\bheart murmur\b",
        ],
    },
    "VenousThromboembolism": {
        "pos": [
            r"\bdeep vein thrombosis\b",
            r"\bdvt\b",
            r"\bpulmonary embol(ism)?\b",
            r"\bpe\b",
            r"\bvte\b",
        ],
        "exclude": [],
    },
    "Steroid": {
        "pos": [
            r"\bprednisone\b",
            r"\bdexamethasone\b",
            r"\bmethylprednisolone\b",
            r"\bsolu[- ]medrol\b",
            r"\bmedrol\b",
            r"\bhydrocortisone\b",
            r"\bchronic steroid(s)?\b",
            r"\blong[- ]term steroid(s)?\b",
            r"\bsystemic steroid(s)?\b",
        ],
        "exclude": [],
    },
}

DM_MED_STRONG = [
    r"\binsulin\b",
    r"\blantus\b",
    r"\bhumalog\b",
    r"\bnovolog\b",
    r"\blevemir\b",
    r"\bmetformin\b",
]

# ---------------------------------
# Helpers
# ---------------------------------

def _emit(field, value, status, evid, section, note, conf):
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

def _section_rank(section):
    s = (section or "").strip().upper()
    if s in PREFERRED_SECTIONS:
        return 0
    if s in LOW_VALUE_SECTIONS:
        return 2
    return 1

def _iter_sections(note):
    keys = list(note.sections.keys())
    keys.sort(key=_section_rank)
    for k in keys:
        if k in SUPPRESS_SECTIONS:
            continue
        txt = note.sections.get(k, "") or ""
        if txt.strip():
            yield k, txt

def _is_negated(evid):
    low = evid.lower()
    return bool(NEGATION_RX.search(low))

def _family_context(evid):
    return bool(FAMILY_RX.search(evid))

def _has_any(patterns, text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def _find_first(patterns, text):
    best = None
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if best is None or m.start() < best.start():
                best = m
    return best

def _status_from_context(evid):
    low = evid.lower()
    if _is_negated(low):
        return "denied"
    if HISTORICAL_ONLY_RX.search(low):
        return "history"
    return "history"

def _concept_confidence(section, base):
    rank = _section_rank(section)
    if rank == 0:
        return min(0.98, base + 0.05)
    if rank == 2:
        return max(0.55, base - 0.08)
    return base

# ---------------------------------
# Main concept extraction
# ---------------------------------

def _extract_concept(field, note):
    cfg = CONCEPTS[field]
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(cfg["pos"], text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        low = evid.lower()

        if _family_context(low):
            continue

        if cfg.get("exclude") and _has_any(cfg["exclude"], low):
            continue

        if field == "VenousThromboembolism" and VTE_PROPHYLAXIS_RX.search(low):
            continue

        if field == "Steroid":
            if SYSTEMIC_STEROID_EXCLUDE_RX.search(low):
                continue
            if STEROID_NEG_CONTEXT_RX.search(low):
                continue

        status = _status_from_context(evid)
        value = False if status == "denied" else True
        conf = _concept_confidence(section, 0.84)

        cands.append(_emit(
            field=field,
            value=value,
            status=status,
            evid=evid,
            section=section,
            note=note,
            conf=conf
        ))

        if value is True and _section_rank(section) == 0:
            break

    return cands

def _extract_diabetes_med_inference(note):
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(DM_MED_STRONG, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        low = evid.lower()

        if _family_context(low):
            continue
        if _has_any(CONCEPTS["Diabetes"]["exclude"], low):
            continue
        if _is_negated(low):
            continue

        conf = _concept_confidence(section, 0.76)
        cands.append(_emit(
            field="Diabetes",
            value=True,
            status="history",
            evid=evid,
            section=section,
            note=note,
            conf=conf
        ))

        if _section_rank(section) == 0:
            break

    return cands

def extract_comorbidities(note):
    cands = []
    cands.extend(_extract_concept("Diabetes", note))
    cands.extend(_extract_diabetes_med_inference(note))
    cands.extend(_extract_concept("Hypertension", note))
    cands.extend(_extract_concept("CardiacDisease", note))
    cands.extend(_extract_concept("VenousThromboembolism", note))
    cands.extend(_extract_concept("Steroid", note))
    return cands
