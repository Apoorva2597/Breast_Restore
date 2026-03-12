# extractors/complications_outcomes.py
# Python 3.6.8 compatible

import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around


SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
    "REVIEW OF SYSTEMS",
}

PREFERRED_SECTIONS = {
    "ASSESSMENT",
    "ASSESSMENT AND PLAN",
    "HOSPITAL COURSE",
    "IMPRESSION",
    "POSTOPERATIVE DIAGNOSIS",
    "POST-OPERATIVE DIAGNOSIS",
    "DIAGNOSIS",
    "PROBLEM LIST",
    "PLAN",
    "BRIEF HOSPITAL COURSE",
}

LOW_VALUE_SECTIONS = {
    "PAST MEDICAL HISTORY",
    "PMH",
    "PAST SURGICAL HISTORY",
    "PSH",
    "MEDICATIONS",
    "SOCIAL HISTORY",
}

NEGATION_RX = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|free\s+of|absence\s+of)\b",
    re.IGNORECASE
)

PLAN_RX = re.compile(
    r"\b(plan|planned|planning|scheduled|schedule|will|would|to be|consider|considered|candidate for)\b",
    re.IGNORECASE
)

HISTORICAL_RX = re.compile(
    r"\b(history of|hx of|h/o|prior|previously|previous)\b",
    re.IGNORECASE
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.IGNORECASE
)

RECON_CONTEXT_RX = re.compile(
    r"\b("
    r"breast reconstruction|reconstruction|recon|expander|implant|te\b|tissue expander|"
    r"diep|tram|latissimus|flap|mastectomy site|breast wound|breast incision"
    r")\b",
    re.IGNORECASE
)

DONOR_CONTEXT_RX = re.compile(
    r"\b("
    r"donor site|abdomen|abdominal|abdominal wall|hernia|bulge|laxity|fat necrosis at donor"
    r")\b",
    re.IGNORECASE
)

COMPLICATION_CONTEXT_RX = re.compile(
    r"\b("
    r"hematoma|dehiscence|infection|infected|necrosis|seroma|extrusion|rupture|deflation|"
    r"malposition|contracture|flap loss|flap failure|implant loss|expander loss|"
    r"cellulitis|abscess|wound breakdown|wound complication|complication"
    r")\b",
    re.IGNORECASE
)

MINOR_COMP_POS = [
    r"\bhematoma\b",
    r"\bwound dehiscence\b",
    r"\bdehiscence\b",
    r"\bwound infection\b",
    r"\binfection\b",
    r"\bcellulitis\b",
    r"\babscess\b",
    r"\bmastectomy skin flap necrosis\b",
    r"\bskin flap necrosis\b",
    r"\bnecrosis\b",
    r"\bseroma\b",
    r"\bcapsular contracture\b",
    r"\bimplant malposition\b",
    r"\bimplant rupture\b",
    r"\bimplant leakage\b",
    r"\bimplant deflation\b",
    r"\bexpander extrusion\b",
    r"\bimplant extrusion\b",
    r"\bflap congestion\b",
    r"\bpartial flap necrosis\b",
    r"\bwound breakdown\b",
    r"\bdrainage from\b.{0,20}\bwound\b",
]

REOP_POS = [
    r"\breturn(ed)?\s+to\s+(the\s+)?or\b",
    r"\bback\s+to\s+(the\s+)?or\b",
    r"\btake\s*back\b",
    r"\bre-?operation\b",
    r"\bre-?operat(ed|ion)\b",
    r"\bre-?exploration\b",
    r"\bwashout\b",
    r"\bincision\s+and\s+drainage\b",
    r"\bi\s*&\s*d\b",
    r"\bdebridement\b",
    r"\boperative debridement\b",
    r"\bclosure in the or\b",
    r"\bsurgical intervention\b",
]

REHOSP_POS = [
    r"\breadmit(ted|sion)?\b",
    r"\bre-?admit(ted|sion)?\b",
    r"\brehospitali[sz](ed|ation)?\b",
    r"\breturn(ed)?\s+to\s+hospital\b",
    r"\badmitted\s+for\b.{0,40}\b(cellulitis|infection|seroma|hematoma|necrosis|wound)\b",
]

FAILURE_POS = [
    r"\bflap\s+(loss|failed|failure)\b",
    r"\btotal\s+flap\s+loss\b",
    r"\bcomplete\s+flap\s+necrosis\b",
    r"\bimplant\s+(loss|removed|removal)\b",
    r"\bexpander\s+(loss|removed|removal)\b",
    r"\bprosthesis\s+removed\b",
    r"\bexplant(ed|ation)?\b",
    r"\bremoved\s+the\s+(implant|expander|flap)\b",
]

REVISION_POS = [
    r"\brevision\s+surgery\b",
    r"\brevision\s+procedure\b",
    r"\bscar\s+revision\b",
    r"\bfat\s+grafting\b",
    r"\blipofill(ing)?\b",
    r"\bcapsulectomy\b",
    r"\bcontour\s+revision\b",
    r"\bdog[- ]ear\s+revision\b",
    r"\bstanding\s+cutaneous\s+deformit(y|ies)\b",
    r"\bcontralateral\s+mastopexy\b",
    r"\bsymmetry\s+procedure\b",
]

NOT_REVISION_RX = re.compile(
    r"\bnipple reconstruction\b",
    re.IGNORECASE
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
        ku = (k or "").strip().upper()
        if ku in SUPPRESS_SECTIONS:
            continue
        txt = note.sections.get(k, "") or ""
        if txt.strip():
            yield ku, txt


def _find_first(patterns, text):
    best = None
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            if best is None or m.start() < best.start():
                best = m
    return best


def _has_any(patterns, text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _is_negated(evid):
    return bool(NEGATION_RX.search(evid))


def _is_family(evid):
    return bool(FAMILY_RX.search(evid))


def _is_planned(evid):
    return bool(PLAN_RX.search(evid))


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


def _base_conf(section, note_type):
    rank = _section_rank(section)
    nt = (note_type or "").lower()
    conf = 0.72
    if rank == 0:
        conf += 0.08
    elif rank == 2:
        conf -= 0.08
    if "op" in nt or "operative" in nt or "operation" in nt:
        conf += 0.06
    return max(0.55, min(0.95, conf))


def _extract_flag(field, pos_patterns, note, require_comp_context=False, require_recon_context=False):
    cands = []

    for section, text in _iter_sections(note):
        m = _find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 220)
        low = evid.lower()

        if _is_family(low):
            continue
        if _is_negated(low):
            status = "denied"
            value = False
        elif _is_planned(low):
            status = "planned"
            value = False
        else:
            status = "performed"
            value = True

        if require_comp_context and not COMPLICATION_CONTEXT_RX.search(low):
            continue

        if require_recon_context:
            if not RECON_CONTEXT_RX.search(low) and not DONOR_CONTEXT_RX.search(low):
                if not COMPLICATION_CONTEXT_RX.search(low):
                    continue

        conf = _base_conf(section, note.note_type)

        if status == "performed":
            cands.append(_emit(field, value, "history", evid, section, note, conf))
        elif status == "denied":
            cands.append(_emit(field, value, "denied", evid, section, note, max(0.55, conf - 0.10)))
        else:
            cands.append(_emit(field, value, "planned", evid, section, note, max(0.50, conf - 0.15)))

        if value is True and _section_rank(section) == 0:
            break

    return cands


def extract_complication_outcomes(note):
    cands = []  # type: List[Candidate]

    cands.extend(_extract_flag(
        "ComplicationSignal",
        MINOR_COMP_POS,
        note,
        require_comp_context=False,
        require_recon_context=True
    ))

    cands.extend(_extract_flag(
        "StageOutcome_Reoperation",
        REOP_POS,
        note,
        require_comp_context=False,
        require_recon_context=True
    ))

    cands.extend(_extract_flag(
        "StageOutcome_Rehospitalization",
        REHOSP_POS,
        note,
        require_comp_context=False,
        require_recon_context=True
    ))

    cands.extend(_extract_flag(
        "StageOutcome_Failure",
        FAILURE_POS,
        note,
        require_comp_context=False,
        require_recon_context=True
    ))

    rev = _extract_flag(
        "StageOutcome_Revision",
        REVISION_POS,
        note,
        require_comp_context=False,
        require_recon_context=False
    )
    rev_clean = []
    for c in rev:
        if NOT_REVISION_RX.search((c.evidence or "").lower()):
            continue
        rev_clean.append(c)
    cands.extend(rev_clean)

    return cands
