# extractors/complications.py
# Python 3.6.8 compatible

import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around


# --------------------------------------------------
# Section controls
# --------------------------------------------------
SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
    "REVIEW OF SYSTEMS",
}

PREFERRED_SECTIONS = {
    "ASSESSMENT",
    "ASSESSMENT AND PLAN",
    "HOSPITAL COURSE",
    "BRIEF HOSPITAL COURSE",
    "IMPRESSION",
    "POSTOPERATIVE DIAGNOSIS",
    "POST-OPERATIVE DIAGNOSIS",
    "DIAGNOSIS",
    "PROBLEM LIST",
    "PLAN",
    "OPERATIVE REPORT",
    "PROCEDURES",
    "OP NOTE",
    "BRIEF OP NOTE",
}

LOW_VALUE_SECTIONS = {
    "PAST MEDICAL HISTORY",
    "PMH",
    "MEDICATIONS",
    "SOCIAL HISTORY",
}

# sections to strongly suppress for revision/history leakage
REVISION_SUPPRESS_SECTIONS = {
    "PAST SURGICAL HISTORY",
    "PSH",
    "SURGICAL HISTORY",
    "PROCEDURE HISTORY",
    "PAST SURGICAL HISTORY/PROCEDURE HISTORY",
    "PAST SURGICAL HISTORY / PROCEDURE HISTORY",
}


# --------------------------------------------------
# Shared context
# --------------------------------------------------
NEGATION_RX = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|free\s+of|absence\s+of|none)\b",
    re.IGNORECASE
)

PLAN_RX = re.compile(
    r"\b("
    r"plan|planned|planning|scheduled|schedule|will|would|to be|consider|considered|"
    r"candidate for|upcoming|electing to proceed|discussed today|request was entered|"
    r"preoperative history and physical|pre-op|preop"
    r")\b",
    re.IGNORECASE
)

REVISION_PLAN_RX = re.compile(
    r"\b("
    r"will discuss|discuss revision|discuss nipple reconstruction|interested in revision|"
    r"interested in undergoing|candidate for revision|may require revision|"
    r"would like revision|wants revision|plan revision|planning revision|"
    r"future revision|next stage of surgery|will include|upcoming procedure"
    r")\b",
    re.IGNORECASE
)

FAMILY_RX = re.compile(
    r"\b(family history|mother|father|sister|brother|aunt|uncle|grandmother|grandfather)\b",
    re.IGNORECASE
)

BREAST_RECON_CONTEXT_RX = re.compile(
    r"\b("
    r"breast|reconstruction|recon|mastectomy|implant|implants|expander|expanders|"
    r"tissue expander|breast pocket|capsule|capsular|flap|diep|tram|latissimus|"
    r"alloderm|adm|breast incision|mastectomy site|chest wall|nipple[- ]areolar"
    r")\b",
    re.IGNORECASE
)

HISTORICAL_ONLY_RX = re.compile(
    r"\b(history of|hx of|h/o|prior|previous|previously|remote)\b",
    re.IGNORECASE
)

RESOLVED_RX = re.compile(
    r"\b("
    r"resolved|healed|healing well|doing well|well healed|improved|no further issues|"
    r"stable|without recurrent|no evidence of|follow up only"
    r")\b",
    re.IGNORECASE
)

ACUTE_EVENT_RX = re.compile(
    r"\b("
    r"postop|post-op|post operative|postoperative|complicated by|developed|presented with|"
    r"required|underwent|warranting|necessitating|was treated for|returned to the or|"
    r"readmitted|rehospitalized|taken back|takeback|washout|debridement|i\s*&\s*d|"
    r"incision and drainage|evacuation|explantation|explant|removed because of|"
    r"procedure performed|operations performed|underwent revision|surgery date"
    r")\b",
    re.IGNORECASE
)

COMPLICATION_REASON_RX = re.compile(
    r"\b("
    r"hematoma|seroma|infection|infected|cellulitis|abscess|dehiscence|wound breakdown|"
    r"necrosis|skin flap necrosis|mastectomy skin flap necrosis|exposure|exposed|"
    r"extrusion|rupture|leakage|deflation|malposition|contracture|capsular contracture|"
    r"flap compromise|venous congestion|arterial insufficiency|flap loss|flap failure|"
    r"open wound|drainage|erythema|purulence|purulent|soft tissue infection|fat necrosis"
    r")\b",
    re.IGNORECASE
)

RISK_DISCUSSION_RX = re.compile(
    r"\b("
    r"risk of|risks of|risks include|risk includes|possible complications|potential complications|"
    r"potential consequences|possible need for|need for additional surgeries|"
    r"complications include|signs and symptoms of|counsel(ed|ing)? on|educated on|"
    r"warning signs|what to expect|long-term maintenance/failure rate|maintenance/failure rate|"
    r"if she develops|future procedures|we discussed the risks|risks and benefits"
    r")\b",
    re.IGNORECASE
)

COSMETIC_REVISION_ONLY_RX = re.compile(
    r"\b("
    r"symmetry|asymmetry|contour deformit(y|ies)|cosmetic|balancing|contralateral mastopexy|"
    r"contralateral reduction|mastopexy|reduction|fat grafting|fat transfer|lipofilling|"
    r"dog[- ]ear|standing cutaneous deformit(y|ies)|recontouring|scar revision"
    r")\b",
    re.IGNORECASE
)

ROUTINE_STAGE2_EXCHANGE_RX = re.compile(
    r"\b("
    r"implant exchange|exchange of (the )?(tissue )?expanders? for (silicone|saline )?implants?|"
    r"tissue expander exchange|expander[- ]implant exchange|second stage reconstruction|"
    r"stage 2 reconstruction|stage ii reconstruction|delayed exchange|exchange for permanent implant"
    r")\b",
    re.IGNORECASE
)

REVISION_PERFORMED_RX = re.compile(
    r"\b("
    r"revision of (the )?(reconstructed )?breast|revision of reconstructed breast|"
    r"underwent revision|status post revision|s/p revision|"
    r"procedure performed|operations performed|operation performed|"
    r"capsulectomy|capsulotomy|capsulorrhaphy|fat grafting|fat transfer|lipofilling|"
    r"scar revision|mastopexy|reduction|dog[- ]ear revision|standing cutaneous deformity|"
    r"implant exchange|placement of new implants|exchange of .* implant|"
    r"exchange of .* expander|revised bilateral breast reconstruction|"
    r"breast lateralis?ation|capsulopexy"
    r")\b",
    re.IGNORECASE
)

# exclude nipple reconstruction by itself
NIPPLE_ONLY_RX = re.compile(
    r"\b("
    r"nipple reconstruction|nipple-areola reconstruction|nipple areolar reconstruction|"
    r"areolar reconstruction|skate flap|nipple-areolar complex reconstruction"
    r")\b",
    re.IGNORECASE
)

NIPPLE_WITH_REAL_REVISION_RX = re.compile(
    r"\b("
    r"implant|fat graft|fat transfer|capsulotomy|capsulectomy|capsulorrhaphy|"
    r"exchange|revision of reconstructed breast|mastopexy|reduction"
    r")\b",
    re.IGNORECASE
)

# pathology / cancer necrosis leakage for minor comp
PATHOLOGY_CONTEXT_RX = re.compile(
    r"\b("
    r"core biopsy|microcalcifications|dcis|pathology|margins|ductal carcinoma|"
    r"lobular carcinoma|cribriform|solid patterns|grade\s+[1-4]|retroareolar|"
    r"invasive ductal|carcinoma"
    r")\b",
    re.IGNORECASE
)

# strong history-list pattern
SURGICAL_HISTORY_LINE_RX = re.compile(
    r"(\b\d{1,2}/\d{1,2}/\d{2,4}\b.*\b(procedure|surgery|mastectomy|reconstruction|flap|implant|expander)\b)|"
    r"(\b[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b.*\b(procedure|surgery|mastectomy|reconstruction|flap|implant|expander)\b)|"
    r"(\b(?:port placement|mastectomy|hysterectomy|oophorectomy|biopsy|diep flap|tram flap)\b.*(?:\||□))",
    re.IGNORECASE
)


# --------------------------------------------------
# Concept patterns
# --------------------------------------------------
MINOR_COMP_POS = [
    r"\bhematoma\b",
    r"\bseroma\b",
    r"\bwound dehiscence\b",
    r"\bdehiscence\b",
    r"\bwound infection\b",
    r"\binfection\b",
    r"\bcellulitis\b",
    r"\babscess\b",
    r"\bmastectomy skin flap necrosis\b",
    r"\bskin flap necrosis\b",
    r"\bnecrosis\b",
    r"\bcapsular contracture\b",
    r"\bimplant malposition\b",
    r"\bimplant rupture\b",
    r"\bimplant leakage\b",
    r"\bimplant deflation\b",
    r"\bexpander extrusion\b",
    r"\bimplant extrusion\b",
    r"\bexposed implant\b",
    r"\bexposed expander\b",
    r"\bwound breakdown\b",
    r"\bopen wound\b",
    r"\bfat necrosis\b",
]

REOP_PROCEDURE_POS = [
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
    r"\bhematoma evacuation\b",
    r"\bseroma drainage\b",
    r"\bdrain placement\b",
    r"\bdrainage procedure\b",
    r"\bexplantation\b",
    r"\bexplant(ed)?\b",
    r"\bremoval of (the )?(implant|expander|flap)\b",
    r"\bimplant removal\b",
    r"\bexpander removal\b",
]

REHOSP_POS = [
    r"\breadmit(ted|sion)?\b",
    r"\bre-?admit(ted|sion)?\b",
    r"\brehospitali[sz](ed|ation)?\b",
    r"\breturn(ed)?\s+to\s+hospital\b",
]

FAILURE_REMOVAL_POS = [
    r"\bflap\s+loss\b",
    r"\bflap\s+failure\b",
    r"\btotal\s+flap\s+loss\b",
    r"\bcomplete\s+flap\s+necrosis\b",
    r"\bimplant\s+removed\b",
    r"\bimplant\s+removal\b",
    r"\bexpander\s+removed\b",
    r"\bexpander\s+removal\b",
    r"\bprosthesis\s+removed\b",
    r"\bexplant(ed|ation)?\b",
    r"\bremoval of (the )?(implant|expander|flap)\b",
]

FAILURE_REASON_RX = re.compile(
    r"\b("
    r"infection|infected|cellulitis|exposure|exposed|extrusion|rupture|leakage|deflation|"
    r"necrosis|flap loss|flap failure|venous congestion|arterial insufficiency|"
    r"implant malposition|capsular contracture|hematoma|seroma|dehiscence|open wound"
    r")\b",
    re.IGNORECASE
)

REVISION_POS = [
    r"\brevision\s+surgery\b",
    r"\brevision\s+procedure\b",
    r"\bscar\s+revision\b",
    r"\bfat\s+grafting\b",
    r"\bfat\s+transfer\b",
    r"\blipofill(ing)?\b",
    r"\bcapsulectomy\b",
    r"\bcapsulotomy\b",
    r"\bcapsulorrhaphy\b",
    r"\bcontour\s+revision\b",
    r"\bcontour\s+deformit(y|ies)\b",
    r"\bdog[- ]ear\s+revision\b",
    r"\bstanding\s+cutaneous\s+deformit(y|ies)\b",
    r"\bcontralateral\s+mastopexy\b",
    r"\bcontralateral\s+reduction\b",
    r"\bsymmetry\s+procedure\b",
    r"\bbalancing\s+procedure\b",
    r"\bmastopexy\b",
    r"\breduction\b",
    r"\bimplant exchange\b",
    r"\bexchange of .* implant\b",
    r"\bexchange of .* expander\b",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _section_rank(section):
    s = (section or "").strip().upper()
    if s in PREFERRED_SECTIONS:
        return 0
    if s in LOW_VALUE_SECTIONS:
        return 2
    if s in REVISION_SUPPRESS_SECTIONS:
        return 3
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


def _is_negated(evid):
    return bool(NEGATION_RX.search(evid))


def _is_family(evid):
    return bool(FAMILY_RX.search(evid))


def _is_planned(evid):
    return bool(PLAN_RX.search(evid))


def _has_breast_recon_context(evid):
    return bool(BREAST_RECON_CONTEXT_RX.search(evid))


def _is_historical_only(evid):
    return bool(HISTORICAL_ONLY_RX.search(evid))


def _is_resolved_only(evid):
    return bool(RESOLVED_RX.search(evid))


def _has_acute_event(evid):
    return bool(ACUTE_EVENT_RX.search(evid))


def _is_risk_discussion(evid):
    return bool(RISK_DISCUSSION_RX.search(evid))


def _base_conf(section, note_type):
    rank = _section_rank(section)
    nt = (note_type or "").lower()
    conf = 0.68
    if rank == 0:
        conf += 0.08
    elif rank == 2:
        conf -= 0.08
    elif rank == 3:
        conf -= 0.12
    if "op" in nt or "operative" in nt or "operation" in nt:
        conf += 0.08
    return max(0.55, min(0.95, conf))


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


def _current_complication_ok(low):
    if _is_risk_discussion(low):
        return False
    if PATHOLOGY_CONTEXT_RX.search(low) and not _has_breast_recon_context(low):
        return False
    if (_is_historical_only(low) or _is_resolved_only(low)) and not _has_acute_event(low):
        return False
    return True


def _has_major_treatment_context(low):
    return bool(re.search(
        r"\b("
        r"washout|debridement|admitted|iv antibiotics|readmitted|rehospitalized|"
        r"explant|removal|evacuation|returned to the or|drain placed|aspirated"
        r")\b",
        low,
        re.IGNORECASE
    ))


def _revision_history_block(section, low):
    if (section or "").strip().upper() in REVISION_SUPPRESS_SECTIONS:
        return True
    if SURGICAL_HISTORY_LINE_RX.search(low):
        return True
    return False


# --------------------------------------------------
# Core extractors
# --------------------------------------------------
def _extract_minor_comp(note):
    cands = []  # type: List[Candidate]

    for section, text in _iter_sections(note):
        m = _find_first(MINOR_COMP_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 260)
        low = evid.lower()

        if _is_family(low):
            continue
        if not _has_breast_recon_context(low):
            continue

        if _is_negated(low):
            cands.append(_emit("ComplicationSignal", False, "denied", evid, section, note, 0.60))
            continue

        # allow true acute/treated complication even if note has some planning language elsewhere
        if _is_planned(low) and not re.search(
            r"\b(treated with|started on|prescribed|drain placed|aspirated|developed|cellulitis|seroma|hematoma|infection)\b",
            low,
            re.IGNORECASE
        ):
            cands.append(_emit("ComplicationSignal", False, "planned", evid, section, note, 0.55))
            continue

        if not _current_complication_ok(low):
            continue

        # if clearly major-treatment context, still emit raw signal;
        # minor outcome will be derived later in patch script
        conf = _base_conf(section, note.note_type)
        if _has_acute_event(low):
            conf += 0.03

        cands.append(_emit("ComplicationSignal", True, "history", evid, section, note, min(conf, 0.95)))

        if _section_rank(section) == 0:
            break

    return cands


def _extract_reoperation(note):
    cands = []  # type: List[Candidate]

    for section, text in _iter_sections(note):
        m = _find_first(REOP_PROCEDURE_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 280)
        low = evid.lower()

        if _is_family(low):
            continue
        if not _has_breast_recon_context(low):
            continue

        if _is_negated(low):
            cands.append(_emit("StageOutcome_Reoperation", False, "denied", evid, section, note, 0.60))
            continue
        if _is_planned(low):
            cands.append(_emit("StageOutcome_Reoperation", False, "planned", evid, section, note, 0.55))
            continue

        if ROUTINE_STAGE2_EXCHANGE_RX.search(low) and not COMPLICATION_REASON_RX.search(low):
            continue

        if COSMETIC_REVISION_ONLY_RX.search(low) and not COMPLICATION_REASON_RX.search(low):
            continue

        if not COMPLICATION_REASON_RX.search(low):
            continue

        conf = _base_conf(section, note.note_type) + 0.05
        cands.append(_emit("StageOutcome_Reoperation", True, "history", evid, section, note, min(conf, 0.95)))

        if _section_rank(section) == 0:
            break

    return cands


def _extract_rehospitalization(note):
    cands = []  # type: List[Candidate]

    for section, text in _iter_sections(note):
        m = _find_first(REHOSP_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 260)
        low = evid.lower()

        if _is_family(low):
            continue
        if _is_negated(low):
            cands.append(_emit("StageOutcome_Rehospitalization", False, "denied", evid, section, note, 0.60))
            continue
        if _is_planned(low):
            cands.append(_emit("StageOutcome_Rehospitalization", False, "planned", evid, section, note, 0.55))
            continue

        if not _has_breast_recon_context(low) and not COMPLICATION_REASON_RX.search(low):
            continue

        conf = _base_conf(section, note.note_type) + 0.06
        cands.append(_emit("StageOutcome_Rehospitalization", True, "history", evid, section, note, min(conf, 0.95)))

        if _section_rank(section) == 0:
            break

    return cands


def _extract_failure(note):
    cands = []  # type: List[Candidate]

    for section, text in _iter_sections(note):
        m = _find_first(FAILURE_REMOVAL_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 280)
        low = evid.lower()

        if _is_family(low):
            continue
        if not _has_breast_recon_context(low):
            continue

        if _is_negated(low):
            cands.append(_emit("StageOutcome_Failure", False, "denied", evid, section, note, 0.60))
            continue
        if _is_planned(low):
            cands.append(_emit("StageOutcome_Failure", False, "planned", evid, section, note, 0.55))
            continue

        if ROUTINE_STAGE2_EXCHANGE_RX.search(low) and not FAILURE_REASON_RX.search(low):
            continue

        if not FAILURE_REASON_RX.search(low):
            continue

        conf = _base_conf(section, note.note_type) + 0.06
        cands.append(_emit("StageOutcome_Failure", True, "history", evid, section, note, min(conf, 0.95)))

        if _section_rank(section) == 0:
            break

    return cands


def _extract_revision(note):
    cands = []  # type: List[Candidate]

    for section, text in _iter_sections(note):
        m = _find_first(REVISION_POS, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 300)
        low = evid.lower()

        if _is_family(low):
            continue
        if not _has_breast_recon_context(low):
            continue

        # suppress surgical history blocks
        if _revision_history_block(section, low):
            continue

        if _is_negated(low):
            cands.append(_emit("StageOutcome_Revision", False, "denied", evid, section, note, 0.60))
            continue

        # stronger planned suppression for revision
        if _is_planned(low) or REVISION_PLAN_RX.search(low):
            cands.append(_emit("StageOutcome_Revision", False, "planned", evid, section, note, 0.55))
            continue

        if _is_risk_discussion(low):
            continue

        # historical mention alone should not count
        if (_is_historical_only(low) or _is_resolved_only(low)) and not (
            REVISION_PERFORMED_RX.search(low) or _has_acute_event(low)
        ):
            continue

        # nipple reconstruction alone should not count
        if NIPPLE_ONLY_RX.search(low) and not NIPPLE_WITH_REAL_REVISION_RX.search(low):
            continue

        # require performed revision signal
        if not REVISION_PERFORMED_RX.search(low):
            continue

        conf = _base_conf(section, note.note_type) + 0.02
        cands.append(_emit("StageOutcome_Revision", True, "history", evid, section, note, min(conf, 0.95)))

        if _section_rank(section) == 0:
            break

    return cands


def extract_complication_outcomes(note):
    cands = []  # type: List[Candidate]
    cands.extend(_extract_minor_comp(note))
    cands.extend(_extract_reoperation(note))
    cands.extend(_extract_rehospitalization(note))
    cands.extend(_extract_failure(note))
    cands.extend(_extract_revision(note))
    return cands


# backward-compatible alias
def extract_stage1_outcomes(note):
    return extract_complication_outcomes(note)
