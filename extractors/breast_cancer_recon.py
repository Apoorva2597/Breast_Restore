# extractors/breast_cancer_recon.py
# Python 3.6.8 compatible

import re
from typing import List, Optional, Tuple

from models import Candidate, SectionedNote
from .utils import window_around

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "REVIEW OF SYSTEMS",
    "ALLERGIES",
}

LEFT_RX = re.compile(r"\b(left|lt|l)\b", re.IGNORECASE)
RIGHT_RX = re.compile(r"\b(right|rt|r)\b", re.IGNORECASE)
BILAT_RX = re.compile(r"\b(bilateral|bilat)\b", re.IGNORECASE)

MASTECTOMY_RX = re.compile(
    r"\b("
    r"mastectomy|"
    r"simple\s+mastectomy|"
    r"total\s+mastectomy|"
    r"skin[- ]sparing\s+mastectomy|"
    r"nipple[- ]sparing\s+mastectomy|"
    r"modified\s+radical\s+mastectomy|"
    r"\bMRM\b"
    r")\b",
    re.IGNORECASE
)

RECON_RX = re.compile(
    r"\b("
    r"breast\s+reconstruction|"
    r"reconstruction|"
    r"diep|tram|latissimus|flap|"
    r"tissue\s+expander|expander|"
    r"implant|alloderm|acellular\s+dermal\s+matrix"
    r")\b",
    re.IGNORECASE
)

SLNB_RX = re.compile(
    r"\b("
    r"sentinel\s+lymph\s+node\s+biopsy|"
    r"sentinel\s+node\s+biopsy|"
    r"\bSLNB\b|"
    r"lymphatic\s+mapping"
    r")\b",
    re.IGNORECASE
)

ALND_RX = re.compile(
    r"\b("
    r"axillary\s+lymph\s+node\s+dissection|"
    r"axillary\s+dissection|"
    r"\bALND\b"
    r")\b",
    re.IGNORECASE
)

RADIATION_RX = re.compile(
    r"\b("
    r"radiation|radiation\s+therapy|radiotherapy|xrt|pmrt"
    r")\b",
    re.IGNORECASE
)

CHEMO_RX = re.compile(
    r"\b("
    r"chemotherapy|chemo|"
    r"adriamycin|doxorubicin|"
    r"cyclophosphamide|cytoxan|"
    r"taxol|paclitaxel|docetaxel|taxotere|"
    r"carboplatin|cisplatin|"
    r"trastuzumab|herceptin|pertuzumab|perjeta|"
    r"\bTCHP?\b|\bAC\b|\bTC\b"
    r")\b",
    re.IGNORECASE
)

ENDOCRINE_ONLY_RX = re.compile(
    r"\b("
    r"tamoxifen|letrozole|anastrozole|exemestane|"
    r"fulvestrant|arimidex|femara|aromasin"
    r")\b",
    re.IGNORECASE
)

NEGATION_RX = re.compile(
    r"\b("
    r"no|denies|denied|without|not|never"
    r")\b",
    re.IGNORECASE
)

PLANNED_RX = re.compile(
    r"\b("
    r"plan|planned|planning|will|scheduled|schedule|candidate|consider"
    r")\b",
    re.IGNORECASE
)

PROPHYLAXIS_RX = re.compile(
    r"\b("
    r"prophylactic|risk[- ]reducing|risk\s+reducing|preventive|contralateral\s+prophylactic"
    r")\b",
    re.IGNORECASE
)

CANCER_RX = re.compile(
    r"\b("
    r"breast\s+cancer|carcinoma|malignancy|malignant|"
    r"invasive\s+ductal|invasive\s+lobular|dcis|lcis|cancer"
    r")\b",
    re.IGNORECASE
)


def _is_operation_note(note_type: str) -> bool:
    s = (note_type or "").lower()
    return (
        ("brief op" in s) or
        ("op note" in s) or
        ("operative" in s) or
        ("operation" in s) or
        ("oper report" in s)
    )


def _clean(x):
    return str(x).strip() if x is not None else ""


def _emit(field, value, text, m, section, note, conf):
    return Candidate(
        field=field,
        value=value,
        status="history",
        evidence=window_around(text, m.start(), m.end(), 220),
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=conf,
    )


def _window(text: str, start: int, end: int, width: int = 160) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return text[lo:hi]


def _infer_laterality(text: str) -> Optional[str]:
    low = (text or "").lower()
    if BILAT_RX.search(low):
        return "BILATERAL"
    has_left = bool(re.search(r"\b(left|lt)\b", low))
    has_right = bool(re.search(r"\b(right|rt)\b", low))
    if has_left and has_right:
        return "BILATERAL"
    if has_left:
        return "LEFT"
    if has_right:
        return "RIGHT"
    return None


def _looks_negated_or_planned(ctx: str, op_note: bool) -> bool:
    low = (ctx or "").lower()
    if NEGATION_RX.search(low):
        return True
    if PLANNED_RX.search(low) and not op_note:
        return True
    return False


def _infer_recon_type_and_class(text: str) -> Tuple[Optional[str], Optional[str]]:
    low = (text or "").lower()

    autologous_terms = [
        "diep", "tram", "latissimus", "flap"
    ]
    implant_terms = [
        "implant", "expander", "tissue expander", "alloderm", "acellular dermal matrix"
    ]

    has_auto = any(t in low for t in autologous_terms)
    has_implant = any(t in low for t in implant_terms)

    if "diep" in low:
        rtype = "DIEP"
    elif "tram" in low:
        rtype = "TRAM"
    elif "latissimus" in low:
        rtype = "LATISSIMUS"
    elif "tissue expander" in low or "expander" in low:
        rtype = "EXPANDER"
    elif "implant" in low:
        rtype = "IMPLANT"
    elif "flap" in low:
        rtype = "FLAP"
    else:
        rtype = None

    if has_auto and has_implant:
        rclass = "Hybrid"
    elif has_auto:
        rclass = "Autologous"
    elif has_implant:
        rclass = "ImplantBased"
    else:
        rclass = None

    return rtype, rclass


def _infer_indications(text: str, lat: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    low = (text or "").lower()

    left_val = None
    right_val = None

    left_pro = bool(re.search(r"(left|lt).{0,80}(prophylactic|risk[- ]reducing|preventive)", low)) or \
               bool(re.search(r"(prophylactic|risk[- ]reducing|preventive).{0,80}(left|lt)", low))
    right_pro = bool(re.search(r"(right|rt).{0,80}(prophylactic|risk[- ]reducing|preventive)", low)) or \
                bool(re.search(r"(prophylactic|risk[- ]reducing|preventive).{0,80}(right|rt)", low))

    left_cancer = bool(re.search(r"(left|lt).{0,120}(cancer|carcinoma|dcis|lcis|malignan|invasive)", low)) or \
                  bool(re.search(r"(cancer|carcinoma|dcis|lcis|malignan|invasive).{0,120}(left|lt)", low))
    right_cancer = bool(re.search(r"(right|rt).{0,120}(cancer|carcinoma|dcis|lcis|malignan|invasive)", low)) or \
                   bool(re.search(r"(cancer|carcinoma|dcis|lcis|malignan|invasive).{0,120}(right|rt)", low))

    note_has_cancer = bool(CANCER_RX.search(low))
    note_has_pro = bool(PROPHYLAXIS_RX.search(low))

    if left_pro:
        left_val = "Prophylactic"
    elif left_cancer:
        left_val = "Therapeutic"
    elif lat == "LEFT" and note_has_cancer and not note_has_pro:
        left_val = "Therapeutic"

    if right_pro:
        right_val = "Prophylactic"
    elif right_cancer:
        right_val = "Therapeutic"
    elif lat == "RIGHT" and note_has_cancer and not note_has_pro:
        right_val = "Therapeutic"

    if lat == "BILATERAL":
        if left_val is None and left_cancer:
            left_val = "Therapeutic"
        if right_val is None and right_cancer:
            right_val = "Therapeutic"

    return left_val, right_val


def extract_breast_cancer_recon(note: SectionedNote) -> List[Candidate]:
    cands = []
    op_note = _is_operation_note(note.note_type)

    for section, text in note.sections.items():
        if section in SUPPRESS_SECTIONS:
            continue
        if not text:
            continue

        low = text.lower()

        # -----------------------
        # Mastectomy block
        # -----------------------
        m = MASTECTOMY_RX.search(text)
        if m:
            ctx = _window(text, m.start(), m.end(), 220)
            if not _looks_negated_or_planned(ctx, op_note):
                lat = _infer_laterality(ctx) or _infer_laterality(text)
                if lat:
                    cands.append(_emit("Mastectomy_Laterality", lat, text, m, section, note, 0.90 if op_note else 0.74))

                if _clean(note.note_date):
                    cands.append(_emit("Mastectomy_Date", _clean(note.note_date), text, m, section, note, 0.88 if op_note else 0.68))

                left_ind, right_ind = _infer_indications(text, lat)
                if left_ind:
                    cands.append(_emit("Indication_Left", left_ind, text, m, section, note, 0.80 if op_note else 0.66))
                if right_ind:
                    cands.append(_emit("Indication_Right", right_ind, text, m, section, note, 0.80 if op_note else 0.66))

                if ALND_RX.search(text):
                    mm = ALND_RX.search(text)
                    cands.append(_emit("LymphNode", "ALND", text, mm, section, note, 0.86 if op_note else 0.72))
                elif SLNB_RX.search(text):
                    mm = SLNB_RX.search(text)
                    cands.append(_emit("LymphNode", "SLNB", text, mm, section, note, 0.82 if op_note else 0.70))

        # -----------------------
        # Reconstruction block
        # -----------------------
        r = RECON_RX.search(text)
        if r:
            ctx = _window(text, r.start(), r.end(), 220)
            if not _looks_negated_or_planned(ctx, op_note):
                lat = _infer_laterality(ctx) or _infer_laterality(text)
                if lat:
                    cands.append(_emit("Recon_Laterality", lat, text, r, section, note, 0.90 if op_note else 0.76))

                rtype, rclass = _infer_recon_type_and_class(text)
                if rtype:
                    cands.append(_emit("Recon_Type", rtype, text, r, section, note, 0.88 if op_note else 0.74))
                if rclass:
                    cands.append(_emit("Recon_Classification", rclass, text, r, section, note, 0.88 if op_note else 0.74))

                if op_note and MASTECTOMY_RX.search(text):
                    cands.append(_emit("Recon_Timing", "Immediate", text, r, section, note, 0.92))

        # -----------------------
        # Radiation block
        # -----------------------
        rr = RADIATION_RX.search(text)
        if rr:
            ctx = _window(text, rr.start(), rr.end(), 220)
            if not _looks_negated_or_planned(ctx, False):
                cands.append(_emit("Radiation", True, text, rr, section, note, 0.80))

        # -----------------------
        # Chemo block
        # -----------------------
        cc = CHEMO_RX.search(text)
        if cc:
            ctx = _window(text, cc.start(), cc.end(), 220)
            low_ctx = ctx.lower()

            if ENDOCRINE_ONLY_RX.search(low_ctx) and not re.search(r"\b(chemo|chemotherapy)\b", low_ctx):
                pass
            elif not _looks_negated_or_planned(ctx, False):
                cands.append(_emit("Chemo", True, text, cc, section, note, 0.80))

    return cands
