# extractors/breast_cancer_recon.py
# Python 3.6.8 compatible
import re
from models import Candidate, SectionedNote
from .utils import window_around

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "REVIEW OF SYSTEMS",
    "ALLERGIES",
}

LOW_VALUE_SECTIONS = {
    "PAST MEDICAL HISTORY",
    "PAST SURGICAL HISTORY",
    "SURGICAL HISTORY",
    "HISTORY",
    "PMH",
    "PSH",
    "GYNECOLOGIC HISTORY",
    "OB HISTORY",
    "FAMILY HISTORY",
}

LEFT_RX = re.compile(r"\b(left|lt)\b", re.IGNORECASE)
RIGHT_RX = re.compile(r"\b(right|rt)\b", re.IGNORECASE)
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
    re.IGNORECASE,
)
RECON_RX = re.compile(
    r"\b("
    r"breast\s+reconstruction|"
    r"reconstruction|"
    r"diep|tram|siea|latissimus|flap|"
    r"tissue\s+expander|expander|"
    r"implant|alloderm|acellular\s+dermal\s+matrix|"
    r"direct[- ]to[- ]implant|"
    r"sgap|igap|gap\s+flap|gluteal\s+artery\s+perforator"
    r")\b",
    re.IGNORECASE,
)
SLNB_RX = re.compile(
    r"\b("
    r"sentinel\s+lymph\s+node\s+biopsy|"
    r"sentinel\s+node\s+biopsy|"
    r"\bSLNB\b|"
    r"lymphatic\s+mapping"
    r")\b",
    re.IGNORECASE,
)
ALND_RX = re.compile(
    r"\b("
    r"axillary\s+lymph\s+node\s+dissection|"
    r"axillary\s+dissection|"
    r"\bALND\b"
    r")\b",
    re.IGNORECASE,
)
RADIATION_RX = re.compile(
    r"\b("
    r"radiation|radiation\s+therapy|radiotherapy|xrt|pmrt"
    r")\b",
    re.IGNORECASE,
)
CHEMO_RX = re.compile(
    r"\b("
    r"chemotherapy|chemo|"
    r"adriamycin|doxorubicin|"
    r"cyclophosphamide|cytoxan|"
    r"taxol|paclitaxel|docetaxel|taxotere|"
    r"carboplatin|cisplatin|"
    r"trastuzumab|herceptin|pertuzumab|perjeta|"
    r"\bTCHP?\b|\bAC\b|\bTC\b|\bACT\b"
    r")\b",
    re.IGNORECASE,
)
ENDOCRINE_ONLY_RX = re.compile(
    r"\b("
    r"tamoxifen|letrozole|anastrozole|exemestane|"
    r"fulvestrant|arimidex|femara|aromasin"
    r")\b",
    re.IGNORECASE,
)
NEGATION_RX = re.compile(r"\b(no|denies|denied|without|not|never)\b", re.IGNORECASE)
PLANNED_RX = re.compile(
    r"\b(plan|planned|planning|will|scheduled|schedule|candidate|consider|recommend|discuss)\b",
    re.IGNORECASE,
)
PROPHYLAXIS_RX = re.compile(
    r"\b(prophylactic|risk[- ]reducing|risk\s+reducing|preventive|contralateral\s+prophylactic)\b",
    re.IGNORECASE,
)
CANCER_RX = re.compile(
    r"\b(breast\s+cancer|carcinoma|malignancy|malignant|invasive\s+ductal|invasive\s+lobular|dcis|lcis|recurrent\s+cancer|cancer)\b",
    re.IGNORECASE,
)
TREATMENT_RECEIVED_RX = re.compile(
    r"\b(s/p|status\s+post|history\s+of|hx\s+of|prior|previous|completed|received|underwent|treated\s+with|adjuvant|neoadjuvant|postmastectomy)\b",
    re.IGNORECASE,
)
STRONG_RADIATION_HISTORY_RX = re.compile(
    r"\b(s/p\s+radiation|status\s+post\s+radiation|history\s+of\s+radiation|prior\s+radiation|previous\s+radiation|completed\s+radiation|received\s+radiation|adjuvant\s+radiation|neoadjuvant\s+radiation|radiation\s+therapy\s+completed|postmastectomy\s+radiation)\b",
    re.IGNORECASE,
)
STRONG_CHEMO_HISTORY_RX = re.compile(
    r"\b(s/p\s+chemo|status\s+post\s+chemo|history\s+of\s+chemo|prior\s+chemo|previous\s+chemo|completed\s+chemo|completed\s+chemotherapy|received\s+chemo|received\s+chemotherapy|adjuvant\s+chemo|neoadjuvant\s+chemo|treated\s+with\s+chemotherapy|treated\s+with\s+chemo)\b",
    re.IGNORECASE,
)
WEAK_TREATMENT_EXCLUDE_RX = re.compile(
    r"\b(consider|candidate|discussion|discussed|recommend|recommended|plan|planned|planning|will\s+start|may\s+need|may\s+require|referred\s+to\s+radiation\s+oncology|radiation\s+oncology\s+consult)\b",
    re.IGNORECASE,
)


def _is_operation_note(note_type):
    s = (note_type or "").lower()
    return (("brief op" in s) or ("op note" in s) or ("operative" in s) or ("operation" in s) or ("oper report" in s))


def _is_clinic_like(note_type):
    s = (note_type or "").lower()
    pats = ["clinic", "progress", "office", "follow up", "follow-up", "consult", "pre-op", "preop", "history and physical", "h&p", "oncology"]
    for p in pats:
        if p in s:
            return True
    return False


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


def _window(text, start, end, width=160):
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return text[lo:hi]


def _infer_laterality(text):
    low = (text or "").lower()
    if BILAT_RX.search(low):
        return "BILATERAL"
    has_left = bool(LEFT_RX.search(low))
    has_right = bool(RIGHT_RX.search(low))
    if has_left and has_right:
        return "BILATERAL"
    if has_left:
        return "LEFT"
    if has_right:
        return "RIGHT"
    return None


def _looks_negated_or_planned(ctx, op_note):
    low = (ctx or "").lower()
    if NEGATION_RX.search(low):
        return True
    if PLANNED_RX.search(low) and not op_note:
        return True
    return False


def _infer_recon_type_and_class(text):
    low = (text or "").lower()
    flap_types_found = []
    if "diep" in low:
        flap_types_found.append("DIEP")
    if "tram" in low:
        flap_types_found.append("TRAM")
    if "siea" in low:
        flap_types_found.append("SIEA")
    if ("gluteal artery perforator" in low or "gap flap" in low or re.search(r"\bsgap\b", low) or re.search(r"\bigap\b", low)):
        flap_types_found.append("gluteal artery perforator flap")
    if "latissimus" in low:
        flap_types_found.append("latissimus dorsi")
    has_any_flap = (len(flap_types_found) > 0 or (" flap" in low) or low.startswith("flap") or ("mixed flaps" in low))
    has_direct_to_implant = bool(re.search(r"\bdirect[- ]to[- ]implant\b", low))
    has_expander = ("tissue expander" in low) or ("expander" in low)
    has_implant = ("implant" in low)
    rtype = None
    if "mixed flaps" in low:
        rtype = "mixed flaps"
    elif len(set(flap_types_found)) >= 2:
        rtype = "mixed flaps"
    elif "DIEP" in flap_types_found:
        rtype = "DIEP"
    elif "TRAM" in flap_types_found:
        rtype = "TRAM"
    elif "SIEA" in flap_types_found:
        rtype = "SIEA"
    elif "gluteal artery perforator flap" in flap_types_found:
        rtype = "gluteal artery perforator flap"
    elif "latissimus dorsi" in flap_types_found:
        rtype = "latissimus dorsi"
    elif has_direct_to_implant:
        rtype = "direct-to-implant"
    elif has_expander or (has_implant and not has_any_flap):
        rtype = "expander/implant"
    elif has_any_flap:
        rtype = "other"
    elif has_implant:
        rtype = "expander/implant"
    rclass = None
    if has_any_flap:
        rclass = "autologous"
    elif has_direct_to_implant or has_expander or has_implant:
        rclass = "implant"
    elif rtype == "other":
        rclass = "other"
    return rtype, rclass


def _infer_indications(text, lat):
    low = (text or "").lower()
    left_val = None
    right_val = None

    paired_patterns = [
        (r"left.{0,100}(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent).{0,100}right.{0,100}(prophylactic|risk[- ]reducing|preventive)", ("Therapeutic", "Prophylactic")),
        (r"right.{0,100}(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent).{0,100}left.{0,100}(prophylactic|risk[- ]reducing|preventive)", ("Prophylactic", "Therapeutic")),
    ]
    for pat, vals in paired_patterns:
        if re.search(pat, low):
            return vals

    left_pro = bool(re.search(r"(left|lt).{0,80}(prophylactic|risk[- ]reducing|preventive)", low)) or bool(re.search(r"(prophylactic|risk[- ]reducing|preventive).{0,80}(left|lt)", low))
    right_pro = bool(re.search(r"(right|rt).{0,80}(prophylactic|risk[- ]reducing|preventive)", low)) or bool(re.search(r"(prophylactic|risk[- ]reducing|preventive).{0,80}(right|rt)", low))
    left_cancer = bool(re.search(r"(left|lt).{0,120}(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent)", low)) or bool(re.search(r"(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent).{0,120}(left|lt)", low))
    right_cancer = bool(re.search(r"(right|rt).{0,120}(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent)", low)) or bool(re.search(r"(cancer|carcinoma|dcis|lcis|malignan|invasive|recurrent).{0,120}(right|rt)", low))

    if "contralateral prophylactic" in low:
        if left_cancer and not right_cancer:
            right_pro = True
        elif right_cancer and not left_cancer:
            left_pro = True

    if left_pro:
        left_val = "Prophylactic"
    elif left_cancer:
        left_val = "Therapeutic"

    if right_pro:
        right_val = "Prophylactic"
    elif right_cancer:
        right_val = "Therapeutic"

    if lat == "BILATERAL":
        if left_val is None and left_cancer:
            left_val = "Therapeutic"
        if right_val is None and right_cancer:
            right_val = "Therapeutic"

    return left_val, right_val


def _strong_radiation_history(ctx):
    low = (ctx or "").lower()
    if WEAK_TREATMENT_EXCLUDE_RX.search(low):
        return False
    if STRONG_RADIATION_HISTORY_RX.search(low):
        return True
    if RADIATION_RX.search(low) and TREATMENT_RECEIVED_RX.search(low):
        return True
    return False


def _strong_chemo_history(ctx):
    low = (ctx or "").lower()
    if WEAK_TREATMENT_EXCLUDE_RX.search(low):
        return False
    if ENDOCRINE_ONLY_RX.search(low) and not CHEMO_RX.search(low):
        return False
    if STRONG_CHEMO_HISTORY_RX.search(low):
        return True
    if CHEMO_RX.search(low) and TREATMENT_RECEIVED_RX.search(low):
        return True
    return False


def extract_breast_cancer_recon(note):
    cands = []
    op_note = _is_operation_note(note.note_type)
    clinic_like = _is_clinic_like(note.note_type)

    for section, text in note.sections.items():
        if section in SUPPRESS_SECTIONS:
            continue
        if not text:
            continue

        section_low_value = section in LOW_VALUE_SECTIONS

        m = MASTECTOMY_RX.search(text)
        if m:
            ctx = _window(text, m.start(), m.end(), 220)
            if not _looks_negated_or_planned(ctx, op_note):
                lat = _infer_laterality(ctx) or _infer_laterality(text)
                if lat:
                    conf = 0.90 if op_note else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Mastectomy_Laterality", lat, text, m, section, note, conf))
                if _clean(note.note_date):
                    conf = 0.88 if op_note else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Mastectomy_Date", _clean(note.note_date), text, m, section, note, conf))
                left_ind, right_ind = _infer_indications(text, lat)
                if left_ind is not None:
                    conf = 0.80 if op_note else 0.66
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Left", left_ind, text, m, section, note, conf))
                if right_ind is not None:
                    conf = 0.80 if op_note else 0.66
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Right", right_ind, text, m, section, note, conf))

            if ALND_RX.search(text):
                mm = ALND_RX.search(text)
                conf = 0.86 if op_note else 0.72
                if section_low_value and not op_note:
                    conf -= 0.10
                cands.append(_emit("LymphNode", "ALND", text, mm, section, note, conf))
            elif SLNB_RX.search(text):
                mm = SLNB_RX.search(text)
                conf = 0.82 if op_note else 0.70
                if section_low_value and not op_note:
                    conf -= 0.10
                cands.append(_emit("LymphNode", "SLNB", text, mm, section, note, conf))
            else:
                conf = 0.60 if op_note else 0.50
                if section_low_value and not op_note:
                    conf -= 0.10
                cands.append(_emit("LymphNode", "none", text, m, section, note, conf))

        r = RECON_RX.search(text)
        if r:
            ctx = _window(text, r.start(), r.end(), 220)
            if not _looks_negated_or_planned(ctx, op_note):
                lat = _infer_laterality(ctx) or _infer_laterality(text)
                if lat:
                    conf = 0.90 if op_note else 0.76
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Laterality", lat, text, r, section, note, conf))
                rtype, rclass = _infer_recon_type_and_class(text)
                if rtype:
                    conf = 0.88 if op_note else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Type", rtype, text, r, section, note, conf))
                if rclass:
                    conf = 0.88 if op_note else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Classification", rclass, text, r, section, note, conf))
                if op_note and MASTECTOMY_RX.search(text):
                    cands.append(_emit("Recon_Timing", "Immediate", text, r, section, note, 0.92))

        rr = RADIATION_RX.search(text)
        if rr:
            ctx = _window(text, rr.start(), rr.end(), 260)
            should_emit = False
            conf = 0.74
            if _strong_radiation_history(ctx):
                should_emit = True
                conf = 0.86 if clinic_like else 0.80
                if section_low_value and not op_note:
                    conf -= 0.10
            elif op_note:
                should_emit = False
            if should_emit:
                cands.append(_emit("Radiation", True, text, rr, section, note, conf))

        cc = CHEMO_RX.search(text)
        if cc:
            ctx = _window(text, cc.start(), cc.end(), 260)
            low_ctx = ctx.lower()
            should_emit = False
            conf = 0.74
            if ENDOCRINE_ONLY_RX.search(low_ctx) and not CHEMO_RX.search(low_ctx):
                should_emit = False
            elif _strong_chemo_history(ctx):
                should_emit = True
                conf = 0.86 if clinic_like else 0.80
                if section_low_value and not op_note:
                    conf -= 0.10
            elif op_note:
                should_emit = False
            if should_emit:
                cands.append(_emit("Chemo", True, text, cc, section, note, conf))
    return cands
