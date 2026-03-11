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

LEFT_WORDS = r"(left|lt)"
RIGHT_WORDS = r"(right|rt)"

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
    r"diep|tram|siea|latissimus|flap|"
    r"tissue\s+expander|expander|"
    r"implant|alloderm|acellular\s+dermal\s+matrix|"
    r"direct[- ]to[- ]implant|"
    r"sgap|igap|gap\s+flap|gluteal\s+artery\s+perforator"
    r")\b",
    re.IGNORECASE
)

# -----------------------------
# LN patterns
# -----------------------------
ALND_RX = re.compile(
    r"\b("
    r"axillary\s+lymph\s+node\s+dissection|"
    r"axillary\s+node\s+dissection|"
    r"axillary\s+dissection|"
    r"axillary\s+clearance|"
    r"completion\s+axillary\s+dissection|"
    r"completion\s+axillary\s+lymph\s+node\s+dissection|"
    r"completion\s+node\s+dissection|"
    r"complete\s+axillary\s+dissection|"
    r"completion\s+alnd|"
    r"level\s*i/?ii\s+dissection|"
    r"level\s*1/?2\s+dissection|"
    r"non[- ]sentinel\s+axillary\s+dissection|"
    r"full\s+axillary\s+dissection|"
    r"axillary\s+contents\s+removed|"
    r"\bALND\b"
    r")\b",
    re.IGNORECASE
)

SLNB_RX = re.compile(
    r"\b("
    r"sentinel\s+lymph\s+node\s+biopsy|"
    r"sentinel\s+node\s+biopsy|"
    r"sentinel\s+lymph\s+node\s+excision|"
    r"sentinel\s+node\s+excision|"
    r"sentinel\s+lymphadenectomy|"
    r"lymphatic\s+mapping|"
    r"sentinel\s+node\s+mapping|"
    r"sentinel\s+mapping|"
    r"sentinel\s+nodes?\s+removed|"
    r"sentinel\s+node\s+removed|"
    r"sln\s+biopsy|"
    r"sln\s+bx|"
    r"sentinel\s+node\s+bx|"
    r"sentinel\s+lymph\s+node\s+bx|"
    r"hot\s+node\s+removed|"
    r"blue\s+node|"
    r"blue\s+dye\s+mapping|"
    r"radioisotope\s+mapping|"
    r"\bSLNB\b|"
    r"\bSLN\b"
    r")\b",
    re.IGNORECASE
)

LN_DONE_RX = re.compile(
    r"\b("
    r"s/p|status\s+post|underwent|completed|received|"
    r"done|performed|had|has\s+had|"
    r"was\s+performed|"
    r"has\s+completed"
    r")\b",
    re.IGNORECASE
)

LN_HISTORY_RX = re.compile(
    r"\b("
    r"hx\s+of|history\s+of|prior|previous|"
    r"s/p|status\s+post|"
    r"had|has\s+had|underwent|completed|received|"
    r"done|performed|"
    r"nodes?\s+removed|"
    r"sentinel\s+nodes?\s+removed|"
    r"axillary\s+dissection\s+performed|"
    r"axillary\s+clearance\s+performed"
    r")\b",
    re.IGNORECASE
)

LN_PLAN_RX = re.compile(
    r"\b("
    r"plan|planned|planning|"
    r"possible|possibly|"
    r"consider|considered|candidate|"
    r"may\s+need|might\s+need|"
    r"if\s+positive|if\s+needed|"
    r"potential|"
    r"could\s+require|would\s+require|"
    r"pending|depending\s+on|awaiting|"
    r"recommend|recommended|discussed|"
    r"would\s+do|will\s+do|"
    r"in\s+the\s+future"
    r")\b",
    re.IGNORECASE
)

LN_FAIL_MAP_RX = re.compile(
    r"\b("
    r"failed\s+to\s+map|"
    r"mapping\s+failed|"
    r"unable\s+to\s+map|"
    r"no\s+sentinel\s+node\s+identified|"
    r"no\s+sentinel\s+nodes?\s+identified|"
    r"initial\s+attempt\s+at\s+sentinel\s+lymph\s+node\s+biopsy\s+was\s+apparently\s+unsuccessful|"
    r"unsuccessful\s+sentinel|"
    r"non[- ]sentinel\s+axillary\s+dissection\s+was\s+performed"
    r")\b",
    re.IGNORECASE
)

LN_SCAR_SITE_RX = re.compile(
    r"\b("
    r"scar|"
    r"site|"
    r"biopsy\s+scar|"
    r"slnb\s+scar|"
    r"sentinel\s+lymph\s+node\s+biopsy\s+scar|"
    r"tethering|"
    r"seroma|"
    r"axillary\s+dissection\s+site|"
    r"at\s+the\s+site\s+of|"
    r"tenderness|"
    r"deforming\s+the\s+breast\s+contour"
    r")\b",
    re.IGNORECASE
)

LN_REMOTE_HISTORY_RX = re.compile(
    r"\b("
    r"history\s+dates\s+back|"
    r"treated\s+in\s+\d{4}|"
    r"diagnosed\s+in\s+\d{4}|"
    r"back\s+to\s+\d{4}|"
    r"years\s+ago|"
    r"remote\s+past|"
    r"past\s+surgical\s+history|"
    r"past\s+medical\s+history"
    r")\b",
    re.IGNORECASE
)

LN_CURRENT_PROCEDURE_CUE_RX = re.compile(
    r"\b("
    r"procedure|procedures|operation|operative|surgery|surgical|"
    r"performed|we\s+performed|"
    r"intraoperative|preoperative|postoperative|"
    r"mastectomy\s+with|"
    r"operation\s+performed|"
    r"intraop|"
    r"date\s+of\s+surgery|"
    r"date\s+of\s+service"
    r")\b",
    re.IGNORECASE
)

# -----------------------------
# Other treatment patterns
# -----------------------------
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
    r"\bTCHP?\b|\bAC\b|\bTC\b|\bACT\b"
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
    r"no|denies|denied|without|not|never|negative\s+for"
    r")\b",
    re.IGNORECASE
)

PLANNED_RX = re.compile(
    r"\b("
    r"plan|planned|planning|will|scheduled|schedule|candidate|consider|recommend|discuss|"
    r"possible|possibly|may\s+need|might\s+need|if\s+positive|if\s+needed|potential"
    r")\b",
    re.IGNORECASE
)

PROPHYLAXIS_RX = re.compile(
    r"\b("
    r"prophylactic|risk[- ]reducing|risk\s+reducing|preventive|"
    r"contralateral\s+prophylactic|"
    r"\bCPM\b"
    r")\b",
    re.IGNORECASE
)

CANCER_RX = re.compile(
    r"\b("
    r"breast\s+cancer|carcinoma|malignancy|malignant|"
    r"invasive\s+ductal|invasive\s+lobular|dcis|lcis|"
    r"idc|ilc|recurrent\s+cancer|cancer"
    r")\b",
    re.IGNORECASE
)

TREATMENT_RECEIVED_RX = re.compile(
    r"\b("
    r"s/p|status\s+post|history\s+of|hx\s+of|prior|previous|"
    r"completed|received|underwent|treated\s+with|"
    r"adjuvant|neoadjuvant|postmastectomy"
    r")\b",
    re.IGNORECASE
)

STRONG_RADIATION_HISTORY_RX = re.compile(
    r"\b("
    r"s/p\s+radiation|status\s+post\s+radiation|"
    r"history\s+of\s+radiation|prior\s+radiation|previous\s+radiation|"
    r"completed\s+radiation|received\s+radiation|"
    r"adjuvant\s+radiation|neoadjuvant\s+radiation|"
    r"radiation\s+therapy\s+completed|postmastectomy\s+radiation"
    r")\b",
    re.IGNORECASE
)

STRONG_CHEMO_HISTORY_RX = re.compile(
    r"\b("
    r"s/p\s+chemo|status\s+post\s+chemo|"
    r"history\s+of\s+chemo|prior\s+chemo|previous\s+chemo|"
    r"completed\s+chemo|completed\s+chemotherapy|"
    r"received\s+chemo|received\s+chemotherapy|"
    r"adjuvant\s+chemo|neoadjuvant\s+chemo|"
    r"treated\s+with\s+chemotherapy|treated\s+with\s+chemo"
    r")\b",
    re.IGNORECASE
)

WEAK_TREATMENT_EXCLUDE_RX = re.compile(
    r"\b("
    r"consider|candidate|discussion|discussed|recommend|recommended|"
    r"plan|planned|planning|will\s+start|may\s+need|may\s+require|"
    r"referred\s+to\s+radiation\s+oncology|radiation\s+oncology\s+consult"
    r")\b",
    re.IGNORECASE
)

REVISION_RX = re.compile(
    r"\b("
    r"revision|scar\s+revision|capsulotomy|capsulectomy|fat\s+graft|fat\s+grafting|"
    r"lipofilling|liposuction|nipple\s+reconstruction|nipple[- ]areolar|tattoo|"
    r"symmetry|symmetrization|dog\s+ear|exchange\s+of\s+implant|implant\s+exchange|"
    r"removal\s+of\s+intact\s+silicone|removal\s+of\s+implant|capsulorrhaphy|"
    r"reposition|mastopexy|adjacent\s+tissue\s+transfer"
    r")\b",
    re.IGNORECASE
)

ANCHOR_RECON_RX = re.compile(
    r"\b("
    r"tissue\s+expander\s+placement|expander\s+placement|"
    r"implant\s+placement|implant[- ]based\s+reconstruction|"
    r"direct[- ]to[- ]implant|"
    r"diep\s+flap|tram\s+flap|siea\s+flap|latissimus\s+dorsi\s+flap|"
    r"free\s+flap|autologous\s+reconstruction|"
    r"immediate\s+reconstruction|delayed\s+reconstruction|"
    r"breast\s+reconstruction"
    r")\b",
    re.IGNORECASE
)

SIDE_CANCER_RX = re.compile(
    r"\b("
    r"left\s+breast\s+cancer|right\s+breast\s+cancer|"
    r"left\s+dcis|right\s+dcis|"
    r"left\s+idc|right\s+idc|"
    r"left\s+ilc|right\s+ilc|"
    r"cancer\s+on\s+the\s+left|cancer\s+on\s+the\s+right|"
    r"left[- ]sided\s+breast\s+cancer|right[- ]sided\s+breast\s+cancer"
    r")\b",
    re.IGNORECASE
)


def _is_operation_note(note_type):
    s = (note_type or "").lower()
    return (
        ("brief op" in s) or
        ("op note" in s) or
        ("operative" in s) or
        ("operation" in s) or
        ("oper report" in s)
    )


def _is_clinic_like(note_type):
    s = (note_type or "").lower()
    pats = [
        "clinic", "progress", "office", "follow up", "follow-up",
        "consult", "pre-op", "preop", "history and physical", "h&p",
        "oncology"
    ]
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


def _split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!\;])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


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


def _has_side_specific_cancer(text, side):
    low = (text or "").lower()
    if side == "LEFT":
        side_pat = LEFT_WORDS
    else:
        side_pat = RIGHT_WORDS

    cancer_terms = r"(breast\s+cancer|carcinoma|dcis|lcis|idc|ilc|malignan|invasive|recurrent)"
    p1 = re.search(side_pat + r".{0,90}" + cancer_terms, low)
    p2 = re.search(cancer_terms + r".{0,90}" + side_pat, low)
    return bool(p1 or p2)


def _has_side_specific_prophylaxis(text, side):
    low = (text or "").lower()
    if side == "LEFT":
        side_pat = LEFT_WORDS
    else:
        side_pat = RIGHT_WORDS
    pro_terms = r"(prophylactic|risk[- ]reducing|risk\s+reducing|preventive|cpm|contralateral\s+prophylactic)"
    p1 = re.search(side_pat + r".{0,70}" + pro_terms, low)
    p2 = re.search(pro_terms + r".{0,70}" + side_pat, low)
    return bool(p1 or p2)


def _paired_side_indications(text):
    low = (text or "").lower()

    left_val = None
    right_val = None

    patterns = [
        (
            re.compile(
                r"right.{0,80}(simple\s+mastectomy|total\s+mastectomy|mastectomy).{0,80}"
                r"left.{0,60}(prophylactic|risk[- ]reducing|preventive)",
                re.IGNORECASE
            ),
            ("Therapeutic", "Prophylactic")
        ),
        (
            re.compile(
                r"left.{0,80}(simple\s+mastectomy|total\s+mastectomy|mastectomy).{0,80}"
                r"right.{0,60}(prophylactic|risk[- ]reducing|preventive)",
                re.IGNORECASE
            ),
            ("Prophylactic", "Therapeutic")
        ),
        (
            re.compile(
                r"left.{0,60}(prophylactic|risk[- ]reducing|preventive).{0,80}"
                r"right.{0,80}(simple\s+mastectomy|total\s+mastectomy|mastectomy)",
                re.IGNORECASE
            ),
            ("Prophylactic", "Therapeutic")
        ),
        (
            re.compile(
                r"right.{0,60}(prophylactic|risk[- ]reducing|preventive).{0,80}"
                r"left.{0,80}(simple\s+mastectomy|total\s+mastectomy|mastectomy)",
                re.IGNORECASE
            ),
            ("Therapeutic", "Prophylactic")
        ),
    ]

    for rx, vals in patterns:
        if rx.search(low):
            left_val, right_val = vals
            break

    return left_val, right_val


def _infer_indications(text, lat, op_note=False):
    low = (text or "").lower()

    left_val = None
    right_val = None

    pair_left, pair_right = _paired_side_indications(text)
    if pair_left is not None:
        left_val = pair_left
    if pair_right is not None:
        right_val = pair_right

    left_cancer = _has_side_specific_cancer(text, "LEFT")
    right_cancer = _has_side_specific_cancer(text, "RIGHT")
    left_pro = _has_side_specific_prophylaxis(text, "LEFT")
    right_pro = _has_side_specific_prophylaxis(text, "RIGHT")

    if left_val is None:
        if left_pro:
            left_val = "Prophylactic"
        elif left_cancer:
            left_val = "Therapeutic"

    if right_val is None:
        if right_pro:
            right_val = "Prophylactic"
        elif right_cancer:
            right_val = "Therapeutic"

    # contralateral prophylactic inference only when the therapeutic side is explicit
    if "contralateral prophylactic" in low:
        if left_cancer and right_val is None:
            right_val = "Prophylactic"
            if left_val is None:
                left_val = "Therapeutic"
        elif right_cancer and left_val is None:
            left_val = "Prophylactic"
            if right_val is None:
                right_val = "Therapeutic"

    # only allow single-side default when note itself is single-sided and cancer is side-specific
    if lat == "LEFT" and left_val is None and left_cancer and not left_pro:
        left_val = "Therapeutic"
    if lat == "RIGHT" and right_val is None and right_cancer and not right_pro:
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


def _infer_side_from_local_ctx(text, match_obj):
    if match_obj is None:
        return _infer_laterality(text)
    ctx = _window(text, match_obj.start(), match_obj.end(), 120)
    return _infer_laterality(ctx) or _infer_laterality(text)


def _lymphnode_value_from_text(text, op_note, clinic_like):
    low = (text or "").lower()

    if LN_SCAR_SITE_RX.search(low):
        return None, None
    if LN_PLAN_RX.search(low) and not LN_DONE_RX.search(low) and not LN_HISTORY_RX.search(low) and not op_note:
        return None, None

    if LN_FAIL_MAP_RX.search(low) and ALND_RX.search(low):
        mm = ALND_RX.search(text)
        return "ALND", mm

    alnd_match = ALND_RX.search(text)
    if alnd_match:
        ctx = _window(low, alnd_match.start(), alnd_match.end(), 180)
        if not _looks_negated_or_planned(ctx, op_note):
            if clinic_like:
                if LN_DONE_RX.search(ctx) or LN_HISTORY_RX.search(ctx):
                    return "ALND", alnd_match
            if op_note and (LN_CURRENT_PROCEDURE_CUE_RX.search(ctx) or LN_DONE_RX.search(ctx) or LN_FAIL_MAP_RX.search(ctx)):
                return "ALND", alnd_match

    slnb_match = SLNB_RX.search(text)
    if slnb_match:
        ctx = _window(low, slnb_match.start(), slnb_match.end(), 180)
        if not _looks_negated_or_planned(ctx, op_note):
            if clinic_like:
                if LN_DONE_RX.search(ctx) or LN_HISTORY_RX.search(ctx):
                    return "SLNB", slnb_match
            if op_note and (LN_CURRENT_PROCEDURE_CUE_RX.search(ctx) or LN_DONE_RX.search(ctx)):
                return "SLNB", slnb_match

    return None, None


def _infer_recon_type_and_class(text):
    low = (text or "").lower()

    flap_types_found = []

    if "diep" in low:
        flap_types_found.append("DIEP")
    if "tram" in low:
        flap_types_found.append("TRAM")
    if "siea" in low:
        flap_types_found.append("SIEA")
    if (
        "gluteal artery perforator" in low or
        "gap flap" in low or
        re.search(r"\bsgap\b", low) or
        re.search(r"\bigap\b", low)
    ):
        flap_types_found.append("gluteal artery perforator flap")
    if "latissimus" in low:
        flap_types_found.append("latissimus dorsi")

    has_any_flap = (
        len(flap_types_found) > 0 or
        (" flap" in low) or
        low.startswith("flap") or
        ("mixed flaps" in low)
    )

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


def _is_revision_only_recon_context(text):
    low = (text or "").lower()
    if REVISION_RX.search(low) and not ANCHOR_RECON_RX.search(low):
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

        section_upper = (section or "").upper()
        section_low_value = section_upper in LOW_VALUE_SECTIONS

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

                left_ind, right_ind = _infer_indications(text, lat, op_note=op_note)
                if left_ind is not None:
                    conf = 0.84 if op_note else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Left", left_ind, text, m, section, note, conf))
                if right_ind is not None:
                    conf = 0.84 if op_note else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Right", right_ind, text, m, section, note, conf))

                lymph_value, lymph_match = _lymphnode_value_from_text(text, op_note, clinic_like)
                if lymph_value == "ALND":
                    conf = 0.88 if clinic_like else 0.80
                    if op_note:
                        conf = 0.83
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("LymphNode", "ALND", text, lymph_match, section, note, conf))
                elif lymph_value == "SLNB":
                    conf = 0.86 if clinic_like else 0.78
                    if op_note:
                        conf = 0.81
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("LymphNode", "SLNB", text, lymph_match, section, note, conf))

        r = RECON_RX.search(text)
        if r:
            ctx = _window(text, r.start(), r.end(), 220)

            if not _looks_negated_or_planned(ctx, op_note):
                if not _is_revision_only_recon_context(text):
                    lat = _infer_laterality(ctx) or _infer_laterality(text)
                    if lat:
                        conf = 0.90 if op_note else 0.74
                        if section_low_value and not op_note:
                            conf -= 0.10
                        cands.append(_emit("Recon_Laterality", lat, text, r, section, note, conf))

                    rtype, rclass = _infer_recon_type_and_class(text)
                    if rtype:
                        conf = 0.88 if op_note else 0.70
                        if section_low_value and not op_note:
                            conf -= 0.10
                        cands.append(_emit("Recon_Type", rtype, text, r, section, note, conf))
                    if rclass:
                        conf = 0.88 if op_note else 0.70
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
