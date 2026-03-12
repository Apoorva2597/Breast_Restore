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

OPERATIVE_SECTION_HINTS = {
    "PROCEDURE",
    "PROCEDURES",
    "OPERATIVE FINDINGS",
    "OPERATIVE NOTE",
    "OPERATION",
    "BRIEF OP NOTE",
    "SURGICAL PROCEDURE",
    "HOSPITAL COURSE",
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
    r"sgap|igap|gap\s+flap|gluteal\s+artery\s+perforator|"
    r"implant\s+exchange|expander\s+exchange|exchange\s+of\s+tissue\s+expander|"
    r"placement\s+of\s+tissue\s+expander|placement\s+of\s+implant"
    r")\b",
    re.IGNORECASE,
)

SLNB_RX = re.compile(
    r"\b("
    r"sentinel\s+lymph\s+node\s+biopsy|"
    r"sentinel\s+node\s+biopsy|"
    r"\bSLNB\b|"
    r"lymphatic\s+mapping|"
    r"sentinel\s+lymphadenectomy|"
    r"sentinel\s+node\s+excision"
    r")\b",
    re.IGNORECASE,
)

ALND_RX = re.compile(
    r"\b("
    r"axillary\s+lymph\s+node\s+dissection|"
    r"axillary\s+dissection|"
    r"\bALND\b|"
    r"level\s+[i1l]+(?:\s*/\s*[i1l]+)?\s+axillary\s+dissection"
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

NEGATION_RX = re.compile(
    r"\b(no|denies|denied|without|not|never|none|negative\s+for|declined|deferred)\b",
    re.IGNORECASE,
)

PLANNED_RX = re.compile(
    r"\b(plan|planned|planning|will|scheduled|schedule|candidate|consider|recommend|discuss|discussion|possible|may\s+need|may\s+require)\b",
    re.IGNORECASE,
)

PROPHYLAXIS_RX = re.compile(
    r"\b(prophylactic|risk[- ]reducing|risk\s+reducing|preventive|contralateral\s+prophylactic|cpm)\b",
    re.IGNORECASE,
)

CANCER_RX = re.compile(
    r"\b("
    r"breast\s+cancer|carcinoma|malignancy|malignant|"
    r"invasive\s+ductal|invasive\s+lobular|"
    r"dcis|lcis|idc|ilc|recurrent\s+cancer|cancer"
    r")\b",
    re.IGNORECASE,
)

TREATMENT_RECEIVED_RX = re.compile(
    r"\b("
    r"s/p|status\s+post|history\s+of|hx\s+of|prior|previous|"
    r"completed|received|underwent|treated\s+with|"
    r"adjuvant|neoadjuvant|postmastectomy"
    r")\b",
    re.IGNORECASE,
)

STRONG_RADIATION_HISTORY_RX = re.compile(
    r"\b("
    r"s/p\s+radiation|status\s+post\s+radiation|"
    r"history\s+of\s+radiation|prior\s+radiation|previous\s+radiation|"
    r"completed\s+radiation|received\s+radiation|"
    r"adjuvant\s+radiation|neoadjuvant\s+radiation|"
    r"radiation\s+therapy\s+completed|postmastectomy\s+radiation"
    r")\b",
    re.IGNORECASE,
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
    re.IGNORECASE,
)

WEAK_TREATMENT_EXCLUDE_RX = re.compile(
    r"\b("
    r"consider|candidate|discussion|discussed|recommend|recommended|"
    r"plan|planned|planning|will\s+start|may\s+need|may\s+require|"
    r"referred\s+to\s+radiation\s+oncology|radiation\s+oncology\s+consult"
    r")\b",
    re.IGNORECASE,
)

HISTORY_ONLY_RX = re.compile(
    r"\b(history\s+of|hx\s+of|prior|previous|remote|past\s+history|past\s+surgical\s+history)\b",
    re.IGNORECASE,
)

PERFORMED_RX = re.compile(
    r"\b(performed|underwent|completed|done|was\s+done|was\s+performed|status\s+post|s/p)\b",
    re.IGNORECASE,
)

SENT_SPLIT_RX = re.compile(r"(?<=[\.\?\!\;])\s+|\n+")


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
    parts = SENT_SPLIT_RX.split(text)
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


def _chunk_sentence(text):
    parts = re.split(r"[,:;]|\band\b|\bwith\b", text, flags=re.IGNORECASE)
    out = []
    for p in parts:
        s = p.strip()
        if s:
            out.append(s)
    return out if out else [text]


def _contains_side(text, side):
    if side == "LEFT":
        return bool(LEFT_RX.search(text)) or bool(BILAT_RX.search(text))
    if side == "RIGHT":
        return bool(RIGHT_RX.search(text)) or bool(BILAT_RX.search(text))
    return False


def _same_side_chunk(chunk, side):
    if side == "LEFT":
        if LEFT_RX.search(chunk):
            return True
        if BILAT_RX.search(chunk) and not RIGHT_RX.search(chunk):
            return True
    elif side == "RIGHT":
        if RIGHT_RX.search(chunk):
            return True
        if BILAT_RX.search(chunk) and not LEFT_RX.search(chunk):
            return True
    return False


def _is_low_value_section(section):
    return section in LOW_VALUE_SECTIONS


def _is_operative_section(section):
    s = (section or "").upper()
    if s in OPERATIVE_SECTION_HINTS:
        return True
    for hint in OPERATIVE_SECTION_HINTS:
        if hint in s:
            return True
    return False


def _local_negated(text, start=None, end=None, width=80):
    if text is None:
        return False
    low = text.lower()
    if start is None or end is None:
        return bool(NEGATION_RX.search(low))
    lo = max(0, start - width)
    hi = min(len(low), end + 20)
    ctx = low[lo:hi]
    return bool(NEGATION_RX.search(ctx))


def _local_planned(text, start=None, end=None, width=80):
    if text is None:
        return False
    low = text.lower()
    if start is None or end is None:
        return bool(PLANNED_RX.search(low))
    lo = max(0, start - width)
    hi = min(len(low), end + 40)
    ctx = low[lo:hi]
    return bool(PLANNED_RX.search(ctx))


def _history_only_context(text, start=None, end=None, width=80):
    if text is None:
        return False
    low = text.lower()
    if start is None or end is None:
        return bool(HISTORY_ONLY_RX.search(low))
    lo = max(0, start - width)
    hi = min(len(low), end + 40)
    ctx = low[lo:hi]
    return bool(HISTORY_ONLY_RX.search(ctx))


def _looks_negated_or_planned(ctx, op_note):
    low = (ctx or "").lower()
    if NEGATION_RX.search(low):
        return True
    if PLANNED_RX.search(low) and not op_note:
        return True
    return False


def _indication_vote(sentence, chunk, side):
    low = chunk.lower()
    if not _contains_side(chunk, side):
        return None

    cancer_here = bool(CANCER_RX.search(low))
    prophy_here = bool(PROPHYLAXIS_RX.search(low))
    mast_here = bool(MASTECTOMY_RX.search(low))

    if prophy_here and mast_here:
        return "Prophylactic"
    if prophy_here and _contains_side(chunk, side):
        return "Prophylactic"
    if cancer_here:
        return "Therapeutic"
    return None


def _infer_indications_local(text, lat):
    low = (text or "").lower()
    left_votes = []
    right_votes = []

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    for sent in sentences:
        sent_low = sent.lower()
        if not MASTECTOMY_RX.search(sent_low) and not CANCER_RX.search(sent_low) and not PROPHYLAXIS_RX.search(sent_low):
            continue
        chunks = _chunk_sentence(sent)

        for chunk in chunks:
            chunk_low = chunk.lower()

            left_vote = _indication_vote(sent, chunk, "LEFT")
            right_vote = _indication_vote(sent, chunk, "RIGHT")

            if left_vote is not None:
                left_votes.append(left_vote)
            if right_vote is not None:
                right_votes.append(right_vote)

            if "contralateral prophylactic" in chunk_low:
                left_cancer = bool(re.search(r"(left|lt).{0,120}(cancer|carcinoma|dcis|lcis|idc|ilc|malignan|invasive|recurrent)", sent_low))
                right_cancer = bool(re.search(r"(right|rt).{0,120}(cancer|carcinoma|dcis|lcis|idc|ilc|malignan|invasive|recurrent)", sent_low))

                if left_cancer and not right_cancer:
                    left_votes.append("Therapeutic")
                    right_votes.append("Prophylactic")
                elif right_cancer and not left_cancer:
                    right_votes.append("Therapeutic")
                    left_votes.append("Prophylactic")

    # fallback only when one clear side is implied
    if lat == "LEFT" and not left_votes:
        if CANCER_RX.search(low) and not PROPHYLAXIS_RX.search(low):
            left_votes.append("Therapeutic")
        elif PROPHYLAXIS_RX.search(low):
            left_votes.append("Prophylactic")

    if lat == "RIGHT" and not right_votes:
        if CANCER_RX.search(low) and not PROPHYLAXIS_RX.search(low):
            right_votes.append("Therapeutic")
        elif PROPHYLAXIS_RX.search(low):
            right_votes.append("Prophylactic")

    def _resolve(votes):
        if not votes:
            return None
        if "Therapeutic" in votes:
            return "Therapeutic"
        if "Prophylactic" in votes:
            return "Prophylactic"
        return None

    return _resolve(left_votes), _resolve(right_votes)


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

    has_any_flap = (
        len(flap_types_found) > 0 or
        (" flap" in low) or
        low.startswith("flap") or
        ("mixed flaps" in low)
    )

    has_direct_to_implant = bool(re.search(r"\bdirect[- ]to[- ]implant\b", low))
    has_expander = (
        ("tissue expander" in low) or
        ("expander" in low) or
        ("expander exchange" in low) or
        ("placement of tissue expander" in low) or
        ("exchange of tissue expander" in low)
    )
    has_implant = (
        ("implant" in low) or
        ("implant exchange" in low) or
        ("placement of implant" in low)
    )

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


def _best_local_recon_type_and_class(text):
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    ranked_hits = []

    for sent in sentences:
        sent_low = sent.lower()
        if not RECON_RX.search(sent_low):
            continue

        if _local_negated(sent_low):
            continue
        if _local_planned(sent_low) and not MASTECTOMY_RX.search(sent_low):
            continue

        rtype, rclass = _infer_recon_type_and_class(sent)
        if rtype is None and rclass is None:
            continue

        rank = 5
        if re.search(r"\bdirect[- ]to[- ]implant\b", sent_low):
            rank = 1
        elif re.search(r"\b(diep|tram|siea|latissimus|sgap|igap|gap\s+flap|gluteal artery perforator)\b", sent_low):
            rank = 1
        elif re.search(r"\b(expander|implant exchange|expander exchange|tissue expander)\b", sent_low):
            rank = 2
        elif "reconstruction" in sent_low:
            rank = 3

        ranked_hits.append((rank, sent, rtype, rclass))

    if not ranked_hits:
        return _infer_recon_type_and_class(text)

    ranked_hits = sorted(ranked_hits, key=lambda x: x[0])
    _, _, rtype, rclass = ranked_hits[0]
    return rtype, rclass


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

        section_low_value = _is_low_value_section(section)
        operative_section = _is_operative_section(section)

        # -----------------------------
        # Mastectomy / indications / lymph node
        # -----------------------------
        m = MASTECTOMY_RX.search(text)
        if m:
            ctx = _window(text, m.start(), m.end(), 220)
            local_bad = _looks_negated_or_planned(ctx, op_note)

            if not local_bad:
                lat = _infer_laterality(ctx) or _infer_laterality(text)

                if lat:
                    conf = 0.90 if (op_note or operative_section) else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Mastectomy_Laterality", lat, text, m, section, note, conf))

                if _clean(note.note_date):
                    conf = 0.88 if (op_note or operative_section) else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Mastectomy_Date", _clean(note.note_date), text, m, section, note, conf))

                # use local context, not full section-only logic
                local_text = ctx
                left_ind, right_ind = _infer_indications_local(local_text, lat)

                if left_ind is not None:
                    conf = 0.84 if (op_note or operative_section) else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Left", left_ind, text, m, section, note, conf))

                if right_ind is not None:
                    conf = 0.84 if (op_note or operative_section) else 0.68
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Indication_Right", right_ind, text, m, section, note, conf))

        # lymph node should not depend on mastectomy match only
        best_ln_match = None
        best_ln_label = None
        best_ln_conf = None

        for rx, label, base_conf in [
            (ALND_RX, "ALND", 0.88),
            (SLNB_RX, "SLNB", 0.84),
        ]:
            for mm in rx.finditer(text):
                ln_ctx = _window(text, mm.start(), mm.end(), 120)
                ln_ctx_low = ln_ctx.lower()

                if _local_negated(text, mm.start(), mm.end()):
                    continue
                if _local_planned(text, mm.start(), mm.end()) and not (op_note or operative_section):
                    continue
                if _history_only_context(text, mm.start(), mm.end()) and not (op_note or operative_section):
                    continue

                conf = base_conf if (op_note or operative_section) else (base_conf - 0.12)
                if section_low_value and not op_note:
                    conf -= 0.10
                if PERFORMED_RX.search(ln_ctx_low):
                    conf += 0.02

                if best_ln_match is None:
                    best_ln_match = mm
                    best_ln_label = label
                    best_ln_conf = conf
                else:
                    # ALND outranks SLNB
                    if label == "ALND" and best_ln_label != "ALND":
                        best_ln_match = mm
                        best_ln_label = label
                        best_ln_conf = conf
                    elif label == best_ln_label and conf > best_ln_conf:
                        best_ln_match = mm
                        best_ln_label = label
                        best_ln_conf = conf

        if best_ln_match is not None:
            cands.append(_emit("LymphNode", best_ln_label, text, best_ln_match, section, note, best_ln_conf))

        # -----------------------------
        # Reconstruction
        # -----------------------------
        r = RECON_RX.search(text)
        if r:
            ctx = _window(text, r.start(), r.end(), 220)

            if not _looks_negated_or_planned(ctx, op_note):
                lat = _infer_laterality(ctx) or _infer_laterality(text)

                if lat:
                    conf = 0.90 if (op_note or operative_section) else 0.76
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Laterality", lat, text, r, section, note, conf))

                local_recon_text = ctx if (op_note or operative_section) else text
                rtype, rclass = _best_local_recon_type_and_class(local_recon_text)

                if rtype:
                    conf = 0.90 if (op_note or operative_section) else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Type", rtype, text, r, section, note, conf))

                if rclass:
                    conf = 0.90 if (op_note or operative_section) else 0.74
                    if section_low_value and not op_note:
                        conf -= 0.10
                    cands.append(_emit("Recon_Classification", rclass, text, r, section, note, conf))

                if (op_note or operative_section) and MASTECTOMY_RX.search(text):
                    cands.append(_emit("Recon_Timing", "Immediate", text, r, section, note, 0.92))

        # -----------------------------
        # Radiation
        # -----------------------------
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

        # -----------------------------
        # Chemo
        # -----------------------------
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
