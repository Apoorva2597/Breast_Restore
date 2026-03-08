# extractors/bmi.py

import re
from models import Candidate

# ----------------------------------------------
# UPDATE:
# Rewritten BMI extractor for reconstruction workflow.
#
# Key changes:
# - Scans the whole note and collects ALL valid BMI mentions.
# - Keeps explicit measured BMI values from operative / peri-op notes.
# - Rejects threshold / policy / eligibility statements such as:
#     "BMI >= 35"
#     "BMI greater than or equal to 35"
#     "patients with BMI 30-35"
# - Supports common real note patterns:
#     "BMI 37.59"
#     "BMI: 37.59"
#     "BMI=37.59"
#     "BMI is 37.59"
#     "BMI was 37.59"
#     "BMI of 37.59"
#     "body mass index is 23.72 kg/m2"
#     "Obesity, BMI 35.9"
#     "(BMI 37.28)"
# - Returns multiple candidates per note so build/ranking logic
#   can choose the best reconstruction-linked value.
# - NEW FALLBACK:
#   If explicit BMI is absent, compute BMI from height + weight found
#   in the same note.
#
# Python 3.6.8 compatible.
# ----------------------------------------------

# -----------------------
# Explicit BMI patterns
# -----------------------
BMI_PATTERNS = [
    re.compile(
        r"\bBMI\s*(?:[:=]|\bis\b|\bwas\b|\bof\b)?\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bbody\s+mass\s+index\s*(?:[:=]|\bis\b|\bwas\b|\bof\b)?\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bobesity\s*,?\s*BMI\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\bmorbid\s+obesity\s*,?\s*BMI\s*\(?\s*(\d{2,3}(?:\.\d+)?)\s*\)?\b",
        re.IGNORECASE
    ),
]

# -----------------------
# Threshold / non-measured BMI contexts to reject
# -----------------------
THRESHOLD_FALSE_POS = re.compile(
    r"(?:"
    r"\bBMI\s*(?:>=|=>|>|<=|=<|<)\s*\d+(?:\.\d+)?"
    r"|\bBMI\s*(?:greater|less)\s+than\b"
    r"|\bBMI\s*(?:greater|less)\s+than\s+or\s+equal\s+to\b"
    r"|\bBMI\s*(?:over|under|above|below)\b"
    r"|\bBMI\s*\d+(?:\.\d+)?\s*(?:to|\-)\s*\d+(?:\.\d+)?\b"
    r"|\bminimum\s+BMI\b"
    r"|\bmaximum\s+BMI\b"
    r"|\btarget\s+BMI\b"
    r"|\bgoal\s+BMI\b"
    r"|\bacceptable\s+BMI\b"
    r"|\brequired\s+BMI\b"
    r"|\beligibility\b"
    r"|\bcriteria\b"
    r"|\bqualif(?:y|ies|ied|ication)\b"
    r")",
    re.IGNORECASE
)

CONDITIONAL_FALSE_POS = re.compile(
    r"(?:"
    r"\bif\s+BMI\b"
    r"|\bwhen\s+BMI\b"
    r"|\bunless\s+BMI\b"
    r"|\bfor\s+BMI\s*(?:>=|=>|>|<=|=<|<)\b"
    r"|\bpatients?\s+with\s+BMI\b"
    r"|\bpts?\s+with\s+BMI\b"
    r"|\bfor\s+patients?\s+with\s+BMI\b"
    r"|\bdue\s+to\s+obesity\s+and\s+BMI\s+(?:>=|=>|>|<=|=<|<)"
    r")",
    re.IGNORECASE
)

# -----------------------
# Height / Weight fallback patterns
# -----------------------

# Height in meters: Ht 1.753 m / Height 1.626 m
HEIGHT_M_PATTERNS = [
    re.compile(r"\bHt\s*(?:is|=|:)?\s*(\d(?:\.\d+)?)\s*m\b", re.IGNORECASE),
    re.compile(r"\bHeight\s*(?:is|=|:)?\s*(\d(?:\.\d+)?)\s*m\b", re.IGNORECASE),
    re.compile(r"\bheight\s*(?:is|=|:)?\s*(\d(?:\.\d+)?)\s*m\b", re.IGNORECASE),
]

# Height in centimeters: Height 168 cm / Ht 170 cm
HEIGHT_CM_PATTERNS = [
    re.compile(r"\bHt\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", re.IGNORECASE),
    re.compile(r"\bHeight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", re.IGNORECASE),
    re.compile(r"\bheight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*cm\b", re.IGNORECASE),
]

# Height in feet/inches:
# 5' 8"
# 5'8"
# 5 ft 8 in
# 5 ft 8
HEIGHT_FT_IN_PATTERNS = [
    # 5'1 or 5' 1 or 5'1"
    re.compile(r"\b(\d)\s*'\s*(\d{1,2})\s*(?:\"|in\b|inches\b)?", re.IGNORECASE),

    # 5 ft 1 in / 5 ft 1 inch / 5 ft 1 inches
    re.compile(r"\b(\d)\s*ft\.?\s*(\d{1,2})\s*(?:in|inch|inches|\" )?\b", re.IGNORECASE),

    # 5 feet 1 inch / 5 feet 1 inches
    re.compile(r"\b(\d)\s*feet\s*(\d{1,2})\s*(?:in|inch|inches)?\b", re.IGNORECASE),

    # height is 5 feet 1 inch
    re.compile(r"\bheight\s*(?:is|=|:)?\s*(\d)\s*feet\s*(\d{1,2})\s*(?:in|inch|inches)?\b", re.IGNORECASE),

    # height is 5 ft 1 in
    re.compile(r"\bheight\s*(?:is|=|:)?\s*(\d)\s*ft\.?\s*(\d{1,2})\s*(?:in|inch|inches)?\b", re.IGNORECASE),
]

# Weight in kg: Wt 73.2 kg / Weight 130 kg
WEIGHT_KG_PATTERNS = [
    re.compile(r"\bWt\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*kg\b", re.IGNORECASE),
    re.compile(r"\bWeight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*kg\b", re.IGNORECASE),
    re.compile(r"\bweight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*kg\b", re.IGNORECASE),
]

# Weight in lb:
# Wt 161 lb
# Weight 220 lb
# Weight: (!) 130 kg handled by kg regex above, so okay
WEIGHT_LB_PATTERNS = [
    # Wt 139 lb / Wt: 139 lbs
    re.compile(r"\bWt\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", re.IGNORECASE),

    # Weight 139 lb / Weight: 139 lbs / Weight is 139 pounds
    re.compile(r"\bWeight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", re.IGNORECASE),

    # generic lower-case phrasing often seen in prose
    re.compile(r"\bweight\s*(?:is|=|:)?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds)\b", re.IGNORECASE),
]

PREFERRED_SECTIONS = set([
    "FULL",
    "OPERATIVE NOTE",
    "BRIEF OPERATIVE NOTE",
    "OPERATIVE FINDINGS",
    "PROCEDURE",
    "PROCEDURES",
    "PREOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DIAGNOSIS",
    "INDICATIONS FOR PROCEDURE",
    "INDICATIONS",
    "CODING NOTE",
    "MICROSURGICAL DETAILS",
    "HISTORY",
    "HISTORY OF PRESENT ILLNESS",
    "HPI",
    "VITALS",
    "PHYSICAL EXAM",
])

SUPPRESS_SECTIONS = set([
    "PLAN",
    "ASSESSMENT",
    "INSTRUCTIONS",
    "DISPOSITION",
])

def _normalize_text(text):
    text = str(text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _section_order(note):
    order = []
    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            order.append(s)
    for s in note.sections.keys():
        if s not in order and s not in SUPPRESS_SECTIONS:
            order.append(s)
    return order

def _confidence_for_context(note_type, section, evidence, source_kind):
    nt = str(note_type or "").lower().strip()
    sec = str(section or "").upper().strip()
    ev = str(evidence or "").lower()

    if source_kind == "computed":
        score = 0.88
    else:
        score = 0.90

    if (
        "brief op" in nt or
        "operative" in nt or
        "operation" in nt or
        "op note" in nt
    ):
        score += 0.06
    elif (
        "anesthesia" in nt or
        "pre-op" in nt or
        "preop" in nt
    ):
        score += 0.04
    elif (
        "progress" in nt or
        "clinic" in nt or
        "office" in nt or
        "consult" in nt or
        "h&p" in nt
    ):
        score += 0.02

    if sec in {
        "VITALS",
        "PHYSICAL EXAM",
        "HPI",
        "HISTORY OF PRESENT ILLNESS",
        "PREOPERATIVE DIAGNOSIS",
        "POSTOPERATIVE DIAGNOSIS",
        "INDICATIONS FOR PROCEDURE",
        "INDICATIONS",
        "OPERATIVE NOTE",
        "BRIEF OPERATIVE NOTE",
    }:
        score += 0.01

    if source_kind == "explicit":
        if "kg/m2" in ev or "kg/m^2" in ev or "kg/m²" in ev:
            score += 0.01

    if score > 0.99:
        score = 0.99

    return score

def window_around(text, start, end, width):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].strip()

def _candidate_sort_key(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    evid = str(getattr(c, "evidence", "") or "")
    source_kind = str(getattr(c, "status", "") or "")
    explicit_bonus = 0 if source_kind == "measured" else 1
    return (explicit_bonus, -conf, -len(evid))

def _find_all_height_candidates(text):
    vals_m = []

    for rx in HEIGHT_M_PATTERNS:
        for m in rx.finditer(text):
            try:
                h_m = float(m.group(1))
            except Exception:
                continue
            if h_m >= 1.0 and h_m <= 2.5:
                vals_m.append((h_m, m.start(), m.end()))

    for rx in HEIGHT_CM_PATTERNS:
        for m in rx.finditer(text):
            try:
                h_cm = float(m.group(1))
            except Exception:
                continue
            if h_cm >= 100 and h_cm <= 250:
                vals_m.append((h_cm / 100.0, m.start(), m.end()))

    for rx in HEIGHT_FT_IN_PATTERNS:
        for m in rx.finditer(text):
            try:
                ft = float(m.group(1))
                inch = float(m.group(2))
            except Exception:
                continue
            total_inches = (ft * 12.0) + inch
            if total_inches >= 48 and total_inches <= 90:
                vals_m.append((total_inches * 0.0254, m.start(), m.end()))

    return vals_m

def _find_all_weight_candidates(text):
    vals_kg = []

    for rx in WEIGHT_KG_PATTERNS:
        for m in rx.finditer(text):
            try:
                w_kg = float(m.group(1))
            except Exception:
                continue
            if w_kg >= 25 and w_kg <= 350:
                vals_kg.append((w_kg, m.start(), m.end()))

    for rx in WEIGHT_LB_PATTERNS:
        for m in rx.finditer(text):
            try:
                w_lb = float(m.group(1))
            except Exception:
                continue
            if w_lb >= 55 and w_lb <= 800:
                vals_kg.append((w_lb * 0.45359237, m.start(), m.end()))

    return vals_kg

def _pair_height_weight(height_candidates, weight_candidates):
    """
    Pair nearest height and weight mentions in same note section.
    """
    pairs = []
    for h_val, h_start, h_end in height_candidates:
        best = None
        best_dist = None
        for w_val, w_start, w_end in weight_candidates:
            dist = abs(h_start - w_start)
            if best is None or dist < best_dist:
                best = (w_val, w_start, w_end)
                best_dist = dist
        if best is not None:
            pairs.append((h_val, h_start, h_end, best[0], best[1], best[2], best_dist))
    return pairs

def _has_explicit_bmi_in_text(text):
    for rx in BMI_PATTERNS:
        if rx.search(text):
            return True
    return False

def extract_bmi(note):
    """
    Extract measured BMI values from a note.
    Priority:
    1) Explicit BMI mentions
    2) If none found in the note, compute BMI from height + weight
    """
    candidates = []
    seen = set()
    explicit_found_anywhere = False
    section_order = _section_order(note)

    # -----------------------
    # Pass 1: explicit BMI
    # -----------------------
    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)
        if not text:
            continue

        for rx in BMI_PATTERNS:
            for m in rx.finditer(text):
                raw_val = m.group(1)

                try:
                    bmi_val = float(raw_val)
                except Exception:
                    continue

                if bmi_val < 10 or bmi_val > 80:
                    continue

                ctx = window_around(text, m.start(), m.end(), 160)
                ctx_low = ctx.lower()

                if THRESHOLD_FALSE_POS.search(ctx):
                    continue

                if CONDITIONAL_FALSE_POS.search(ctx):
                    allow_measured = (
                        ("bmi is " in ctx_low) or
                        ("bmi was " in ctx_low) or
                        ("bmi of " in ctx_low) or
                        ("body mass index is " in ctx_low) or
                        ("body mass index was " in ctx_low) or
                        ("body mass index of " in ctx_low)
                    )
                    if not allow_measured:
                        continue

                bmi_val = round(bmi_val, 1)
                explicit_found_anywhere = True

                key = "{0}|{1}|{2}|{3}|{4}|explicit".format(
                    bmi_val,
                    section,
                    note.note_id,
                    note.note_date,
                    ctx
                )
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(
                    Candidate(
                        field="BMI",
                        value=bmi_val,
                        status="measured",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=_confidence_for_context(note.note_type, section, ctx, "explicit"),
                    )
                )

    if explicit_found_anywhere:
        candidates = sorted(candidates, key=_candidate_sort_key)
        return candidates

    # -----------------------
    # Pass 2: compute from height + weight fallback
    # Only if NO explicit BMI found anywhere in note
    # -----------------------
    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)
        if not text:
            continue

        # extra safety
        if _has_explicit_bmi_in_text(text):
            continue

        height_candidates = _find_all_height_candidates(text)
        weight_candidates = _find_all_weight_candidates(text)

        if not height_candidates or not weight_candidates:
            continue

        pairs = _pair_height_weight(height_candidates, weight_candidates)

        for h_m, h_start, h_end, w_kg, w_start, w_end, pair_dist in pairs:
            if h_m <= 0:
                continue

            try:
                bmi_val = w_kg / (h_m ** 2)
            except Exception:
                continue

            if bmi_val < 10 or bmi_val > 80:
                continue

            bmi_val = round(bmi_val, 1)

            start = min(h_start, w_start)
            end = max(h_end, w_end)
            ctx = window_around(text, start, end, 180)

            # avoid computing from clearly non-current or junk contexts
            if THRESHOLD_FALSE_POS.search(ctx):
                continue

            key = "{0}|{1}|{2}|{3}|{4}|computed".format(
                bmi_val,
                section,
                note.note_id,
                note.note_date,
                ctx
            )
            if key in seen:
                continue
            seen.add(key)

            evidence = "BMI_COMPUTED_FROM_HEIGHT_WEIGHT | height_m={0:.3f} | weight_kg={1:.1f} | {2}".format(
                h_m, w_kg, ctx
            )

            candidates.append(
                Candidate(
                    field="BMI",
                    value=bmi_val,
                    status="computed",
                    evidence=evidence,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=_confidence_for_context(note.note_type, section, evidence, "computed"),
                )
            )

    if not candidates:
        return []

    candidates = sorted(candidates, key=_candidate_sort_key)
    return candidates
