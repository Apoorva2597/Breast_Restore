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
#
# Python 3.6.8 compatible.
# ----------------------------------------------

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

def _confidence_for_context(note_type, section, evidence):
    nt = str(note_type or "").lower().strip()
    sec = str(section or "").upper().strip()
    ev = str(evidence or "").lower()

    score = 0.90

    if (
        "brief op" in nt or
        "operative" in nt or
        "operation" in nt or
        "op note" in nt
    ):
        score = 0.96
    elif (
        "anesthesia" in nt or
        "pre-op" in nt or
        "preop" in nt
    ):
        score = 0.94
    elif (
        "progress" in nt or
        "clinic" in nt or
        "office" in nt or
        "consult" in nt or
        "h&p" in nt
    ):
        score = 0.92

    if sec in {
        "OPERATIVE NOTE",
        "BRIEF OPERATIVE NOTE",
        "OPERATIVE FINDINGS",
        "PREOPERATIVE DIAGNOSIS",
        "POSTOPERATIVE DIAGNOSIS",
        "INDICATIONS FOR PROCEDURE",
        "INDICATIONS",
        "CODING NOTE",
        "MICROSURGICAL DETAILS",
        "VITALS",
    }:
        score += 0.01

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
    return (-conf, -len(evid))

def extract_bmi(note):
    """
    Extract measured BMI values from a note.

    Returns:
        list[Candidate]
    """
    candidates = []
    seen = set()
    section_order = _section_order(note)

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

                key = "{0}|{1}|{2}|{3}|{4}".format(
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
                        confidence=_confidence_for_context(note.note_type, section, ctx),
                    )
                )

    if not candidates:
        return []

    candidates = sorted(candidates, key=_candidate_sort_key)
    return candidates
