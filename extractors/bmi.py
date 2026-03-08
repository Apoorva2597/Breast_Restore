# extractors/bmi.py

import re
from models import Candidate

# ----------------------------------------------
# UPDATE:
# BMI extraction is no longer restricted to OP notes only.
# It now extracts from peri-reconstruction notes already
# filtered by the build script (op note, brief op note,
# anesthesia, pre-op, progress note, clinic, H&P, etc.).
#
# Captures:
# BMI 30.12
# BMI: 30.12
# BMI=30.12
# BMI of 30.12
# body mass index 30.12
# body mass index of 30.12
#
# Rejects threshold / policy / comparison language:
# BMI >= 35
# BMI greater than or equal to 35
# BMI < 30
# if BMI ...
#
# Python 3.6.8 compatible.
# ----------------------------------------------

BMI_PATTERNS = [
    re.compile(r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bBMI\s+of\s+(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s+of\s+(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bmorbid obesity\s*[-(]?\s*BMI\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bobesity\s*,?\s*BMI\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
]

THRESHOLD_FALSE_POS = re.compile(
    r"(?:"
    r"\bBMI\s*(?:>=|=>|>|<=|=<|<)\s*\d+"
    r"|\bBMI\s*(?:greater|less)\s+than\b"
    r"|\bBMI\s*(?:greater|less)\s+than\s+or\s+equal\s+to\b"
    r"|\bBMI\s*(?:over|under|above|below)\b"
    r"|\bminimum\s+BMI\b"
    r"|\bmaximum\s+BMI\b"
    r"|\btarget\s+BMI\b"
    r"|\bgoal\s+BMI\b"
    r"|\bacceptable\s+BMI\b"
    r"|\brequired\s+BMI\b"
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
    r")",
    re.IGNORECASE
)

PREFERRED_SECTIONS = set([
    "FULL",
    "VITALS",
    "PHYSICAL EXAM",
    "HISTORY",
    "HISTORY OF PRESENT ILLNESS",
    "HPI",
    "PRE-OP",
    "PREOP",
    "ANESTHESIA",
    "PROCEDURES",
    "PROCEDURE",
    "OPERATIVE FINDINGS",
    "OPERATIVE NOTE",
    "BRIEF OPERATIVE NOTE",
])

SUPPRESS_SECTIONS = set([
    "PLAN",
    "ASSESSMENT",
    "INSTRUCTIONS",
    "DISPOSITION",
])

def _normalize_text(text):
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

def _confidence_for_note_type(note_type):
    nt = str(note_type or "").lower().strip()
    if (
        "brief op" in nt or
        "operative" in nt or
        "operation" in nt or
        "anesthesia" in nt or
        "pre-op" in nt or
        "preop" in nt
    ):
        return 0.95
    if (
        "progress" in nt or
        "clinic" in nt or
        "office" in nt or
        "consult" in nt or
        "h&p" in nt
    ):
        return 0.93
    return 0.90

def window_around(text, start, end, width):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].strip()

def extract_bmi(note):
    """
    High-precision BMI extraction:
    - build script handles recon-date anchoring
    - extractor only looks for explicit measured BMI text
    - rejects threshold/comparison language
    - rounds to 1 decimal
    Returns at most one candidate per note.
    """
    section_order = _section_order(note)

    for section in section_order:
        raw_text = note.sections.get(section, "") or ""
        if not raw_text:
            continue

        text = _normalize_text(raw_text)

        for rx in BMI_PATTERNS:
            for m in rx.finditer(text):
                try:
                    bmi_val = float(m.group(1))
                except ValueError:
                    continue

                if bmi_val < 10 or bmi_val > 80:
                    continue

                ctx = window_around(text, m.start(), m.end(), 140)
                ctx_low = ctx.lower()

                if THRESHOLD_FALSE_POS.search(ctx):
                    continue

                if CONDITIONAL_FALSE_POS.search(ctx):
                    if "bmi of" not in ctx_low and "body mass index of" not in ctx_low:
                        continue

                bmi_val = round(bmi_val, 1)

                return [
                    Candidate(
                        field="BMI",
                        value=bmi_val,
                        status="measured",
                        evidence=ctx,
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=_confidence_for_note_type(note.note_type),
                    )
                ]

    return []
