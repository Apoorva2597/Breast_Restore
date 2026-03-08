# extractors/bmi.py

import re
from models import Candidate

# ----------------------------------------------
# UPDATE:
# Restrict BMI extraction to operative / operation
# notes only, because gold BMI should reflect BMI
# at reconstruction rather than any clinic BMI.
#
# Captures:
# BMI 30.12
# BMI: 30.12
# BMI=30.12
# BMI of 30.12
# body mass index 30.12
#
# Rejects threshold / policy / comparison language:
# BMI >= 35
# BMI greater than or equal to 35
# BMI < 30
# if BMI ...
#
# BMI rounded to ONE decimal to match gold file.
# Python 3.6.8 compatible.
# ----------------------------------------------

BMI_PATTERNS = [
    re.compile(r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bBMI\s+of\s+(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
]

THRESHOLD_FALSE_POS = re.compile(
    r"(?:"
    r"bmi\s*(?:>=|=>|>|<=|=<|<)\s*\d+"
    r"|bmi\s*(?:greater|less)\s+than"
    r"|bmi\s*(?:greater|less)\s+than\s+or\s+equal\s+to"
    r"|bmi\s*(?:over|under|above|below)"
    r"|minimum\s+bmi"
    r"|maximum\s+bmi"
    r"|target\s+bmi"
    r"|goal\s+bmi"
    r"|acceptable\s+bmi"
    r"|required\s+bmi"
    r")",
    re.IGNORECASE
)

CONDITIONAL_FALSE_POS = re.compile(
    r"(?:"
    r"\bif\s+bmi\b"
    r"|\bwhen\s+bmi\b"
    r"|\bunless\s+bmi\b"
    r"|\bfor\s+bmi\s*(?:>=|=>|>|<=|=<|<)\b"
    r"|\bpatients?\s+with\s+bmi\b"
    r"|\bpts?\s+with\s+bmi\b"
    r"|\bfor\s+patients?\s+with\s+bmi\b"
    r")",
    re.IGNORECASE
)

PREFERRED_SECTIONS = set([
    "FULL",
    "PROCEDURES",
    "PROCEDURE",
    "OPERATIVE FINDINGS",
    "OPERATIVE NOTE",
    "BRIEF OPERATIVE NOTE",
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

def _is_operation_note(note_type):
    nt = str(note_type or "").lower().strip()
    return (
        "op" in nt or
        "operation" in nt or
        "operative" in nt
    )

def _confidence_for_note_type(note_type):
    nt = str(note_type or "").lower().strip()
    if nt in {
        "op note",
        "operation notes",
        "brief op notes",
        "operative note",
        "operation note",
        "brief operative note",
        "brief op note"
    }:
        return 0.95
    if "op" in nt or "operative" in nt or "operation" in nt:
        return 0.93
    return 0.90

def window_around(text, start, end, width):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].strip()

def extract_bmi(note):
    """
    High-precision BMI extraction:
    - only from operative / operation notes
    - requires explicit numeric BMI mention
    - rejects threshold/comparison language
    - rounds to 1 decimal
    Returns at most one candidate per note.
    """
    if not _is_operation_note(note.note_type):
        return []

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

                if THRESHOLD_FALSE_POS.search(ctx):
                    continue

                if CONDITIONAL_FALSE_POS.search(ctx) and "bmi of" not in ctx.lower():
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
