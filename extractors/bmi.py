# extractors/bmi.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# UPDATE:
# Refined BMI extraction to target measured BMI
# mentions and avoid threshold / eligibility /
# counseling language.
#
# Captures:
#   BMI 30.12
#   BMI: 30.12
#   BMI=30.12
#   BMI of 30.12
#   body mass index 30.12
#
# Rejects examples like:
#   BMI is greater than or equal to 35
#   BMI > 35
#   BMI < 30
#   if BMI ...
#
# BMI rounded to ONE decimal to match gold file.
# Python 3.6.8 compatible.
# ----------------------------------------------

# Explicit measured BMI mentions
BMI_PATTERNS = [
    re.compile(r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bBMI\s+of\s+(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(r"\bbody\s+mass\s+index\s+of\s+(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
]

# Reject threshold / comparison / rule language
THRESHOLD_FALSE_POS = re.compile(
    r"\bBMI\b.{0,40}\b("
    r"greater\s+than|less\s+than|greater\s+than\s+or\s+equal\s+to|"
    r"less\s+than\s+or\s+equal\s+to|at\s+least|more\s+than|under|over|"
    r"above|below|minimum|maximum|threshold|criteria|cutoff|eligib"
    r")\b|"
    r"\bBMI\s*(>=|<=|>|<)\s*\d{1,3}(\.\d+)?\b",
    re.IGNORECASE,
)

# Extra caution for non-measured counseling / conditional language
CONDITIONAL_FALSE_POS = re.compile(
    r"\b(if|when|because|due to|given)\b.{0,40}\bBMI\b",
    re.IGNORECASE,
)

PREFERRED_SECTIONS = {
    "PHYSICAL EXAM",
    "OBJECTIVE",
    "ASSESSMENT",
    "ASSESSMENT/PLAN",
    "PROGRESS NOTE",
    "PROGRESS NOTES",
    "FULL",
}

SUPPRESS_SECTIONS = {
    "FAMILY HISTORY",
    "SOCIAL HISTORY",
    "ALLERGIES",
    "PLAN",
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


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
    if nt in {"op note", "operation notes", "brief op notes", "operative note", "operation note"}:
        return 0.95
    if "op" in nt or "operative" in nt or "operation" in nt:
        return 0.93
    if "progress" in nt or "clinic" in nt:
        return 0.88
    return 0.82


def extract_bmi(note: SectionedNote) -> List[Candidate]:
    """
    High-precision BMI extraction:
      - requires explicit numeric BMI mention
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

        # If whole section is clearly threshold/rule language, skip it
        if THRESHOLD_FALSE_POS.search(text):
            # still allow extraction later only if an explicit measured BMI exists
            # outside the threshold window, so do not continue here
            pass

        for rx in BMI_PATTERNS:
            for m in rx.finditer(text):
                try:
                    bmi_val = float(m.group(1))
                except ValueError:
                    continue

                if bmi_val < 10 or bmi_val > 80:
                    continue

                ctx = window_around(text, m.start(), m.end(), 140)

                # reject if local context is threshold / comparative language
                if THRESHOLD_FALSE_POS.search(ctx):
                    continue

                # reject obvious conditional / policy wording around BMI
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
