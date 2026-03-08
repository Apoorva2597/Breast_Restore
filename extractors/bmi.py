# extractors/bmi.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

# ----------------------------------------------
# UPDATE:
# Improved BMI extraction for clinic progress notes.
# Handles common formats such as:
#   BMI 30.12
#   BMI: 30.12
#   BMI=30.12
#   BMI 30.12 kg/m2
#   BMI 30.1
#
# Also normalizes note text spacing so matches still work
# in long vitals lines or broken line formatting.
#
# BMI is rounded to ONE decimal to match the gold file.
# Python 3.6.8 compatible.
# ----------------------------------------------

BMI_PATTERNS = [
    re.compile(r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\b", re.IGNORECASE),
]

PREFERRED_SECTIONS = {
    "VITALS", "OBJECTIVE", "PROGRESS NOTES", "PROGRESS NOTE",
    "FULL"
}

SUPPRESS_SECTIONS = {
    "LABS", "IMAGING"
}


def _normalize_text(text):
    text = text or ""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_bmi(note: SectionedNote) -> List[Candidate]:
    """
    High-precision BMI extraction:
      - Looks for explicit BMI mentions
      - Keeps values in plausible adult range
      - Rounds to one decimal to match gold
    Returns at most one candidate per note.
    """

    section_order = []

    for s in note.sections.keys():
        if s in PREFERRED_SECTIONS and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

    for s in note.sections.keys():
        if s not in section_order and s not in SUPPRESS_SECTIONS:
            section_order.append(s)

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

                bmi_val = round(bmi_val, 1)
                ctx = window_around(text, m.start(), m.end(), 120)

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
                        confidence=0.95,
                    )
                ]

    return []
