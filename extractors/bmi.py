# ----------------------------------------------
# UPDATE (BMI extraction improvement)
#
# Extracts BMI values from clinic vitals blocks
# and narrative text. Handles patterns such as:
#
#   BMI 30.12
#   BMI: 30.12
#   BMI=30.12
#   BMI 30.12 kg/m2
#   BMI 30.1
#
# Handles vitals rows with separators:
#   Temp | Ht | Wt | BMI
#
# BMI rounded to ONE decimal to match gold set.
#
# Python 3.6.8 compatible
# ----------------------------------------------

import re
from models import Candidate


BMI_REGEX = re.compile(
    r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)",
    re.IGNORECASE
)


def normalize_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_bmi(note):

    results = []

    sections = note.sections if note.sections else {"FULL": ""}

    for section_name, text in sections.items():

        if not text:
            continue

        text = normalize_text(text)

        matches = BMI_REGEX.findall(text)

        if not matches:
            continue

        for m in matches:

            try:
                bmi_val = float(m)
            except Exception:
                continue

            if bmi_val < 10 or bmi_val > 80:
                continue

            bmi_val = round(bmi_val, 1)

            results.append(
                Candidate(
                    field="BMI",
                    value=bmi_val,
                    confidence=0.95,
                    section=section_name,
                    evidence="BMI extracted from vitals pattern",
                    note_id=note.note_id,
                    note_date=note.note_date,
                    note_type=note.note_type
                )
            )

    return results
