# ----------------------------------------------
# UPDATE (NLP refinement):
# Improved BMI extraction using vitals-style regex
# capturing patterns such as:
#   "BMI 30.12"
#   "BMI: 30.12"
#   "BMI=30.12"
#   "BMI 30 kg/m2"
#
# These frequently appear in clinic progress notes
# in the vitals block.
#
# Python 3.6.8 compatible.
# ----------------------------------------------

import re
from models import Candidate


BMI_REGEX = re.compile(
    r"\bBMI\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)",
    re.IGNORECASE
)


def extract_bmi(note):

    results = []

    sections = note.sections if note.sections else {"FULL": ""}

    for section_name, text in sections.items():

        if not text:
            continue

        matches = BMI_REGEX.findall(text)

        if not matches:
            continue

        for m in matches:

            try:
                bmi_val = float(m)
            except Exception:
                continue

            # sanity bounds
            if bmi_val < 10 or bmi_val > 80:
                continue

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
