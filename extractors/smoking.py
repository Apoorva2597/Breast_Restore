# ----------------------------------------------
# UPDATE (Smoking extraction improvement)
#
# Expanded phrase detection for smoking status.
# Captures common oncology clinic documentation:
#
# Current:
#   "current smoker"
#   "smokes"
#
# Former:
#   "former smoker"
#   "quit smoking"
#   "quit smoking 7 years ago"
#   "previous history of tobacco use"
#   "smoking status: former"
#
# Never:
#   "never smoked"
#   "never smoker"
#   "never tobacco user"
#   "lifetime nonsmoker"
#   "nonsmoker"
#   "non-smoker"
#   "denies tobacco"
#
# Output labels match gold dataset:
#   Current / Former / Never
#
# Python 3.6.8 compatible.
# ----------------------------------------------

import re
from models import Candidate


CURRENT_PATTERNS = [
    r"current smoker",
    r"\bsmokes\b",
    r"active smoker"
]

FORMER_PATTERNS = [
    r"former smoker",
    r"quit smoking",
    r"quit tobacco",
    r"smoking status:\s*former",
    r"previous history of tobacco",
]

NEVER_PATTERNS = [
    r"never smoked",
    r"never smoker",
    r"never tobacco",
    r"never tobacco user",
    r"lifetime nonsmoker",
    r"nonsmoker",
    r"non-smoker",
    r"denies tobacco"
]


CURRENT_REGEX = [re.compile(p, re.IGNORECASE) for p in CURRENT_PATTERNS]
FORMER_REGEX = [re.compile(p, re.IGNORECASE) for p in FORMER_PATTERNS]
NEVER_REGEX = [re.compile(p, re.IGNORECASE) for p in NEVER_PATTERNS]


def extract_smoking(note):

    results = []

    sections = note.sections if note.sections else {"FULL": ""}

    for section_name, text in sections.items():

        if not text:
            continue

        label = None
        evidence = None

        for r in CURRENT_REGEX:
            m = r.search(text)
            if m:
                label = "Current"
                evidence = m.group(0)
                break

        if label is None:
            for r in FORMER_REGEX:
                m = r.search(text)
                if m:
                    label = "Former"
                    evidence = m.group(0)
                    break

        if label is None:
            for r in NEVER_REGEX:
                m = r.search(text)
                if m:
                    label = "Never"
                    evidence = m.group(0)
                    break

        if label is None:
            continue

        results.append(
            Candidate(
                field="SmokingStatus",
                value=label,
                confidence=0.9,
                section=section_name,
                evidence=evidence,
                note_id=note.note_id,
                note_date=note.note_date,
                note_type=note.note_type
            )
        )

    return results
