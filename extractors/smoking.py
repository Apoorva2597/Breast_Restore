# ----------------------------------------------
# UPDATE (Smoking extraction improvement)
#
# Robust smoking status extraction for oncology
# clinic notes. Handles narrative and structured
# documentation including:
#
# Current:
#   current smoker
#   smokes
#   active smoker
#
# Former:
#   former smoker
#   quit smoking
#   quit tobacco
#   smoking status: former
#   previous history of tobacco use
#
# Never:
#   never smoker
#   never smoked
#   never tobacco user
#   lifetime nonsmoker
#   nonsmoker / non-smoker
#   denies smoking
#   denies tobacco
#   denies tobacco use
#   no history of tobacco use
#
# Output labels:
#   Current / Former / Never
#
# Python 3.6.8 compatible
# ----------------------------------------------

import re
from models import Candidate


CURRENT_PATTERNS = [
    r"current smoker",
    r"\bsmokes\b",
    r"active smoker",
]

FORMER_PATTERNS = [
    r"former smoker",
    r"quit smoking",
    r"quit tobacco",
    r"smoking status\s*:\s*former",
    r"previous history of tobacco",
]

NEVER_PATTERNS = [
    r"never smoker",
    r"never smoked",
    r"never tobacco",
    r"never tobacco user",
    r"lifetime nonsmoker",
    r"\bnonsmoker\b",
    r"\bnon[- ]smoker\b",
    r"denies smoking",
    r"denies tobacco",
    r"denies tobacco use",
    r"no history of tobacco",
]


CURRENT_REGEX = [re.compile(p, re.IGNORECASE) for p in CURRENT_PATTERNS]
FORMER_REGEX = [re.compile(p, re.IGNORECASE) for p in FORMER_PATTERNS]
NEVER_REGEX = [re.compile(p, re.IGNORECASE) for p in NEVER_PATTERNS]


def normalize_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_smoking(note):

    results = []

    sections = note.sections if note.sections else {"FULL": ""}

    for section_name, text in sections.items():

        if not text:
            continue

        text = normalize_text(text)

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
