# ----------------------------------------------
# UPDATE (NLP refinement):
# Expanded smoking extraction rules to capture
# tobacco-related synonyms commonly found under
# SOCIAL HISTORY in clinic notes.
#
# Recognizes phrases such as:
#   "current smoker"
#   "former smoker"
#   "quit smoking"
#   "never smoked"
#   "never tobacco user"
#   "nonsmoker"
#   "denies tobacco"
#   "previous history of tobacco use"
#
# Outputs standardized labels:
#   Current
#   Former
#   Never
#
# Python 3.6.8 compatible.
# ----------------------------------------------

import re
from models import Candidate


CURRENT_PATTERNS = [
    r"\bcurrent smoker\b",
    r"\bsmokes\b",
    r"\bactive smoker\b",
]

FORMER_PATTERNS = [
    r"\bformer smoker\b",
    r"\bquit smoking\b",
    r"\bquit tobacco\b",
    r"\bsmoking status:\s*former\b",
    r"\bprevious history of tobacco\b",
]

NEVER_PATTERNS = [
    r"\bnever smoker\b",
    r"\bnever smoked\b",
    r"\bnever tobacco\b",
    r"\bnonsmoker\b",
    r"\bnon[- ]smoker\b",
    r"\blifetime nonsmoker\b",
    r"\bnever tobacco user\b",
    r"\bdenies tobacco\b",
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

        lower_text = text.lower()

        label = None
        evidence = None

        # order matters
        for r in CURRENT_REGEX:
            m = r.search(lower_text)
            if m:
                label = "Current"
                evidence = m.group(0)
                break

        if label is None:
            for r in FORMER_REGEX:
                m = r.search(lower_text)
                if m:
                    label = "Former"
                    evidence = m.group(0)
                    break

        if label is None:
            for r in NEVER_REGEX:
                m = r.search(lower_text)
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
                confidence=0.90,
                section=section_name,
                evidence=evidence,
                note_id=note.note_id,
                note_date=note.note_date,
                note_type=note.note_type
            )
        )

    return results
