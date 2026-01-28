import re
from typing import List

from models import Candidate, SectionedNote
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from .utils import window_around, classify_status, find_first, should_skip_block

# ----------------------------
# Radiation patterns
# ----------------------------
RADIATION_POS = [
    r"\bradiation\b",
    r"\bradiation\s+therapy\b",
    r"\bradiotherapy\b",
    r"\bxrt\b",
    r"\bRT\b",  # beware: also "RT breast" etc. We'll validate via evidence review
    r"\bPMRT\b",  # post-mastectomy RT (still indicates radiation history)
]

# ----------------------------
# Chemo patterns (include regimen words later after grep pass)
# ----------------------------
CHEMO_POS = [
    r"\bchemotherapy\b",
    r"\bchemo\b",
    r"\bneoadjuvant\s+chemo\b",
    r"\badjuvant\s+chemo\b",
    r"\bchemo\s+therapy\b",
]

# Endocrine therapy is NOT chemotherapy (per your spec)
ENDOCRINE_EXCLUDE = [
    r"\btamoxifen\b",
    r"\bletrozole\b",
    r"\banastrozole\b",
    r"\bexemestane\b",
    r"\bfulvestrant\b",
    r"\barimidex\b",
    r"\bfemara\b",
    r"\baromasin\b",
]

def _extract_flag(field, pos_patterns, note, exclude_patterns=None):
    cands = []  # type: List[Candidate]
    exclude_patterns = exclude_patterns or []

    for section, text in note.sections.items():
        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 160)

        # skip family history/allergies blocks etc.
        if should_skip_block(section, evid):
            continue

        # endocrine exclusion for chemo: if the only signal is endocrine meds, suppress
        evid_lower = evid.lower()
        if exclude_patterns:
            # If endocrine words present AND no "chemo/chemotherapy" anchor besides them, suppress.
            # (Keeps it conservative for your study definition.)
            if any(re.search(pat, evid_lower) for pat in exclude_patterns):
                # If chemo anchor not present strongly, skip
                if not re.search(r"\bchemo\b|\bchemotherapy\b", evid_lower):
                    continue

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)
        if status == "performed":
            status = "history"

        value = False if status == "denied" else True

        cands.append(Candidate(
            field=field,
            value=value,
            status=status,
            evidence=evid,
            section=section,
            note_type=note.note_type,
            note_id=note.note_id,
            note_date=note.note_date,
            confidence=0.75
        ))

    return cands


def extract_cancer_treatment(note: SectionedNote) -> List[Candidate]:
    cands = []  # type: List[Candidate]
    cands += _extract_flag("Radiation", RADIATION_POS, note)
    cands += _extract_flag("Chemo", CHEMO_POS, note, exclude_patterns=ENDOCRINE_EXCLUDE)
    return cands
