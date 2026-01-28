import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around, classify_status, find_first, should_skip_block
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES


# ----------------------------
# Stage 1 outcomes (FIRST PASS)
# ----------------------------
# Keep patterns conservative; expand later via grep + QA.

REOP_POS = [
    r"\breturn(ed)?\s+to\s+(the\s+)?or\b",
    r"\bback\s+to\s+(the\s+)?or\b",
    r"\btake\s*back\b",
    r"\bre-?operation\b",
    r"\bre-?exploration\b",
    r"\bwashout\b",
    r"\bincision\s+and\s+drainage\b",
    r"\bi\s*&\s*d\b",
    r"\bdebridement\b",
]

REHOSP_POS = [
    r"\breadmit(ted|sion)?\b",
    r"\bre-?admit(ted|sion)?\b",
    r"\brehospitali[sz]ed\b",
    r"\breturn(ed)?\s+to\s+hospital\b",
]

# “Failure” is tricky; start with high-signal losses/removals
FAILURE_POS = [
    r"\bflap\s+(loss|failed|failure|necrosis)\b",
    r"\b(total|complete)\s+flap\s+necrosis\b",
    r"\bimplant\s+(loss|removed)\b",
    r"\bexpander\s+(loss|removed)\b",
    r"\bprosthesis\s+removed\b",
    r"\bexplant(ed)?\b",
]

# If someone says “planned return to OR” it’s not an outcome yet
OUTCOME_PLAN_EXCLUDE = [
    r"\bplanned\b",
    r"\bscheduled\b",
    r"\bwill\b",
    r"\bto\s+be\b",
    r"\bconsider(ing|ed)?\b",
]


def _extract_flag(field, pos_patterns, note):
    cands = []  # type: List[Candidate]

    for section, text in note.sections.items():
        m = find_first(pos_patterns, text)
        if not m:
            continue

        evid = window_around(text, m.start(), m.end(), 180)

        # Skip junk blocks (family history/allergies etc)
        if should_skip_block(section, evid):
            continue

        evid_l = evid.lower()

        # Exclude obvious future/plan language
        for pat in OUTCOME_PLAN_EXCLUDE:
            if re.search(pat, evid_l):
                # BUT: don't exclude if the phrase is clearly retrospective ("returned to OR on POD2")
                # We'll keep this simple for v1: if plan words appear, downgrade to planned.
                pass

        status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)

        # Outcomes are events; treat "performed" as happened
        # If planned language is detected, keep planned (conservative)
        value = False if status == "denied" else True

        # Heuristic: if plan-ish language appears near the hit, force planned unless op-note
        if note.note_type != "op note":
            for pat in OUTCOME_PLAN_EXCLUDE:
                if re.search(pat, evid_l) and status not in {"denied"}:
                    status = "planned"
                    break

        # Confidence: op note strongest, progress notes decent, others lower
        conf = 0.8
        nt = (note.note_type or "").lower()
        if nt == "op note" or nt == "brief op notes":
            conf = 0.9
        elif nt == "progress notes" or nt == "h&p":
            conf = 0.75
        else:
            conf = 0.6

        cands.append(Candidate(
            field=field,
            value=value,
            status=status,
            evidence=evid,
            section=section,
            note_type=note.note_type,
            note_id=note.note_id,
            note_date=note.note_date,
            confidence=conf
        ))

    return cands


def extract_stage1_outcomes(note: SectionedNote) -> List[Candidate]:
    cands = []  # type: List[Candidate]
    cands += _extract_flag("Stage1_Reoperation", REOP_POS, note)
    cands += _extract_flag("Stage1_Rehospitalization", REHOSP_POS, note)
    cands += _extract_flag("Stage1_Failure", FAILURE_POS, note)
    return cands
