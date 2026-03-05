# extractors/mastectomy.py
import re
from typing import List

from models import Candidate, SectionedNote
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from .utils import window_around, classify_status

MASTECTOMY_RX = re.compile(r"\bmastectomy\b", re.IGNORECASE)

MASTECTOMY_TYPE_PATTERNS = [
    (r"\bnipple[- ]sparing\b", "nipple-sparing"),
    (r"\bskin[- ]sparing\b", "skin-sparing"),
    (r"\bsimple\s+mastectomy\b", "simple"),
    (r"\btotal\s+mastectomy\b", "simple"),
    (r"\bmodified\s+radical\b|\bMRM\b", "modified radical"),
    (r"\bradical\s+mastectomy\b", "radical"),
]

# High-FP template phrases
TEMPLATE_EXCLUDES = [
    r"\bplanned\b",
    r"\bwill undergo\b",
    r"\bscheduled for\b",
    r"\bdiscussed\b.*\bmastectomy\b",
]

# Context cues that indicate it actually happened
PERFORMED_CONTEXT = re.compile(r"\b(performed|underwent|completed|status post|s/p)\b", re.IGNORECASE)


def _infer_laterality(ctx: str):
    c = (ctx or "").lower()
    if "bilateral" in c:
        return "bilateral"
    has_left = "left" in c or "lt" in c
    has_right = "right" in c or "rt" in c
    if has_left and has_right:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _infer_type(ctx: str):
    for pat, label in MASTECTOMY_TYPE_PATTERNS:
        if re.search(pat, ctx, re.IGNORECASE):
            return label
    return None


def extract_mastectomy(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    for section, text in note.sections.items():
        if not text:
            continue

        for m in MASTECTOMY_RX.finditer(text):
            ctx = window_around(text, m.start(), m.end(), 220)
            low = ctx.lower()

            # exclude template/plan-y contexts
            if any(re.search(p, low) for p in TEMPLATE_EXCLUDES):
                # allow if op note OR explicit performed cue in same window
                if not (str(note.note_type).lower() in {"op note", "operation notes", "brief op notes"} or PERFORMED_CONTEXT.search(ctx)):
                    continue

            status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)

            # op-note default: if not denied/planned, treat as performed
            if str(note.note_type).lower() in {"op note", "operation notes", "brief op notes"} and status not in {"denied", "planned"}:
                status = "performed"

            if status == "planned":
                # conservative: don’t mark mastectomy performed from planned text
                continue
            if status == "denied":
                continue

            lat = _infer_laterality(ctx)
            if lat:
                cands.append(Candidate(
                    field="Mastectomy_Laterality",
                    value=lat,
                    status=status if status != "performed" else "history",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.80 if str(note.note_type).lower() in {"op note", "operation notes", "brief op notes"} else 0.70
                ))

            mtype = _infer_type(ctx)
            if mtype:
                cands.append(Candidate(
                    field="Mastectomy_Type",
                    value=mtype,
                    status=status if status != "performed" else "history",
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.78
                ))

            cands.append(Candidate(
                field="Mastectomy_Performed",
                value=True,
                status=status if status != "performed" else "history",
                evidence=ctx,
                section=section,
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.82
            ))

    return cands
