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


def _infer_laterality(ctx):
    ctx = ctx.lower()
    has_left = "left" in ctx
    has_right = "right" in ctx

    if has_left and has_right:
        return "bilateral"
    if "bilateral" in ctx:
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _infer_type(ctx):
    for pat, label in MASTECTOMY_TYPE_PATTERNS:
        if re.search(pat, ctx, re.IGNORECASE):
            return label
    return None


def extract_mastectomy(note: SectionedNote) -> List[Candidate]:
    cands = []  # type: List[Candidate]

    for section, text in note.sections.items():
        for m in MASTECTOMY_RX.finditer(text):
            ctx = window_around(text, m.start(), m.end(), 140)

            status = classify_status(
                text, m.start(), m.end(),
                PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES
            )

            # op-note default
            if note.note_type == "op_note" and status not in {"denied", "planned"}:
                status = "performed"

            # Laterality
            lat = _infer_laterality(ctx)
            if lat:
                cands.append(Candidate(
                    field="Mastectomy_Laterality",
                    value=lat,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.75
                ))

            # Type
            mtype = _infer_type(ctx)
            if mtype:
                cands.append(Candidate(
                    field="Mastectomy_Type",
                    value=mtype,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.75
                ))

            # Mastectomy_Performed flag
            if status in {"performed", "history"}:
                cands.append(Candidate(
                    field="Mastectomy_Performed",
                    value=True,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.8
                ))

    return cands
