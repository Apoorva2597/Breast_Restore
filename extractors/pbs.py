# extractors/pbs.py
import re
from typing import List

from models import Candidate, SectionedNote
from .utils import window_around

SUPPRESS_SECTIONS = {"FAMILY HISTORY", "REVIEW OF SYSTEMS", "ALLERGIES"}

NEGATE_PATTERNS = [
    r"\bno\s+prior\s+breast\s+surgery\b",
    r"\bno\s+history\s+of\s+breast\s+surgery\b",
    r"\bdenies\s+prior\s+breast\s+surgery\b",
    r"\bnever\s+had\s+breast\s+surgery\b",
]

LUMP_PATTERNS = [
    r"\blumpectomy\b",
    r"\bpartial\s+mastectomy\b",
    r"\bsegmental\s+mastectomy\b",
    r"\bbreast[- ]conserving\s+surgery\b",
    r"\bwide\s+local\s+excision\b",
]

REDUCTION_PATTERNS = [
    r"\bbreast\s+reduction\b",
    r"\breduction\s+mammaplasty\b",
]

MASTOPEXY_PATTERNS = [
    r"\bmastopexy\b",
    r"\bbreast\s+lift\b",
]

AUGMENT_PATTERNS = [
    r"\bbreast\s+augmentation\b",
    r"\baugmentation\s+mammaplasty\b",
    r"\bimplants?\b.*\baugmentation\b",
]

OTHER_PATTERNS = [
    r"\bexcisional\s+biopsy\b",
    r"\bbenign\s+excision\b",
    r"\bbiopsy\b",
]


def _emit(field, text, m, section, note, conf):
    return Candidate(
        field=field,
        value=True,
        status="history",
        evidence=window_around(text, m.start(), m.end(), 180),
        section=section,
        note_type=note.note_type,
        note_id=note.note_id,
        note_date=note.note_date,
        confidence=conf,
    )


def extract_pbs(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []

    for section, text in note.sections.items():
        if section in SUPPRESS_SECTIONS:
            continue
        if not text:
            continue

        low = text.lower()

        # whole-block negation
        if any(re.search(p, low) for p in NEGATE_PATTERNS):
            continue

        # Lumpectomy
        for p in LUMP_PATTERNS:
            m = re.search(p, low)
            if m:
                cands.append(_emit("PBS_Lumpectomy", text, m, section, note, 0.80))
                break

        # Reduction
        for p in REDUCTION_PATTERNS:
            m = re.search(p, low)
            if m:
                cands.append(_emit("PBS_Breast Reduction", text, m, section, note, 0.78))
                break

        # Mastopexy
        for p in MASTOPEXY_PATTERNS:
            m = re.search(p, low)
            if m:
                cands.append(_emit("PBS_Mastopexy", text, m, section, note, 0.78))
                break

        # Augmentation
        for p in AUGMENT_PATTERNS:
            m = re.search(p, low)
            if m:
                cands.append(_emit("PBS_Augmentation", text, m, section, note, 0.78))
                break

        # Other prior breast procedures
        for p in OTHER_PATTERNS:
            m = re.search(p, low)
            if m:
                cands.append(_emit("PBS_Other", text, m, section, note, 0.72))
                break

    # Optional: PastBreastSurgery flag if ANY PBS_* true
    if any(c.field.startswith("PBS_") and c.value is True for c in cands):
        # make a single summary candidate
        cands.append(
            Candidate(
                field="PastBreastSurgery",
                value=True,
                status="history",
                evidence="Derived from PBS_* hits in note.",
                section="DERIVED",
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.85,
            )
        )

    return cands
