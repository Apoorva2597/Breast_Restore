from __future__ import annotations
import re
from typing import Optional, List

def find_first(patterns: List[str], text: str, flags=re.IGNORECASE) -> Optional[re.Match]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m
    return None

def has_any(patterns: List[str], text: str, flags=re.IGNORECASE) -> bool:
    return find_first(patterns, text, flags) is not None

def window_around(text: str, start: int, end: int, window: int = 80) -> str:
    a = max(0, start - window)
    b = min(len(text), end + window)
    return text[a:b].replace("\n", " ").strip()

def classify_status(text: str, start: int, end: int, performed_cues, planned_cues, negation_cues) -> str:
    """
    Determine status for a mention based on local context.

    IMPORTANT hardening:
    - Many templated EHR fields express absence as "(None)" / "None" / "No".
      Treat these as explicit negation even if config NEGATION_CUES misses them.
    """
    ctx = window_around(text, start, end, window=120).lower()

    # Hard-stop for common templated negations (prevents "Diabetes: (None)" -> True)
    none_negation_re = r"\(\s*none\s*\)|\b(none|no|denies|denied|negative)\b"
    if re.search(none_negation_re, ctx):
        return "denied"

    if any(re.search(p, ctx) for p in negation_cues):
        return "denied"
    if any(re.search(p, ctx) for p in planned_cues):
        return "planned"
    if any(re.search(p, ctx) for p in performed_cues):
        return "performed"
    return "history"

def should_skip_block(section: str, evidence: str) -> bool:
    """
    Returns True if this text block should be ignored entirely for extraction,
    because it represents a Family History or Allergies table/block.
    This is implementation-agnostic and does not depend on perfect sectioning.
    """
    sec = (section or "").upper()
    ev = (evidence or "").lower()

    # Hard stop if the section is explicitly family history or allergies
    if sec in {"FAMILY HISTORY", "ALLERGIES"}:
        return True

    # Family history table/phrasing often embedded in other sections
    family_cues = [
        "paternal", "maternal", "grandmother", "grandfather",
        "mother", "father", "sister", "brother", "relation", "family history"
    ]

    # Allergy blocks often embedded in other sections as tables
    allergy_cues = [
        "allergen", "reaction", "severity", "allergies", "rash", "anaphyl"
    ]

    # If either table signature is present, skip
    if any(cue in ev for cue in family_cues):
        return True
    if any(cue in ev for cue in allergy_cues):
        return True

    return False
