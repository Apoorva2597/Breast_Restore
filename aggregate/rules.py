from __future__ import annotations
from typing import Dict, List, Optional
from ..models import Candidate, FinalField
from ..config import STATUS_PRECEDENCE, NOTE_TYPE_PRECEDENCE, SECTION_PRECEDENCE, PHASE1_FIELDS

def _rank(value: str, order: List[str], default: int = 10_000) -> int:
    try:
        return order.index(value)
    except ValueError:
        return default

def choose_best(cands: List[Candidate], field: str) -> Optional[FinalField]:
    if not cands:
        return None

    performed_only = {"Recon_Performed", "Recon_Type", "Recon_Laterality", "Recon_Timing", "LymphNodeMgmt_Performed"}
    filtered = [c for c in cands if not (field in performed_only and c.status == "planned")]
    if not filtered:
        filtered = cands

    def key(c: Candidate):
        return (
            _rank(c.status, STATUS_PRECEDENCE),
            _rank(c.note_type, NOTE_TYPE_PRECEDENCE),
            _rank(c.section, SECTION_PRECEDENCE),
            -(c.confidence or 0.0),
        )

    win = sorted(filtered, key=key)[0]
    rule = f"status>{win.status}; note_type>{win.note_type}; section>{win.section}; conf>{win.confidence}"
    return FinalField(
        field=field,
        value=win.value,
        status=win.status,
        evidence=win.evidence,
        section=win.section,
        note_type=win.note_type,
        note_id=win.note_id,
        note_date=win.note_date,
        rule=rule,
    )

def aggregate_patient(candidates: List[Candidate]) -> Dict[str, FinalField]:
    out: Dict[str, FinalField] = {}

    # First pass: compute all fields normally
    for field in PHASE1_FIELDS:
        ff = choose_best([c for c in candidates if c.field == field], field)
        if ff:
            out[field] = ff

    # --- HARD CLINICAL RULES ---
    # If reconstruction was performed, drop planned
    if "Recon_Performed" in out:
        out.pop("Recon_Planned", None)

    return out
