from __future__ import annotations
import re

def guess_note_type(note_id: str, text: str) -> str:
    s = (note_id or "").lower()
    if "op note" in s or "operative" in s:
        return "op_note"
    if "pre op" in s or "preop" in s or "anesthesia" in s:
        return "preop_note"
    if "surg onc" in s or "surgical oncology" in s:
        return "surg_onc_note"
    if "plastic" in s:
        return "plastics_consult"
    if "follow" in s:
        return "followup_note"
    if "clinic" in s or "breast clinic" in s:
        return "clinic_note"

    t = text.lower()
    if re.search(r"\boperative\s+report\b|\bprocedure\b", t) and re.search(r"\bpostoperative\b|\bpreoperative\b", t):
        return "op_note"
    if re.search(r"\banesthesia\b|\bpre[- ]op\b|\basa\b", t):
        return "preop_note"
    if re.search(r"\bsurgical\s+oncology\b|\bbreast\s+surgery\b", t):
        return "surg_onc_note"
    if re.search(r"\bplastic\s+surgery\b|\breconstruction\s+options\b", t):
        return "plastics_consult"
    return "unknown"
