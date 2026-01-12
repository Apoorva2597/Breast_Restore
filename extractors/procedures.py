import re
from typing import List

from ..models import Candidate, SectionedNote
from ..config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from .utils import window_around, classify_status

RECON_PATTERNS = [
    (r"\bdiep\s+flap\b", "diep flap"),
    (r"\btram\s+flap\b", "tram flap"),
    (r"\bsiea\s+flap\b", "siea flap"),
    (r"\blatissimus\s+dorsi\s+flap\b", "latissimus dorsi flap"),
    (r"\bdirect\s*[- ]\s*to\s*[- ]\s*implant\b", "direct-to-implant"),
    (r"\btissue\s+expander\b", "tissue expander/implant"),
]

LATERALITY_PATTERNS = [
    (r"\bbilateral\b", "bilateral"),
    (r"\bleft\b", "left"),
    (r"\bright\b", "right"),
]

def extract_reconstruction(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    for section, text in note.sections.items():
        t = text.lower()
        for pat, recon_type in RECON_PATTERNS:
            m = re.search(pat, t, re.IGNORECASE)
            if not m:
                continue

            status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)

            # Operative note default: reconstruction mentioned = performed unless negated/planned
            if note.note_type == "op_note" and status not in {"denied", "planned"}:
                status = "performed"

            # Non-op notes: "performed" language often means prior history
            if note.note_type != "op_note" and status == "performed":
                status = "history"

            evid140 = window_around(text, m.start(), m.end(), 140)

            # Recon type
            cands.append(Candidate(
                field="Recon_Type",
                value=recon_type,
                status=status if status != "denied" else "unknown",
                evidence=evid140,
                section=section,
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.8 if note.note_type == "op_note" else 0.6
            ))

            # Laterality
            lat_val = None
            has_left = re.search(r"\bleft\b", t, re.IGNORECASE) is not None
            has_right = re.search(r"\bright\b", t, re.IGNORECASE) is not None

            if has_left and has_right:
                lat_val = "bilateral"
            elif re.search(r"\bbilateral\b", t, re.IGNORECASE):
                lat_val = "bilateral"
            elif has_left:
                lat_val = "left"
            elif has_right:
                lat_val = "right"

            if lat_val:
                cands.append(Candidate(
                    field="Recon_Laterality",
                    value=lat_val,
                    status=status,
                    evidence=window_around(text, m.start(), m.end(), 140),
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.7
                ))


            # Timing cue
            if re.search(r"\bimmediate\b", window_around(text, m.start(), m.end(), 120), re.IGNORECASE):
                cands.append(Candidate(
                    field="Recon_Timing",
                    value="immediate",
                    status=status,
                    evidence=evid140,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.65
                ))

            # Do not emit Recon_Performed=False (misleading). Emit planned explicitly.
            if status in {"performed", "history"}:
                cands.append(Candidate(
                    field="Recon_Performed",
                    value=True,
                    status=status,
                    evidence=evid140,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.8 if note.note_type == "op_note" else 0.5
                ))
            else:
                cands.append(Candidate(
                    field="Recon_Planned",
                    value=True,
                    status=status,  # typically planned
                    evidence=evid140,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.7 if note.note_type == "op_note" else 0.55
                ))

            break  # stop after first recon pattern in a section

    return cands

def extract_lymph_node_mgmt(note: SectionedNote) -> List[Candidate]:
    cands: List[Candidate] = []
    patterns = [
        (r"\bsentinel\s+lymph\s+node\b|\bsln\b|\bslnb\b", "SLNB"),
        (r"\baxillary\s+lymph\s+node\s+dissection\b|\balnd\b", "ALND"),
        (r"\binternal\s+mammary\s+lymph\s+node\b", "InternalMammaryLN"),
    ]
    for section, text in note.sections.items():
        t = text.lower()
        for pat, val in patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if not m:
                continue

            status = classify_status(text, m.start(), m.end(), PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES)

            # Operative note default: if mentioned and not negated/planned, treat as performed
            if note.note_type == "op_note" and status not in {"denied", "planned"}:
                status = "performed"

            # Non-op note: performed language often indicates planned (consult note)
            if note.note_type != "op_note" and status == "performed":
                status = "planned"

            cands.append(Candidate(
                field="LymphNodeMgmt",
                value=val,
                status=status,
                evidence=window_around(text, m.start(), m.end(), 140),
                section=section,
                note_type=note.note_type,
                note_id=note.note_id,
                note_date=note.note_date,
                confidence=0.8 if note.note_type == "op_note" else 0.55
            ))
            break
    return cands
