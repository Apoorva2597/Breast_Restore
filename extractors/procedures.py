import re
from typing import List

from models import Candidate, SectionedNote
from config import NEGATION_CUES, PLANNED_CUES, PERFORMED_CUES
from .utils import window_around, classify_status


# ---------------------------------------------------------
# Reconstruction (existing)
# ---------------------------------------------------------
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
    cands = []  # type: List[Candidate]
    for section, text in note.sections.items():
        t = text.lower()
        for pat, recon_type in RECON_PATTERNS:
            m = re.search(pat, t, re.IGNORECASE)
            if not m:
                continue

            status = classify_status(
                text, m.start(), m.end(),
                PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES
            )

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

            # Laterality (section-level; good enough for now)
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

            # Stop after first recon pattern in this section
            break

    return cands


def extract_lymph_node_mgmt(note: SectionedNote) -> List[Candidate]:
    cands = []  # type: List[Candidate]
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

            status = classify_status(
                text, m.start(), m.end(),
                PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES
            )

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


# ---------------------------------------------------------
# NEW: Prior breast surgery (PBS) – Tier 2
#   14.a PBS_Lumpectomy
#   14.b PBS_Other
# ---------------------------------------------------------

# Very simple pattern sets to start; we can harden later with your gold set.
PBS_LUMP_PATTERNS = [
    r"\blumpectomy\b",
    r"\bpartial\s+mastectomy\b",
    r"\bsegmental\s+mastectomy\b",
    r"\bbreast[- ]conserving\s+surgery\b",
    r"\bwide\s+local\s+excision\b",
]

PBS_OTHER_PATTERNS = [
    r"\bbreast\s+reduction\b",
    r"\breduction\s+mammaplasty\b",
    r"\bbenign\s+excision\b",
    r"\bexcisional\s+biopsy\b",
    r"\bmastopexy\b",
]

PBS_NEGATE_PATTERNS = [
    r"no\s+prior\s+breast\s+surgery",
    r"no\s+history\s+of\s+breast\s+surgery",
    r"denies\s+prior\s+breast\s+surgery",
]


def extract_prior_breast_surgery(note: SectionedNote) -> List[Candidate]:
    """
    Emit:
      - PBS_Lumpectomy = True when lumpectomy-like prior surgery mentioned
      - PBS_Other      = True when other prior breast procedures mentioned

    For now we only emit POSITIVE evidence; absence of a candidate does NOT mean "no".
    """
    cands = []  # type: List[Candidate]

    for section, text in note.sections.items():
        lower = text.lower()

        # If block explicitly says "no prior breast surgery", skip that block entirely
        for pat in PBS_NEGATE_PATTERNS:
            if re.search(pat, lower):
                lower = ""  # suppress both PBS_Lumpectomy and PBS_Other
                break
        if not lower:
            continue

        # Lumpectomy-like prior surgery
        for pat in PBS_LUMP_PATTERNS:
            m = re.search(pat, lower)
            if m:
                ctx = window_around(text, m.start(), m.end(), 140)
                status = "history"  # by definition this is prior breast surgery
                cands.append(Candidate(
                    field="PBS_Lumpectomy",
                    value=True,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.75
                ))
                break  # one per section is fine for now

        # Other benign/prior breast surgery types
        for pat in PBS_OTHER_PATTERNS:
            m = re.search(pat, lower)
            if m:
                ctx = window_around(text, m.start(), m.end(), 140)
                status = "history"
                cands.append(Candidate(
                    field="PBS_Other",
                    value=True,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.7
                ))
                break

    return cands


# ---------------------------------------------------------
# NEW: Mastectomy features – Tier 2
#   16. Mastectomy_Laterality
#   Mastectomy_Type
# ---------------------------------------------------------

MASTECTOMY_CORE_RX = re.compile(r"\bmastectomy\b", re.IGNORECASE)

MASTECTOMY_TYPE_PATTERNS = [
    (r"\bnipple[- ]sparing\b", "nipple-sparing"),
    (r"\bskin[- ]sparing\b", "skin-sparing"),
    (r"\bsimple\s+mastectomy\b", "simple"),
    (r"\btotal\s+mastectomy\b", "simple"),  # map total -> simple bucket
    (r"\bmodified\s+radical\s+mastectomy\b|\bMRM\b", "modified radical"),
    (r"\bradical\s+mastectomy\b", "radical"),
]


def _infer_laterality_near(text, span_start, span_end):
    """Look for left/right/bilateral in a local window around a mastectomy mention."""
    ctx = window_around(text, span_start, span_end, 80).lower()

    has_left = re.search(r"\bleft\b", ctx) is not None
    has_right = re.search(r"\bright\b", ctx) is not None

    if has_left and has_right:
        return "bilateral"
    if re.search(r"\bbilateral\b", ctx):
        return "bilateral"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _infer_mastectomy_type(ctx):
    """Map detailed phrases to a coarse mastectomy type label."""
    for pat, label in MASTECTOMY_TYPE_PATTERNS:
        if re.search(pat, ctx, re.IGNORECASE):
            return label
    return None


def extract_mastectomy_features(note: SectionedNote) -> List[Candidate]:
    """
    Emit candidates for:
      - Mastectomy_Laterality
      - Mastectomy_Type
      - Mastectomy_Performed (True when clearly done)

    We use classify_status for context; op notes default to 'performed' if not negated/planned.
    """
    cands = []  # type: List[Candidate]

    for section, text in note.sections.items():
        for m in MASTECTOMY_CORE_RX.finditer(text):
            status = classify_status(
                text, m.start(), m.end(),
                PERFORMED_CUES, PLANNED_CUES, NEGATION_CUES
            )

            # Op note default – if mastectomy appears and not negated/planned, assume performed
            if note.note_type == "op_note" and status not in {"denied", "planned"}:
                status = "performed"

            ctx = window_around(text, m.start(), m.end(), 140)

            # Laterality
            lat_val = _infer_laterality_near(text, m.start(), m.end())
            if lat_val:
                cands.append(Candidate(
                    field="Mastectomy_Laterality",
                    value=lat_val,
                    status=status,
                    evidence=ctx,
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.75 if note.note_type == "op_note" else 0.6
                ))

            # Type
            mtype = _infer_mastectomy_type(ctx)
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
                    confidence=0.75 if note.note_type == "op_note" else 0.6
                ))

            # Performed flag (only if not clearly denied)
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
                    confidence=0.8 if note.note_type == "op_note" else 0.6
                ))

            # For now we allow multiple mastectomy mentions per section
            # (e.g., "left and right mastectomy"); aggregation will de-duplicate later.

    return cands
