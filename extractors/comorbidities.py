# extractors/comorbidities.py

import re
from typing import List
from models import Candidate, SectionedNote
from .utils import window_around, classify_status

NEG_SECTIONS = {
    "FAMILY HISTORY",
    "ALLERGIES",
    "REVIEW OF SYSTEMS"
}

# -------------------------
# Diabetes
# -------------------------

DM_POS = [
    r"\bdiabetes\b",
    r"\bdiabetes mellitus\b",
    r"\btype\s*[12]\s*diabetes\b",
    r"\bIDDM\b",
    r"\bNIDDM\b"
]

DM_MEDS = [
    r"\binsulin\b",
    r"\blantus\b",
    r"\bnovolog\b",
]

DM_EXCLUDE = [
    r"\bgestational\b",
    r"\bdiabetes insipidus\b",
    r"\bprediabetes\b",
]

# -------------------------
# Hypertension
# -------------------------

HTN_POS = [
    r"\bhypertension\b",
    r"\bHTN\b"
]

HTN_EXCLUDE = [
    r"\bgestational\b",
    r"\bpulmonary hypertension\b",
    r"\bportal hypertension\b",
]

# -------------------------
# Cardiac disease
# -------------------------

CARDIAC_POS = [
    r"\bcoronary artery disease\b",
    r"\bCAD\b",
    r"\bcongestive heart failure\b",
    r"\bCHF\b",
    r"\bmyocardial infarction\b",
    r"\bprior MI\b"
]

# -------------------------
# VTE
# -------------------------

VTE_POS = [
    r"\bdeep vein thrombosis\b",
    r"\bDVT\b",
    r"\bpulmonary embolism\b",
    r"\bPE\b"
]

VTE_EXCLUDE = [
    r"\bprophylaxis\b",
    r"\bppx\b",
]

# -------------------------
# Steroids
# -------------------------

STEROID_POS = [
    r"\bprednisone\b",
    r"\bdexamethasone\b",
    r"\bmethylprednisolone\b",
]

STEROID_EXCLUDE = [
    r"\binhaled\b",
    r"\btopical\b"
]


def extract_comorbidities(note: SectionedNote) -> List[Candidate]:

    cands = []

    for section, text in note.sections.items():

        if section in NEG_SECTIONS:
            continue

        lower = text.lower()

        # -------------------------
        # Diabetes
        # -------------------------

        if any(re.search(p, lower) for p in DM_POS):

            if not any(re.search(p, lower) for p in DM_EXCLUDE):

                cands.append(
                    Candidate(
                        field="Diabetes",
                        value=True,
                        status="history",
                        evidence=window_around(text, 0, 100, 200),
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.85
                    )
                )

        # medication inference

        if any(re.search(p, lower) for p in DM_MEDS):

            cands.append(
                Candidate(
                    field="Diabetes",
                    value=True,
                    status="history",
                    evidence=text[:200],
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.80
                )
            )

        # -------------------------
        # Hypertension
        # -------------------------

        if any(re.search(p, lower) for p in HTN_POS):

            if not any(re.search(p, lower) for p in HTN_EXCLUDE):

                cands.append(
                    Candidate(
                        field="Hypertension",
                        value=True,
                        status="history",
                        evidence=text[:200],
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.85
                    )
                )

        # -------------------------
        # Cardiac disease
        # -------------------------

        if any(re.search(p, lower) for p in CARDIAC_POS):

            cands.append(
                Candidate(
                    field="CardiacDisease",
                    value=True,
                    status="history",
                    evidence=text[:200],
                    section=section,
                    note_type=note.note_type,
                    note_id=note.note_id,
                    note_date=note.note_date,
                    confidence=0.85
                )
            )

        # -------------------------
        # VTE
        # -------------------------

        if any(re.search(p, lower) for p in VTE_POS):

            if not any(re.search(p, lower) for p in VTE_EXCLUDE):

                cands.append(
                    Candidate(
                        field="VTE",
                        value=True,
                        status="history",
                        evidence=text[:200],
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.85
                    )
                )

        # -------------------------
        # Steroids
        # -------------------------

        if any(re.search(p, lower) for p in STEROID_POS):

            if not any(re.search(p, lower) for p in STEROID_EXCLUDE):

                cands.append(
                    Candidate(
                        field="SteroidUse",
                        value=True,
                        status="current",
                        evidence=text[:200],
                        section=section,
                        note_type=note.note_type,
                        note_id=note.note_id,
                        note_date=note.note_date,
                        confidence=0.80
                    )
                )

    return cands
