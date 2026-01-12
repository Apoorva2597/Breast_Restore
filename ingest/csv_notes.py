from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import pandas as pd

from models import NoteDocument


# -------------------------------------------------------------------
# Column map: ONLY encrypted IDs + safe fields
# -------------------------------------------------------------------
DEFAULT_COLMAP: Dict[str, str] = {
    "note_id": "NOTE_ID",
    "patient_id": "ENCRYPTED_PAT_ID",
    "encounter_id": "PAT_ENC_CSN_ID",
    "note_type": "NOTE_TYPE",
    "note_date": "NOTE_DATE_OF_SERVICE",
    "line": "LINE",
    "text": "NOTE_TEXT",
}


# -------------------------------------------------------------------
# Encoding cleanup
# -------------------------------------------------------------------
def clean_encoding_artifacts(text: str) -> str:
    if not text:
        return text

    replacements = {
        "<95>": "-",
        "<97>": "-",
        "<8D>": "-",
        "<B0>": " ",
        "<91>": "'",
        "<92>": "'",
        "<93>": '"',
        "<94>": '"',
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse triple newlines to double
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text


# -------------------------------------------------------------------
# Insert line breaks before/after known Breast RESTORE headings
# (needed because many notes are "flattened" into one line)
# -------------------------------------------------------------------
def insert_heading_newlines(text: str) -> str:
    if not text:
        return text

    headings = [
        # Clinic
        "REASON FOR VISIT:",
        "CHIEF COMPLAINT:",
        "CC:",
        "HPI:",
        "HISTORY OF PRESENT ILLNESS:",
        "INTERVAL HISTORY AND REVIEW OF SYSTEMS:",
        "ROS:",
        "REVIEW OF SYSTEMS:",
        "Past Medical History:",
        "PAST MEDICAL HISTORY:",
        "Past Surgical History:",
        "PAST SURGICAL HISTORY:",
        "FAMILY HSTORY:",
        "FAMILY HISTORY:",
        "Social History:",
        "SOCIAL HISTORY:",
        "Physical Exam:",
        "PHYSICAL EXAM:",
        "PATHOLOY:",
        "PATHOLOGY:",
        "RADIOLOGY:",
        "LABS:",
        "ASSESSMENT:",
        "ASSESSMENT AND PLAN:",
        "Assessment and Plan:",
        "ASSESSMENT/PLAN:",
        "PLAN:",
        "TREATMENT:",

        # Inpatient
        "S/P Procedures(s):",
        "Subjective:",
        "Interval History:",
        "Objective:",
        "Diagnosis:",
        "History:",
        "Assessment/Plan:",

        # Op notes
        "OP NOTE:",
        "OPERATIVE REPORT:",
        "PREOPERATIVE DIAGNOSIS:",
        "POSTOPERATIVE DIAGNOSIS:",
        "PROCEDURE:",
        "ATTENDING SURGEON:",
        "ASSISTANT:",
        "ANESTHESIA:",
        "IV FLUIDS:",
        "ESTIMATED BLOOD LOSS:",
        "URINE OUTPUT:",
        "MICRO SURGICAL DETAILS:",
        "COMPLICATIONS:",
        "CONDITION AT THE END OF THE PROCEDURE:",
        "DISPOSITION:",
        "INDICATIONS FOR OPERATION:",
        "DETAILS OF OPERATION:",
    ]

    for h in headings:
        pattern = r"\s*" + re.escape(h)
        replacement = "\n" + h + "\n"
        text = re.sub(pattern, replacement, text)

    return text


# -------------------------------------------------------------------
# Load notes from CSV
# -------------------------------------------------------------------
def load_notes_from_csv(
    path: Union[str, Path],
    *,
    note_source: str = "unknown",
    colmap: Optional[Dict[str, str]] = None,
    min_short_chars: int = 30,
) -> List[NoteDocument]:

    csv_path = Path(path)
    if colmap is None:
        colmap = DEFAULT_COLMAP

    note_id_col = colmap["note_id"]
    text_col = colmap["text"]
    line_col = colmap["line"]
    patient_col = colmap.get("patient_id")
    enc_col = colmap.get("encounter_id")
    type_col = colmap.get("note_type")
    date_col = colmap.get("note_date")

    # Load CSV
    df = pd.read_csv(str(csv_path), encoding="cp1252")

    # Basic cleaning
    df = df.dropna(subset=[note_id_col])
    df[text_col] = df[text_col].fillna("")

    if line_col not in df.columns:
        raise ValueError("Expected line column '{}' missing".format(line_col))

    df = df.sort_values([note_id_col, line_col])

    notes: List[NoteDocument] = []
    short_count = 0

    for note_id, group in df.groupby(note_id_col):
        first = group.iloc[0]

        parts = [t for t in group[text_col].tolist() if isinstance(t, str)]
        raw_text = "\n".join(parts).strip()

        # Clean + restore structure
        clean_text = clean_encoding_artifacts(raw_text).strip()
        clean_text = insert_heading_newlines(clean_text)

        if len(clean_text) < min_short_chars:
            short_count += 1

        metadata = {
            "source_path": str(csv_path),
            "source_kind": note_source,
        }

        if patient_col in group.columns:
            metadata["patient_id"] = first[patient_col]
        if enc_col in group.columns:
            metadata["encounter_id"] = first[enc_col]
        if type_col in group.columns:
            metadata["note_type_raw"] = first[type_col]
        if date_col in group.columns:
            metadata["note_date_raw"] = first[date_col]

        notes.append(
            NoteDocument(
                note_id=str(note_id),
                text=clean_text,
                metadata=metadata,
            )
        )

    print(
        "[csv_notes] Loaded {} notes from {} (short < {} chars: {})".format(
            len(notes), csv_path.name, min_short_chars, short_count
        )
    )

    return notes
