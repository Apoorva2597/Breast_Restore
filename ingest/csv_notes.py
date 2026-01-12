from pathlib import Path
from typing import Dict, List, Optional, Union
import re

import pandas as pd

from models import NoteDocument


# We intentionally ONLY map encrypted IDs here.
# Do NOT add MRN or any direct PHI identifiers to this map.
DEFAULT_COLMAP: Dict[str, str] = {
    "note_id": "NOTE_ID",
    "patient_id": "ENCRYPTED_PAT_ID",      # pseudonymous
    "encounter_id": "PAT_ENC_CSN_ID",      # or ENCRYPTED_CSN if that's the field name
    "note_type": "NOTE_TYPE",
    "note_date": "NOTE_DATE_OF_SERVICE",   # or OPERATION_DATE / ADMIT_DATE, etc.
    "line": "LINE",
    "text": "NOTE_TEXT",
}


def clean_encoding_artifacts(text: str) -> str:
    """
    Clean cp1252/Windows-1252 artifacts that appear as <95>, <B0>, etc.
    Operates only on placeholder tokens, does not touch clinical content semantics.
    """
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
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text


def insert_heading_newlines(text: str) -> str:
    """
    Insert line breaks before and after known headings so that the sectionizer
    can see them even when the original CSV flattened everything onto one line.

    This does NOT change clinical content – it only adds '\n' around headings
    that you have confirmed exist in Breast RESTORE notes.
    """
    if not text:
        return text

    # Headings actually observed in Breast RESTORE data (clinic + inpatient + op).
    # We include both ALL-CAPS and Title Case variants where you reported them.
    headings = [
        # Clinic / outpatient
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
        # Physical exam appears in both clinic + inpatient
        # so we reuse "Physical Exam:" / "PHYSICAL EXAM:" above.

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
        # Allow extra spaces immediately before the heading.
        # We replace that whitespace + the heading with:
        #   '\nHEADING:\n'
        pattern = r"\s*" + re.escape(h)
        replacement = "\n" + h + "\n"
        text = re.sub(pattern, replacement, text)

    return text


def load_notes_from_csv(
    path: Union[str, Path],
    *,
    note_source: str = "unknown",  # e.g., "operation", "clinic", "inpatient"
    colmap: Optional[Dict[str, str]] = None,
    min_short_chars: int = 30,
) -> List[NoteDocument]:
    """
    Read a Breast RESTORE CSV file and reconstruct full notes by NOTE_ID.

    PHI boundary:
      - Raw CSV may contain MRN and other identifiers.
      - This function intentionally ONLY propagates encrypted IDs
        (ENCRYPTED_PAT_ID, encrypted encounter ID) into NoteDocument.metadata.
      - MRN and similar identifiers are NEVER stored in NoteDocument.

    Logic:
      - Group by NOTE_ID
      - Sort by LINE
      - Concatenate NOTE_TEXT into a single string per note
      - Clean encoding artifacts
      - Insert line breaks around known headings so the sectionizer can work
      - Log count of very short notes (< min_short_chars) but do NOT drop them.
    """
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

    # Read full CSV for now; we can move to chunked processing later if needed
    df = pd.read_csv(str(csv_path), encoding="cp1252")

    # Drop rows without NOTE_ID; NOTE_TEXT missing becomes empty string
    df = df.dropna(subset=[note_id_col])
    if text_col not in df.columns:
        raise ValueError(
            "Expected text column '{}' not found in {}".format(text_col, csv_path)
        )
    df[text_col] = df[text_col].fillna("")

    if line_col not in df.columns:
        raise ValueError(
            "Expected line column '{}' not found in {}".format(line_col, csv_path)
        )
    df = df.sort_values([note_id_col, line_col])

    notes: List[NoteDocument] = []
    short_count = 0

    for note_id, group in df.groupby(note_id_col):
        first = group.iloc[0]

        parts = [t for t in group[text_col].tolist() if isinstance(t, str)]
        raw_text = "\n".join(parts).strip()

        # 1) Clean encoding artifacts
        clean_text = clean_encoding_artifacts(raw_text).strip()
        # 2) Restore structure by putting headings onto their own lines
        clean_text = insert_heading_newlines(clean_text)

        if len(clean_text) < min_short_chars:
            short_count += 1

        # Metadata is strictly pseudonymous here
        metadata = {
            "source_path": str(csv_path),
            "source_kind": note_source,
        }  # type: Dict[str, object]

        if patient_col and patient_col in group.columns:
            metadata["patient_id"] = first[patient_col]          # encrypted
        if enc_col and enc_col in group.columns:
            metadata["encounter_id"] = first[enc_col]            # encrypted
        if type_col and type_col in group.columns:
            metadata["note_type_raw"] = first[type_col]
        if date_col and date_col in group.columns:
            metadata["note_date_raw"] = first[date_col]

        notes.append(
            NoteDocument(
                note_id=str(note_id),
                text=clean_text,
                metadata=metadata,
            )
        )

    total = len(notes)
    # Log only counts, not IDs or text → avoids leaking PHI in logs
    print(
        "[csv_notes] Loaded {} notes from {} (short notes with < {} chars: {})".format(
            total, csv_path.name, min_short_chars, short_count
        )
    )

    return notes
