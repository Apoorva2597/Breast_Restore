from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..models import NoteDocument


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


def load_notes_from_csv(
    path: str | Path,
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
        raise ValueError(f"Expected text column '{text_col}' not found in {csv_path}")
    df[text_col] = df[text_col].fillna("")

    if line_col not in df.columns:
        raise ValueError(f"Expected line column '{line_col}' not found in {csv_path}")
    df = df.sort_values([note_id_col, line_col])

    notes: List[NoteDocument] = []
    short_count = 0

    for note_id, group in df.groupby(note_id_col):
        first = group.iloc[0]

        parts = [t for t in group[text_col].tolist() if isinstance(t, str)]
        raw_text = "\n".join(parts).strip()
        clean_text = clean_encoding_artifacts(raw_text).strip()

        if len(clean_text) < min_short_chars:
            short_count += 1

        # Metadata is strictly pseudonymous here
        metadata: Dict[str, object] = {
            "source_path": str(csv_path),
            "source_kind": note_source,
        }
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
        f"[csv_notes] Loaded {total} notes from {csv_path.name} "
        f"(short notes with < {min_short_chars} chars: {short_count})"
    )

    return notes

