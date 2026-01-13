import os
import sys
from pathlib import Path

# Make the Breast_Restore repo root importable (Python 3.6-safe)
ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

import argparse
import csv
from typing import List, Dict

from ingest.csv_notes import load_notes_from_csv
from normalize.sectionizer import sectionize
from normalize.note_type import guess_note_type
from models import SectionedNote, Candidate
from extractors import extract_all


def build_sectioned_note(doc):
    """
    Convert a NoteDocument (from ingest) into a SectionedNote
    using the sectionizer and note-type guesser.
    """
    sections = sectionize(doc.text)

    raw_type = doc.metadata.get("note_type_raw", "") or ""
    guessed_type = guess_note_type(raw_type, doc.text)
    note_date = doc.metadata.get("note_date_raw")

    return SectionedNote(
        note_id=doc.note_id,
        note_type=guessed_type,
        sections=sections,
        note_date=str(note_date) if note_date is not None else None,
    )


def write_candidates_to_csv(candidates, out_path, note_to_patient_id):
    """
    Write candidates to CSV.

    IMPORTANT:
    - Writes only structured fields and short evidence snippets.
    - Includes patient_id (encrypted) derived from ingest metadata.
    - No full note text or MRN is written.
    """
    fieldnames = [
        "patient_id",
        "note_id",
        "note_type",
        "note_date",
        "field",
        "value",
        "status",
        "section",
        "confidence",
        "evidence",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:  # type: Candidate
            writer.writerow(
                {
                    "patient_id": note_to_patient_id.get(c.note_id, ""),
                    "note_id": c.note_id,
                    "note_type": c.note_type,
                    "note_date": c.note_date,
                    "field": c.field,
                    "value": c.value,
                    "status": c.status,
                    "section": c.section,
                    "confidence": c.confidence,
                    "evidence": c.evidence,
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 1 abstraction (BMI, Age_DOS, smoking, comorbidities, "
            "reconstruction, lymph nodes, PBS, mastectomy) from CSV notes."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Path to the notes CSV (e.g. HPI11526 Operation Notes.csv)",
    )
    parser.add_argument(
        "--note_source",
        default="unknown",
        help='Source kind label, e.g. "operation", "clinic", "inpatient".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of notes to process (0 = all).",
    )
    parser.add_argument(
        "--output",
        default="phase1_candidates.csv",
        help="Output CSV file path (default: phase1_candidates.csv).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    notes = load_notes_from_csv(
        csv_path,
        note_source=args.note_source,
        min_short_chars=10,
    )

    if args.limit and args.limit > 0:
        notes = notes[: args.limit]

    # Build note_id → patient_id map from ingest metadata
    note_to_patient_id = {}  # type: Dict[str, str]
    for doc in notes:
        pid = doc.metadata.get("patient_id")
        if pid is not None:
            note_to_patient_id[str(doc.note_id)] = str(pid)

    all_candidates = []  # type: List[Candidate]

    for doc in notes:
        sn = build_sectioned_note(doc)
        cands = extract_all(sn)
        all_candidates.extend(cands)

    out_path = Path(args.output)
    write_candidates_to_csv(all_candidates, out_path, note_to_patient_id)

    print(
        "Phase 1: wrote {} candidates from {} notes to {}".format(
            len(all_candidates), len(notes), out_path
        )
    )


if __name__ == "__main__":
    main()
