from __future__ import annotations

import argparse
from pathlib import Path

from phase1_pipeline.ingest.docx_reader import read_docx
from phase1_pipeline.normalize.sectionizer import sectionize
from phase1_pipeline.normalize.note_type import guess_note_type
from phase1_pipeline.extractors import extract_all
from phase1_pipeline.aggregate.rules import aggregate_patient
from phase1_pipeline.outputs.json_writer import write_note_json, write_patient_json
from phase1_pipeline.models import SectionedNote


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 rule-based extraction (note-level -> patient-level)."
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing DOCX notes for ONE patient (pilot).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for JSON artifacts.",
    )
    ap.add_argument(
        "--patient_id",
        default="patient_001",
        help="Non-PHI patient id for output.",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(in_dir.glob("*.docx"))
    if not docx_files:
        raise SystemExit(f"No .docx files found in {in_dir}")

    all_candidates = []

    for f in docx_files:
        # Read raw note text (NoteDocument)
        raw_note = read_docx(str(f), note_id=f.stem)

        # Sectionize -> Dict[str, str]
        sec_dict = sectionize(raw_note.text)

        # Infer note type from note_id + full text
        note_type = guess_note_type(raw_note.note_id, raw_note.text)

        # Note date (None unless you set it in metadata later)
        note_date = None

        # Build SectionedNote for extractors
        sec_note = SectionedNote(
            note_id=raw_note.note_id,
            note_type=note_type,
            sections=sec_dict,
            note_date=note_date,
        )

        # Extract + write note-level output
        cands = extract_all(sec_note)
        all_candidates.extend(cands)

        write_note_json(
            str(out_dir / f"{f.stem}.note_level.json"),
            f.stem,
            cands,
        )

    # Aggregate across all notes for the patient
    final = aggregate_patient(all_candidates)
    write_patient_json(
        str(out_dir / f"{args.patient_id}.patient_level.json"),
        args.patient_id,
        final,
    )

    print(
        f"Wrote {len(docx_files)} note-level JSON files + patient-level JSON to: {out_dir}"
    )


if __name__ == "__main__":
    main()
