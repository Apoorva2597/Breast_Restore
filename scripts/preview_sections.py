import argparse
from pathlib import Path

from breast_restore.ingest.csv_notes import load_notes_from_csv
from breast_restore.sectioning.sectionizer import sectionize


def preview_sections(
    csv_path: str,
    note_source: str = "unknown",
    limit: int = 5,
    min_short_chars: int = 30,
) -> None:
    """
    Load notes from a CSV, run the sectionizer, and print a PHI-safe summary:

      - NOTE_ID
      - source kind
      - total note length (chars)
      - section names
      - section lengths (chars)

    No raw note text or identifiers beyond NOTE_ID/encrypted IDs are printed.
    """
    notes = load_notes_from_csv(
        csv_path,
        note_source=note_source,
        min_short_chars=min_short_chars,
    )

    total_notes = len(notes)
    print(f"[preview_sections] Loaded {total_notes} notes from {Path(csv_path).name}")
    print(f"[preview_sections] Showing first {min(limit, total_notes)} notes\n")

    for idx, doc in enumerate(notes[:limit], start=1):
        sections = sectionize(doc.text)

        # Basic summary stats
        total_chars = len(doc.text)
        n_sections = len(sections)

        print(f"--- Note {idx} / {limit} ---")
        print(f"NOTE_ID        : {doc.note_id}")
        print(f"source_kind    : {doc.metadata.get('source_kind', note_source)}")
        print(f"note_type_raw  : {doc.metadata.get('note_type_raw', 'NA')}")
        print(f"note_date_raw  : {doc.metadata.get('note_date_raw', 'NA')}")
        print(f"total_chars    : {total_chars}")
        print(f"n_sections     : {n_sections}")

        # List section names + lengths only (no text)
        print("sections:")
        for name, body in sections.items():
            print(f"  - {name} (chars={len(body)})")

        print("")  # blank line between notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview sectionization for a CSV notes file (PHI-safe)."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the CSV file (e.g., HPI11526 Operation Notes.csv)",
    )
    parser.add_argument(
        "--note_source",
        default="unknown",
        help='Source kind label, e.g. "operation", "clinic", "inpatient".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of notes to preview.",
    )
    parser.add_argument(
        "--min_short_chars",
        type=int,
        default=30,
        help="Threshold (in chars) below which notes are counted as extra-short (ingest still keeps them).",
    )

    args = parser.parse_args()
    preview_sections(
        csv_path=args.csv_path,
        note_source=args.note_source,
        limit=args.limit,
        min_short_chars=args.min_short_chars,
    )


if __name__ == "__main__":
    main()
