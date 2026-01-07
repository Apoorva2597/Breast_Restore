#!/usr/bin/env python
# run_diagnostics.py
#
# PHI-safe diagnostics for NOTES CSVs.
# - Works on: Operation, Clinic, Inpatient Notes
# - Uses chunked reading (no full-table load)
# - NEVER prints NOTE_TEXT or identifiers, only counts & summary stats.

from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import math
import argparse


# -------------------------------------------------------------------
# Config: filenames + anchor date column per note type
# -------------------------------------------------------------------

NOTE_FILE_CONFIGS = {
    "operation": {
        "filename": "HPI11526 Operation Notes.csv",
        "anchor_col": "OPERATION_DATE",
    },
    "clinic": {
        "filename": "HPI11526 Clinic Notes.csv",
        "anchor_col": "ADMIT_DATE",
    },
    "inpatient": {
        "filename": "HPI11526 Inpatient Notes.csv",
        "anchor_col": "HOSP_ADMSN_TIME",
    },
}

# Columns we always need from the notes tables
BASE_USECOLS = [
    "ENCRYPTED_PAT_ID",
    "ENCRYPTED_CSN",
    "NOTE_ID",
    "LINE",
    "NOTE_TEXT",
    "NOTE_TYPE",
    "NOTE_DATE_OF_SERVICE",
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def safe_pct(numer, denom):
    """Return percentage safely, avoiding divide-by-zero."""
    return 0.0 if denom == 0 else (float(numer) / float(denom)) * 100.0


def percentile_from_sorted(arr, p):
    """
    Given a sorted list of ints, return the p-th percentile (0–100).
    """
    if not arr:
        return None
    if p <= 0:
        return arr[0]
    if p >= 100:
        return arr[-1]

    k = (len(arr) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return arr[f]

    d0 = arr[f] * (c - k)
    d1 = arr[c] * (k - f)
    return int(round(d0 + d1))


# -------------------------------------------------------------------
# Core diagnostics for a single NOTES file
# -------------------------------------------------------------------

def diagnose_one_notes_file(data_dir, out_dir, kind, chunksize=200000):
    """
    Run PHI-safe diagnostics for a single notes CSV.

    kind: "operation" | "clinic" | "inpatient"
    """
    cfg = NOTE_FILE_CONFIGS[kind]
    csv_path = Path(data_dir) / cfg["filename"]
    anchor_col = cfg["anchor_col"]

    if not csv_path.exists():
        raise FileNotFoundError("Missing file: {}".format(csv_path))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global counters
    total_rows = 0
    missing_note_text = 0
    missing_note_type = 0
    missing_note_date = 0
    missing_anchor = 0

    # Unique IDs (in memory; should be manageable)
    unique_pat = set()
    unique_csn = set()
    unique_note = set()

    # Per-note accumulators
    note_type_counts = Counter()
    note_lines = defaultdict(int)     # NOTE_ID -> line count
    note_lengths = defaultdict(int)   # NOTE_ID -> total char length

    usecols = BASE_USECOLS + [anchor_col]

    # Stream in chunks
    for chunk in pd.read_csv(
    csv_path,
    usecols=usecols,
    chunksize=chunksize,
    encoding="cp1252",
    errors="replace"     # replaces bad bytes instead of erroring
):

        # Missingness (PHI-safe)
        missing_note_text += int(chunk["NOTE_TEXT"].isna().sum())
        missing_note_type += int(chunk["NOTE_TYPE"].isna().sum())
        missing_note_date += int(chunk["NOTE_DATE_OF_SERVICE"].isna().sum())
        missing_anchor += int(chunk[anchor_col].isna().sum())

        # Unique IDs
        unique_pat.update(chunk["ENCRYPTED_PAT_ID"].dropna().astype(str).unique())
        unique_csn.update(chunk["ENCRYPTED_CSN"].dropna().astype(str).unique())
        unique_note.update(chunk["NOTE_ID"].dropna().astype(str).unique())

        # NOTE_TYPE distribution (counts only)
        note_type_counts.update(chunk["NOTE_TYPE"].dropna().astype(str).tolist())

        # Lines per note: count rows per NOTE_ID in this chunk
        lines_per_note_chunk = chunk.groupby("NOTE_ID").size()
        for note_id, cnt in lines_per_note_chunk.items():
            if pd.isna(note_id):
                continue
            note_lines[str(note_id)] += int(cnt)

        # Note length per note: sum of text length in this chunk
        text_lens = chunk["NOTE_TEXT"].fillna("").astype(str).str.len()
        lens_per_note_chunk = text_lens.groupby(chunk["NOTE_ID"]).sum()
        for note_id, nchar in lens_per_note_chunk.items():
            if pd.isna(note_id):
                continue
            note_lengths[str(note_id)] += int(nchar)

    # Convert accumulators to sorted lists for stats
    lines_list = sorted(note_lines.values())
    lens_list = sorted(note_lengths.values())

    lines_stats = {}
    if lines_list:
        lines_stats = {
            "min": lines_list[0],
            "median": percentile_from_sorted(lines_list, 50),
            "p90": percentile_from_sorted(lines_list, 90),
            "max": lines_list[-1],
            "n": len(lines_list),
        }

    len_stats = {}
    if lens_list:
        len_stats = {
            "min": lens_list[0],
            "median": percentile_from_sorted(lens_list, 50),
            "p90": percentile_from_sorted(lens_list, 90),
            "max": lens_list[-1],
            "n": len(lens_list),
        }

    # ----------------------------------------------------------------
    # Write PHI-safe summary to text file
    # ----------------------------------------------------------------
    summary_txt = out_dir / "notes_{}_summary.txt".format(kind)
    with summary_txt.open("w") as f:
        f.write("=== NOTES DIAGNOSTICS: {} ===\n".format(kind.upper()))
        f.write("File: {}\n".format(csv_path))
        f.write("Anchor date column: {}\n\n".format(anchor_col))

        f.write("--- Basic counts ---\n")
        f.write("Total rows: {}\n".format(total_rows))
        f.write("Unique patients (ENCRYPTED_PAT_ID): {}\n".format(len(unique_pat)))
        f.write("Unique encounters (ENCRYPTED_CSN): {}\n".format(len(unique_csn)))
        f.write("Unique notes (NOTE_ID): {}\n\n".format(len(unique_note)))

        f.write("--- Missingness (%) ---\n")
        f.write("NOTE_TEXT missing: {:.2f}%\n".format(safe_pct(missing_note_text, total_rows)))
        f.write("NOTE_TYPE missing: {:.2f}%\n".format(safe_pct(missing_note_type, total_rows)))
        f.write("NOTE_DATE_OF_SERVICE missing: {:.2f}%\n".format(safe_pct(missing_note_date, total_rows)))
        f.write("{} missing: {:.2f}%\n\n".format(anchor_col, safe_pct(missing_anchor, total_rows)))

        f.write("--- Lines per note (segments per NOTE_ID) ---\n")
        if lines_stats:
            f.write(
                "min/median/p90/max: {}/{}/{}/{}\n".format(
                    lines_stats["min"],
                    lines_stats["median"],
                    lines_stats["p90"],
                    lines_stats["max"],
                )
            )
        else:
            f.write("No line stats available.\n")

        f.write("\n--- Reconstructed note length (characters; sum of line lengths) ---\n")
        if len_stats:
            f.write(
                "min/median/p90/max: {}/{}/{}/{}\n".format(
                    len_stats["min"],
                    len_stats["median"],
                    len_stats["p90"],
                    len_stats["max"],
                )
            )
        else:
            f.write("No length stats available.\n")

        f.write("\n--- NOTE_TYPE top 20 (counts) ---\n")
        for note_type, cnt in note_type_counts.most_common(20):
            f.write("{}: {}\n".format(note_type, cnt))

    # Also write NOTE_TYPE counts to CSV (still PHI-safe)
    types_csv = out_dir / "notes_{}_note_type_counts.csv".format(kind)
    pd.DataFrame(
        note_type_counts.most_common(),
        columns=["NOTE_TYPE", "COUNT"],
    ).to_csv(types_csv, index=False)

    return summary_txt


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PHI-safe diagnostics for Operation/Clinic/Inpatient notes CSVs."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the HPI11526 * Notes.csv files",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write diagnostics outputs",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Rows per chunk when reading CSV (tune for memory).",
    )
    parser.add_argument(
        "--which",
        choices=["operation", "clinic", "inpatient", "all"],
        default="all",
        help="Which notes file to diagnose.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.which == "all":
        targets = ["operation", "clinic", "inpatient"]
    else:
        targets = [args.which]

    for kind in targets:
        print("[run_diagnostics] Running notes diagnostics for: {}".format(kind))
        summary_path = diagnose_one_notes_file(data_dir, out_dir, kind, args.chunksize)
        print("[run_diagnostics] Wrote summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
