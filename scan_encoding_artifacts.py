import argparse
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import re


HEX_TOKEN_RE = re.compile(r"<[0-9A-Fa-f]{2}>")


def scan_csv_for_artifacts(csv_path: Path, text_column: str = "NOTE_TEXT") -> Dict[str, int]:
    """
    Scan a single CSV for hex-like encoding artifacts such as <95>, <B0>, etc.
    Returns a dict: {token: count}.
    """
    artifacts: Dict[str, int] = {}

    try:
        df = pd.read_csv(str(csv_path), encoding="cp1252")
    except Exception as e:
        print("[scan] WARNING: failed to read {}: {}".format(csv_path, e))
        return artifacts

    if text_column not in df.columns:
        print("[scan] WARNING: {} has no '{}' column, skipping".format(csv_path, text_column))
        return artifacts

    # We only touch NOTE_TEXT; we never print any actual note content
    for val in df[text_column].dropna():
        if not isinstance(val, str):
            continue
        for match in HEX_TOKEN_RE.findall(val):
            artifacts[match] = artifacts.get(match, 0) + 1

    return artifacts


def merge_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def scan_directory(data_dir: Path, text_column: str = "NOTE_TEXT") -> Dict[str, Dict[str, int]]:
    """
    Scan all CSV files under data_dir (non-recursive) for encoding artifacts.

    Returns:
      {
        "TOTAL": {token: total_count_across_all_files},
        "<file_name_1.csv>": {token: count_in_that_file},
        ...
      }
    """
    results: Dict[str, Dict[str, int]] = {}
    total_counts: Dict[str, int] = {}

    csv_files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".csv")

    if not csv_files:
        print("[scan] No CSV files found in {}".format(data_dir))
        return {"TOTAL": {}}

    for csv_path in csv_files:
        print("[scan] Scanning {}".format(csv_path.name))
        counts = scan_csv_for_artifacts(csv_path, text_column=text_column)
        results[csv_path.name] = counts
        total_counts = merge_counts(total_counts, counts)

    results["TOTAL"] = total_counts
    return results


def write_report(
    results: Dict[str, Dict[str, int]],
    out_file: Path,
    data_dir: Path,
    text_column: str = "NOTE_TEXT",
) -> None:
    """
    Write a PHI-safe report: only tokens and counts, no note text or IDs.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        f.write("Encoding artifact scan report\n")
        f.write("Data directory: {}\n".format(data_dir))
        f.write("Text column   : {}\n".format(text_column))
        f.write("\n")

        # Total summary
        total = results.get("TOTAL", {})
        f.write("=== TOTAL COUNTS ACROSS ALL FILES ===\n")
        if not total:
            f.write("(No hex-like tokens <XX> found.)\n\n")
        else:
            for token, count in sorted(total.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write("{:<8} {}\n".format(token, count))
            f.write("\n")

        # Per-file breakdown
        for fname, counts in results.items():
            if fname == "TOTAL":
                continue
            f.write("=== {} ===\n".format(fname))
            if not counts:
                f.write("(No hex-like tokens found.)\n\n")
                continue
            for token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write("{:<8} {}\n".format(token, count))
            f.write("\n")

    print("[scan] Wrote report to {}".format(out_file))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan CSV note files for cp1252-like encoding artifacts such as <95>, <B0>, etc."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing the note CSV files (e.g., .../HPI-11526/HPI11256).",
    )
    parser.add_argument(
        "--out_file",
        default="diagnostics/encoding_artifacts_report.txt",
        help="Path to write the report file (relative or absolute).",
    )
    parser.add_argument(
        "--text_column",
        default="NOTE_TEXT",
        help="Name of the text column to scan (default: NOTE_TEXT).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print("[scan] ERROR: data_dir '{}' is not a directory".format(data_dir))
        return

    out_file = Path(args.out_file)

    results = scan_directory(data_dir, text_column=args.text_column)
    write_report(results, out_file, data_dir, text_column=args.text_column)


if __name__ == "__main__":
    main()
