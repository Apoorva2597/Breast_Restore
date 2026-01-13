#!/usr/bin/env python3
"""
Phase 2: resolve Phase 1 candidates into a single "best" value per
(note_id, field).

Input  : one or more *_phase1_candidates.csv files
Output : one CSV with final fields (one row per note_id + field)

Compatible with Python 3.6.x.

NOTE:
- If Phase 1 included 'patient_id' (encrypted), it is preserved here
  and written as a column in the output.
"""

import argparse
import csv
from collections import defaultdict

# Higher is better
STATUS_RANK = {
    "performed": 4,
    "history": 3,
    "planned": 2,
    "denied": 1,
    "unknown": 0,
    "": 0,
}


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _score(row):
    """
    Composite score used to pick the "best" candidate in a group.

      1st: status rank (performed > history > planned > denied > unknown)
      2nd: confidence (higher is better)
    """
    status = (row.get("status") or "").strip().lower()
    conf = _safe_float(row.get("confidence"))
    return (STATUS_RANK.get(status, 0), conf)


def _resolve_group(candidates):
    """
    Given a list of candidate rows (dicts) for the same (note_id, field),
    pick the single best one and tag it with a rule label.

    All other columns (e.g., patient_id, note_type, note_date, section)
    are preserved from the winning row.
    """
    best = None
    best_score = None

    for row in candidates:
        s = _score(row)
        if best is None or s > best_score:
            best = row
            best_score = s

    # Copy so we don't mutate the original dict
    final_row = dict(best)
    final_row["rule"] = "phase2_max_status_conf"
    return final_row


def run_phase2(input_paths, output_path):
    # ---------- 1) Read all input rows ----------
    all_rows = []

    for path in input_paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Phase 1 should have at least:
                # patient_id (optional but preferred),
                # note_id, note_type, note_date, field, value, status,
                # section, confidence, evidence
                all_rows.append(row)

    if not all_rows:
        print("No rows found in Phase 1 inputs; nothing to do.")
        return

    # ---------- 2) Group by (note_id, field) ----------
    groups = defaultdict(list)
    for row in all_rows:
        note_id = row.get("note_id")
        field = row.get("field")
        if note_id is None or field is None:
            # Skip malformed rows
            continue
        groups[(note_id, field)].append(row)

    # ---------- 3) Resolve each group ----------
    finals = []
    for key, cand_list in groups.items():
        final_row = _resolve_group(cand_list)
        finals.append(final_row)

    # ---------- 4) Write output ----------
    # Collect all possible column names, then impose a stable order
    all_keys = set()
    for r in finals:
        all_keys.update(r.keys())

    # Preferred column order (others go at the end)
    preferred = [
        "patient_id",   # encrypted; may or may not be present
        "note_id",
        "note_type",
        "note_date",
        "field",
        "value",
        "status",
        "section",
        "confidence",
        "evidence",
        "rule",
    ]
    fieldnames = []
    for k in preferred:
        if k in all_keys:
            fieldnames.append(k)
    for k in sorted(all_keys):
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in finals:
            writer.writerow(row)

    print(
        "Phase 2: read {} Phase 1 rows, produced {} final rows → {}".format(
            len(all_rows), len(finals), output_path
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resolve Phase 1 candidates into final fields "
            "(one row per note_id + field)."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more *_phase1_candidates.csv files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for final resolved fields.",
    )

    args = parser.parse_args()
    run_phase2(args.input, args.output)


if __name__ == "__main__":
    main()
