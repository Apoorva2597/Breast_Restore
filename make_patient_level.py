#!/usr/bin/env python3
"""
Phase 3: aggregate Phase 2 note-level fields into a single
patient-level dataset.

Input : all_phase2_final.csv (one row per note_id + field)
Output: patient_level_fields.csv (one row per patient_id,
        one column per field name)

Assumptions:
- There is a column 'patient_id' in the Phase 2 file.
- 'field' and 'value' columns are present (from Phase 1/2).
- Python 3.6 compatible.
"""

import csv
from collections import defaultdict

INPUT = "all_phase2_final.csv"
OUTPUT = "patient_level_fields.csv"

PATIENT_COL = "patient_id"
FIELD_COL = "field"
VALUE_COL = "value"
STATUS_COL = "status"
CONF_COL = "confidence"

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
    Composite score to pick best candidate for a (patient_id, field):

      1st: status rank (performed > history > planned > denied > unknown)
      2nd: confidence (higher is better)
    """
    status = (row.get(STATUS_COL) or "").strip().lower()
    conf = _safe_float(row.get(CONF_COL))
    return (STATUS_RANK.get(status, 0), conf)


def _resolve_patient_field(candidates):
    """
    Given all candidate rows for the same (patient_id, field),
    return the single "best" row.
    """
    best = None
    best_score = None

    for row in candidates:
        s = _score(row)
        if best is None or s > best_score:
            best = row
            best_score = s

    return best


def main():
    # ---------- 1) Read Phase 2 file ----------
    with open(INPUT, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows found in {}. Nothing to do.".format(INPUT))
        return

    if PATIENT_COL not in rows[0]:
        print(
            "ERROR: '{}' column not found in {}. "
            "Make sure Phase 1/2 wrote patient_id.".format(PATIENT_COL, INPUT)
        )
        return

    # ---------- 2) Group by (patient_id, field) ----------
    groups = defaultdict(list)
    for row in rows:
        pid = row.get(PATIENT_COL)
        field = row.get(FIELD_COL)
        if not pid or not field:
            continue
        groups[(pid, field)].append(row)

    # ---------- 3) Resolve each group ----------
    resolved = []  # list of best rows
    for key, cand_list in groups.items():
        best_row = _resolve_patient_field(cand_list)
        resolved.append(best_row)

    # ---------- 4) Pivot to patient-level wide table ----------
    # Collect all distinct field names
    field_names = set()
    for r in resolved:
        field_names.add(r.get(FIELD_COL))

    # Build per-patient dict
    per_patient = {}  # pid -> { 'patient_id': pid, field1: value1, ... }

    for r in resolved:
        pid = r.get(PATIENT_COL)
        field = r.get(FIELD_COL)
        value = r.get(VALUE_COL)
        if not pid or not field:
            continue

        # Initialize row if needed
        if pid not in per_patient:
            per_patient[pid] = {PATIENT_COL: pid}

        per_patient[pid][field] = value

    # ---------- 5) Write output ----------
    # Ordered columns: patient_id, then sorted fields
    sorted_fields = sorted(field_names)
    out_cols = [PATIENT_COL] + sorted_fields

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        for pid, row in per_patient.items():
            # Ensure all fields exist (missing -> empty string)
            out_row = {}
            for col in out_cols:
                out_row[col] = row.get(col, "")
            writer.writerow(out_row)

    print(
        "Phase 3: read {} Phase 2 rows, produced {} patients → {}".format(
            len(rows), len(per_patient), OUTPUT
        )
    )


if __name__ == "__main__":
    main()
