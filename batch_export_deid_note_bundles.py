#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch export de-identified note bundles for a list of patient_ids.

Input:
  - A CSV that contains a column named: patient_id

Output:
  - One folder per patient under OUT_DIR (created if missing)
  - Whatever your single-patient exporter writes (e.g., timeline.csv,
    ALL_NOTES_COMBINED.txt, encounters_timeline.csv, stage2_anchor_summary.txt)

Prereq:
  - export_deid_clinic_notes_fulltext.py exists and can run for ONE patient.

Notes:
  - This wrapper is compatible with Python 3.6 (uses universal_newlines=True).
"""

import os
import sys
import subprocess
import pandas as pd

# ---------- HARD-CODED PATHS (edit only these if needed) ----------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
EXPORTER = os.path.join(BREAST_RESTORE_DIR, "export_deid_clinic_notes_fulltext.py")

# Where your raw de-id note CSVs live (the ones the exporter reads).
# If the exporter already hardcodes/ignores this, it won't hurt to pass it.
DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

# Output directory for QA bundles
OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "QA_DEID_BUNDLES")
# ---------------------------------------------------------------


def read_patient_ids(csv_path: str):
    """Read patient_id values from csv_path, return unique IDs preserving order."""
    df = pd.read_csv(csv_path, dtype=str, engine="python")

    if "patient_id" not in df.columns:
        raise ValueError(
            "Expected column 'patient_id' in {}. Found columns: {}".format(
                csv_path, list(df.columns)
            )
        )

    raw = (
        df["patient_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Unique while preserving order
    seen = set()
    out = []
    for pid in raw:
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)

    return out


def run_one(pid: str) -> int:
    """
    Run the single-patient exporter for one patient_id.

    Assumes exporter supports:
      python export_deid_clinic_notes_fulltext.py --patient-id <PID> --out-dir <OUT_DIR> --data-dir <DATA_DIR>

    If your exporter uses different args, change cmd below.
    """
    cmd = [
        sys.executable,
        EXPORTER,
        "--patient-id", pid,
        "--out-dir", OUT_DIR,
        "--data-dir", DATA_DIR,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # Python 3.6-compatible replacement for text=True
    )

    # Stream output for visibility
    if proc.stdout:
        print(proc.stdout.rstrip())

    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)

    return proc.returncode


def main():
    if len(sys.argv) != 2:
        print("Usage: {} /path/to/patient_list.csv".format(sys.argv[0]), file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]

    if not os.path.exists(EXPORTER):
        print("ERROR: Exporter not found at: {}".format(EXPORTER), file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)

    pids = read_patient_ids(csv_path)
    print("Patients to export: {}".format(len(pids)))
    print("OUT_DIR: {}".format(OUT_DIR))

    ok = 0
    bad = 0
    failures = []

    for i, pid in enumerate(pids, 1):
        print("\n[{}/{}] Exporting: {}".format(i, len(pids), pid))
        rc = run_one(pid)
        if rc == 0:
            ok += 1
        else:
            bad += 1
            failures.append(pid)

    print("\n==== SUMMARY ====")
    print("Success: {}".format(ok))
    print("Failed : {}".format(bad))

    if failures:
        fail_path = os.path.join(OUT_DIR, "FAILED_PATIENT_IDS.txt")
        with open(fail_path, "w") as f:
            for pid in failures:
                f.write(pid + "\n")
        print("Wrote failures: {}".format(fail_path))


if __name__ == "__main__":
    main()
