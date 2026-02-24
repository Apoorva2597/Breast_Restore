#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch export de-identified note bundles for a list of patient_ids.

Inputs:
  - CSV with a column named 'patient_id'
Outputs:
  - One folder per patient under OUT_DIR
  - timeline.csv
  - ALL_NOTES_COMBINED.txt (de-identified)
  - encounters_timeline.csv (if your exporter produces it)
  - stage2_anchor_summary.txt (if your exporter produces it)

You MUST already have your working single-patient exporter script:
  export_deid_clinic_notes_fulltext.py

This wrapper runs it repeatedly and logs successes/failures.
"""

import os
import sys
import subprocess
import pandas as pd

# ---- HARD-CODED PATHS (edit only these if needed) ----
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
EXPORTER = os.path.join(BREAST_RESTORE_DIR, "export_deid_clinic_notes_fulltext.py")

# Where your raw de-id note CSVs live (the ones the exporter reads)
# If your exporter already hardcodes these internally, leave these alone.
DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

# Output directory for QA bundles
OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "QA_DEID_BUNDLES")

# -----------------------------------------------------

def read_patient_ids(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    if "patient_id" not in df.columns:
        raise ValueError(f"Expected column 'patient_id' in {csv_path}. Found: {list(df.columns)}")
    pids = (
        df["patient_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    pids = [p for p in pids if p]
    # unique while preserving order
    seen = set()
    out = []
    for p in pids:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def run_one(pid: str) -> int:
    """
    Calls the exporter for a single patient_id.

    Assumes exporter supports:
      python export_deid_clinic_notes_fulltext.py --patient-id <PID> --out-dir <OUT_DIR> --data-dir <DATA_DIR>

    If your exporter uses different args, adjust the cmd below.
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
    universal_newlines=True
)
    # Stream output for logging visibility
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/patient_list.csv", file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)

    pids = read_patient_ids(csv_path)
    print(f"Patients to export: {len(pids)}")
    print(f"OUT_DIR: {OUT_DIR}")

    ok = 0
    bad = 0
    failures = []

    for i, pid in enumerate(pids, 1):
        print(f"\n[{i}/{len(pids)}] Exporting: {pid}")
        rc = run_one(pid)
        if rc == 0:
            ok += 1
        else:
            bad += 1
            failures.append(pid)

    print("\n==== SUMMARY ====")
    print(f"Success: {ok}")
    print(f"Failed : {bad}")

    if failures:
        fail_path = os.path.join(OUT_DIR, "FAILED_PATIENT_IDS.txt")
        with open(fail_path, "w") as f:
            for pid in failures:
                f.write(pid + "\n")
        print(f"Wrote failures: {fail_path}")

if __name__ == "__main__":
    main()
