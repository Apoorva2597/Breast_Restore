#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch export de-identified note bundles for a list of patient_ids.

Inputs:
  - CSV with a column named 'patient_id'
Outputs (under OUT_DIR):
  - One folder per patient (whatever the exporter creates)
  - logs/<patient_id>.out.txt
  - logs/<patient_id>.err.txt
  - FAILED_PATIENT_IDS.txt
  - FAILED_PATIENT_IDS_with_rc.csv

Requires:
  export_deid_clinic_notes_fulltext.py to exist and support:
    --patient_id <PID>
    --out_dir <OUT_DIR>

NOTE:
  Do NOT pass --data-dir unless the exporter explicitly supports it.
"""

import os
import sys
import subprocess
import pandas as pd

BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"
EXPORTER = os.path.join(BREAST_RESTORE_DIR, "export_deid_clinic_notes_fulltext.py")
OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "QA_DEID_BUNDLES")
LOG_DIR = os.path.join(OUT_DIR, "logs")


def read_patient_ids(csv_path: str):
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    if "patient_id" not in df.columns:
        raise ValueError(f"Expected column 'patient_id' in {csv_path}. Found: {list(df.columns)}")
    pids = df["patient_id"].fillna("").astype(str).str.strip()
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
    Calls the exporter for a single patient_id using the exporter's actual CLI.
    """
    cmd = [
        sys.executable,
        EXPORTER,
        "--patient_id", pid,
        "--out_dir", OUT_DIR,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,  # python3.6 compatible
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{pid}.out.txt")
    err_path = os.path.join(LOG_DIR, f"{pid}.err.txt")

    with open(out_path, "w") as f:
        f.write(proc.stdout or "")
    with open(err_path, "w") as f:
        f.write(proc.stderr or "")

    # Print minimal progress to console
    if proc.returncode != 0:
        print(f"  FAIL rc={proc.returncode}  (see {err_path})", file=sys.stderr)
    else:
        print(f"  OK   (see {out_path})")

    return proc.returncode


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/patient_list.csv", file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]
    if not os.path.exists(EXPORTER):
        print(f"ERROR: exporter not found: {EXPORTER}", file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUT_DIR, exist_ok=True)

    pids = read_patient_ids(csv_path)
    print(f"Patients to export: {len(pids)}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"EXPORTER: {EXPORTER}")

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
            failures.append((pid, rc))

    print("\n==== SUMMARY ====")
    print(f"Success: {ok}")
    print(f"Failed : {bad}")

    if failures:
        fail_txt = os.path.join(OUT_DIR, "FAILED_PATIENT_IDS.txt")
        fail_csv = os.path.join(OUT_DIR, "FAILED_PATIENT_IDS_with_rc.csv")

        with open(fail_txt, "w") as f:
            for pid, _rc in failures:
                f.write(pid + "\n")

        pd.DataFrame(failures, columns=["patient_id", "return_code"]).to_csv(fail_csv, index=False)

        print(f"Wrote failures: {fail_txt}")
        print(f"Wrote failure codes: {fail_csv}")


if __name__ == "__main__":
    main()
