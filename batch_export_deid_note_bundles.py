#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch export de-identified note bundles for a list of patient_ids
AND copy encounter/staging artifacts from PATIENT_BUNDLES into QA bundle dirs.

Expected input:
  - CSV with column: patient_id

Produces:
  QA_DEID_BUNDLES/<patient_id>/
      ALL_NOTES_COMBINED.txt
      timeline.csv
      note_*.txt
      encounters_timeline.csv            (copied from PATIENT_BUNDLES if present)
      stage2_anchor_summary.txt          (copied from PATIENT_BUNDLES if present)

Also writes logs:
  QA_DEID_BUNDLES/logs/<patient_id>.out.txt
  QA_DEID_BUNDLES/logs/<patient_id>.err.txt
"""

import os
import sys
import shutil
import subprocess
import pandas as pd


# ---------------- HARD-CODED PATHS ----------------
BREAST_RESTORE_DIR = "/home/apokol/Breast_Restore"

# The existing single-patient de-id exporter you already have:
EXPORTER = os.path.join(BREAST_RESTORE_DIR, "export_deid_clinic_notes_fulltext.py")

# Where the full patient bundles live (encounters + anchor summary):
PATIENT_BUNDLES_DIR = os.path.join(BREAST_RESTORE_DIR, "PATIENT_BUNDLES")

# Where QA exports go:
OUT_DIR = os.path.join(BREAST_RESTORE_DIR, "QA_DEID_BUNDLES")

# --------------------------------------------------


def read_patient_ids(csv_path):
    df = pd.read_csv(csv_path, dtype=str, engine="python")
    if "patient_id" not in df.columns:
        raise ValueError(
            "Expected column 'patient_id' in {}. Found: {}".format(csv_path, list(df.columns))
        )

    series = df["patient_id"].fillna("").astype(str).str.strip()
    pids = [p for p in series.tolist() if p]

    # unique while preserving order
    seen = set()
    out = []
    for p in pids:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def safe_copy(src, dst_dir):
    """
    Copy src file into dst_dir if it exists.
    Returns True if copied, False otherwise.
    """
    if not src or not os.path.exists(src):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    return True


def run_exporter(pid, log_out_path, log_err_path):
    """
    Run your de-id exporter for one patient.
    NOTE: matches the help output you showed:
      export_deid_clinic_notes_fulltext.py [--out_dir OUT_DIR] [--patient_id PATIENT_ID]
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
        universal_newlines=True
    )

    # Write logs
    with open(log_out_path, "w") as f:
        if proc.stdout:
            f.write(proc.stdout)
    with open(log_err_path, "w") as f:
        if proc.stderr:
            f.write(proc.stderr)

    return proc.returncode


def copy_bundle_artifacts(pid):
    """
    Copy encounter + anchor summary outputs from:
      PATIENT_BUNDLES/<pid>/
    into:
      QA_DEID_BUNDLES/<pid>/
    """
    src_dir = os.path.join(PATIENT_BUNDLES_DIR, pid)
    dst_dir = os.path.join(OUT_DIR, pid)

    copied = {
        "encounters_timeline.csv": False,
        "stage2_anchor_summary.txt": False,
        "stage2_anchor_summary_v2.txt": False,
    }

    copied["encounters_timeline.csv"] = safe_copy(
        os.path.join(src_dir, "encounters_timeline.csv"),
        dst_dir
    )

    copied["stage2_anchor_summary.txt"] = safe_copy(
        os.path.join(src_dir, "stage2_anchor_summary.txt"),
        dst_dir
    )

    # optional alternate name if you ever used it
    copied["stage2_anchor_summary_v2.txt"] = safe_copy(
        os.path.join(src_dir, "stage2_anchor_summary_v2.txt"),
        dst_dir
    )

    return copied


def main():
    if len(sys.argv) != 2:
        print("Usage: {} /path/to/patient_list.csv".format(sys.argv[0]), file=sys.stderr)
        sys.exit(2)

    csv_path = sys.argv[1]

    os.makedirs(OUT_DIR, exist_ok=True)
    logs_dir = os.path.join(OUT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    pids = read_patient_ids(csv_path)
    print("Patients to export: {}".format(len(pids)))
    print("OUT_DIR: {}".format(OUT_DIR))
    print("PATIENT_BUNDLES_DIR: {}".format(PATIENT_BUNDLES_DIR))

    ok = 0
    bad = 0
    failures = []

    # Track copy stats
    copy_counts = {
        "encounters_timeline.csv": 0,
        "stage2_anchor_summary.txt": 0,
        "stage2_anchor_summary_v2.txt": 0
    }

    for i, pid in enumerate(pids, 1):
        print("\n[{}/{}] Exporting: {}".format(i, len(pids), pid))

        log_out = os.path.join(logs_dir, "{}.out.txt".format(pid))
        log_err = os.path.join(logs_dir, "{}.err.txt".format(pid))

        rc = run_exporter(pid, log_out, log_err)
        if rc != 0:
            bad += 1
            failures.append(pid)
            print("  FAIL (see {})".format(log_err))
            continue

        # Export ok -> copy artifacts
        copied = copy_bundle_artifacts(pid)
        for k, v in copied.items():
            if v:
                copy_counts[k] += 1

        ok += 1
        print("  OK  (see {})".format(log_out))
        print("  copied: {}".format(
            ", ".join(["{}={}".format(k, "Y" if v else "N") for k, v in copied.items()])
        ))

    print("\n==== SUMMARY ====")
    print("Success: {}".format(ok))
    print("Failed : {}".format(bad))

    print("\n==== COPIED ARTIFACTS (count of patients) ====")
    for k in ["encounters_timeline.csv", "stage2_anchor_summary.txt", "stage2_anchor_summary_v2.txt"]:
        print("  {}: {}".format(k, copy_counts[k]))

    if failures:
        fail_path = os.path.join(OUT_DIR, "FAILED_PATIENT_IDS.txt")
        with open(fail_path, "w") as f:
            for pid in failures:
                f.write(pid + "\n")
        print("\nWrote failures: {}".format(fail_path))


if __name__ == "__main__":
    main()
