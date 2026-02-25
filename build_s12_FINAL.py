#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_s12_FINAL.py  (Python 3.6.8 compatible)

Run this FROM ~/Breast_Restore:

    cd ~/Breast_Restore
    python build_s12_FINAL.py

Optional (recommended if your data is not under the auto-discovered locations):
    python build_s12_FINAL.py --data_dir ~/my_data_Breast/HPI-11526/HPI11256

What it does (right now):
- Locates the key CSVs (Operation Notes + others if present)
- Prints exactly where it found them
- Exits early with a helpful error + directory listing if not found

You can plug your Stage1/Stage2 extraction logic after the "LOAD/PROCESS" section.
"""

from __future__ import print_function

import os
import sys
import glob
import argparse

# -----------------------------
# Utilities
# -----------------------------
def eprint(*args):
    sys.stderr.write(" ".join([str(a) for a in args]) + "\n")

def abspath(p):
    return os.path.abspath(os.path.expanduser(p))

def list_csvs(dir_path, limit=30):
    try:
        paths = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        return paths[:limit]
    except Exception:
        return []

def first_hit(dir_path, patterns):
    """
    Return first matching file for any pattern in patterns (glob patterns),
    preferring shortest basename if multiple hits.
    """
    tried = []
    for pat in patterns:
        g = os.path.join(dir_path, pat)
        tried.append(g)
        hits = glob.glob(g)
        if hits:
            hits = sorted(hits, key=lambda p: (len(os.path.basename(p)), p))
            return hits[0], tried
    return None, tried

# -----------------------------
# Data dir discovery
# -----------------------------
def candidate_data_dirs(script_dir, user_data_dir=None):
    """
    We run from ~/Breast_Restore (script_dir).
    Data is in your screenshot path:
      ~/my_data_Breast/HPI-11526/HPI11256
    but we try a few sensible places.
    """
    cands = []

    if user_data_dir:
        cands.append(abspath(user_data_dir))

    # 1) Where you are running from (Breast_Restore)
    cands.append(abspath(os.getcwd()))

    # 2) Where the script lives (also Breast_Restore if you run it there)
    cands.append(abspath(script_dir))

    # 3) Your known storage location from screenshot
    cands.append(abspath("~/my_data_Breast/HPI-11526/HPI11256"))
    cands.append(abspath("~/my_data_Breast/HPI-11526"))

    # 4) A common pattern if you keep data beside the repo:
    #    ~/Breast_Restore/../my_data_Breast/HPI-11526/HPI11256
    cands.append(abspath(os.path.join(script_dir, "..", "my_data_Breast", "HPI-11526", "HPI11256")))
    cands.append(abspath(os.path.join(script_dir, "..", "my_data_Breast", "HPI-11526")))

    # Deduplicate preserving order
    seen = set()
    out = []
    for d in cands:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

def must_find_files(user_data_dir=None):
    script_dir = os.path.dirname(__file__)
    cands = candidate_data_dirs(script_dir, user_data_dir=user_data_dir)

    eprint("[DEBUG] Running from CWD:", abspath(os.getcwd()))
    eprint("[DEBUG] Script path:", abspath(__file__))
    eprint("[DEBUG] Candidate data dirs:")
    for d in cands:
        eprint("  -", d)

    # You showed these filenames in the directory listing:
    # 'HPI11526 Clinic Encounters.csv'
    # 'HPI11526 Inpatient Encounters.csv'
    # 'HPI11526 Operation Encounters.csv'
    # 'HPI11526 Clinic Notes.csv'
    # 'HPI11526 Inpatient Notes.csv'
    # 'HPI11526 Operation Notes.csv'
    patterns = {
        "operation_notes": [
            "HPI11526 Operation Notes.csv",
            "HPI11526_Operation_Notes.csv",
            "HPI11526*Operation*Notes*.csv",
            "*Operation*Notes*.csv",
        ],
        "clinic_notes": [
            "HPI11526 Clinic Notes.csv",
            "HPI11526_Clinic_Notes.csv",
            "HPI11526*Clinic*Notes*.csv",
            "*Clinic*Notes*.csv",
        ],
        "inpatient_notes": [
            "HPI11526 Inpatient Notes.csv",
            "HPI11526_Inpatient_Notes.csv",
            "HPI11526*Inpatient*Notes*.csv",
            "*Inpatient*Notes*.csv",
        ],
        "operation_encounters": [
            "HPI11526 Operation Encounters.csv",
            "HPI11526_Operation_Encounters.csv",
            "HPI11526*Operation*Encounters*.csv",
            "*Operation*Encounters*.csv",
        ],
        "clinic_encounters": [
            "HPI11526 Clinic Encounters.csv",
            "HPI11526_Clinic_Encounters.csv",
            "HPI11526*Clinic*Encounters*.csv",
            "*Clinic*Encounters*.csv",
        ],
        "inpatient_encounters": [
            "HPI11526 Inpatient Encounters.csv",
            "HPI11526_Inpatient_Encounters.csv",
            "HPI11526*Inpatient*Encounters*.csv",
            "*Inpatient*Encounters*.csv",
        ],
    }

    found = {}
    used_dir = None

    # We require operation_notes to proceed; others are optional.
    required_key = "operation_notes"

    for d in cands:
        if not os.path.isdir(d):
            continue

        # Try to locate required first
        req_path, _ = first_hit(d, patterns[required_key])
        if not req_path:
            csvs = list_csvs(d, limit=12)
            if csvs:
                eprint("[DEBUG] In", d, "I see CSVs (up to 12):")
                for p in csvs:
                    eprint("   ", os.path.basename(p))
            else:
                eprint("[DEBUG] No CSVs visible in", d)
            continue

        # If required exists here, lock in this directory and find others
        used_dir = d
        found[required_key] = req_path

        for key, pats in patterns.items():
            if key == required_key:
                continue
            pth, _ = first_hit(d, pats)
            if pth:
                found[key] = pth

        break

    if not used_dir:
        raise IOError(
            "Could not locate required file: HPI11526 Operation Notes.csv\n\n"
            "Fix: run with explicit --data_dir, e.g.\n"
            "  python build_s12_FINAL.py --data_dir ~/my_data_Breast/HPI-11526/HPI11256\n"
        )

    return used_dir, found

# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing HPI11526 CSVs (e.g., ~/my_data_Breast/HPI-11526/HPI11256)",
    )
    return ap.parse_args()

def main():
    args = parse_args()

    used_dir, paths = must_find_files(user_data_dir=args.data_dir)

    print("\n=== DATA LOCATION ===")
    print("USING_DATA_DIR={}".format(used_dir))

    print("\n=== FOUND FILES ===")
    for k in sorted(paths.keys()):
        print("{:<22} {}".format(k + ":", paths[k]))

    # -----------------------------
    # LOAD/PROCESS (placeholder)
    # -----------------------------
    # At this point you have absolute paths for the CSVs.
    # You can now load Operation Notes and build Stage1/Stage2 detection.
    #
    # Example (if you want pandas):
    #   import pandas as pd
    #   op_df = pd.read_csv(paths["operation_notes"])
    #
    # For now, we just confirm discovery succeeded.
    print("\nOK: File discovery succeeded. Plug extraction logic after this point.\n")

if __name__ == "__main__":
    main()
