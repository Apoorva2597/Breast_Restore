#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGING SCRIPT (Python 3.6.8)

Run from ~/Breast_Restore:
    cd ~/Breast_Restore
    python stage_inputs_FINAL.py

Purpose:
- Fix the file-path error by AUTO-locating the HPI11526 CSVs from common locations
- Create a local staging folder inside Breast_Restore:
      ./_staging_inputs/
  and copy the required file(s) there so downstream scripts can use RELATIVE paths.

It will:
1) Search for "HPI11526 Operation Notes.csv" (required)
2) Optionally pick up the other related CSVs if present
3) Copy found files into ./_staging_inputs/
4) Print the resolved paths + basic row counts
"""

from __future__ import print_function

import os
import sys
import glob
import csv
import shutil

REQUIRED_FILE = "HPI11526 Operation Notes.csv"
OPTIONAL_FILES = [
    "HPI11526 Clinic Notes.csv",
    "HPI11526 Inpatient Notes.csv",
    "HPI11526 Clinic Encounters.csv",
    "HPI11526 Inpatient Encounters.csv",
    "HPI11526 Operation Encounters.csv",
]

STAGING_DIRNAME = "_staging_inputs"


def eprint(*args):
    sys.stderr.write(" ".join([str(a) for a in args]) + "\n")


def abspath(p):
    return os.path.abspath(os.path.expanduser(p))


def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def count_rows_quick(csv_path, max_rows=2000000):
    """
    Fast-ish count (no pandas). Stops at max_rows to avoid runaway on huge files.
    Returns (n_rows_excluding_header, header_list).
    """
    n = 0
    header = None
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, []
        for _ in reader:
            n += 1
            if n >= max_rows:
                break
    return n, header


def candidate_dirs(repo_root):
    """
    We assume you run from Breast_Restore. We'll search:
    - repo_root
    - repo_root/data, repo_root/inputs, repo_root/my_data_Breast (if you have)
    - ~/my_data_Breast/HPI-11526/HPI11256  (your known location)
    - ~/my_data_Breast/HPI-11526
    - a shallow scan under ~/my_data_Breast for any HPI-11526/HPI11256-like folder
    """
    home = abspath("~")
    cands = [
        repo_root,
        abspath(os.path.join(repo_root, "data")),
        abspath(os.path.join(repo_root, "inputs")),
        abspath(os.path.join(repo_root, "my_data_Breast")),
        abspath("~/my_data_Breast/HPI-11526/HPI11256"),
        abspath("~/my_data_Breast/HPI-11526"),
    ]

    # Shallow discovery under ~/my_data_Breast (2 levels deep)
    base = abspath("~/my_data_Breast")
    if os.path.isdir(base):
        # e.g., ~/my_data_Breast/HPI-11526/* and ~/my_data_Breast/*/*
        for pat in [
            os.path.join(base, "HPI-11526", "*"),
            os.path.join(base, "*", "*"),
        ]:
            for d in glob.glob(pat):
                if os.path.isdir(d):
                    cands.append(abspath(d))

    # Deduplicate, preserve order
    seen = set()
    out = []
    for d in cands:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_file(filename, dirs):
    for d in dirs:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p

        # also allow minor naming variants (spaces/underscores)
        variants = [
            filename.replace(" ", "_"),
            filename.replace("_", " "),
        ]
        for v in variants:
            pv = os.path.join(d, v)
            if os.path.isfile(pv):
                return pv

        # last-resort glob in that directory
        # (kept tight to avoid false matches)
        base = os.path.splitext(filename)[0]
        hits = glob.glob(os.path.join(d, base.replace(" ", "*") + "*.csv"))
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            hits = sorted(hits, key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]
    return None


def copy_into_staging(src_path, staging_dir):
    dst_path = os.path.join(staging_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    return dst_path


def main():
    repo_root = abspath(os.getcwd())  # run from Breast_Restore
    staging_dir = abspath(os.path.join(repo_root, STAGING_DIRNAME))
    ensure_dir(staging_dir)

    dirs = candidate_dirs(repo_root)

    eprint("[INFO] Repo root (CWD):", repo_root)
    eprint("[INFO] Staging dir:", staging_dir)
    eprint("[INFO] Searching for required file:", REQUIRED_FILE)

    req_src = find_file(REQUIRED_FILE, dirs)
    if not req_src:
        eprint("\n[ERROR] Could not find '{}' in any searched directory.".format(REQUIRED_FILE))
        eprint("[HINT] Your screenshot suggests it lives under:")
        eprint("       ~/my_data_Breast/HPI-11526/HPI11256/")
        eprint("\n[DEBUG] Directories searched (first 20):")
        for d in dirs[:20]:
            eprint("  -", d)
        sys.exit(1)

    eprint("[OK] Found required file at:", req_src)

    # Copy required
    req_dst = copy_into_staging(req_src, staging_dir)

    # Copy optional if present
    copied = [(REQUIRED_FILE, req_src, req_dst)]
    for fn in OPTIONAL_FILES:
        src = find_file(fn, dirs)
        if src:
            dst = copy_into_staging(src, staging_dir)
            copied.append((fn, src, dst))

    print("\n=== STAGING COMPLETE ===")
    print("Staged files are now available at:")
    print("  {}".format(staging_dir))
    print("\nYou can now reference them with RELATIVE paths from Breast_Restore, e.g.:")
    print("  ./{}_inputs/{}".format("staging" if False else STAGING_DIRNAME, REQUIRED_FILE))

    print("\n=== FILES COPIED ===")
    for fn, src, dst in copied:
        print("- {}".format(fn))
        print("    from: {}".format(src))
        print("    to:   {}".format(dst))

    print("\n=== QUICK VALIDATION ===")
    for _, _, dst in copied:
        n, header = count_rows_quick(dst)
        print("\n{}".format(os.path.basename(dst)))
        print("  rows (excl header): {}".format(n))
        print("  columns: {}".format(len(header)))
        # show first 12 column names
        show = header[:12]
        print("  header (first 12): {}".format(show))

    print("\nOK: Path issue eliminated. Your pipeline can now read from ./{}/...".format(STAGING_DIRNAME))


if __name__ == "__main__":
    main()
