#!/usr/bin/env python3
# build_new_master_ids_plus_new_staging.py
# Python 3.6.8 compatible
#
# Goal:
#   Create a NEW master cohort file that contains ONLY:
#     (A) patient identifiers from the old cohort file
#     (B) your CURRENT staging outputs (new logic)
#
# Rationale:
#   - Prevent contamination from legacy Stage1/Stage2 fields in the old cohort file
#   - Keep a clean, versioned master you can trust for validation + presentation
#
# Outputs:
#   - NEW_MASTER_CSV (clean master: IDs + new staging)
#   - QA_REPORT_TXT  (counts, merge coverage, duplicates, columns kept)
#
# Usage:
#   cd /home/apokol/Breast_Restore
#   python build_new_master_ids_plus_new_staging.py

from __future__ import print_function
import os
import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT THESE PATHS)
# -------------------------
OLD_COHORT_CSV = "/home/apokol/Breast_Restore/cohort_all_patient_level_final_gold_order.csv"
NEW_STAGING_CSV = "/home/apokol/Breast_Restore/patient_recon_staging_refined.csv"

# Make the filename VERY obvious that this is the new clean master
NEW_MASTER_CSV = "/home/apokol/Breast_Restore/MASTER__IDS_PLUS_NEW_STAGING__vNEW.csv"
QA_REPORT_TXT  = "/home/apokol/Breast_Restore/MASTER__IDS_PLUS_NEW_STAGING__vNEW__QA.txt"

# Required key
PID_COL = "patient_id"

# Which identifier columns to keep from the OLD cohort file.
# Keep this minimal on purpose. Add/remove as needed.
# (If a column is missing, the script will skip it and report it.)
ID_COL_CANDIDATES = [
    "patient_id",
    "ENCRYPTED_PAT_ID",
    "ENCRYPTED_PATID",
    "PAT_ID",
    "PATIENT_ID",
    "MRN",             # keep only if allowed in your workflow
    "mrn",
]

# Which columns to keep from the NEW staging file.
# Recommended default: keep EVERYTHING EXCEPT obvious legacy/notes/raw text fields.
# If you want to keep only a subset, set STAGING_KEEP_MODE="whitelist"
STAGING_KEEP_MODE = "auto"   # "auto" or "whitelist"

# If STAGING_KEEP_MODE="whitelist", only keep these from NEW staging (plus patient_id)
STAGING_WHITELIST = [
    "patient_id",
    "has_expander_refined",
    "expander_bucket",
    "stage1_date",
    # add your vNEW outputs here when ready, e.g.:
    # "stage2_date_final",
    # "stage2_confirmed_flag",
    # "stage2_tier_best",
]

# In auto mode, we drop staging columns that look like raw text / note blobs / PHI-ish columns.
STAGING_DROP_REGEX = re.compile(
    r"(note_text|note_txt|free_text|raw_text|snippet|fulltext|body_text|text_deid|deid|provider|author|signature)",
    re.I
)

# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path):
    # Use latin1+replace to avoid UM export encoding landmines.
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object)
    finally:
        try:
            f.close()
        except Exception:
            pass

def norm_str(x):
    if x is None:
        return ""
    try:
        s = str(x).strip()
    except Exception:
        return ""
    # normalize trailing ".0" from Excel-ish exports
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s

def find_existing_cols(df_cols, candidates):
    cols = []
    for c in candidates:
        if c in df_cols:
            cols.append(c)
    return cols

def safe_unique_count(series):
    return int(series.dropna().map(norm_str).replace("", pd.NA).dropna().nunique())

def write_report(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

# -------------------------
# Main
# -------------------------
def main():
    for p in (OLD_COHORT_CSV, NEW_STAGING_CSV):
        if not os.path.exists(p):
            print("ERROR: Missing file:", p)
            sys.exit(1)

    print("Loading OLD cohort:", OLD_COHORT_CSV)
    old = read_csv_safe(OLD_COHORT_CSV)

    print("Loading NEW staging:", NEW_STAGING_CSV)
    stg = read_csv_safe(NEW_STAGING_CSV)

    # Validate patient_id exists
    if PID_COL not in old.columns:
        print("ERROR: patient_id not found in OLD cohort file.")
        print("Columns:", list(old.columns)[:50])
        sys.exit(1)

    if PID_COL not in stg.columns:
        print("ERROR: patient_id not found in NEW staging file.")
        print("Columns:", list(stg.columns)[:50])
        sys.exit(1)

    # Normalize patient_id for stable join
    old[PID_COL] = old[PID_COL].map(norm_str)
    stg[PID_COL] = stg[PID_COL].map(norm_str)

    old = old[old[PID_COL] != ""].copy()
    stg = stg[stg[PID_COL] != ""].copy()

    # OLD: keep only ID columns that exist
    id_cols_keep = find_existing_cols(old.columns, ID_COL_CANDIDATES)
    if PID_COL not in id_cols_keep:
        id_cols_keep = [PID_COL] + id_cols_keep

    # NEW staging: decide which columns to keep
    if STAGING_KEEP_MODE.lower() == "whitelist":
        stg_cols_keep = [c for c in STAGING_WHITELIST if c in stg.columns]
        if PID_COL not in stg_cols_keep:
            stg_cols_keep = [PID_COL] + stg_cols_keep
    else:
        # auto: keep all except obvious text/PHI-ish columns
        stg_cols_keep = []
        for c in stg.columns:
            if c == PID_COL:
                stg_cols_keep.append(c)
                continue
            if STAGING_DROP_REGEX.search(c):
                continue
            stg_cols_keep.append(c)

    # De-dup staging by patient_id (keep first; you can change to last if needed)
    stg_dup = int(stg.duplicated(subset=[PID_COL]).sum())
    stg_dedup = stg.drop_duplicates(subset=[PID_COL], keep="first").copy()

    # Build master by left-joining staging onto IDs backbone
    ids_df = old[id_cols_keep].drop_duplicates(subset=[PID_COL], keep="first").copy()

    n_old_rows = int(len(old))
    n_old_unique = safe_unique_count(old[PID_COL])
    n_ids_rows = int(len(ids_df))
    n_ids_unique = safe_unique_count(ids_df[PID_COL])

    n_stg_rows = int(len(stg))
    n_stg_unique = safe_unique_count(stg[PID_COL])
    n_stg_dedup = int(len(stg_dedup))

    # Merge
    master = ids_df.merge(stg_dedup[stg_cols_keep], on=PID_COL, how="left", indicator=True)

    n_master = int(len(master))
    n_matched = int((master["_merge"] == "both").sum())
    n_missing_stg = int((master["_merge"] == "left_only").sum())
    master = master.drop(columns=["_merge"])

    # QA: staging non-null coverage per staging column (topline only)
    coverage_rows = []
    for c in stg_cols_keep:
        if c == PID_COL:
            continue
        nonblank = master[c].map(norm_str).replace("", pd.NA).notnull().sum()
        coverage_rows.append((c, int(nonblank), float(nonblank) / float(n_master) if n_master else 0.0))
    coverage_rows = sorted(coverage_rows, key=lambda x: (-x[1], x[0]))

    # Write outputs
    master.to_csv(NEW_MASTER_CSV, index=False, encoding="utf-8")

    # Write QA report
    lines = []
    lines.append("NEW MASTER BUILD QA")
    lines.append("===================")
    lines.append("")
    lines.append("OLD cohort file: {}".format(OLD_COHORT_CSV))
    lines.append("NEW staging file: {}".format(NEW_STAGING_CSV))
    lines.append("OUTPUT master: {}".format(NEW_MASTER_CSV))
    lines.append("")
    lines.append("Key column: {}".format(PID_COL))
    lines.append("")
    lines.append("OLD cohort:")
    lines.append("  rows (raw): {}".format(n_old_rows))
    lines.append("  unique patient_id: {}".format(n_old_unique))
    lines.append("")
    lines.append("IDs backbone (from OLD cohort):")
    lines.append("  rows (dedup by patient_id): {}".format(n_ids_rows))
    lines.append("  unique patient_id: {}".format(n_ids_unique))
    lines.append("  ID columns kept ({}):".format(len(id_cols_keep)))
    for c in id_cols_keep:
        lines.append("    - {}".format(c))
    lines.append("")
    lines.append("NEW staging:")
    lines.append("  rows (raw): {}".format(n_stg_rows))
    lines.append("  unique patient_id: {}".format(n_stg_unique))
    lines.append("  duplicates on patient_id (raw): {}".format(stg_dup))
    lines.append("  rows after dedup: {}".format(n_stg_dedup))
    lines.append("  staging keep mode: {}".format(STAGING_KEEP_MODE))
    lines.append("  staging columns kept ({}):".format(len(stg_cols_keep)))
    for c in stg_cols_keep[:80]:
        lines.append("    - {}".format(c))
    if len(stg_cols_keep) > 80:
        lines.append("    ... ({} more)".format(len(stg_cols_keep) - 80))
    lines.append("")
    lines.append("MERGE results (IDs LEFT JOIN staging):")
    lines.append("  master rows: {}".format(n_master))
    lines.append("  matched to staging: {} ({:.1f}%)".format(n_matched, 100.0 * n_matched / n_master if n_master else 0.0))
    lines.append("  missing staging:   {} ({:.1f}%)".format(n_missing_stg, 100.0 * n_missing_stg / n_master if n_master else 0.0))
    lines.append("")
    lines.append("STAGING COLUMN COVERAGE (non-blank counts, top 25):")
    for (c, cnt, pct) in coverage_rows[:25]:
        lines.append("  {:<40} {:>6}  ({:>5.1f}%)".format(c, cnt, 100.0 * pct))
    lines.append("")
    lines.append("Notes:")
    lines.append("  - This master intentionally excludes ALL legacy staging/derived fields from the old cohort.")
    lines.append("  - Later, when you validate demographics/comorbs, you can merge those in deliberately.")
    lines.append("")

    write_report(QA_REPORT_TXT, lines)

    print("\nDONE.")
    print("Wrote master:", NEW_MASTER_CSV)
    print("Wrote QA    :", QA_REPORT_TXT)
    print("Master rows:", n_master, "| matched staging:", n_matched, "| missing staging:", n_missing_stg)

if __name__ == "__main__":
    main()
