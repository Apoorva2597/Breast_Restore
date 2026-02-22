# profile_all_encounters_for_procedures.py
# Python 3.6.8 compatible
#
# Goal:
#   Scan multiple encounter CSV files and find procedure-related columns + top values.
#
# Output:
#   - Prints summary to terminal
#   - Writes one combined summary file: encounter_procedure_profile_summary.csv

from __future__ import print_function
import os
import pandas as pd


# -------------------------
# CONFIG: EDIT THIS
# -------------------------

FILES = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]

MAX_SCAN_ROWS = 50000     # only scan first N rows per file
TOPK = 25                 # print top N values per procedure-like column
OUT_SUMMARY = "encounter_procedure_profile_summary.csv"


# -------------------------
# Helpers
# -------------------------

BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--"])

PAT_COL_CANDIDATES = [
    "ENCRYPTED_PAT_ID", "PATIENT_ID", "PAT_ID", "patient_id", "ENCRYPTED_PATID",
    "MRN", "PAT_MRN_ID"
]


def read_csv_safe(path, nrows=None):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object, nrows=nrows)
    finally:
        try:
            f.close()
        except Exception:
            pass


def norm_str(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()
    if s.lower() in BLANK_TOKENS:
        return ""
    return s


def pick_col(cols, candidates):
    cols_u = {c: str(c).strip().upper() for c in cols}

    # exact match first
    for want in candidates:
        w = want.upper()
        for c in cols:
            if cols_u[c] == w:
                return c

    # contains match fallback
    for want in candidates:
        w = want.upper()
        for c in cols:
            if w in cols_u[c]:
                return c

    return None


def is_procedure_like_col(colname):
    n = str(colname).strip().lower()

    # Strong signals
    strong_tokens = [
        "procedure", "cpt", "hcpcs", "icd", "icd9", "icd10",
        "operative", "surgery", "operation",
        "explant", "implant", "expander",
        "recon", "reconstruction"
    ]

    # Weaker signals (can cause false hits, so keep limited)
    weak_tokens = [
        "proc",   # may match process/procedure fields but can be noisy
        "px",     # sometimes used for procedure code
    ]

    # Avoid obvious non-procedure columns
    bad_tokens = [
        "provider", "department", "clinic", "location", "address", "phone",
        "zip", "city", "state", "insurance", "payer", "guarantor"
    ]

    if any(bt in n for bt in bad_tokens) and not any(st in n for st in ["procedure", "cpt", "hcpcs", "icd"]):
        return False

    if any(t in n for t in strong_tokens):
        return True

    if any(t in n for t in weak_tokens):
        # only accept weak tokens if column also hints code/name
        if "code" in n or "name" in n or "desc" in n or "description" in n:
            return True

    return False


# -------------------------
# Main
# -------------------------

print("\n=== Profile encounter files for procedure info ===")

# Validate file list (show missing)
all_files = []
for p in FILES:
    if p and os.path.exists(p):
        all_files.append(p)
    else:
        print("WARNING: file not found:", p)

if not all_files:
    raise RuntimeError("No input files found. Check FILES paths.")

print("Files to scan:", len(all_files))

summary_rows = []

for path in all_files:
    print("\n--------------------------------------------------")
    print("FILE:", path)

    try:
        df = read_csv_safe(path, nrows=MAX_SCAN_ROWS)
    except Exception as e:
        print("  ERROR reading file:", str(e))
        continue

    print("  Rows scanned:", len(df))
    print("  Columns:", len(df.columns))

    # patient column detection
    pat_col = pick_col(df.columns, PAT_COL_CANDIDATES)
    if pat_col:
        df["_PID_"] = df[pat_col].map(norm_str)
        n_pat_rows = int((df["_PID_"] != "").sum())
        n_pat_uniq = int(df[df["_PID_"] != ""]["_PID_"].nunique())
    else:
        df["_PID_"] = ""
        n_pat_rows = 0
        n_pat_uniq = 0

    proc_cols = [c for c in df.columns if is_procedure_like_col(c)]

    print("  Detected patient column:", pat_col if pat_col else "(none)")
    print("  Procedure-like columns found:", len(proc_cols))

    if not proc_cols:
        summary_rows.append({
            "file": os.path.basename(path),
            "full_path": path,
            "patient_col_detected": pat_col if pat_col else "",
            "patients_unique_in_scan": n_pat_uniq,
            "procedure_col": "",
            "value": "",
            "count_in_first_{}_rows".format(MAX_SCAN_ROWS): 0,
        })
        continue

    # For each procedure-like column, print top values
    for c in proc_cols:
        s = df[c].map(norm_str)
        nonblank = s[s != ""]
        nn = int((s != "").sum())
        if nn == 0:
            continue

        vc = nonblank.value_counts().head(TOPK)

        print("\n  Top values for:", c, "(nonblank rows:", nn, ")")
        for k, v in vc.items():
            kk = k if len(k) <= 140 else (k[:140] + "...")
            print("    {:>6}  {}".format(int(v), kk))

        for k, v in vc.items():
            summary_rows.append({
                "file": os.path.basename(path),
                "full_path": path,
                "patient_col_detected": pat_col if pat_col else "",
                "patients_unique_in_scan": n_pat_uniq,
                "procedure_col": str(c),
                "value": str(k),
                "count_in_first_{}_rows".format(MAX_SCAN_ROWS): int(v),
            })

# Write combined summary CSV
out_df = pd.DataFrame(summary_rows)
out_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")

print("\n==================================================")
print("Wrote summary CSV:", OUT_SUMMARY)
print("Rows in summary:", len(out_df))
print("Done.\n")
