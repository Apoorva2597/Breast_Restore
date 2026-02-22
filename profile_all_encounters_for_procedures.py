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
import re
import pandas as pd


# -------------------------
# CONFIG: EDIT THIS
# -------------------------

# Option A (recommended): list your encounter CSVs explicitly (paths can have spaces)
FILES = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11526 Clinic Encounters.csv", "/home/apokol/my_data_Breast/HPI-11526/HPI11526 Operation Encounters.csv", "/home/apokol/my_data_Breast/HPI-11526/HPI11526 Inpatient Encounters.csv"
  ]

MAX_SCAN_ROWS = 50000     # only scan first N rows per file to get top-values quickly
TOPK = 25                 # print top N values per procedure-like column
OUT_SUMMARY = "encounter_procedure_profile_summary.csv"


# -------------------------
# Helpers
# -------------------------

BLANK_TOKENS = set(["", "nan", "none", "null", "na", "n/a", ".", "-", "--"])

PAT_COL_CANDIDATES = [
    "ENCRYPTED_PAT_ID", "patient_id", "PAT_ID", "PATIENT_ID", "ENCRYPTED_PATID",
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
    for want in candidates:
        w = want.upper()
        for c in cols:
            if cols_u[c] == w:
                return c
    # fallback contains
    for want in candidates:
        w = want.upper()
        for c in cols:
            if w in cols_u[c]:
                return c
    return None

def is_procedure_like_col(colname):
    n = str(colname).strip().lower()
    # procedure signals
    tokens = [
        "procedure", "proc", "cpt", "hcpcs", "icd", "px", "op", "operative",
        "surgery", "operation", "or_", "or ", "explant", "implant", "expander",
        "recon", "reconstruction"
    ]
    # avoid columns that are clearly not procedures
    bad = ["provider", "department", "clinic", "location", "address", "phone", "zip", "city", "state"]
    if any(b in n for b in bad) and not any(t in n for t in ["procedure", "cpt", "hcpcs", "icd"]):
        return False
    return any(t in n for t in tokens)

def list_csvs_from_folder(folder, recursive=False):
    out = []
    if not folder:
        return out
    if recursive:
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".csv"):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            if fn.lower().endswith(".csv"):
                out.append(os.path.join(folder, fn))
    return sorted(out)


# -------------------------
# Main
# -------------------------

print("\n=== Profile encounter files for procedure info ===")

all_files = []
if FILES:
    all_files = FILES[:]
elif FOLDER:
    all_files = list_csvs_from_folder(FOLDER, recursive=FOLDER_RECURSIVE)

all_files = [p for p in all_files if p and os.path.exists(p)]

if not all_files:
    raise RuntimeError("No input files found. Edit FILES or set FOLDER.")

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

    pat_col = pick_col(df.columns, PAT_COL_CANDIDATES)
    if pat_col:
        df["_PID_"] = df[pat_col].map(norm_str)
        n_pat = int((df["_PID_"] != "").sum())
        n_pat_uniq = int(df[df["_PID_"] != ""]["_PID_"].nunique())
    else:
        df["_PID_"] = ""
        n_pat = 0
        n_pat_uniq = 0

    proc_cols = [c for c in df.columns if is_procedure_like_col(c)]
    print("  Detected patient column:", pat_col if pat_col else "(none)")
    print("  Procedure-like columns found:", len(proc_cols))

    if proc_cols:
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

            # store for summary output
            for k, v in vc.items():
                summary_rows.append({
                    "file": os.path.basename(path),
                    "full_path": path,
                    "patient_col_detected": pat_col if pat_col else "",
                    "patients_scanned_nonblank_pid_rows": n_pat,
                    "patients_unique_in_scan": n_pat_uniq,
                    "procedure_col": str(c),
                    "value": str(k),
                    "count_in_first_{}_rows".format(MAX_SCAN_ROWS): int(v),
                })
    else:
        print("  No procedure-like columns detected by header heuristics.")
        summary_rows.append({
            "file": os.path.basename(path),
            "full_path": path,
            "patient_col_detected": pat_col if pat_col else "",
            "patients_scanned_nonblank_pid_rows": n_pat,
            "patients_unique_in_scan": n_pat_uniq,
            "procedure_col": "",
            "value": "",
            "count_in_first_{}_rows".format(MAX_SCAN_ROWS): 0,
        })

# Write combined summary CSV
out_df = pd.DataFrame(summary_rows)
out_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")

print("\n==================================================")
print("Wrote summary CSV:", OUT_SUMMARY)
print("Rows in summary:", len(out_df))
print("Done.\n")
