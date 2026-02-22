# profile_all_encounters_for_procedures.py
# Python 3.6.8 compatible
#
# Goal:
#   Scan multiple encounter CSV files and find procedure-related columns + top values.
#
# Output:
#   - Prints summary to terminal
#   - Writes one combined summary file: encounter_procedure_profile_summary.csv
#
# Improvements vs earlier:
#   - Deduplicates FILES (prevents same file printing twice)
#   - Normalizes date-like values (e.g., "12/27/2016 0:00" -> "2016-12-27") for reporting
#   - Includes both row_count and unique_patients_with_value (if patient column exists)

from __future__ import print_function
import os
import re
import pandas as pd


# -------------------------
# CONFIG: EDIT THIS
# -------------------------

FILES = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]

# Folder mode (optional). If FILES is empty, folder mode is used.
FOLDER = ""
FOLDER_RECURSIVE = False

MAX_SCAN_ROWS = 50000     # scan first N rows per file (set None to scan ALL)
TOPK = 25
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
    tokens = [
        "procedure", "proc", "cpt", "hcpcs", "icd", "px", "op", "operative",
        "surgery", "operation", "explant", "implant", "expander",
        "recon", "reconstruction"
    ]
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

def looks_like_date_col(colname):
    n = str(colname).strip().lower()
    # keep this conservative; we only *normalize values* for these columns
    return ("date" in n) or n.endswith("_dt") or ("dob" in n)

def normalize_value_for_reporting(colname, val):
    """
    For date-like columns, convert many formats to YYYY-MM-DD (string).
    Otherwise return normalized string as-is.
    """
    s = norm_str(val)
    if s == "":
        return ""
    if looks_like_date_col(colname):
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return s  # keep original if unparsable
        return dt.strftime("%Y-%m-%d")
    return s

def safe_value_counts(colname, series, topk):
    """
    Return value_counts for nonblank normalized strings.
    Date-like columns are normalized to YYYY-MM-DD for reporting.
    """
    s = series.map(lambda x: normalize_value_for_reporting(colname, x))
    s = s[s != ""]
    if len(s) == 0:
        return None, 0
    vc = s.value_counts().head(topk)
    return vc, int(len(s))


# -------------------------
# Main
# -------------------------

print("\n=== Profile encounter files for procedure info ===")

# Resolve file list
all_files = []
if FILES:
    all_files = FILES[:]
elif FOLDER:
    all_files = list_csvs_from_folder(FOLDER, recursive=FOLDER_RECURSIVE)

# Deduplicate paths (prevents repeated printing if a file is listed twice)
seen = set()
deduped = []
for p in all_files:
    if not p:
        continue
    ap = os.path.abspath(p)
    if ap in seen:
        continue
    seen.add(ap)
    deduped.append(p)
all_files = deduped

# Keep only existing
all_files = [p for p in all_files if os.path.exists(p)]
if not all_files:
    raise RuntimeError("No input files found. Edit FILES or set FOLDER.")

print("Files to scan:", len(all_files))
print("MAX_SCAN_ROWS:", "ALL" if MAX_SCAN_ROWS is None else str(MAX_SCAN_ROWS))

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
        pid = df[pat_col].map(norm_str)
        nonblank_pid_rows = int((pid != "").sum())
        unique_pats = int(pid[pid != ""].nunique())
    else:
        pid = pd.Series([""] * len(df))
        nonblank_pid_rows = 0
        unique_pats = 0

    proc_cols = [c for c in df.columns if is_procedure_like_col(c)]
    proc_cols = sorted(proc_cols, key=lambda x: str(x))

    print("  Detected patient column:", pat_col if pat_col else "(none)")
    print("  Unique patients in scan:", unique_pats)
    print("  Procedure-like columns found:", len(proc_cols))

    if not proc_cols:
        print("  No procedure-like columns detected by header heuristics.")
        summary_rows.append({
            "file": os.path.basename(path),
            "full_path": path,
            "rows_scanned": int(len(df)),
            "patient_col_detected": pat_col if pat_col else "",
            "patients_nonblank_pid_rows": nonblank_pid_rows,
            "patients_unique_in_scan": unique_pats,
            "procedure_col": "",
            "value": "",
            "row_count": 0,
            "unique_patients_with_value": 0,
        })
        continue

    for c in proc_cols:
        vc, nn = safe_value_counts(c, df[c], TOPK)
        if vc is None:
            continue

        print("\n  Top values for:", c, "(nonblank rows:", nn, ")")
        for k, v in vc.items():
            kk = k if len(str(k)) <= 140 else (str(k)[:140] + "...")
            print("    {:>6}  {}".format(int(v), kk))

        # patient-level counts for same top values (if patient id exists)
        if pat_col:
            c_norm = df[c].map(lambda x: normalize_value_for_reporting(c, x))
            tmp = pd.DataFrame({"_pid": pid, "_val": c_norm})
            tmp = tmp[(tmp["_pid"] != "") & (tmp["_val"] != "")]
            top_vals = set([str(x) for x in vc.index.tolist()])
            tmp = tmp[tmp["_val"].isin(top_vals)]
            pats_per_val = tmp.groupby("_val")["_pid"].nunique().to_dict()
        else:
            pats_per_val = {}

        for k, v in vc.items():
            k_str = str(k)
            summary_rows.append({
                "file": os.path.basename(path),
                "full_path": path,
                "rows_scanned": int(len(df)),
                "patient_col_detected": pat_col if pat_col else "",
                "patients_nonblank_pid_rows": nonblank_pid_rows,
                "patients_unique_in_scan": unique_pats,
                "procedure_col": str(c),
                "value": k_str,
                "row_count": int(v),
                "unique_patients_with_value": int(pats_per_val.get(k_str, 0)) if pat_col else 0,
            })

out_df = pd.DataFrame(summary_rows)
out_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")

print("\n==================================================")
print("Wrote summary CSV:", OUT_SUMMARY)
print("Rows in summary:", len(out_df))
print("Done.\n")
