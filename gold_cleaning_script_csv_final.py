import re
import numpy as np
import pandas as pd

# -------------------------
# CONFIG (edit if needed)
# -------------------------
INPUT_CSV = "Breast-Restore.csv"
OUTPUT_CSV = "gold_cleaned_for_cedar.csv"

# EXCEL row numbers from the original sheet (1-based)
DROP_EXCEL_ROWS = [29, 198, 203, 284]

# Tokens treated as blank
BLANK_TOKENS = {"na", "n/a", "none", "null", ".", "-", "--", "nan"}


# -------------------------
# Helpers
# -------------------------
def clean_header(s):
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^\s*\d+\s*\.\s*", "", s)  # remove leading "34. "
    return s


def norm_cell(x):
    if x is None:
        return np.nan

    # keep numeric values as-is (except NaN)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and np.isnan(x):
            return np.nan
        return x

    s = str(x)
    try:
        s = s.replace("\xa0", " ")
    except Exception:
        pass
    s = s.strip()

    if s == "":
        return np.nan
    if s.lower() in BLANK_TOKENS:
        return np.nan
    return s


def find_true_header_row(path, max_scan=60):
    """
    Scan first max_scan rows with header=None and find the row that looks like the
    real column header. We look for MRN + PatientID (or close variants).
    Returns 0-based row index.
    """
    preview = pd.read_csv(path, header=None, nrows=max_scan, engine="python", encoding="latin1")
    best_i = None

    for i in range(len(preview)):
        row_vals = [clean_header(v) for v in preview.iloc[i].tolist()]
        row_up = [str(v).strip().upper() for v in row_vals if str(v).strip() != ""]
        joined = " | ".join(row_up)

        has_mrn = ("MRN" in row_up) or ("MRN" in joined)
        has_pid = ("PATIENTID" in row_up) or ("PATIENT_ID" in joined) or ("PATIENTID" in joined)

        if has_mrn and has_pid:
            best_i = i
            break

    if best_i is None:
        for i in range(len(preview)):
            row_vals = [clean_header(v) for v in preview.iloc[i].tolist()]
            row_up = [str(v).strip().upper() for v in row_vals if str(v).strip() != ""]
            if "MRN" in row_up or "MRN" in " | ".join(row_up):
                best_i = i
                break

    return 0 if best_i is None else int(best_i)


def looks_like_date_col(colname):
    cl = str(colname).strip().lower()
    if "dob" in cl:
        return True
    if cl.endswith("_dt"):
        return True
    if "date" in cl:
        return True
    return False


# -------------------------
# Main
# -------------------------
print("\n=== Gold CSV Cleaning (CEDAR) ===")
print("Input:", INPUT_CSV)
print("Output:", OUTPUT_CSV)

# 1) Detect real header row
hdr_idx = find_true_header_row(INPUT_CSV, max_scan=60)
print("\nDetected true header row index in CSV (0-based):", hdr_idx)

# 2) Read from that header row
df = pd.read_csv(
    INPUT_CSV,
    header=hdr_idx,
    engine="python",
    encoding="latin1",
    dtype=object
)

original_rows = len(df)

# 3) Clean headers and drop Unnamed
df.columns = [clean_header(c) for c in df.columns]
df = df.loc[:, [c for c in df.columns if c and not str(c).lower().startswith("unnamed")]].copy()

print("\nFinal headers (first 30):")
print(list(df.columns)[:30])

# 4) Normalize cells
for col in df.columns:
    df[col] = df[col].map(norm_cell)

# 5) Drop fully blank rows (all NaN)
df = df[~df.isna().all(axis=1)].copy()

# 6) Drop identifier-only rows (MRN/PatientID/Name/DOB filled but everything else empty)
id_cols = [c for c in df.columns if re.search(r"(mrn|patientid|name|dob)", str(c), re.I)]
var_cols = [c for c in df.columns if c not in id_cols]

if id_cols and var_cols:
    has_id = df[id_cols].notna().any(axis=1)
    no_vars = df[var_cols].isna().all(axis=1)
    df = df[~(has_id & no_vars)].copy()

# 7) Drop flagged red rows
excel_row_col = None
for c in df.columns:
    if str(c).strip().lower() in ["__excel_row__", "_excel_row_", "excel_row", "excelrow"]:
        excel_row_col = c
        break

if excel_row_col is not None:
    df[excel_row_col] = pd.to_numeric(df[excel_row_col], errors="coerce")
    before = len(df)
    df = df[~df[excel_row_col].isin(DROP_EXCEL_ROWS)].copy()
    print("\nDropped red rows using {}: {} rows removed".format(excel_row_col, before - len(df)))
else:
    # reconstruct excel row numbers (best effort)
    first_data_excel_row = (hdr_idx + 1) + 1  # header row (1-based) + 1
    df = df.reset_index(drop=True)
    df.insert(0, "__excel_row__", range(first_data_excel_row, first_data_excel_row + len(df)))
    before = len(df)
    df = df[~df["__excel_row__"].isin(DROP_EXCEL_ROWS)].copy()
    print("\nDropped red rows using reconstructed __excel_row__: {} rows removed".format(before - len(df)))

# 8) Drop rows where all Stage1 outcomes are blank
stage1_cols = [c for c in df.columns if "Stage1" in str(c)]
if stage1_cols:
    df = df[~df[stage1_cols].isna().all(axis=1)].copy()
else:
    print("\nWARNING: No Stage1 columns found.")

# 9) Force MRN / PatientID to string
for c in df.columns:
    if re.search(r"(mrn|patientid)", str(c), re.I):
        df[c] = df[c].fillna("").astype(str).str.strip()

# 10) Add Stage2_Applicable (helper only; does not modify Stage2 columns)
stage2_cols = [c for c in df.columns if "Stage2" in str(c)]
if stage2_cols:
    df["Stage2_Applicable"] = df[stage2_cols].notna().any(axis=1).astype(int)
else:
    print("\nWARNING: No Stage2 columns found.")

# 11) Force ALL date-like columns to YYYY-MM-DD strings (no time)
for c in df.columns:
    if looks_like_date_col(c):
        dt = pd.to_datetime(df[c], errors="coerce")
        df[c] = dt.dt.strftime("%Y-%m-%d")
        df[c] = df[c].fillna("")

# 12) Summary + write
final_rows = len(df)
print("\nRow counts:")
print("Original rows (raw read):", original_rows)
print("Final rows (cleaned):", final_rows)
print("Total dropped:", original_rows - final_rows)
print("Stage1 columns detected:", len(stage1_cols))
print("Stage2 columns detected:", len(stage2_cols))
if stage2_cols:
    print("Stage2_Applicable=1 count:", int(df["Stage2_Applicable"].sum()))

print("\nFirst row after cleaning (after date formatting):")
if final_rows > 0:
    print(df.iloc[0].to_dict())
else:
    print("No rows left.")

# quick DOB peek
if "DOB" in df.columns:
    print("\nDOB sample (first 10):", df["DOB"].head(10).tolist())

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print("\nWrote:", OUTPUT_CSV)
