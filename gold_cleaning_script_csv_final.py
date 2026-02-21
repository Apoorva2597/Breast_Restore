# gold_cleaning_script_csv_final.py
# Reads the CSV export of the gold spreadsheet on CEDAR and reconstructs a true patient-level table.
# No openpyxl needed.

from __future__ import print_function

import re
import sys
import pandas as pd

INPUT_CSV = "Breast-Restore.csv"
OUTPUT_CSV = "gold_cleaned_for_cedar.csv"

# Excel row numbers (confirmed red rows)
DROP_EXCEL_ROWS = [29, 198, 203, 284]

# In the original Excel, the true header row was row 3.
# We still use this for Excel row mapping AFTER we locate the real header row in the CSV.
HEADER_EXCEL_ROW = 3


def clean_header(name):
    s = "" if name is None else str(name)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)

    # Remove leading numbering: "34. X" or "34 X"
    s = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", s)
    s = re.sub(r"^\s*\d+\s+", "", s)

    return s.strip().strip(":").strip()


def normalize_cell(x):
    if x is None:
        return ""
    s = str(x)

    if s.lower() == "nan":
        return ""

    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass

    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)

    if s in ["", ".", "-", "—", "–"]:
        return ""

    return s


def row_has_any_token(row_vals, tokens):
    # row_vals: list of cell strings (already normalized-ish)
    s = " | ".join([v.lower() for v in row_vals if v is not None])
    for t in tokens:
        if t.lower() in s:
            return True
    return False


def find_true_header_row(raw_df):
    # raw_df is read with header=None
    # We find a row that looks like the real patient-level header.
    # Strong signal: contains MRN and PatientID (or at least MRN).
    tokens_strong = ["mrn", "patientid"]
    tokens_ok = ["mrn"]

    for i in range(len(raw_df)):
        row = raw_df.iloc[i].tolist()
        row_norm = [normalize_cell(x) for x in row]
        if row_has_any_token(row_norm, tokens_strong):
            return i

    for i in range(len(raw_df)):
        row = raw_df.iloc[i].tolist()
        row_norm = [normalize_cell(x) for x in row]
        if row_has_any_token(row_norm, tokens_ok):
            return i

    return None


def main():
    # Read raw CSV WITHOUT trusting header
    raw = pd.read_csv(INPUT_CSV, header=None, dtype=str, keep_default_na=False)

    header_idx = find_true_header_row(raw)
    if header_idx is None:
        raise RuntimeError(
            "Could not find a header row containing MRN/PatientID in the CSV. "
            "Open the CSV and confirm it contains MRN somewhere."
        )

    # Build patient-level df using that row as header
    header_row = raw.iloc[header_idx].tolist()
    header_row = [clean_header(normalize_cell(x)) for x in header_row]

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = header_row

    # Drop empty/unnamed columns
    keep_cols = []
    for c in df.columns:
        if not c:
            continue
        if str(c).strip().lower().startswith("unnamed"):
            continue
        keep_cols.append(c)
    df = df.loc[:, keep_cols].copy()

    # Add excel row mapping column:
    # We want this mapping to reflect the original Excel row numbers.
    #
    # IMPORTANT:
    # The CSV contains extra junk rows above the real header, so header_idx will not equal HEADER_EXCEL_ROW-1.
    # We still create an excel row counter in the "patient table" space:
    # First data row after true header corresponds to Excel row (HEADER_EXCEL_ROW + 1)
    #
    # This keeps your original “red row” numbering logic consistent with how the spreadsheet was indexed.
    df["_excel_row_"] = list(range(HEADER_EXCEL_ROW + 1, HEADER_EXCEL_ROW + 1 + len(df)))

    original_rows = len(df)

    # Normalize all cells
    for c in df.columns:
        if c == "_excel_row_":
            continue
        df[c] = df[c].apply(normalize_cell)

    # Drop confirmed red rows
    drop_set = set(DROP_EXCEL_ROWS)
    before_drop = len(df)
    df = df[~df["_excel_row_"].isin(drop_set)].copy()
    dropped_red = before_drop - len(df)

    # Drop rows that are clearly not patients:
    # - MRN empty AND PatientID empty (if those cols exist)
    cols_lower = {str(c).lower(): c for c in df.columns}
    mrn_col = cols_lower.get("mrn", None)
    pid_col = None
    for k in cols_lower.keys():
        if "patientid" == k or k.endswith("patientid"):
            pid_col = cols_lower[k]
            break

    if mrn_col is not None:
        df = df[df[mrn_col].astype(str).str.strip() != ""].copy()

    # Print sanity info
    print("=== Gold CSV Cleaning (CEDAR) ===")
    print("Input:", INPUT_CSV)
    print("Output:", OUTPUT_CSV)
    print("")
    print("Detected true header row index in CSV (0-based):", header_idx)
    print("Final headers:")
    print(list(df.columns))
    print("")
    if len(df) > 0:
        print("First row after cleaning:")
        print(df.iloc[0].to_dict())
    else:
        print("WARNING: No rows left after cleaning.")

    print("")
    print("Patient-table rows (before red-row drop):", original_rows)
    print("Rows dropped (red rows):", dropped_red)
    print("Final rows:", len(df))

    out = df.drop(columns=["_excel_row_"], errors="ignore")
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("")
    print("Wrote:", OUTPUT_CSV)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
