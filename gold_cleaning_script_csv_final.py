# gold_cleaning_script_csv_final.py
# Works on CEDAR using only pandas + stdlib (no openpyxl needed).
# Input is the CSV export of the gold spreadsheet.
#
# What it does:
# - reads Breast-Restore.csv as strings (avoids weird dtype issues)
# - cleans column headers (removes numbering like "34. X", removes "Unnamed")
# - drops known "red rows" by their original Excel row numbers
# - trims whitespace and normalizes blank-like cells to empty
# - writes gold_cleaned_for_cedar.csv
# - prints: cleaned headers, first row, original vs final row counts, rows dropped

from __future__ import print_function

import re
import sys
import pandas as pd

INPUT_CSV = "Breast-Restore.csv"
OUTPUT_CSV = "gold_cleaned_for_cedar.csv"

# These are Excel row numbers (as you confirmed): 29, 198, 203, 284
# We map them to CSV row positions below using HEADER_EXCEL_ROW.
DROP_EXCEL_ROWS = [29, 198, 203, 284]

# In the original Excel, the true header row was row 3.
# That means:
# - Excel row 3  -> CSV header
# - Excel row 4  -> first data row (CSV data index 0)
HEADER_EXCEL_ROW = 3


def clean_header(name):
    s = "" if name is None else str(name)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)

    # Remove leading numbering patterns:
    # "34. Stage1_Reoperation" -> "Stage1_Reoperation"
    # "34 Stage1_Reoperation"  -> "Stage1_Reoperation"
    s = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", s)
    s = re.sub(r"^\s*\d+\s+", "", s)

    # Normalize common weird header artifacts
    s = s.strip().strip(":").strip()
    return s


def normalize_cell(x):
    if x is None:
        return ""
    s = str(x)

    # pandas might already have "nan" as a string if we read dtype=str in older versions
    if s.lower() == "nan":
        return ""

    # normalize whitespace and non-breaking spaces
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"\s+", " ", s)

    # treat common blank-like tokens as empty
    if s in ["", ".", "-", "—", "–"]:
        return ""

    return s


def main():
    # Read everything as string so we don't depend on pandas.NA behavior
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

    original_rows = len(df)

    # Clean headers
    new_cols = []
    for c in df.columns:
        new_cols.append(clean_header(c))
    df.columns = new_cols

    # Drop unnamed columns
    keep_cols = []
    for c in df.columns:
        if not c:
            continue
        if c.strip().lower().startswith("unnamed"):
            continue
        keep_cols.append(c)
    df = df.loc[:, keep_cols].copy()

    # Add a reference column that stores original Excel row number
    # CSV data row 0 corresponds to Excel row (HEADER_EXCEL_ROW + 1)
    df["_excel_row_"] = list(range(HEADER_EXCEL_ROW + 1, HEADER_EXCEL_ROW + 1 + len(df)))

    # Drop the confirmed red rows
    drop_set = set(DROP_EXCEL_ROWS)
    before_drop = len(df)
    df = df[~df["_excel_row_"].isin(drop_set)].copy()
    dropped_red = before_drop - len(df)

    # Normalize all cells
    for c in df.columns:
        if c == "_excel_row_":
            continue
        df[c] = df[c].apply(normalize_cell)

    # Print quick sanity outputs
    print("=== Gold CSV Cleaning (CEDAR) ===")
    print("Input:", INPUT_CSV)
    print("Output:", OUTPUT_CSV)
    print("")
    print("Final headers:")
    print(list(df.columns))
    print("")
    if len(df) > 0:
        first_row = df.iloc[0].to_dict()
        print("First row after cleaning:")
        print(first_row)
    else:
        print("WARNING: No rows left after cleaning.")

    print("")
    print("Original rows:", original_rows)
    print("Rows dropped (red rows):", dropped_red)
    print("Final rows:", len(df))

    # Write output (drop helper column)
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
