#!/usr/bin/env python3
# Python 3.6 compatible

import os
import pandas as pd


def classify_fp(row):
    """
    Very simple heuristic bucket assignment.
    You can refine after reviewing output.
    """

    pattern = str(row.get("STAGE2_MATCH_PATTERN", "")).lower()
    note_type = str(row.get("STAGE2_NOTE_TYPE", "")).lower()

    # 1) Removal-only cases
    if "remove" in pattern or "explant" in pattern:
        if "implant" not in pattern and "replace" not in pattern and "exchange" not in pattern:
            return "removal_only"

    # 2) Exchange but not implant exchange
    if "exchange" in pattern:
        if any(x in pattern for x in ["drain", "dressing", "catheter", "jp"]):
            return "exchange_non_surgical"

    # 3) Immediate implant placement (likely stage1)
    if "implant" in pattern and "exchange" not in pattern and "remove" not in pattern:
        return "implant_placement_only"

    # 4) Non operative note types
    if not any(x in note_type for x in ["op", "operative", "brief"]):
        return "non_op_note"

    return "other"


def main():

    base_dir = os.getcwd()
    mismatches_path = os.path.join(
        base_dir, "_outputs", "validation_mismatches.csv"
    )

    if not os.path.exists(mismatches_path):
        raise IOError("validation_mismatches.csv not found at: {}".format(mismatches_path))

    df = pd.read_csv(mismatches_path, dtype=str, encoding="latin1")

    # Adjust this column name if your mismatch file uses something else
    # Many pipelines use something like "error_type" or similar
    # We assume you have a column that marks FP rows.
    if "error_type" in df.columns:
        fp_df = df[df["error_type"] == "FP"].copy()
    elif "FP" in df.columns:
        fp_df = df[df["FP"] == "1"].copy()
    else:
        # fallback logic
        print("WARNING: Could not auto-detect FP column. Showing all rows.")
        fp_df = df.copy()

    if fp_df.empty:
        print("No FP rows found.")
        return

    # Add classification column
    fp_df["fp_reason"] = fp_df.apply(classify_fp, axis=1)

    print("\nTotal False Positives:", len(fp_df))
    print("\nCounts by reason:")
    print(fp_df["fp_reason"].value_counts())

    print("\nSample FP rows:")
    cols_to_show = [
        "ENCRYPTED_PAT_ID",
        "STAGE2_DATE",
        "STAGE2_NOTE_TYPE",
        "STAGE2_MATCH_PATTERN",
        "fp_reason"
    ]

    cols_to_show = [c for c in cols_to_show if c in fp_df.columns]

    print(fp_df[cols_to_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
