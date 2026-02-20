# stage1_anchor_add_bins.py
# Purpose: Add Stage1 timing bins to Stage1-anchored rows (0-365 days, same-day included).

from __future__ import print_function

import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
ANCHOR_ROWS_CSV = "stage1_complication_anchor_rows.csv"

OUT_WITH_BINS = "stage1_anchor_rows_with_bins.csv"
OUT_EXCLUDE_DELTA0 = "stage1_anchor_rows_excluding_delta0.csv"
OUT_SUMMARY = "stage1_anchor_bins_summary.txt"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def time_bin_days(x):
    if x is None or pd.isnull(x):
        return "NA"
    try:
        d = int(float(x))
    except Exception:
        return "NA"
    if d < 0:
        return "<0"
    if d == 0:
        return "0d"
    if d <= 30:
        return "1-30d"
    if d <= 90:
        return "31-90d"
    if d <= 180:
        return "91-180d"
    if d <= 365:
        return "181-365d"
    return ">365d"


def main():
    ar = read_csv_safe(ANCHOR_ROWS_CSV, dtype=object)
    if ar is None or ar.empty:
        raise RuntimeError("Could not read or empty: {}".format(ANCHOR_ROWS_CSV))

    if "patient_id" not in ar.columns:
        raise RuntimeError("Missing patient_id in {}".format(ANCHOR_ROWS_CSV))

    if "DELTA_DAYS_FROM_STAGE1" not in ar.columns:
        raise RuntimeError("Missing DELTA_DAYS_FROM_STAGE1 in {}".format(ANCHOR_ROWS_CSV))

    # coerce delta to numeric
    ar["DELTA_DAYS_FROM_STAGE1"] = pd.to_numeric(ar["DELTA_DAYS_FROM_STAGE1"], errors="coerce")

    # basic integrity checks
    n_total = len(ar)
    n_null_delta = int(ar["DELTA_DAYS_FROM_STAGE1"].isnull().sum())

    # keep only non-null delta (should already be true)
    ar = ar[ar["DELTA_DAYS_FROM_STAGE1"].notnull()].copy()

    # these should already be 0..365 from anchor step, but check anyway
    n_neg = int((ar["DELTA_DAYS_FROM_STAGE1"] < 0).sum())
    n_gt365 = int((ar["DELTA_DAYS_FROM_STAGE1"] > 365).sum())

    # add bins
    ar["TIME_BIN_STAGE1"] = ar["DELTA_DAYS_FROM_STAGE1"].apply(time_bin_days)

    # optional exclusion file
    ar_no0 = ar[ar["DELTA_DAYS_FROM_STAGE1"] != 0].copy()

    # write outputs
    ar.to_csv(OUT_WITH_BINS, index=False, encoding="utf-8")
    ar_no0.to_csv(OUT_EXCLUDE_DELTA0, index=False, encoding="utf-8")

    # bin counts
    bin_counts = ar["TIME_BIN_STAGE1"].value_counts()

    lines = []
    lines.append("=== Stage1 Anchored Rows + Timing Bins (0-365d) ===")
    lines.append("Input: {}".format(ANCHOR_ROWS_CSV))
    lines.append("")
    lines.append("Row counts:")
    lines.append("  Total rows read: {}".format(n_total))
    lines.append("  Rows with null DELTA_DAYS_FROM_STAGE1 (before filter): {}".format(n_null_delta))
    lines.append("  Rows kept (non-null delta): {}".format(len(ar)))
    lines.append("  Rows with delta < 0 (should be 0): {}".format(n_neg))
    lines.append("  Rows with delta > 365 (should be 0): {}".format(n_gt365))
    lines.append("  Patients with >=1 row: {}".format(int(ar["patient_id"].nunique()) if len(ar) else 0))
    lines.append("")
    lines.append("Timing bins (TIME_BIN_STAGE1):")
    order = ["0d", "1-30d", "31-90d", "91-180d", "181-365d", ">365d", "<0", "NA"]
    for k in order:
        if k in bin_counts.index:
            lines.append("  {:>8}: {}".format(k, int(bin_counts[k])))
    for k, v in bin_counts.items():
        if k not in order:
            lines.append("  {:>8}: {}".format(str(k), int(v)))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_WITH_BINS))
    lines.append("  - {}".format(OUT_EXCLUDE_DELTA0))
    lines.append("  - {}".format(OUT_SUMMARY))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
