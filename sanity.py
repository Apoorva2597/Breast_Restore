# qa_stage2_revision_snippet_sampler.py
# Python 3.6.8+
#
# Purpose:
#   Create a small manual-review file of Stage2 revision hits with snippets.
#   Outputs:
#     - qa_stage2_revision_manual_review.csv
#
# Options:
#   - Change N_PER_RULE or MAX_TOTAL if you want.

from __future__ import print_function
import os
import sys
import pandas as pd

ROW_HITS = "stage2_ab_failure_revision_row_hits.csv"
OUT_CSV = "qa_stage2_revision_manual_review.csv"

N_PER_RULE = 25
MAX_TOTAL = 300  # safety cap


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def main():
    if not os.path.exists(ROW_HITS):
        raise RuntimeError("Missing required file: {}".format(ROW_HITS))

    df = read_csv_safe(ROW_HITS)
    if df.empty:
        print("Row hits file is empty; nothing to sample.")
        pd.DataFrame().to_csv(OUT_CSV, index=False, encoding="utf-8")
        print("Wrote empty:", OUT_CSV)
        return

    # Ensure required columns
    for c in ["patient_id", "S2_Revision_Flag"]:
        if c not in df.columns:
            raise RuntimeError("Missing required column: {}".format(c))

    df["S2_Revision_Flag"] = df["S2_Revision_Flag"].fillna(False).astype(bool)
    rev = df[df["S2_Revision_Flag"]].copy()
    if rev.empty:
        print("No revision rows found.")
        pd.DataFrame().to_csv(OUT_CSV, index=False, encoding="utf-8")
        print("Wrote empty:", OUT_CSV)
        return

    # Normalize event date column selection
    event_col = None
    for cand in ["EVENT_DT", "event_dt", "EVENT_DT_STD"]:
        if cand in rev.columns:
            event_col = cand
            break

    # Minimal useful columns for review
    keep_cols = ["patient_id"]
    if event_col:
        keep_cols.append(event_col)
    for c in ["stage2_dt", "DELTA_DAYS_FROM_STAGE2", "NOTE_TYPE", "NOTE_ID", "file_tag", "S2_Revision_Rule", "NOTE_SNIPPET", "NOTE_TEXT", "snippet"]:
        if c in rev.columns and c not in keep_cols:
            keep_cols.append(c)

    # If no explicit revision rule column, create one bucket
    if "S2_Revision_Rule" not in rev.columns:
        rev["S2_Revision_Rule"] = "REVISION"

    # Sample per rule
    samples = []
    for rule, g in rev.groupby("S2_Revision_Rule"):
        g2 = g.copy()
        # prefer earliest hits if event date exists
        if event_col:
            g2[event_col] = pd.to_datetime(g2[event_col], errors="coerce")
            g2 = g2.sort_values(by=[event_col], ascending=True)
        samples.append(g2.head(N_PER_RULE)[keep_cols])

    out = pd.concat(samples, ignore_index=True)
    # cap total
    if len(out) > MAX_TOTAL:
        out = out.head(MAX_TOTAL)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Revision rows total:", len(rev))
    print("Wrote manual review sample:", OUT_CSV)
    print("Sample rows:", len(out))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
