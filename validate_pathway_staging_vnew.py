#!/usr/bin/env python3
# validate_pathway_staging_vnew.py  (Python 3.6 compatible)

from __future__ import print_function
import os
import pandas as pd

MASTER = "/home/apokol/Breast_Restore/MASTER__STAGING_PATHWAY__vNEW.csv"

def read_csv(path):
    try:
        return pd.read_csv(path, dtype=str, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, engine="python", encoding="latin1")

def to_bool(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "t", "yes", "y"])

def to_int(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def main():
    if not os.path.exists(MASTER):
        raise RuntimeError("Missing master staging file: {}".format(MASTER))

    df = read_csv(MASTER)

    required = ["patient_id", "has_expander", "has_stage2_definitive", "revision_only_flag",
                "stage1_date", "stage2_date",
                "counts_19357", "counts_19364", "counts_19380", "counts_19350",
                "encounter_rows_loaded"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("WARNING: missing columns:", missing)

    # normalize types
    df["has_expander_b"] = to_bool(df.get("has_expander", pd.Series([""]*len(df))))
    df["has_stage2_b"] = to_bool(df.get("has_stage2_definitive", pd.Series([""]*len(df))))
    df["revision_only_b"] = to_bool(df.get("revision_only_flag", pd.Series([""]*len(df))))

    for c in ["counts_19357","counts_19364","counts_19380","counts_19350","encounter_rows_loaded"]:
        df[c] = to_int(df.get(c, 0))

    # parse dates
    df["stage1_dt"] = pd.to_datetime(df.get("stage1_date", ""), errors="coerce")
    df["stage2_dt"] = pd.to_datetime(df.get("stage2_date", ""), errors="coerce")

    # headline counts
    total = len(df)
    n_stage1 = int(df["has_expander_b"].sum())
    n_stage2 = int(df["has_stage2_b"].sum())
    n_rev_only = int(df["revision_only_b"].sum())
    n_zero_enc = int((df["encounter_rows_loaded"] == 0).sum())

    print("\n=== HEADLINE ===")
    print("Rows (patients):", total)
    print("Stage1 (has_expander=True):", n_stage1)
    print("Stage2 definitive (True):", n_stage2)
    print("Revision-only flagged (True):", n_rev_only)
    print("Patients with 0 encounter rows loaded:", n_zero_enc)

    # integrity checks
    print("\n=== INTEGRITY CHECKS ===")

    # A) staged but missing date
    miss_s1_date = df[df["has_expander_b"] & df["stage1_dt"].isnull()]
    miss_s2_date = df[df["has_stage2_b"] & df["stage2_dt"].isnull()]
    print("Stage1 True but missing stage1_date:", len(miss_s1_date))
    print("Stage2 True but missing stage2_date:", len(miss_s2_date))

    # B) Stage2 before Stage1
    order_bad = df[df["has_expander_b"] & df["has_stage2_b"] &
                   df["stage1_dt"].notnull() & df["stage2_dt"].notnull() &
                   (df["stage2_dt"] < df["stage1_dt"])]
    print("Stage2 date < Stage1 date (both present):", len(order_bad))

    # C) Stage2 True but no 19364 count
    s2_count_mismatch = df[df["has_stage2_b"] & (df["counts_19364"] == 0)]
    print("Stage2 True but counts_19364==0:", len(s2_count_mismatch))

    # D) Revision-only but has 19364
    rev_only_has_s2code = df[df["revision_only_b"] & (df["counts_19364"] > 0)]
    print("Revision-only flagged but counts_19364>0:", len(rev_only_has_s2code))

    # E) Stage2 code present but has_stage2 False (should be rare/0)
    s2code_but_false = df[(df["counts_19364"] > 0) & (~df["has_stage2_b"])]
    print("counts_19364>0 but has_stage2_definitive=False:", len(s2code_but_false))

    # Summaries / distributions
    print("\n=== QUICK DISTRIBUTIONS ===")
    print("counts_19364 value counts (top 10):")
    print(df["counts_19364"].value_counts().head(10).to_string())

    print("\nencounter_rows_loaded (quantiles):")
    try:
        q = df["encounter_rows_loaded"].quantile([0, .25, .5, .75, .9, .95, 1.0])
        print(q.to_string())
    except Exception:
        pass

    # Write problem lists (so you can inspect later if validation looks off)
    outdir = "/home/apokol/Breast_Restore/VALIDATION_REPORTS"
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    def write_list(name, sub):
        path = os.path.join(outdir, name)
        cols = ["patient_id","has_expander","stage1_date","has_stage2_definitive","stage2_date",
                "revision_only_flag","counts_19357","counts_19364","counts_19380","counts_19350",
                "encounter_rows_loaded"]
        cols = [c for c in cols if c in sub.columns]
        sub[cols].to_csv(path, index=False, encoding="utf-8")
        print("WROTE:", path, "| rows:", len(sub))

    write_list("stage1_true_missing_date.csv", miss_s1_date)
    write_list("stage2_true_missing_date.csv", miss_s2_date)
    write_list("stage2_before_stage1.csv", order_bad)
    write_list("stage2_true_counts0.csv", s2_count_mismatch)
    write_list("revision_only_but_has_19364.csv", rev_only_has_s2code)
    write_list("counts19364_but_stage2_false.csv", s2code_but_false)

    print("\nDone. Validation artifacts in:", outdir)

if __name__ == "__main__":
    main()
