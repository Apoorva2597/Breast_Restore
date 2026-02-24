#!/usr/bin/env python3
# revision_only_check.py
# Python 3.6.8 compatible

from __future__ import print_function
import os
import argparse
import pandas as pd


def _norm(s):
    return str(s).strip().lower()

def _pick_col(cols, candidates):
    """
    Pick first column whose normalized name matches any candidate exactly,
    or contains candidate as a substring.
    """
    cols_norm = {c: _norm(c) for c in cols}
    # exact match
    for cand in candidates:
        candn = _norm(cand)
        for c, cn in cols_norm.items():
            if cn == candn:
                return c
    # substring match
    for cand in candidates:
        candn = _norm(cand)
        for c, cn in cols_norm.items():
            if candn in cn:
                return c
    return None

def _to_bool_series(s):
    """
    Convert common truthy/falsey strings into booleans safely.
    """
    if s is None:
        return None
    x = s.astype(str).str.strip().str.lower()
    truthy = x.isin(["true", "t", "1", "yes", "y"])
    falsy  = x.isin(["false", "f", "0", "no", "n", ""])
    out = pd.Series([None] * len(x), index=x.index, dtype=object)
    out[truthy] = True
    out[falsy] = False
    # anything else -> False (conservative) or keep None; choose conservative False
    out[out.isnull()] = False
    return out.astype(bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("MASTER__STAGING_PATHWAY__vNEW.csv", required=True, help="Path to MASTER__STAGING_PATHWAY__vNEW.csv")
    args = ap.parse_args()

    path = args.master_csv
    if not os.path.exists(path):
        raise RuntimeError("File not found: {}".format(path))

    # NOTE: no `errors=` kwarg (older pandas compatibility)
    df = pd.read_csv(path, dtype=str, engine="python", encoding="utf-8")

    # Identify columns (robust to naming variations)
    pid_col = _pick_col(df.columns, ["patient_id", "encrypted_pat_id", "pat_id", "pid"])
    has_exp_col = _pick_col(df.columns, ["has_expander", "has_expander_refined", "expander_present"])
    stage2_def_col = _pick_col(df.columns, ["stage2_definitive_present", "has_stage2_flap", "stage2_definitive"])
    rev_only_col = _pick_col(df.columns, ["revision_only_flagged", "revision_only", "rev_only"])

    missing = []
    if pid_col is None: missing.append("patient_id")
    if has_exp_col is None: missing.append("has_expander")
    if stage2_def_col is None: missing.append("stage2_definitive_present")
    if rev_only_col is None: missing.append("revision_only_flagged")

    if missing:
        print("Could not auto-detect required column(s):", ", ".join(missing))
        print("Available columns:")
        for c in df.columns:
            print(" -", c)
        raise SystemExit(1)

    has_exp = _to_bool_series(df[has_exp_col])
    stage2_def = _to_bool_series(df[stage2_def_col])
    rev_only = _to_bool_series(df[rev_only_col])

    total = df[pid_col].nunique()
    n_has_exp = int(has_exp.sum())
    n_stage2_def = int(stage2_def.sum())
    n_rev_only = int(rev_only.sum())
    n_rev_only_and_exp = int((rev_only & has_exp).sum())
    n_rev_only_and_noexp = int((rev_only & (~has_exp)).sum())

    print("\nCOLUMN MAP:")
    print("  pid_col        :", pid_col)
    print("  has_expander   :", has_exp_col)
    print("  stage2_def     :", stage2_def_col)
    print("  revision_only  :", rev_only_col)

    print("\nCOUNTS:")
    print("  total patients                 :", total)
    print("  has_expander=True              :", n_has_exp)
    print("  stage2_definitive=True         :", n_stage2_def)
    print("  revision_only=True             :", n_rev_only)
    print("  revision_only AND has_expander :", n_rev_only_and_exp)
    print("  revision_only AND NO expander  :", n_rev_only_and_noexp)

    # Quick 2x2 breakdown
    breakdown = pd.DataFrame({
        "has_expander": has_exp,
        "revision_only": rev_only,
        "stage2_definitive": stage2_def
    })

    # table: revision_only x has_expander
    tab = breakdown.groupby(["revision_only", "has_expander"]).size().reset_index(name="patients")
    print("\nBREAKDOWN (revision_only x has_expander):")
    print(tab.to_string(index=False))

    # Optional: save the subset list for follow-up QA
    out_subset = os.path.join(os.path.dirname(path), "qa_revision_only_patients.csv")
    subset_df = df.loc[rev_only, [pid_col]].drop_duplicates()
    subset_df.to_csv(out_subset, index=False, encoding="utf-8")
    print("\nWROTE:", out_subset)


if __name__ == "__main__":
    main()
