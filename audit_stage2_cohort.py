#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_stage2_cohort.py (Python 3.6.8 compatible)

Run from: ~/Breast_Restore

Read-only audit:
- Reads Stage2 anchor from newest frozen stage2_patient_clean.csv if present,
  else falls back to ./_outputs/patient_stage_summary.csv
- Optionally reads ./gold_cleaned_for_cedar.csv if present
- Reports cohort leakage:
    - HAS_STAGE2 == 0 (if available)
    - missing/unparseable STAGE2_DATE
    - gold says Stage2 not applicable (if available)
- Writes: ./_outputs/audit_stage2_leakers.csv
"""

from __future__ import print_function

import os
import glob
import re
from datetime import datetime
import pandas as pd


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def normalize_id(x):
    if x is None:
        return ""
    return str(x).strip()


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def parse_date_any(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%y %H:%M:%S",
        "%m/%d/%y %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        token = m.group(1)
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(token, fmt).date()
            except Exception:
                pass
    m = re.search(r"(\d{4}-\d{1,2}-\d{1,2})", s)
    if m:
        token = m.group(1)
        try:
            return datetime.strptime(token, "%Y-%m-%d").date()
        except Exception:
            pass
    return None


def ensure_encpat_col(df, label=""):
    df = normalize_cols(df)
    options = [
        "ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID",
        "Encrypted_Pat_ID", "Encrypted_Patient_ID",
        "encrypted_pat_id", "encrypted_patient_id",
    ]
    found = None
    for c in options:
        if c in df.columns:
            found = c
            break
    if not found:
        raise ValueError("No ENCRYPTED_PAT_ID-like column found in {}. Cols: {}".format(label, list(df.columns)))
    if found != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={found: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    return df


def find_latest_frozen_anchor(root):
    base = os.path.join(root, "_frozen_stage2")
    if not os.path.isdir(base):
        return None
    candidates = sorted(glob.glob(os.path.join(base, "*", "stage2_patient_clean.csv")))
    return os.path.abspath(candidates[-1]) if candidates else None


def load_stage2_anchor(root):
    frozen = find_latest_frozen_anchor(root)
    if frozen:
        df = read_csv_robust(frozen, dtype=str, low_memory=False)
        df = ensure_encpat_col(df, label=os.path.basename(frozen))
        df = normalize_cols(df)
        src = frozen
    else:
        src = os.path.join(root, "_outputs", "patient_stage_summary.csv")
        if not os.path.isfile(src):
            raise IOError("No frozen anchor and missing {}".format(src))
        df = read_csv_robust(src, dtype=str, low_memory=False)
        df = ensure_encpat_col(df, label=os.path.basename(src))
        df = normalize_cols(df)

    if "STAGE2_DATE" not in df.columns:
        raise ValueError("Anchor file missing STAGE2_DATE: {}".format(src))

    df["STAGE2_DATE_PARSED"] = df["STAGE2_DATE"].map(parse_date_any)

    if "HAS_STAGE2" in df.columns:
        df["HAS_STAGE2_01"] = df["HAS_STAGE2"].map(to01)
    else:
        df["HAS_STAGE2_01"] = None

    return df, src


def load_gold(root):
    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    if not os.path.isfile(gold_path):
        return None, None
    g = read_csv_robust(gold_path, dtype=str, low_memory=False)
    g = normalize_cols(g)

    # detect MRN col
    mrn_col = None
    for cand in ["MRN", "Pat_MRN", "PAT_MRN", "PATIENT_MRN", "MEDICAL_RECORD_NUMBER"]:
        if cand in g.columns:
            mrn_col = cand
            break
    if not mrn_col:
        # fallback heuristic
        for c in g.columns:
            if "mrn" in str(c).lower():
                mrn_col = c
                break

    # detect Stage2 applicable col(s)
    stage2_app_col = None
    for c in g.columns:
        if "stage2" in str(c).lower() and "app" in str(c).lower():
            stage2_app_col = c
            break

    has_stage2_col = None
    for c in g.columns:
        if str(c).upper() in ["GOLD_HAS_STAGE2", "HAS_STAGE2", "HAS_STAGE_2"]:
            has_stage2_col = c
            break

    return g, {"path": gold_path, "mrn_col": mrn_col, "stage2_app_col": stage2_app_col, "has_stage2_col": has_stage2_col}


def main():
    root = os.path.abspath(".")
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    anchor_df, anchor_src = load_stage2_anchor(root)
    print("\nANCHOR SOURCE:", anchor_src)
    print("Anchor rows:", len(anchor_df))
    print("Unique ENCRYPTED_PAT_ID:", anchor_df["ENCRYPTED_PAT_ID"].nunique())

    # Gate checks within anchor
    has_has_stage2 = "HAS_STAGE2" in anchor_df.columns
    n_has0 = int((anchor_df["HAS_STAGE2_01"] == 0).sum()) if has_has_stage2 else 0
    n_missing_date = int(anchor_df["STAGE2_DATE_PARSED"].isna().sum())

    print("\nANCHOR GATE CHECKS")
    print("  HAS_STAGE2 col present:", bool(has_has_stage2))
    if has_has_stage2:
        print("  Rows with HAS_STAGE2==0:", n_has0)
    print("  Rows with missing/unparseable STAGE2_DATE:", n_missing_date)

    # gold cross-check (only if we can map)
    gold_df, gold_meta = load_gold(root)
    leakers = anchor_df.copy()
    leakers["LEAK_REASON"] = ""

    if has_has_stage2:
        leakers.loc[leakers["HAS_STAGE2_01"] == 0, "LEAK_REASON"] += "|HAS_STAGE2_0"
    leakers.loc[leakers["STAGE2_DATE_PARSED"].isna(), "LEAK_REASON"] += "|MISSING_STAGE2_DATE"

    if gold_df is not None:
        print("\nGOLD FOUND:", gold_meta["path"])
        if not gold_meta["mrn_col"]:
            print("  WARNING: could not detect MRN column in gold, skipping gold cross-check.")
        else:
            # For gold cross-check you must have MRN in anchor file OR do crosswalk elsewhere.
            # We only do this if anchor_df already has MRN.
            anchor_mrn_col = None
            for c in anchor_df.columns:
                if str(c).upper() == "MRN" or "mrn" in str(c).lower():
                    anchor_mrn_col = c
                    break

            if not anchor_mrn_col:
                print("  NOTE: anchor file has no MRN column; gold cross-check skipped (needs MRN<->ENCRYPTED_PAT_ID mapping).")
            else:
                tmp = anchor_df[[ "ENCRYPTED_PAT_ID", anchor_mrn_col ]].copy()
                tmp[anchor_mrn_col] = tmp[anchor_mrn_col].map(normalize_id)

                g = gold_df[[gold_meta["mrn_col"]]].copy()
                g[gold_meta["mrn_col"]] = g[gold_meta["mrn_col"]].map(normalize_id)

                # bring Stage2 applicability info
                keep_cols = [gold_meta["mrn_col"]]
                if gold_meta["stage2_app_col"]:
                    keep_cols.append(gold_meta["stage2_app_col"])
                if gold_meta["has_stage2_col"] and gold_meta["has_stage2_col"] not in keep_cols:
                    keep_cols.append(gold_meta["has_stage2_col"])
                g = gold_df[keep_cols].copy()
                g[gold_meta["mrn_col"]] = g[gold_meta["mrn_col"]].map(normalize_id)

                merged = tmp.merge(g, left_on=anchor_mrn_col, right_on=gold_meta["mrn_col"], how="left")

                # Decide eligibility based on available columns
                elig = None
                if gold_meta["stage2_app_col"]:
                    elig = merged[gold_meta["stage2_app_col"]].map(to01)
                    merged["GOLD_STAGE2_ELIG_01"] = elig
                elif gold_meta["has_stage2_col"]:
                    elig = merged[gold_meta["has_stage2_col"]].map(to01)
                    merged["GOLD_STAGE2_ELIG_01"] = elig
                else:
                    merged["GOLD_STAGE2_ELIG_01"] = None

                # annotate leakers by gold if available
                if merged["GOLD_STAGE2_ELIG_01"].notnull().any():
                    bad = merged["GOLD_STAGE2_ELIG_01"] == 0
                    bad_pids = set(merged.loc[bad, "ENCRYPTED_PAT_ID"].tolist())
                    leakers.loc[leakers["ENCRYPTED_PAT_ID"].isin(bad_pids), "LEAK_REASON"] += "|GOLD_STAGE2_NOT_ELIG"
                    print("  Rows where gold indicates NOT Stage2-eligible:", int(bad.sum()))
                else:
                    print("  NOTE: gold has no Stage2 eligibility column found; cannot flag gold-based leakage.")
    else:
        print("\nNo gold_cleaned_for_cedar.csv found; skipping gold cross-check.")

    # Emit leakers table
    leakers = leakers[leakers["LEAK_REASON"].str.len() > 0].copy()
    leakers["LEAK_REASON"] = leakers["LEAK_REASON"].str.strip("|")

    out_csv = os.path.join(out_dir, "audit_stage2_leakers.csv")
    # keep only useful columns
    keep = []
    for c in ["ENCRYPTED_PAT_ID", "STAGE2_DATE", "STAGE2_DATE_PARSED", "HAS_STAGE2", "HAS_STAGE2_01", "LEAK_REASON"]:
        if c in leakers.columns:
            keep.append(c)
    # add MRN if present
    for c in leakers.columns:
        if "mrn" in str(c).lower() and c not in keep:
            keep.insert(1, c)
            break

    leakers[keep].to_csv(out_csv, index=False)

    print("\nLEAKERS SUMMARY")
    print("  Total leaker rows:", len(leakers))
    if len(leakers) > 0:
        print("  Wrote:", out_csv)
        print("  Reasons count:")
        print(leakers["LEAK_REASON"].value_counts().head(20).to_string())
    else:
        print("  ✅ No leakers detected by HAS_STAGE2 / STAGE2_DATE gates (and gold check if applied).")

    print("\nDone.")


if __name__ == "__main__":
    main()
