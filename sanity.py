#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_exemplars_all_outcomes.py  (Python 3.6.8 compatible)

Run from:  ~/Breast_Restore

Input:
  ./_outputs/validation_merged.csv

Output:
  ./_outputs/exemplar_cases_by_outcome.csv

What it does:
- For each outcome, picks up to 1 case each for FP/FN/TP/TN:
    FP: gold=0 pred=1
    FN: gold=1 pred=0
    TP: gold=1 pred=1
    TN: gold=0 pred=0
- Includes MRN + ENCRYPTED_PAT_ID (if available) + STAGE2_DATE/window + evidence fields/snippets when present.
"""

from __future__ import print_function
import os
import pandas as pd


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def to01(x):
    if x is None:
        return 0
    s = str(x).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


def pick_first_existing(cols, options):
    for c in options:
        if c in cols:
            return c
    return None


def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def main():
    root = os.path.abspath(".")
    in_path = os.path.join(root, "_outputs", "validation_merged.csv")
    out_path = os.path.join(root, "_outputs", "exemplar_cases_by_outcome.csv")

    if not os.path.isfile(in_path):
        raise IOError("Missing input: {}".format(in_path))

    df = read_csv_robust(in_path, dtype=str, low_memory=False)

    cols = list(df.columns)

    # ID columns (best effort)
    mrn_col = pick_first_existing(cols, ["MRN", "Mrn", "mrn"])
    pid_col = pick_first_existing(cols, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "Encrypted_Pat_ID"])

    # Date/window columns
    stage2_col = pick_first_existing(cols, ["STAGE2_DATE", "Stage2_DATE"])
    wstart_col = pick_first_existing(cols, ["WINDOW_START"])
    wend_col = pick_first_existing(cols, ["WINDOW_END"])

    # Outcomes we care about (gold, pred)
    # NOTE: "MajorComp" pred is Stage2_MajorComp_pred; gold is often GOLD_Stage2_MajorComp
    outcomes = [
        ("MinorComp", "GOLD_Stage2_MinorComp", "Stage2_MinorComp_pred"),
        ("Reoperation", "GOLD_Stage2_Reoperation", "Stage2_Reoperation_pred"),
        ("Rehospitalization", "GOLD_Stage2_Rehospitalization", "Stage2_Rehospitalization_pred"),
        ("MajorComp", "GOLD_Stage2_MajorComp", "Stage2_MajorComp_pred"),
        ("Failure", "GOLD_Stage2_Failure", "Stage2_Failure_pred"),
        ("Revision", "GOLD_Stage2_Revision", "Stage2_Revision_pred"),
    ]

    # Evidence columns (best effort mapping per outcome)
    evidence_map = {
        "MinorComp": ["minor_evidence_date", "minor_evidence_source", "minor_evidence_note_id", "minor_evidence_pattern", "minor_evidence_snippet"],
        "Reoperation": ["reop_evidence_date", "reop_evidence_source", "reop_evidence_note_id", "reop_evidence_pattern", "reop_evidence_snippet"],
        "Rehospitalization": ["rehosp_evidence_date", "rehosp_evidence_source", "rehosp_evidence_note_id", "rehosp_evidence_pattern", "rehosp_evidence_snippet"],
        "Failure": ["failure_evidence_date", "failure_evidence_source", "failure_evidence_note_id", "failure_evidence_pattern", "failure_evidence_snippet"],
        "Revision": ["revision_evidence_date", "revision_evidence_source", "revision_evidence_note_id", "revision_evidence_pattern", "revision_evidence_snippet"],
        # MajorComp usually derived; still show reop/rehosp evidence if present
        "MajorComp": ["reop_evidence_date", "reop_evidence_source", "reop_evidence_pattern", "reop_evidence_snippet",
                      "rehosp_evidence_date", "rehosp_evidence_source", "rehosp_evidence_pattern", "rehosp_evidence_snippet"],
    }

    rows = []

    def add_one(outcome_name, gold_col, pred_col, label, subdf):
        if len(subdf) == 0:
            return
        r = subdf.iloc[0]

        row = {
            "outcome": outcome_name,
            "case_type": label,
            "gold_col": gold_col,
            "pred_col": pred_col,
            "gold": str(r.get(gold_col, "")),
            "pred": str(r.get(pred_col, "")),
        }
        if mrn_col:
            row["MRN"] = str(r.get(mrn_col, ""))
        else:
            row["MRN"] = ""
        if pid_col:
            row["ENCRYPTED_PAT_ID"] = str(r.get(pid_col, ""))
        else:
            row["ENCRYPTED_PAT_ID"] = ""

        row["STAGE2_DATE"] = str(r.get(stage2_col, "")) if stage2_col else ""
        row["WINDOW_START"] = str(r.get(wstart_col, "")) if wstart_col else ""
        row["WINDOW_END"] = str(r.get(wend_col, "")) if wend_col else ""

        ev_cols = evidence_map.get(outcome_name, [])
        for c in ev_cols:
            row[c] = str(r.get(c, ""))

        rows.append(row)

    # Build exemplars
    for name, gold_col, pred_col in outcomes:
        if gold_col not in df.columns or pred_col not in df.columns:
            # Skip if not present
            continue

        tmp = df.copy()
        tmp["_g"] = tmp[gold_col].map(to01)
        tmp["_p"] = tmp[pred_col].map(to01)

        # Prefer rows that have MRN present (if possible) so you can de-id easily
        if mrn_col:
            tmp["_mrn_ok"] = tmp[mrn_col].notna() & (tmp[mrn_col].astype(str).str.strip() != "") & (tmp[mrn_col].astype(str).str.lower() != "nan")
            tmp = tmp.sort_values(["_mrn_ok"], ascending=False)

        add_one(name, gold_col, pred_col, "FP", tmp[(tmp["_g"] == 0) & (tmp["_p"] == 1)])
        add_one(name, gold_col, pred_col, "FN", tmp[(tmp["_g"] == 1) & (tmp["_p"] == 0)])
        add_one(name, gold_col, pred_col, "TP", tmp[(tmp["_g"] == 1) & (tmp["_p"] == 1)])
        add_one(name, gold_col, pred_col, "TN", tmp[(tmp["_g"] == 0) & (tmp["_p"] == 0)])

    out_df = pd.DataFrame(rows)

    # Ensure columns exist consistently (so CSV is stable)
    base_cols = ["outcome", "case_type", "MRN", "ENCRYPTED_PAT_ID", "STAGE2_DATE", "WINDOW_START", "WINDOW_END", "gold_col", "pred_col", "gold", "pred"]
    all_ev = []
    for k in evidence_map:
        all_ev += evidence_map[k]
    # dedupe preserve order
    seen = set()
    all_ev2 = []
    for c in all_ev:
        if c not in seen:
            seen.add(c)
            all_ev2.append(c)

    out_df = ensure_cols(out_df, base_cols + all_ev2)
    out_df = out_df[base_cols + all_ev2]

    out_df.to_csv(out_path, index=False)

    # Terminal output
    print("")
    print("Wrote:", out_path)
    print("Rows (exemplars):", len(out_df))
    if len(out_df) > 0:
        pd.set_option("display.max_colwidth", 120)
        pd.set_option("display.width", 200)
        print("")
        print(out_df[["outcome", "case_type", "MRN", "ENCRYPTED_PAT_ID", "gold", "pred"]].to_string(index=False))
        print("")

    # Also print missing outcomes (if any)
    missing = []
    for name, gold_col, pred_col in outcomes:
        if gold_col not in df.columns or pred_col not in df.columns:
            missing.append((name, gold_col, pred_col))
    if missing:
        print("Skipped outcomes missing columns:")
        for m in missing:
            print("  -", m[0], "| missing:", (m[1] if m[1] not in df.columns else ""), (m[2] if m[2] not in df.columns else ""))

if __name__ == "__main__":
    main()
