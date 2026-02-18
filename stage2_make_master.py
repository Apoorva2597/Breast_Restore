#!/usr/bin/env python3
# stage2_make_master.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Build a single, join-friendly Stage 2 "master" patient-level file from:
#     1) stage2_final_ab_patient_level.csv         (A/B confirmed set)
#     2) stage2_candidates_tierC_patient_level.csv (Tier C candidates for later QA)
#     3) stage2_from_notes_patient_level.csv       (universe; provides patient list)
#
# Outputs:
#   1) stage2_master_patient_level.csv
#   2) stage2_master_summary.txt
#
# Design:
#   - AB_CONFIRMED: stage2_confirmed_ab=1 and stage2_date_ab populated
#   - TIER_C_CANDIDATE: stage2_possible_tierC=1 (NOT treated as confirmed)
#   - NO_EVIDENCE: neither of the above
#
# Notes:
#   - Reads using latin1(errors=replace) to avoid UnicodeDecodeError (0xA0 etc.)
#   - Writes outputs as UTF-8
#
# Usage:
#   python stage2_make_master.py
#
from __future__ import print_function

import os
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
IN_UNIVERSE = "stage2_from_notes_patient_level.csv"
IN_AB = "stage2_final_ab_patient_level.csv"
IN_TIERC = "stage2_candidates_tierC_patient_level.csv"

OUT_MASTER = "stage2_master_patient_level.csv"
OUT_SUMMARY = "stage2_master_summary.txt"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        try:
            f.close()
        except Exception:
            pass


def must_have_cols(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError("Missing columns {} in {}".format(missing, path))


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def main():
    # -------------------------
    # Load universe
    # -------------------------
    if not os.path.exists(IN_UNIVERSE):
        raise RuntimeError("Universe file not found: {}".format(IN_UNIVERSE))

    uni = read_csv_safe(IN_UNIVERSE)
    must_have_cols(uni, ["patient_id"], IN_UNIVERSE)

    uni["patient_id"] = uni["patient_id"].fillna("").astype(str)
    uni = uni[uni["patient_id"] != ""].copy()

    keep_uni = ["patient_id"]
    for c in [
        "stage1_dt", "mrn_from_staging", "expander_bucket",
        "best_note_type", "best_note_id", "best_note_dos", "best_note_op_date",
        "snippet"
    ]:
        if c in uni.columns:
            keep_uni.append(c)

    base = uni[keep_uni].drop_duplicates(subset=["patient_id"]).copy()

    # -------------------------
    # Load AB confirmed
    # -------------------------
    if not os.path.exists(IN_AB):
        raise RuntimeError("AB file not found: {}".format(IN_AB))

    ab = read_csv_safe(IN_AB)
    must_have_cols(ab, ["patient_id"], IN_AB)

    ab["patient_id"] = ab["patient_id"].fillna("").astype(str)
    ab = ab[ab["patient_id"] != ""].copy()

    ab_date_col = None
    for cand in ["stage2_event_dt_best", "stage2_date_ab", "stage2_date", "stage2_dt"]:
        if cand in ab.columns:
            ab_date_col = cand
            break
    if ab_date_col is None:
        raise RuntimeError(
            "No AB stage2 date column found in {}. Expected one of: "
            "stage2_event_dt_best, stage2_date_ab, stage2_date, stage2_dt".format(IN_AB)
        )

    ab_tier_col = None
    for cand in ["stage2_tier_best", "stage2_tier_ab", "tier", "stage2_tier"]:
        if cand in ab.columns:
            ab_tier_col = cand
            break

    ab_out = pd.DataFrame()
    ab_out["patient_id"] = ab["patient_id"]
    ab_out["stage2_confirmed_ab"] = 1
    ab_out["stage2_date_ab"] = to_dt(ab[ab_date_col])
    ab_out["stage2_tier_ab"] = (
        ab[ab_tier_col].fillna("").astype(str).str.strip().str.upper()
        if ab_tier_col is not None else "AB"
    )

    for c in ["stage2_rule_best", "best_has_proc_section", "best_has_implants_section", "stage2_delta_days_from_stage1"]:
        if c in ab.columns:
            ab_out[c] = ab[c]

    ab_out = ab_out.drop_duplicates(subset=["patient_id"])

    # -------------------------
    # Load Tier C candidates
    # -------------------------
    if not os.path.exists(IN_TIERC):
        raise RuntimeError("Tier C file not found: {}".format(IN_TIERC))

    tc = read_csv_safe(IN_TIERC)
    must_have_cols(tc, ["patient_id"], IN_TIERC)

    tc["patient_id"] = tc["patient_id"].fillna("").astype(str)
    tc = tc[tc["patient_id"] != ""].copy()

    tc_date_col = None
    for cand in ["stage2_event_dt_best", "stage2_date_tierC_candidate", "stage2_date", "stage2_dt"]:
        if cand in tc.columns:
            tc_date_col = cand
            break
    if tc_date_col is None:
        raise RuntimeError(
            "No Tier C date column found in {}. Expected one of: "
            "stage2_event_dt_best, stage2_date_tierC_candidate, stage2_date, stage2_dt".format(IN_TIERC)
        )

    tc_out = pd.DataFrame()
    tc_out["patient_id"] = tc["patient_id"]
    tc_out["stage2_possible_tierC"] = 1
    tc_out["stage2_date_tierC_candidate"] = to_dt(tc[tc_date_col])

    tc_out = tc_out.drop_duplicates(subset=["patient_id"])

    # -------------------------
    # Merge into master
    # -------------------------
    master = base.merge(ab_out, on="patient_id", how="left")
    master = master.merge(
        tc_out[["patient_id", "stage2_possible_tierC", "stage2_date_tierC_candidate"]],
        on="patient_id",
        how="left"
    )

    master["stage2_confirmed_ab"] = master["stage2_confirmed_ab"].fillna(0).astype(int)
    master["stage2_possible_tierC"] = master["stage2_possible_tierC"].fillna(0).astype(int)

    def bucket_row(r):
        if int(r.get("stage2_confirmed_ab", 0)) == 1 and pd.notnull(r.get("stage2_date_ab", None)):
            return "AB_CONFIRMED"
        if int(r.get("stage2_possible_tierC", 0)) == 1 and pd.notnull(r.get("stage2_date_tierC_candidate", None)):
            return "TIER_C_CANDIDATE"
        return "NO_EVIDENCE"

    master["stage2_bucket"] = master.apply(bucket_row, axis=1)

    # Convenience: AB date only for downstream use (complication windows)
    master["stage2_date_for_downstream"] = master["stage2_date_ab"]

    master = master.sort_values(by=["stage2_bucket", "patient_id"], ascending=[True, True])
    master.to_csv(OUT_MASTER, index=False, encoding="utf-8")

    # -------------------------
    # Summary
    # -------------------------
    total = int(len(master))
    n_ab = int((master["stage2_bucket"] == "AB_CONFIRMED").sum())
    n_tc = int((master["stage2_bucket"] == "TIER_C_CANDIDATE").sum())
    n_none = int((master["stage2_bucket"] == "NO_EVIDENCE").sum())

    lines = []
    lines.append("=== Stage 2 Master Build ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")
    lines.append("Inputs:")
    lines.append("  - {}".format(IN_UNIVERSE))
    lines.append("  - {}".format(IN_AB))
    lines.append("  - {}".format(IN_TIERC))
    lines.append("")
    lines.append("Total patients (universe): {}".format(total))
    lines.append("AB_CONFIRMED: {} ({:.1f}%)".format(n_ab, (100.0 * n_ab / total) if total else 0.0))
    lines.append("TIER_C_CANDIDATE: {} ({:.1f}%)".format(n_tc, (100.0 * n_tc / total) if total else 0.0))
    lines.append("NO_EVIDENCE: {} ({:.1f}%)".format(n_none, (100.0 * n_none / total) if total else 0.0))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_MASTER))
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
