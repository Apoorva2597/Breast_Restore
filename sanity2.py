#!/usr/bin/env python3

import pandas as pd
import numpy as np

MISMATCH_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_mismatch_reasons.csv"
EVIDENCE_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_only_evidence.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_large_diff_evidence.csv"

MRN_COL = "MRN"
DIFF_THRESHOLD = 2.0


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def is_missing_val(x):
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in ["", "nan", "none", "null", "na"]


def to_float(x):
    try:
        if is_missing_val(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pick_best_evidence_row(group):
    """
    If multiple evidence rows exist for the same MRN/pred value,
    keep the one that is most directly useful for defense:
      1) measured over computed
      2) day0 over pm7 over pm14
      3) higher confidence
    """
    g = group.copy()

    def status_rank(x):
        s = str(x).strip().lower()
        if s == "measured":
            return 0
        if s == "computed":
            return 1
        return 9

    def stage_rank(x):
        s = str(x).strip().lower()
        if s == "day0":
            return 0
        if s == "pm7":
            return 1
        if s == "pm14":
            return 2
        return 9

    g["_status_rank"] = g["STATUS"].apply(status_rank) if "STATUS" in g.columns else 9
    g["_stage_rank"] = g["STAGE_USED"].apply(stage_rank) if "STAGE_USED" in g.columns else 9
    g["_confidence_num"] = pd.to_numeric(g["CONFIDENCE"], errors="coerce").fillna(-1)

    g = g.sort_values(
        by=["_status_rank", "_stage_rank", "_confidence_num"],
        ascending=[True, True, False]
    )
    return g.iloc[0]


print("Loading mismatch file...")
mis = pd.read_csv(MISMATCH_FILE, dtype=str)
mis = clean_cols(mis)

print("Loading evidence file...")
evid = pd.read_csv(EVIDENCE_FILE, dtype=str)
evid = clean_cols(evid)

mis["BMI_gold"] = mis["BMI_gold"].apply(to_float)
mis["BMI_pred"] = mis["BMI_pred"].apply(to_float)
mis["diff_abs"] = mis["diff_abs"].apply(to_float)

# Keep only real large mismatches
mis_large = mis[
    (~mis["BMI_gold"].isna()) &
    (~mis["BMI_pred"].isna()) &
    (mis["diff_abs"] >= DIFF_THRESHOLD)
].copy()

print("Large mismatches found: {0}".format(len(mis_large)))

if len(mis_large) == 0:
    print("No large mismatches found.")
    pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
    raise SystemExit

# Evidence file only for BMI rows
if "FIELD" in evid.columns:
    evid = evid[evid["FIELD"].astype(str).str.strip() == "BMI"].copy()

# Normalize MRN as string
mis_large[MRN_COL] = mis_large[MRN_COL].astype(str).str.strip()
evid[MRN_COL] = evid[MRN_COL].astype(str).str.strip()

# Make evidence numeric for matching on predicted BMI
if "VALUE" not in evid.columns:
    raise RuntimeError("Evidence file missing VALUE column.")
evid["VALUE_num"] = evid["VALUE"].apply(to_float)

rows = []

for _, mrow in mis_large.iterrows():
    mrn = str(mrow[MRN_COL]).strip()
    pred_bmi = mrow["BMI_pred"]

    ev_sub = evid[
        (evid[MRN_COL] == mrn) &
        (evid["VALUE_num"] == pred_bmi)
    ].copy()

    if len(ev_sub) == 0:
        rows.append({
            "MRN": mrn,
            "BMI_gold": mrow["BMI_gold"],
            "BMI_pred": mrow["BMI_pred"],
            "diff_abs": mrow["diff_abs"],
            "reason": mrow.get("reason", ""),
            "note_type": "",
            "note_date": "",
            "anchor_date": "",
            "anchor_type": "",
            "stage_used": "",
            "status": "",
            "section": "",
            "confidence": "",
            "snippet": "NO_MATCHING_EVIDENCE_ROW_FOUND"
        })
        continue

    best = pick_best_evidence_row(ev_sub)

    rows.append({
        "MRN": mrn,
        "BMI_gold": mrow["BMI_gold"],
        "BMI_pred": mrow["BMI_pred"],
        "diff_abs": mrow["diff_abs"],
        "reason": mrow.get("reason", ""),
        "note_type": best.get("NOTE_TYPE", ""),
        "note_date": best.get("NOTE_DATE", ""),
        "anchor_date": best.get("ANCHOR_DATE", ""),
        "anchor_type": best.get("ANCHOR_TYPE", ""),
        "stage_used": best.get("STAGE_USED", ""),
        "status": best.get("STATUS", ""),
        "section": best.get("SECTION", ""),
        "confidence": best.get("CONFIDENCE", ""),
        "snippet": best.get("EVIDENCE", "")
    })

out = pd.DataFrame(rows)
out = out.sort_values(by="diff_abs", ascending=False)

out.to_csv(OUTPUT_FILE, index=False)

print("")
print("Detailed large-difference evidence table written to:")
print(OUTPUT_FILE)
print("")

print("Preview:")
print("MRN | gold | pred | diff | note_type | note_date | anchor_date | stage_used")
print("-" * 110)

for _, r in out.iterrows():
    print(
        "{0} | {1} | {2} | {3} | {4} | {5} | {6} | {7}".format(
            r["MRN"],
            r["BMI_gold"],
            r["BMI_pred"],
            r["diff_abs"],
            r["note_type"],
            r["note_date"],
            r["anchor_date"],
            r["stage_used"]
        )
    )
