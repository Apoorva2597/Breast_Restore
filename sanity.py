#!/usr/bin/env python3

import pandas as pd
import numpy as np

BASE = "/home/apokol/Breast_Restore/_outputs"

VALIDATION_FILE = BASE + "/bmi_gold_vs_pred.csv"
EVIDENCE_FILE = BASE + "/bmi_only_evidence.csv"

MRN_COL = "MRN"


def to_float(x):
    try:
        return float(x)
    except:
        return np.nan


def obesity(bmi):
    if pd.isna(bmi):
        return np.nan
    return 1 if bmi >= 30 else 0


print("Loading files...")

df = pd.read_csv(VALIDATION_FILE)
evid = pd.read_csv(EVIDENCE_FILE)

df["BMI_gold"] = df["BMI_gold"].apply(to_float)
df["BMI_pred"] = df["BMI_pred"].apply(to_float)

df["gold_obesity"] = df["BMI_gold"].apply(obesity)
df["pred_obesity"] = df["BMI_pred"].apply(obesity)

# remove rows where prediction missing
df = df[~df["BMI_pred"].isna()]

# obesity mismatches
mis = df[df["gold_obesity"] != df["pred_obesity"]].copy()

print("")
print("Total obesity mismatches:", len(mis))
print("")

rows = []

for _, r in mis.iterrows():

    mrn = str(r[MRN_COL]).strip()
    pred = r["BMI_pred"]

    ev = evid[
        (evid["MRN"].astype(str) == mrn) &
        (evid["VALUE"].astype(float) == pred)
    ]

    if len(ev) == 0:
        rows.append({
            "MRN": mrn,
            "BMI_gold": r["BMI_gold"],
            "BMI_pred": r["BMI_pred"],
            "gold_obesity": r["gold_obesity"],
            "pred_obesity": r["pred_obesity"],
            "note_type": "",
            "note_date": "",
            "anchor_date": "",
            "stage": "",
            "snippet": ""
        })
        continue

    ev = ev.iloc[0]

    rows.append({
        "MRN": mrn,
        "BMI_gold": r["BMI_gold"],
        "BMI_pred": r["BMI_pred"],
        "gold_obesity": r["gold_obesity"],
        "pred_obesity": r["pred_obesity"],
        "note_type": ev.get("NOTE_TYPE", ""),
        "note_date": ev.get("NOTE_DATE", ""),
        "anchor_date": ev.get("ANCHOR_DATE", ""),
        "stage": ev.get("STAGE_USED", ""),
        "snippet": ev.get("EVIDENCE", "")
    })

out = pd.DataFrame(rows)

print("Preview:\n")

for _, r in out.iterrows():

    print(
        r["MRN"],
        "| gold_bmi:", r["BMI_gold"],
        "| pred_bmi:", r["BMI_pred"],
        "| gold_ob:", r["gold_obesity"],
        "| pred_ob:", r["pred_obesity"]
    )

print("\nTotal mismatches:", len(out))

out.to_csv(BASE + "/qa_obesity_mismatches.csv", index=False)

print("\nSaved to:", BASE + "/qa_obesity_mismatches.csv")
