#!/usr/bin/env python3
# qa_smoking_mismatches.py
#
# Finds smoking mismatches vs gold and shows the note evidence used.

import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
EVID_FILE = "_outputs/bmi_smoking_only_evidence.csv"

MRN = "MRN"


def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_smoking(x):

    s = clean(x).lower()

    if s in ["current", "current smoker", "smoker"]:
        return "Current"

    if s in ["former", "former smoker", "ex-smoker", "quit smoking"]:
        return "Former"

    if s in ["never", "never smoker", "never smoked", "nonsmoker", "non-smoker"]:
        return "Never"

    return clean(x)


print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)
evid = pd.read_csv(EVID_FILE, dtype=str)

master[MRN] = master[MRN].astype(str).str.strip()
gold[MRN] = gold[MRN].astype(str).str.strip()

merged = pd.merge(master, gold, on=MRN, suffixes=("_pred", "_gold"))

merged["Smoking_pred"] = merged["SmokingStatus_pred"].apply(normalize_smoking)
merged["Smoking_gold"] = merged["SmokingStatus_gold"].apply(normalize_smoking)

mismatches = merged[merged["Smoking_pred"] != merged["Smoking_gold"]].copy()

print("\nSmoking mismatches:", len(mismatches))

rows = []

for _, r in mismatches.iterrows():

    mrn = r[MRN]

    ev = evid[
        (evid[MRN].astype(str).str.strip() == mrn) &
        (evid["FIELD"] == "SmokingStatus")
    ]

    if len(ev) > 0:
        ev = ev.sort_values(by="CONFIDENCE", ascending=False)
        e = ev.iloc[0]

        rows.append({
            "MRN": mrn,
            "Gold": r["Smoking_gold"],
            "Pred": r["Smoking_pred"],
            "Note_Date": clean(e.get("NOTE_DATE")),
            "Note_Type": clean(e.get("NOTE_TYPE")),
            "Section": clean(e.get("SECTION")),
            "Value": clean(e.get("VALUE")),
            "Evidence": clean(e.get("EVIDENCE"))
        })

qa = pd.DataFrame(rows)

out = "_outputs/qa_smoking_mismatches.csv"
qa.to_csv(out, index=False)

print("Saved:", out)
