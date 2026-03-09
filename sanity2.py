#!/usr/bin/env python3

import pandas as pd
import numpy as np

MISMATCH_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_mismatch_reasons.csv"
EVIDENCE_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_only_evidence.csv"
NOTES_FILE = "/home/apokol/Breast_Restore/_outputs/reconstructed_notes.csv"

OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/bmi_large_diff_full_notes.csv"

DIFF_THRESHOLD = 2.0


def to_float(x):
    try:
        return float(x)
    except:
        return np.nan


print("Loading files...")

mis = pd.read_csv(MISMATCH_FILE)
evid = pd.read_csv(EVIDENCE_FILE)
notes = pd.read_csv(NOTES_FILE)

mis["BMI_gold"] = mis["BMI_gold"].apply(to_float)
mis["BMI_pred"] = mis["BMI_pred"].apply(to_float)
mis["diff_abs"] = mis["diff_abs"].apply(to_float)

# large mismatches only
mis_large = mis[
    (~mis["BMI_gold"].isna()) &
    (~mis["BMI_pred"].isna()) &
    (mis["diff_abs"] >= DIFF_THRESHOLD)
].copy()

print("Large mismatches:", len(mis_large))

rows = []

for _, r in mis_large.iterrows():

    mrn = str(r["MRN"]).strip()
    pred = r["BMI_pred"]

    ev = evid[
        (evid["MRN"].astype(str) == mrn) &
        (evid["VALUE"].astype(float) == pred)
    ]

    if len(ev) == 0:
        continue

    ev = ev.iloc[0]

    note_id = ev["NOTE_ID"]

    note_text = ""

    note_row = notes[notes["NOTE_ID"] == note_id]

    if len(note_row) > 0:
        note_text = note_row.iloc[0]["NOTE_TEXT"]

    rows.append({
        "MRN": mrn,
        "BMI_gold": r["BMI_gold"],
        "BMI_pred": r["BMI_pred"],
        "diff": r["diff_abs"],
        "note_type": ev.get("NOTE_TYPE", ""),
        "note_date": ev.get("NOTE_DATE", ""),
        "anchor_date": ev.get("ANCHOR_DATE", ""),
        "stage_used": ev.get("STAGE_USED", ""),
        "snippet": ev.get("EVIDENCE", ""),
        "NOTE_TEXT": note_text
    })

out = pd.DataFrame(rows)

out = out.sort_values("diff", ascending=False)

out.to_csv(OUTPUT_FILE, index=False)

print("")
print("Output written to:")
print(OUTPUT_FILE)
