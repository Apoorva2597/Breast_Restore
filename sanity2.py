#!/usr/bin/env python3

import pandas as pd
import numpy as np
from glob import glob

BASE = "/home/apokol/Breast_Restore"

MISMATCH_FILE = BASE + "/_outputs/bmi_mismatch_reasons.csv"
EVIDENCE_FILE = BASE + "/_outputs/bmi_only_evidence.csv"

NOTE_GLOBS = [
    BASE + "/**/*Clinic Notes.csv",
    BASE + "/**/*Inpatient Notes.csv",
    BASE + "/**/*Operation Notes.csv"
]

DIFF_THRESHOLD = 2.0


def to_float(x):
    try:
        return float(x)
    except:
        return np.nan


def load_notes():

    files = []
    for g in NOTE_GLOBS:
        files.extend(glob(g, recursive=True))

    notes = []

    for f in files:

        try:
            df = pd.read_csv(f, dtype=str, engine="python", on_bad_lines="skip")
        except:
            df = pd.read_csv(f, dtype=str, engine="python", encoding="latin1", on_bad_lines="skip")

        cols = [c.upper().strip() for c in df.columns]
        df.columns = cols

        if "NOTE_ID" not in df.columns:
            continue

        if "NOTE_TEXT" not in df.columns:
            continue

        keep = ["NOTE_ID", "NOTE_TEXT"]

        if "NOTE_TYPE" in df.columns:
            keep.append("NOTE_TYPE")

        if "NOTE_DATE_OF_SERVICE" in df.columns:
            keep.append("NOTE_DATE_OF_SERVICE")

        df = df[keep]

        notes.append(df)

    notes = pd.concat(notes, ignore_index=True)

    return notes


print("Loading mismatch table...")
mis = pd.read_csv(MISMATCH_FILE)

print("Loading BMI evidence...")
evid = pd.read_csv(EVIDENCE_FILE)

print("Reconstructing notes...")
notes = load_notes()

mis["BMI_gold"] = mis["BMI_gold"].apply(to_float)
mis["BMI_pred"] = mis["BMI_pred"].apply(to_float)
mis["diff_abs"] = mis["diff_abs"].apply(to_float)

mis_large = mis[
    (~mis["BMI_gold"].isna()) &
    (~mis["BMI_pred"].isna()) &
    (mis["diff_abs"] >= DIFF_THRESHOLD)
]

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

    n = notes[notes["NOTE_ID"] == note_id]

    if len(n) > 0:
        note_text = n.iloc[0]["NOTE_TEXT"]

    rows.append({
        "MRN": mrn,
        "BMI_gold": r["BMI_gold"],
        "BMI_pred": r["BMI_pred"],
        "diff": r["diff_abs"],
        "note_type": ev.get("NOTE_TYPE", ""),
        "note_date": ev.get("NOTE_DATE", ""),
        "anchor_date": ev.get("ANCHOR_DATE", ""),
        "stage": ev.get("STAGE_USED", ""),
        "snippet": ev.get("EVIDENCE", ""),
        "NOTE_TEXT": note_text
    })

out = pd.DataFrame(rows)

out = out.sort_values("diff", ascending=False)

print("")
print("RESULTS")
print("")

for _, r in out.iterrows():

    print("MRN:", r["MRN"])
    print("gold:", r["BMI_gold"], "pred:", r["BMI_pred"], "diff:", r["diff"])
    print("note_type:", r["note_type"], "note_date:", r["note_date"])
    print("anchor_date:", r["anchor_date"], "stage:", r["stage"])
    print("snippet:", r["snippet"])
    print("\nNOTE TEXT:\n")
    print(r["NOTE_TEXT"][:2000])
    print("\n-----------------------------------------------------\n")
