#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py
Creates patient-level abstraction table using:

Priority:
1) Structured encounter data
2) Clinical notes fallback

Outputs:
_outputs/patient_master.csv
"""

from __future__ import print_function
import os
import csv
import re
import sys
import pandas as pd

# -----------------------------
# Paths
# -----------------------------

STAGING_DIR = "_staging_inputs"
OUTPUT_DIR = "_outputs"

ENCOUNTER_FILE = os.path.join(STAGING_DIR, "encounters.csv")
NOTE_FILE = os.path.join(STAGING_DIR, "notes.csv")

OUT_MASTER = os.path.join(OUTPUT_DIR, "patient_master.csv")

# -----------------------------
# Helper
# -----------------------------

def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).lower().strip()


def safe_float(x):
    try:
        return float(x)
    except:
        return None


# -----------------------------
# Note Extractors
# -----------------------------

BMI_RE = re.compile(r"\bbmi[: ]*([0-9]{2}\.?[0-9]?)\b", re.I)
SMOKING_RE = re.compile(r"\b(smoker|smoking|tobacco use)\b", re.I)
FORMER_SMOKER_RE = re.compile(r"\bformer smoker\b", re.I)

DIABETES_RE = re.compile(r"\bdiabetes\b", re.I)
HTN_RE = re.compile(r"\bhypertension\b", re.I)
CARDIAC_RE = re.compile(r"\b(coronary artery disease|cad|heart failure)\b", re.I)
VTE_RE = re.compile(r"\b(dvt|pe|venous thromboembolism)\b", re.I)
STEROID_RE = re.compile(r"\bsteroid\b", re.I)

LUMPECTOMY_RE = re.compile(r"\blumpectomy\b", re.I)
REDUCTION_RE = re.compile(r"\breduction mammoplasty\b", re.I)
MASTOPEXY_RE = re.compile(r"\bmastopexy\b", re.I)
AUGMENT_RE = re.compile(r"\baugmentation\b", re.I)

RADIATION_RE = re.compile(r"\bradiation\b", re.I)
CHEMO_RE = re.compile(r"\bchemotherapy|chemo\b", re.I)


def extract_bmi(text):
    m = BMI_RE.search(text)
    if m:
        return safe_float(m.group(1))
    return None


def extract_smoking(text):
    if SMOKING_RE.search(text):
        return "Current"
    if FORMER_SMOKER_RE.search(text):
        return "Former"
    return None


def extract_flag(regex, text):
    return 1 if regex.search(text) else None


# -----------------------------
# Main
# -----------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    encounters = pd.read_csv(ENCOUNTER_FILE)
    notes = pd.read_csv(NOTE_FILE)

    pid_col = "ENCRYPTED_PAT_ID"

    patients = encounters[[pid_col]].drop_duplicates().copy()

    # -----------------------------
    # Structured Demographics
    # -----------------------------

    demo = encounters.groupby(pid_col).first().reset_index()

    patients["Race"] = demo["RACE"]
    patients["Ethnicity"] = demo["ETHNICITY"]
    patients["Age"] = demo["AGE_AT_ENCOUNTER"]

    # -----------------------------
    # Initialize abstraction fields
    # -----------------------------

    fields = [
        "BMI","Obesity","SmokingStatus",
        "Diabetes","Hypertension","CardiacDisease",
        "VenousThromboembolism","Steroid",
        "PBS_Lumpectomy","PBS_Reduction","PBS_Mastopexy","PBS_Augmentation",
        "Radiation","Chemo"
    ]

    for f in fields:
        patients[f] = None

    # -----------------------------
    # Process notes
    # -----------------------------

    for _, row in notes.iterrows():

        pid = row[pid_col]
        text = normalize_text(row["NOTE_TEXT"])

        idx = patients.index[patients[pid_col] == pid]

        if len(idx) == 0:
            continue

        idx = idx[0]

        bmi = extract_bmi(text)
        if bmi:
            patients.loc[idx, "BMI"] = bmi
            patients.loc[idx, "Obesity"] = 1 if bmi >= 30 else 0

        sm = extract_smoking(text)
        if sm:
            patients.loc[idx, "SmokingStatus"] = sm

        patients.loc[idx,"Diabetes"] = extract_flag(DIABETES_RE,text) or patients.loc[idx,"Diabetes"]
        patients.loc[idx,"Hypertension"] = extract_flag(HTN_RE,text) or patients.loc[idx,"Hypertension"]
        patients.loc[idx,"CardiacDisease"] = extract_flag(CARDIAC_RE,text) or patients.loc[idx,"CardiacDisease"]
        patients.loc[idx,"VenousThromboembolism"] = extract_flag(VTE_RE,text) or patients.loc[idx,"VenousThromboembolism"]
        patients.loc[idx,"Steroid"] = extract_flag(STEROID_RE,text) or patients.loc[idx,"Steroid"]

        patients.loc[idx,"PBS_Lumpectomy"] = extract_flag(LUMPECTOMY_RE,text) or patients.loc[idx,"PBS_Lumpectomy"]
        patients.loc[idx,"PBS_Reduction"] = extract_flag(REDUCTION_RE,text) or patients.loc[idx,"PBS_Reduction"]
        patients.loc[idx,"PBS_Mastopexy"] = extract_flag(MASTOPEXY_RE,text) or patients.loc[idx,"PBS_Mastopexy"]
        patients.loc[idx,"PBS_Augmentation"] = extract_flag(AUGMENT_RE,text) or patients.loc[idx,"PBS_Augmentation"]

        patients.loc[idx,"Radiation"] = extract_flag(RADIATION_RE,text) or patients.loc[idx,"Radiation"]
        patients.loc[idx,"Chemo"] = extract_flag(CHEMO_RE,text) or patients.loc[idx,"Chemo"]

    patients.to_csv(OUT_MASTER, index=False)

    print("Build complete.")
    print("Output:", OUT_MASTER)


if __name__ == "__main__":
    main()
