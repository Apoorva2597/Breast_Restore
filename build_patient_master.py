#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patient-level abstraction pipeline

Reads original HPI11526 encounter and note files and builds
a patient-level abstraction table.

Input directory:
/home/apokol/my_data_Breast/HPI-11526/HPI11256

Output:
_outputs/patient_master.csv
"""

import pandas as pd
import os
import re

# ---------------------------------------------------
# DATA DIRECTORY (FULL PATH)
# ---------------------------------------------------

DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

# ---------------------------------------------------
# Encounter files
# ---------------------------------------------------

ENCOUNTER_FILES = [
    os.path.join(DATA_DIR, "HPI11526 Clinic Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Inpatient Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Operation Encounters.csv")
]

# ---------------------------------------------------
# Note files
# ---------------------------------------------------

NOTE_FILES = [
    os.path.join(DATA_DIR, "HPI11526 Clinic Notes.csv"),
    os.path.join(DATA_DIR, "HPI11526 Inpatient Notes.csv"),
    os.path.join(DATA_DIR, "HPI11526 Operation Notes.csv")
]

# ---------------------------------------------------
# Output
# ---------------------------------------------------

OUTPUT_DIR = "_outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "patient_master.csv")

PATIENT_ID = "ENCRYPTED_PAT_ID"
NOTE_TEXT_COL = "NOTE_TEXT"


# ---------------------------------------------------
# Regex patterns
# ---------------------------------------------------

BMI_RE = re.compile(r"\bbmi[: ]*([0-9]{2}\.?[0-9]?)\b", re.I)

SMOKING_CURRENT_RE = re.compile(r"\b(smoker|smoking|tobacco use)\b", re.I)
SMOKING_FORMER_RE = re.compile(r"\bformer smoker\b", re.I)

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
CHEMO_RE = re.compile(r"\bchemo|chemotherapy\b", re.I)


# ---------------------------------------------------
# Helper extraction functions
# ---------------------------------------------------

def extract_bmi(text):
    m = BMI_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


def extract_smoking(text):
    if SMOKING_FORMER_RE.search(text):
        return "Former"
    if SMOKING_CURRENT_RE.search(text):
        return "Current"
    return None


def flag(regex, text):
    return 1 if regex.search(text) else 0


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------

def main():

    print("Loading encounter tables...")

    encounters = pd.concat(
        [pd.read_csv(f) for f in ENCOUNTER_FILES],
        ignore_index=True
    )

    print("Loading note tables...")

    notes = pd.concat(
        [pd.read_csv(f) for f in NOTE_FILES],
        ignore_index=True
    )

    print("Total encounters:", len(encounters))
    print("Total notes:", len(notes))

    # ---------------------------------------------------
    # Create patient list
    # ---------------------------------------------------

    patients = encounters[[PATIENT_ID]].drop_duplicates().copy()

    # ---------------------------------------------------
    # Structured demographics (if available)
    # ---------------------------------------------------

    demo = encounters.groupby(PATIENT_ID).first().reset_index()

    if "RACE" in demo.columns:
        patients["Race"] = demo["RACE"]

    if "ETHNICITY" in demo.columns:
        patients["Ethnicity"] = demo["ETHNICITY"]

    if "AGE_AT_ENCOUNTER" in demo.columns:
        patients["Age"] = demo["AGE_AT_ENCOUNTER"]

    # ---------------------------------------------------
    # Initialize variables
    # ---------------------------------------------------

    fields = [
        "BMI",
        "Obesity",
        "SmokingStatus",
        "Diabetes",
        "Hypertension",
        "CardiacDisease",
        "VenousThromboembolism",
        "Steroid",
        "PBS_Lumpectomy",
        "PBS_Reduction",
        "PBS_Mastopexy",
        "PBS_Augmentation",
        "Radiation",
        "Chemo"
    ]

    for f in fields:
        patients[f] = None

    # ---------------------------------------------------
    # Aggregate notes per patient
    # ---------------------------------------------------

    notes[NOTE_TEXT_COL] = notes[NOTE_TEXT_COL].fillna("").str.lower()

    patient_notes = notes.groupby(PATIENT_ID)[NOTE_TEXT_COL].apply(
        lambda x: " ".join(x)
    ).reset_index()

    print("Patients with notes:", len(patient_notes))

    # ---------------------------------------------------
    # Run extraction
    # ---------------------------------------------------

    for _, row in patient_notes.iterrows():

        pid = row[PATIENT_ID]
        text = row[NOTE_TEXT_COL]

        idx = patients.index[patients[PATIENT_ID] == pid]

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

        patients.loc[idx, "Diabetes"] = flag(DIABETES_RE, text)
        patients.loc[idx, "Hypertension"] = flag(HTN_RE, text)
        patients.loc[idx, "CardiacDisease"] = flag(CARDIAC_RE, text)
        patients.loc[idx, "VenousThromboembolism"] = flag(VTE_RE, text)
        patients.loc[idx, "Steroid"] = flag(STEROID_RE, text)

        patients.loc[idx, "PBS_Lumpectomy"] = flag(LUMPECTOMY_RE, text)
        patients.loc[idx, "PBS_Reduction"] = flag(REDUCTION_RE, text)
        patients.loc[idx, "PBS_Mastopexy"] = flag(MASTOPEXY_RE, text)
        patients.loc[idx, "PBS_Augmentation"] = flag(AUGMENT_RE, text)

        patients.loc[idx, "Radiation"] = flag(RADIATION_RE, text)
        patients.loc[idx, "Chemo"] = flag(CHEMO_RE, text)

    # ---------------------------------------------------
    # Save output
    # ---------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patients.to_csv(OUTPUT_FILE, index=False)

    print("Build complete.")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
