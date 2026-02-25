#!/usr/bin/env python3

import sys
import pandas as pd

# === EDIT PATH IF NEEDED ===
CROSSWALK_PATH = "CROSSWALK/CROSSWALK__MRN_to_patient_id__vNEW.csv"

if len(sys.argv) != 2:
    print("Usage: python mrn_to_pid.py <MRN>")
    sys.exit(1)

mrn_input = sys.argv[1]

# Read as strings to avoid type mismatch
df = pd.read_csv(CROSSWALK_PATH, dtype=str)

match = df[df["MRN"] == mrn_input]

if match.empty:
    print(f"No encrypted patient_id found for MRN {mrn_input}")
    sys.exit(1)

patient_id = match.iloc[0]["patient_id"]

print(f"MRN: {mrn_input}")
print(f"Encrypted patient_id: {patient_id}")
