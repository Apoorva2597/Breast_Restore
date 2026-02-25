#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path

# === EDIT THESE PATHS IF NEEDED ===
CROSSWALK_PATH = "CROSSWALK/CROSSWALK__MRN_to_patient_id__vNEW.csv"
BUNDLES_ROOT = Path("QA_DEID_BUNDLES")

# === GET MRN FROM COMMAND LINE ===
if len(sys.argv) != 2:
    print("Usage: python find_bundle_by_mrn.py <MRN>")
    sys.exit(1)

mrn_input = sys.argv[1]

# === LOAD CROSSWALK ===
df = pd.read_csv(CROSSWALK_PATH, dtype=str)

match = df[df["MRN"] == mrn_input]

if match.empty:
    print(f"No patient_id found for MRN {mrn_input}")
    sys.exit(1)

patient_id = match.iloc[0]["patient_id"]

print(f"MRN {mrn_input} → patient_id {patient_id}")

# === BUILD BUNDLE PATH ===
bundle_dir = BUNDLES_ROOT / patient_id

if not bundle_dir.exists():
    print(f"Bundle directory not found: {bundle_dir}")
    sys.exit(1)

print(f"\nBundle directory:\n{bundle_dir}")

# === LIST NOTE FILES ===
print("\nFiles in bundle:")
for file in bundle_dir.glob("*"):
    print(" -", file.name)
