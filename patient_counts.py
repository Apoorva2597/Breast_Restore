#!/usr/bin/env python3
"""
patient_counts.py

Reports:
  - unique patients per raw note CSV
  - total unique patients across all 3 files
"""

import csv

# ===== EDIT THESE PATHS TO MATCH YOUR SYSTEM =====
INPATIENT_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11526/HPI11526 Inpatient Notes.csv"
CLINIC_CSV    = "/home/apokol/my_data_Breast/HPI-11526/HPI11526/HPI11526 Clinic Notes.csv"
OP_CSV        = "/home/apokol/my_data_Breast/HPI-11526/HPI11526/HPI11526 Operation Notes.csv"

# Column name for the (encrypted) patient ID in those CSVs
PATIENT_ID_COL = "ENCRYPTED_PAT_ID"
# ================================================


def unique_patients(path, label):
    """Return the set of unique patient IDs in the given CSV."""
    patients = set()
    with open(path, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get(PATIENT_ID_COL) or "").strip()
            if pid:
                patients.add(pid)

    print("{}: {:>6} unique patients".format(label, len(patients)))
    return patients


def main():
    print("=== Unique patients per note file ===")
    inpatient_patients = unique_patients(INPATIENT_CSV, "Inpatient")
    clinic_patients    = unique_patients(CLINIC_CSV,    "Clinic   ")
    op_patients        = unique_patients(OP_CSV,        "Operation")

    # union of all three sets
    all_patients = inpatient_patients | clinic_patients | op_patients
    print("\nTOTAL unique patients across ALL notes: {}".format(len(all_patients)))


if __name__ == "__main__":
    main()
