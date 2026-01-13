# summarize_counts.py
# Python 3.6-compatible

from __future__ import print_function
import pandas as pd

# ------------------------
# EDIT THESE IF NEEDED
# ------------------------

# Patient-level extraction file (the one you just generated)
EXTRACTED_PATIENT = "extracted_patient_variables.csv"

# Raw note CSVs
OP_CSV     = "HPI11526 Operation Notes.csv"
CLINIC_CSV = "HPI11526 Clinic Notes.csv"
INPAT_CSV  = "HPI11526 Inpatient Notes.csv"

# Encrypted patient ID column name in the raw note CSVs
NOTE_PID_COL = "ENCRYPTED_PAT_ID"   # change if yours is different


def load_df(path, encoding="cp1252"):
    print("\n--- Loading {} ---".format(path))
    df = pd.read_csv(path, encoding=encoding)
    print("Rows: {}, Columns: {}".format(len(df), len(df.columns)))
    return df


def summarize_extracted_patient_level():
    print("\n==============================")
    print("  PATIENT-LEVEL EXTRACTION")
    print("==============================")

    # patient-level file is UTF-8 (you wrote it), not cp1252
    df = pd.read_csv(EXTRACTED_PATIENT, encoding="utf-8")

    # Guess PatientID column name
    pid_candidates = ["PatientID", "patient_id", "PATIENT_ID", "Patient_ID"]
    patient_col = None
    for c in pid_candidates:
        if c in df.columns:
            patient_col = c
            break

    if patient_col is None:
        print("WARNING: could not find a PatientID column.")
        print("Columns are:", list(df.columns))
    else:
        n_patients = df[patient_col].nunique()
        print("Patient ID column:", patient_col)
        print("Unique patients in extracted_patient_variables.csv: {}".format(n_patients))

    if patient_col is not None:
        variable_cols = [c for c in df.columns if c != patient_col]
    else:
        variable_cols = list(df.columns)

    print("\nVariables (columns) in patient-level extraction:")
    for c in variable_cols:
        print("  - {}".format(c))
    print("Total #variables (excluding PatientID): {}".format(len(variable_cols)))


def summarize_raw_note_files():
    print("\n==============================")
    print("  RAW NOTE FILES (PER SOURCE)")
    print("==============================")

    sources = [
        ("operation", OP_CSV),
        ("clinic", CLINIC_CSV),
        ("inpatient", INPAT_CSV),
    ]

    all_pids = set()

    for label, path in sources:
        df = load_df(path, encoding="cp1252")

        if NOTE_PID_COL not in df.columns:
            print("ERROR: column '{}' not found in {}."
                  .format(NOTE_PID_COL, path))
            print("Columns:", list(df.columns))
            continue

        pids = set(df[NOTE_PID_COL].dropna().astype(str))
        all_pids.update(pids)

        print("Source: {:10s} -> unique patients: {}".format(label, len(pids)))

    print("\nTotal unique patients across ALL three files: {}".format(len(all_pids)))


def main():
    summarize_extracted_patient_level()
    summarize_raw_note_files()


if __name__ == "__main__":
    main()
