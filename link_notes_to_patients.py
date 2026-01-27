import pandas as pd
import os

# --------------------------------------------------
# BASE DIRECTORY (sibling of Breast_Restore)
# --------------------------------------------------
BASE_DIR = os.path.join("..", "my_data_Breast", "HPI-11526", "HPI11256")

# --------------------------------------------------
# HARD-CODED NOTE FILES
# --------------------------------------------------
note_files = [
    os.path.join(BASE_DIR, "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "HPI11526 Inpatient Notes.csv"),
    os.path.join(BASE_DIR, "HPI11526 Operation Notes.csv"),
]

# --------------------------------------------------
# CHECK FILES EXIST
# --------------------------------------------------
print("Checking files...")
for f in note_files:
    print(" -", f, "exists?", os.path.exists(f))
    if not os.path.exists(f):
        raise RuntimeError("File not found: {}".format(f))

print("\nAll files found.\n")

# --------------------------------------------------
# LOAD AND STANDARDIZE
# --------------------------------------------------
dfs = []

for f in note_files:
    print("Loading:", f)
    df = pd.read_csv(f)

    df = df.rename(columns={
        "ENCRYPTED_PAT_ID": "patient_id",
        "NOTE_ID": "note_id",
        "NOTE DATE OF SERVICE": "note_date",
        "NOTE TYPE": "note_type",
        "NOTE TEXT": "note_text"
    })

    df["source_file"] = os.path.basename(f)

    dfs.append(df[["patient_id", "note_id", "note_date", "note_type", "note_text", "source_file"]])

# --------------------------------------------------
# COMBINE ALL NOTES
# --------------------------------------------------
all_notes = pd.concat(dfs, ignore_index=True)

print("Total notes:", len(all_notes))
print("Total unique patients:", all_notes["patient_id"].nunique())

# --------------------------------------------------
# SAVE MASTER PATIENT-NOTE INDEX
# --------------------------------------------------
all_notes.to_csv("patient_note_index.csv", index=False)
print("\nSaved patient_note_index.csv")
