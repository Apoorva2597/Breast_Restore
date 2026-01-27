import pandas as pd
import glob
import os

BASE_DIR = "/home/apoko/my_data_Breast/HPI-11526/HPI11256"

note_files = glob.glob(os.path.join(BASE_DIR, "*Notes*.csv"))

# Keep only the note tables (not encounters)
note_files = [f for f in note_files if ("Clinic Notes" in f or "Inpatient Notes" in f or "Operation Notes" in f)]

if not note_files:
    raise RuntimeError("No note CSV files found in {}".format(BASE_DIR))

print("Using note files:")
for f in note_files:
    print(" -", f)

dfs = []
for f in note_files:
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

all_notes = pd.concat(dfs, ignore_index=True)

print("Total notes:", len(all_notes))
print("Total patients:", all_notes["patient_id"].nunique())

all_notes.to_csv("patient_note_index.csv", index=False)
print("Saved patient_note_index.csv")
