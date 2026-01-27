import pandas as pd
import os

BASE_DIR = os.path.join("..", "my_data_Breast", "HPI-11526", "HPI11256")

note_files = [
    os.path.join(BASE_DIR, "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "HPI11526 Inpatient Notes.csv"),
    os.path.join(BASE_DIR, "HPI11526 Operation Notes.csv"),
]

print("Checking files...")
for f in note_files:
    print(" -", f, "exists?", os.path.exists(f))
    if not os.path.exists(f):
        raise RuntimeError("File not found: {}".format(f))

print("\nAll files found.\n")

dfs = []
for f in note_files:
    print("Loading:", f)
    df = pd.read_csv(f, encoding="cp1252")

    # IMPORTANT: your headers use underscores
    df = df.rename(columns={
        "ENCRYPTED_PAT_ID": "patient_id",
        "NOTE_ID": "note_id",
        "NOTE_DATE_OF_SERVICE": "note_date",
        "NOTE_TYPE": "note_type",
        "NOTE_TEXT": "note_text"
    })

    # sanity check
    needed = ["patient_id", "note_id", "note_date", "note_type", "note_text"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError("Missing columns {} in file {}. Columns seen: {}".format(
            missing, f, df.columns.tolist()
        ))

    df["source_file"] = os.path.basename(f)
    dfs.append(df[needed + ["source_file"]])

all_notes = pd.concat(dfs, ignore_index=True)

print("Total notes:", len(all_notes))
print("Total unique patients:", all_notes["patient_id"].nunique())

all_notes.to_csv("patient_note_index.csv", index=False)
print("\nSaved patient_note_index.csv")
