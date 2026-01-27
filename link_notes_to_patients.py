import pandas as pd
from glob import glob

# Files containing notes
note_files = [
    "HPI11526 Clinic Notes.csv",
    "HPI11526 Inpatient Notes.csv",
    "HPI11526 Operation Notes.csv"
]

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

    df["source_file"] = f

    dfs.append(df[["patient_id", "note_id", "note_date", "note_type", "note_text", "source_file"]])

all_notes = pd.concat(dfs, ignore_index=True)

print("Total notes:", len(all_notes))
print("Total patients:", all_notes["patient_id"].nunique())

all_notes.to_csv("patient_note_index.csv", index=False)
print("Saved patient_note_index.csv")
