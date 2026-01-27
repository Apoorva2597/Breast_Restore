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
# LOAD AND STANDARDIZE (CP1252 ENCODING)
# --------------------------------------------------
dfs = []

for f in note_files:
    print("Loading:", f)
    df = pd.read_csv(f, encoding="cp1252")   # ← FIXED HERE

    df = df.rename(columns={
        "ENCRYPTED_PAT_ID":
