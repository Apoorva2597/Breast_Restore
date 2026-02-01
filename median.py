# extract_unique_procedures.py
# Python 3.6 compatible
# PURPOSE: Explore procedure vocabulary for staging logic

import pandas as pd
import re

FILE_PATH = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

def normalize_proc(text):
    if pd.isnull(text):
        return ""
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def main():
    print("Reading file...")
    try:
        df = pd.read_csv(FILE_PATH, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="cp1252", engine="python")

    required_cols = ["ENCRYPTED_PAT_ID", "OPERATION_DATE", "PROCEDURE"]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError("Missing column: {}".format(c))

    print("Rows:", df.shape[0])

    df["proc_norm"] = df["PROCEDURE"].apply(normalize_proc)

    # Count unique procedure names
    proc_counts = (
        df.groupby("proc_norm")
        .size()
        .reset_index(name="encounter_rows")
        .sort_values("encounter_rows", ascending=False)
    )

    print("\n=== TOP 30 PROCEDURE NAMES ===")
    print(proc_counts.head(30).to_string(index=False))

    # Focus on likely Stage 2 / revision-related words
    keywords = [
        "revision",
        "exchange",
        "expander",
        "implant",
        "flap",
        "takeback",
        "removal",
        "explant",
        "re-exploration",
        "washout"
    ]

    mask = df["proc_norm"].str.contains("|".join(keywords), regex=True)
    subset = df[mask]["proc_norm"].value_counts().reset_index()
    subset.columns = ["proc_norm", "count"]

    print("\n=== PROCEDURES WITH STAGING-RELEVANT WORDS ===")
    print(subset.head(40).to_string(index=False))

    # Save full list for review
    proc_counts.to_csv("qa_unique_procedures.csv", index=False)
    print("\nSaved: qa_unique_procedures.csv")

    print("\nDone.")

if __name__ == "__main__":
    main()
