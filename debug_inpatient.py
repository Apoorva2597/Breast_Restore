# debug_inpatient.py
# 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

import pandas as pd
from pathlib import Path

# CHANGE THIS PATH IF NEEDED
csv_path = Path("/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv")

print("\n=== Reading CSV ===")
df = pd.read_csv(str(csv_path), encoding="cp1252")
print("Columns:", list(df.columns))

# Count rows with non-empty NOTE_TEXT
txt = df["NOTE_TEXT"].astype(str)
non_empty = (txt.str.strip() != "").sum()

print("\nNon-empty NOTE_TEXT rows:", non_empty)

# Show a few sample entries if available
sample = df.loc[txt.str.strip() != "", "NOTE_TEXT"].head(5)

print("\n=== Sample NOTE_TEXT (first 5) ===")
for i, v in sample.items():
    s = str(v).strip()
    print(f"Row {i}, length {len(s)} → {s[:200]!r}")

print("\nDone.\n")
