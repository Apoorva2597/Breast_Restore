# qa_no_note_patients.py

import pandas as pd

INPUT_FILE = "_outputs/qa_smoking_mismatches_categorized.csv"
OUTPUT_FILE = "_outputs/no_note_available_mrns.csv"

df = pd.read_csv(INPUT_FILE)

# Filter the category
no_note_df = df[df["category"] == "no_evidence_row"]

# Extract MRNs
mrns = no_note_df["mrn"].dropna().unique()

print("\nPatients with NO NOTE evidence:")
print("--------------------------------")
for m in mrns:
    print(m)

print("\nTotal patients:", len(mrns))

# Save for manual review
out_df = pd.DataFrame({"mrn": mrns})
out_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved MRN list to:", OUTPUT_FILE)
