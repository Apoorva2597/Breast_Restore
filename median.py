import pandas as pd
df = pd.read_csv("patient_note_index.csv", encoding="utf-8", engine="python")
print(sorted(df["note_type"].dropna().unique()))
