import pandas as pd
df = pd.read_csv("patient_note_index.csv", encoding="utf-8", engine="python")
print(df["note_type"].value_counts().head(30))
