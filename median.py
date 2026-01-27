import pandas as pd
df = pd.read_csv("patient_note_index.csv", encoding="cp1252")
s = df.groupby("patient_id").size()
print("Patients:", s.shape[0])
print("Notes per patient -> min:", int(s.min()), "median:", float(s.median()), "max:", int(s.max()))

