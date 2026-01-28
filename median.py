import pandas as pd
df = pd.read_csv("evidence_log_phase1_p50.csv", encoding="utf-8", engine="python")
fields = sorted(df["field"].dropna().unique().tolist())
print([f for f in fields if f in ("Chemo","Radiation")])
