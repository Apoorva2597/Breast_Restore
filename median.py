import pandas as pd
df = pd.read_csv("evidence_log_phase1_p50.csv", encoding="utf-8", engine="python")
print(sorted(df["field"].dropna().unique().tolist()))
