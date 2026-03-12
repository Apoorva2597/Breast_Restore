
import pandas as pd
df = pd.read_csv("_outputs/validation_merged.csv", dtype=str)
for c in df.columns:
    if "Stage1_" in c or "Stage2_" in c or "PRED_" in c or "GOLD_" in c:
        print(c)
