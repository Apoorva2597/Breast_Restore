
import pandas as pd
df = pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv", dtype=str)
cols = [
    "Stage2_MinorComp","Stage2_Reoperation","Stage2_Rehospitalization",
    "Stage2_MajorComp","Stage2_Failure","Stage2_Revision"
]
for c in cols:
    print(c, df[c].fillna("").astype(str).str.strip().value_counts(dropna=False).to_dict())
