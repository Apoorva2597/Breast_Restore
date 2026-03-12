
import pandas as pd
df = pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv", dtype=str)
want = [
    "Stage1_MinorComp","Stage1_Reoperation","Stage1_Rehospitalization",
    "Stage1_MajorComp","Stage1_Failure","Stage1_Revision",
    "Stage2_MinorComp","Stage2_Reoperation","Stage2_Rehospitalization",
    "Stage2_MajorComp","Stage2_Failure","Stage2_Revision"
]
for c in want:
    print(c, c in df.columns)
