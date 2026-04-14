import pandas as pd
df = pd.read_csv('_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv', dtype=str)
print('Stage2_Rehospitalization value counts:')
print(df['Stage2_Rehospitalization'].value_counts())
print('Stage2_MinorComp value counts:')
print(df['Stage2_MinorComp'].value_counts())
print('RAW signal check - patients with PRED_HAS_STAGE2=1:', df[df['PRED_HAS_STAGE2'].astype(str)=='1']['Stage2_Rehospitalization'].value_counts())
