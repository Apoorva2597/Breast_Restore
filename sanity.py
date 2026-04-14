import pandas as pd
df = pd.read_csv('_frozen_stage2/20260228_200052/validation_merged.csv', dtype=str, nrows=5)
print(df[['PRED_HAS_STAGE2','STAGE2_DATE','WINDOW_START']].to_string())
