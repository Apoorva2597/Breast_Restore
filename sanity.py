import pandas as pd 

df = pd.read_csv("pred_with_mrn.csv", dtype=object)
print(df.columns.tolist())
print(df.head())
