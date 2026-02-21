import pandas as pd

df_check = pd.read_csv("gold_cleaned_for_cedar.csv", dtype=str)
print("Literal 'nan' strings count:",
      (df_check == "nan").sum().sum())
