
import pandas as pd
p="/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
df=pd.read_csv(p, dtype=str, engine="python")
col="Stage2_Applicable"
print("unique Stage2_Applicable values:")
print(df[col].fillna("").value_counts(dropna=False).head(30))
