import pandas as pd

gold = pd.read_csv("/home/apokol/Breast_Restore/_outputs/patient_master.csv", dtype=str)
pred = pd.read_csv("/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv", dtype=str)

print("GOLD COLUMNS:")
print(list(gold.columns))
print("\nPRED COLUMNS:")
print(list(pred.columns))
