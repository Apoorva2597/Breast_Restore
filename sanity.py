import pandas as pd

gold = pd.read_csv("/home/apokol/Breast_Restore/_outputs/patient_master.csv", dtype=str)
pred = pd.read_csv("/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv", dtype=str)

print("GOLD COLUMNS:")
print(list(gold.columns))
print("\nPRED COLUMNS:")
print(list(pred.columns))

cols = ["MRN","Race","Ethnicity","Age","BMI","SmokingStatus","Diabetes","Hypertension","CardiacDisease","VenousThromboembolism","Steroid"]

print("\nGOLD SAMPLE:")
print(gold[[c for c in cols if c in gold.columns]].head(20).to_string())

print("\nPRED SAMPLE:")
print(pred[[c for c in cols if c in pred.columns]].head(20).to_string())
