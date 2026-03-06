import pandas as pd

gold = pd.read_csv("/home/apokol/Breast_Restore/_outputs/patient_master.csv", dtype=str)
pred = pd.read_csv("/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv", dtype=str)

print("GOLD COLUMNS:")
print(list(gold.columns))
print("\nPRED COLUMNS:")
print(list(pred.columns))

cols = ["Race","Ethnicity","Age","BMI","SmokingStatus","Diabetes","Hypertension","CardiacDisease","VenousThromboembolism","Steroid"]

print("\nGOLD SAMPLE:")
print(gold[[c for c in cols if c in gold.columns]].head(20).to_string())

print("\nPRED SAMPLE:")
print(pred[[c for c in cols if c in pred.columns]].head(20).to_string())

print("\nGOLD DTYPES:")
print(gold.dtypes)

print("\nPRED DTYPES:")
print(pred.dtypes)

g = gold.copy()
p = pred.copy()

g["MRN"] = g["MRN"].astype(str).str.strip()
p["MRN"] = p["MRN"].astype(str).str.strip()

m = g.merge(p, on="MRN", how="inner", suffixes=("_gold","_pred"))

want = [
    "Race_gold","Race_pred",
    "Ethnicity_gold","Ethnicity_pred",
    "Age_gold","Age_pred",
    "BMI_gold","BMI_pred",
    "SmokingStatus_gold","SmokingStatus_pred",
    "Diabetes_gold","Diabetes_pred",
    "Hypertension_gold","Hypertension_pred",
    "CardiacDisease_gold","CardiacDisease_pred",
    "VenousThromboembolism_gold","VenousThromboembolism_pred",
    "Steroid_gold","Steroid_pred"
]

print(m[[c for c in want if c in m.columns]].head(10).to_string())
