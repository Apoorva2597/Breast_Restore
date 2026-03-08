import pandas as pd

GOLD = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"   
PRED = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

gold = pd.read_csv(GOLD, dtype=str)
pred = pd.read_csv(PRED, dtype=str)


gold["MRN"] = gold["MRN"].astype(str)
pred["MRN"] = pred["MRN"].astype(str)

merged = gold.merge(pred, on="MRN", suffixes=("_gold","_pred"))

# ---------- Smoking mismatches ----------
smoking_mismatch = merged[
    merged["SmokingStatus_gold"] != merged["SmokingStatus_pred"]
]

smoking_mismatch = smoking_mismatch[
    ["MRN","SmokingStatus_gold","SmokingStatus_pred"]
]

smoking_mismatch.to_csv(
    "_outputs/smoking_mismatches.csv",
    index=False
)

print("Smoking mismatches:", len(smoking_mismatch))


# ---------- BMI mismatches ----------
# convert to numeric for comparison

merged["BMI_gold"] = pd.to_numeric(merged["BMI_gold"], errors="coerce")
merged["BMI_pred"] = pd.to_numeric(merged["BMI_pred"], errors="coerce")

bmi_mismatch = merged[
    merged["BMI_gold"] != merged["BMI_pred"]
]

bmi_mismatch = bmi_mismatch[
    ["MRN","BMI_gold","BMI_pred"]
]

bmi_mismatch.to_csv(
    "_outputs/bmi_mismatches.csv",
    index=False
)

print("BMI mismatches:", len(bmi_mismatch))
