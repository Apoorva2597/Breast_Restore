import pandas as pd

GOLD = "gold_cleaned_for_cedar.csv"
PRED = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

gold = pd.read_csv(GOLD, dtype=str)
pred = pd.read_csv(PRED, dtype=str)

gold["MRN"] = gold["MRN"].astype(str).str.strip()
pred["MRN"] = pred["MRN"].astype(str).str.strip()

merged = gold.merge(pred, on="MRN", suffixes=("_gold", "_pred"))

# -----------------------------
# Smoking mismatches
# -----------------------------
smoking = merged[["MRN", "SmokingStatus_gold", "SmokingStatus_pred"]].copy()

smoking["SmokingStatus_gold"] = smoking["SmokingStatus_gold"].astype(str).str.strip()
smoking["SmokingStatus_pred"] = smoking["SmokingStatus_pred"].astype(str).str.strip()

smoking_mismatch = smoking[
    smoking["SmokingStatus_gold"] != smoking["SmokingStatus_pred"]
].copy()

print("\nSMOKING MISMATCHES\n")
if len(smoking_mismatch) == 0:
    print("No smoking mismatches found.")
else:
    print(smoking_mismatch.to_string(index=False))

smoking_mismatch.to_csv("_outputs/smoking_mismatches.csv", index=False)

# -----------------------------
# BMI mismatches
# -----------------------------
bmi = merged[["MRN", "BMI_gold", "BMI_pred"]].copy()

bmi["BMI_gold_num"] = pd.to_numeric(bmi["BMI_gold"], errors="coerce")
bmi["BMI_pred_num"] = pd.to_numeric(bmi["BMI_pred"], errors="coerce")

# mismatch if either missing or outside tolerance
bmi_mismatch = bmi[
    (bmi["BMI_gold_num"].isna() & bmi["BMI_pred_num"].notna()) |
    (bmi["BMI_gold_num"].notna() & bmi["BMI_pred_num"].isna()) |
    (
        bmi["BMI_gold_num"].notna() &
        bmi["BMI_pred_num"].notna() &
        ((bmi["BMI_gold_num"] - bmi["BMI_pred_num"]).abs() > 0.2)
    )
].copy()

print("\nBMI MISMATCHES\n")
if len(bmi_mismatch) == 0:
    print("No BMI mismatches found.")
else:
    print(bmi_mismatch[["MRN", "BMI_gold", "BMI_pred"]].to_string(index=False))

bmi_mismatch[["MRN", "BMI_gold", "BMI_pred"]].to_csv("_outputs/bmi_mismatches.csv", index=False)

print("\nSaved:")
print("_outputs/smoking_mismatches.csv")
print("_outputs/bmi_mismatches.csv")
