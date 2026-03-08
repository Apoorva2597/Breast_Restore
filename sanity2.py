import pandas as pd

GOLD = "gold_cleaned_for_cedar.csv"
PRED = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

gold = pd.read_csv(GOLD, dtype=str)
pred = pd.read_csv(PRED, dtype=str)

gold["MRN"] = gold["MRN"].astype(str).str.strip()
pred["MRN"] = pred["MRN"].astype(str).str.strip()

merged = gold.merge(pred, on="MRN", how="inner", suffixes=("_gold", "_pred"))

# keep BMI columns
df = merged[["MRN", "BMI_gold", "BMI_pred"]].copy()

# numeric versions
df["BMI_gold_num"] = pd.to_numeric(df["BMI_gold"], errors="coerce")
df["BMI_pred_num"] = pd.to_numeric(df["BMI_pred"], errors="coerce")

# rounded-to-integer versions
df["BMI_gold_round"] = df["BMI_gold_num"].round(0)
df["BMI_pred_round"] = df["BMI_pred_num"].round(0)

# match flag after rounding
df["match_after_round"] = df["BMI_gold_round"] == df["BMI_pred_round"]

print("\nBMI ROUNDING CHECK\n")
print(
    df[
        ["MRN", "BMI_gold", "BMI_pred", "BMI_gold_round", "BMI_pred_round", "match_after_round"]
    ].head(100).to_string(index=False)
)

df.to_csv("_outputs/bmi_rounding_check.csv", index=False)

print("\nSaved:")
print("_outputs/bmi_rounding_check.csv")

print("\nSummary:")
print("Total rows with gold BMI:", df["BMI_gold_num"].notna().sum())
print("Total rows with pred BMI:", df["BMI_pred_num"].notna().sum())
print("Matches after rounding:", df["match_after_round"].fillna(False).sum())
