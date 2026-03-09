python - <<'PY'
import pandas as pd

master = pd.read_csv("_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv", dtype=str)
gold = pd.read_csv("gold_cleaned_for_cedar.csv", dtype=str)

master["MRN"] = master["MRN"].astype(str).str.strip()
gold["MRN"] = gold["MRN"].astype(str).str.strip()

master = master.drop_duplicates(subset=["MRN"])
gold = gold.drop_duplicates(subset=["MRN"])

merged = pd.merge(master, gold, on="MRN", how="inner", suffixes=("_pred","_gold"))

gold_bmi = pd.to_numeric(merged["BMI_gold"], errors="coerce")
pred_bmi = pd.to_numeric(merged["BMI_pred"], errors="coerce")

print("gold present only:", gold_bmi.notna().sum())
print("gold and pred present:", (gold_bmi.notna() & pred_bmi.notna()).sum())
PY
