import pandas as pd

GOLD = "gold_cleaned_for_cedar.csv"
PRED = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
EVID = "_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv"

gold = pd.read_csv(GOLD, dtype=str)
pred = pd.read_csv(PRED, dtype=str)
evid = pd.read_csv(EVID, dtype=str)

gold["MRN"] = gold["MRN"].astype(str).str.strip()
pred["MRN"] = pred["MRN"].astype(str).str.strip()
evid["MRN"] = evid["MRN"].astype(str).str.strip()

# Keep only BMI evidence rows
bmi_evid = evid[evid["FIELD"].astype(str).str.strip() == "BMI"].copy()

# Keep one evidence row per MRN (first one)
bmi_evid = bmi_evid[["MRN", "EVIDENCE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "VALUE"]].copy()
bmi_evid = bmi_evid.drop_duplicates(subset=["MRN"], keep="first")

merged = gold.merge(pred, on="MRN", suffixes=("_gold", "_pred"))
merged = merged.merge(bmi_evid, on="MRN", how="left")

bmi = merged[["MRN", "BMI_gold", "BMI_pred", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "EVIDENCE"]].copy()

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

print("\nBMI MISMATCHES WITH EVIDENCE\n")
if len(bmi_mismatch) == 0:
    print("No BMI mismatches found.")
else:
    print(
        bmi_mismatch[
            ["MRN", "BMI_gold", "BMI_pred", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "EVIDENCE"]
        ].head(50).to_string(index=False)
    )

bmi_mismatch[
    ["MRN", "BMI_gold", "BMI_pred", "VALUE", "NOTE_DATE", "NOTE_TYPE", "SECTION", "EVIDENCE"]
].to_csv("_outputs/bmi_mismatches_with_evidence.csv", index=False)

print("\nSaved:")
print("_outputs/bmi_mismatches_with_evidence.csv")
