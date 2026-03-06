import pandas as pd

MASTER = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD = "gold_cleaned_for_cedar.csv"

master = pd.read_csv(MASTER, dtype=str)
gold = pd.read_csv(GOLD, dtype=str)

master["MRN"] = master["MRN"].astype(str).str.strip()
gold["MRN"] = gold["MRN"].astype(str).str.strip()

merged = pd.merge(master, gold, on="MRN", suffixes=("_pred", "_gold"))

mask = (
    merged["Race_gold"].notna() &
    (merged["Race_gold"].str.strip() != "") &
    (merged["Race_pred"].str.strip() != merged["Race_gold"].str.strip())
)

print(merged.loc[mask, ["MRN", "Race_gold", "Race_pred"]])
