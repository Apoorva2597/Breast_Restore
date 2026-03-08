#!/usr/bin/env python3

import pandas as pd

INPUT = "/home/apokol/Breast_Restore/_outputs/bmi_mismatch_reasons.csv"

df = pd.read_csv(INPUT)

# keep only real comparisons
df = df[
    (~df["gold_missing"]) &
    (~df["pred_missing"])
]

# large differences
df_large = df[df["diff_abs"] >= 2].copy()

df_large = df_large.sort_values("diff_abs", ascending=False)

print("\nLarge BMI differences (>=2)\n")

for _, r in df_large.iterrows():

    print(
        r["MRN"],
        "| gold=", r["BMI_gold"],
        "| pred=", r["BMI_pred"],
        "| diff=", round(r["diff_abs"],2)
    )

print("\nTotal large mismatches:", len(df_large))
