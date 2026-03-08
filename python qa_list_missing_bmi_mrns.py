#!/usr/bin/env python3

import pandas as pd

MASTER_FILE = "/home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
OUTPUT_FILE = "/home/apokol/Breast_Restore/_outputs/missing_bmi_pred_mrns.csv"

MRN_COL = "MRN"
BMI_COL = "BMI"


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    for k in ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
        if k in df.columns:
            if k != MRN_COL:
                df = df.rename(columns={k: MRN_COL})
            break
    df[MRN_COL] = df[MRN_COL].astype(str).str.strip()
    return df


def is_missing(series):
    s = series.astype(str).str.strip().str.lower()
    return series.isna() | s.isin(["", "nan", "none", "null", "na"])


print("Loading files...")

master = pd.read_csv(MASTER_FILE, dtype=str)
gold = pd.read_csv(GOLD_FILE, dtype=str)

master = clean_cols(master)
gold = clean_cols(gold)

master = normalize_mrn(master)
gold = normalize_mrn(gold)

if BMI_COL not in gold.columns:
    raise RuntimeError("Gold file missing BMI column.")

if BMI_COL not in master.columns:
    master[BMI_COL] = ""

gold_nonmissing = gold[~is_missing(gold[BMI_COL])][[MRN_COL, BMI_COL]].copy()

merged = gold_nonmissing.merge(
    master[[MRN_COL, BMI_COL]],
    on=MRN_COL,
    how="left",
    suffixes=("_gold", "_pred")
)

missing_pred = merged[is_missing(merged["BMI_pred"])].copy()
missing_pred = missing_pred.sort_values(MRN_COL)

missing_pred[[MRN_COL, "BMI_gold"]].to_csv(OUTPUT_FILE, index=False)

print("")
print("Gold BMI present but pred missing count: {0}".format(len(missing_pred)))
print("")
print("MRN | GOLD_BMI")
print("----------------------")

for _, row in missing_pred.iterrows():
    print("{0} | {1}".format(row[MRN_COL], row["BMI_gold"]))

print("")
print("CSV written to:")
print(OUTPUT_FILE)
