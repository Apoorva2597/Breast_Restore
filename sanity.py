#!/usr/bin/env python3
# qa_smoking_old_vs_new.py
#
# QA script to compare OLD vs NEW smoking predictions against gold.
#
# Purpose:
# 1. Identify MRNs where smoking result changed
# 2. Show whether change improved / worsened / unchanged vs gold
# 3. Pull evidence rows from old and new evidence files
# 4. Save a detailed CSV for manual review
#
# Adjust OLD/NEW evidence paths below as needed.
#
# Python 3.6.8 compatible

import os
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
MRN = "MRN"

OLD_MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_OLD.csv".format(BASE_DIR)
NEW_MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_FILE = "{0}/gold_cleaned_for_cedar.csv".format(BASE_DIR)

OLD_EVID_FILE = "{0}/_outputs/bmi_only_evidence_OLD.csv".format(BASE_DIR)
NEW_EVID_FILE = "{0}/_outputs/bmi_smoking_only_evidence.csv".format(BASE_DIR)

OUT_FILE = "{0}/_outputs/qa_smoking_old_vs_new.csv".format(BASE_DIR)


def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        return pd.read_csv(path, encoding="latin1", dtype=str)


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MRN:
                df = df.rename(columns={k: MRN})
            break
    if MRN not in df.columns:
        raise RuntimeError("MRN column not found.")
    df[MRN] = df[MRN].astype(str).str.strip()
    return df


def clean_val(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def normalize_smoking(x):
    s = clean_val(x).lower()

    if s in {"current", "current smoker", "smoker", "active smoker", "smokes", "currently smoking"}:
        return "Current"

    if s in {
        "former", "former smoker", "ex-smoker", "quit smoking", "quit tobacco",
        "stopped smoking", "history of tobacco use", "prior tobacco use"
    }:
        return "Former"

    if s in {
        "never", "never smoker", "never smoked", "non-smoker", "nonsmoker",
        "lifetime nonsmoker", "denies smoking", "denies tobacco", "no tobacco use"
    }:
        return "Never"

    if s == "":
        return ""

    return str(x).strip()


def load_master_subset(path, label):
    df = clean_cols(safe_read_csv(path))
    df = normalize_mrn(df)
    if "SmokingStatus" not in df.columns:
        raise RuntimeError("SmokingStatus column not found in {0}".format(path))
    out = df[[MRN, "SmokingStatus"]].copy()
    out["SmokingStatus"] = out["SmokingStatus"].apply(normalize_smoking)
    out = out.rename(columns={"SmokingStatus": "SmokingStatus_{0}".format(label)})
    out = out.drop_duplicates(subset=[MRN])
    return out


def load_gold_subset(path):
    df = clean_cols(safe_read_csv(path))
    df = normalize_mrn(df)
    if "SmokingStatus" not in df.columns:
        raise RuntimeError("SmokingStatus column not found in gold file.")
    out = df[[MRN, "SmokingStatus"]].copy()
    out["SmokingStatus"] = out["SmokingStatus"].apply(normalize_smoking)
    out = out.rename(columns={"SmokingStatus": "SmokingStatus_gold"})
    out = out.drop_duplicates(subset=[MRN])
    return out


def pick_best_evidence(evid_df, mrn):
    tmp = evid_df[evid_df[MRN].astype(str).str.strip() == str(mrn).strip()].copy()
    if len(tmp) == 0:
        return None

    # keep smoking rows only
    if "FIELD" in tmp.columns:
        tmp = tmp[tmp["FIELD"].astype(str).str.strip() == "SmokingStatus"].copy()

    if len(tmp) == 0:
        return None

    # prioritize final-stage-ish rows by stage/date/confidence
    if "STAGE_USED" not in tmp.columns:
        tmp["STAGE_USED"] = ""
    if "CONFIDENCE" not in tmp.columns:
        tmp["CONFIDENCE"] = ""
    if "NOTE_DATE" not in tmp.columns:
        tmp["NOTE_DATE"] = ""

    stage_order = {"day0": 1, "pm7": 2, "pm14": 3, "": 9}
    tmp["_stage_rank"] = tmp["STAGE_USED"].astype(str).map(lambda x: stage_order.get(str(x).strip(), 9))

    def to_float_safe(x):
        try:
            return float(str(x).strip())
        except Exception:
            return -1.0

    tmp["_conf_num"] = tmp["CONFIDENCE"].apply(to_float_safe)
    tmp = tmp.sort_values(by=["_stage_rank", "_conf_num"], ascending=[True, False])

    row = tmp.iloc[0]
    return {
        "NOTE_ID": clean_val(row.get("NOTE_ID", "")),
        "NOTE_DATE": clean_val(row.get("NOTE_DATE", "")),
        "NOTE_TYPE": clean_val(row.get("NOTE_TYPE", "")),
        "VALUE": clean_val(row.get("VALUE", "")),
        "SECTION": clean_val(row.get("SECTION", "")),
        "STAGE_USED": clean_val(row.get("STAGE_USED", "")),
        "ANCHOR_DATE": clean_val(row.get("ANCHOR_DATE", "")),
        "EVIDENCE": clean_val(row.get("EVIDENCE", ""))
    }


def load_evidence(path):
    if not os.path.exists(path):
        return None
    df = clean_cols(safe_read_csv(path))
    df = normalize_mrn(df)
    return df


def main():
    print("Loading files...")

    old_df = load_master_subset(OLD_MASTER_FILE, "old")
    new_df = load_master_subset(NEW_MASTER_FILE, "new")
    gold_df = load_gold_subset(GOLD_FILE)

    merged = gold_df.merge(old_df, on=MRN, how="left").merge(new_df, on=MRN, how="left")

    merged["old_match"] = (merged["SmokingStatus_old"] == merged["SmokingStatus_gold"]).astype(int)
    merged["new_match"] = (merged["SmokingStatus_new"] == merged["SmokingStatus_gold"]).astype(int)
    merged["changed_prediction"] = (merged["SmokingStatus_old"] != merged["SmokingStatus_new"]).astype(int)

    def classify_change(row):
        oldm = int(row["old_match"])
        newm = int(row["new_match"])
        if oldm == 0 and newm == 1:
            return "improved"
        if oldm == 1 and newm == 0:
            return "worsened"
        if oldm == 1 and newm == 1:
            return "both_match"
        if oldm == 0 and newm == 0:
            return "both_mismatch"
        return "unknown"

    merged["change_vs_gold"] = merged.apply(classify_change, axis=1)

    old_evid = load_evidence(OLD_EVID_FILE)
    new_evid = load_evidence(NEW_EVID_FILE)

    rows = []

    for _, row in merged.iterrows():
        mrn = clean_val(row[MRN])

        if row["changed_prediction"] != 1:
            continue

        out = {
            MRN: mrn,
            "SmokingStatus_gold": clean_val(row.get("SmokingStatus_gold", "")),
            "SmokingStatus_old": clean_val(row.get("SmokingStatus_old", "")),
            "SmokingStatus_new": clean_val(row.get("SmokingStatus_new", "")),
            "old_match": int(row.get("old_match", 0)),
            "new_match": int(row.get("new_match", 0)),
            "change_vs_gold": clean_val(row.get("change_vs_gold", "")),
        }

        if old_evid is not None:
            old_best = pick_best_evidence(old_evid, mrn)
        else:
            old_best = None

        if new_evid is not None:
            new_best = pick_best_evidence(new_evid, mrn)
        else:
            new_best = None

        if old_best is not None:
            out["old_NOTE_ID"] = old_best["NOTE_ID"]
            out["old_NOTE_DATE"] = old_best["NOTE_DATE"]
            out["old_NOTE_TYPE"] = old_best["NOTE_TYPE"]
            out["old_VALUE"] = old_best["VALUE"]
            out["old_SECTION"] = old_best["SECTION"]
            out["old_STAGE_USED"] = old_best["STAGE_USED"]
            out["old_ANCHOR_DATE"] = old_best["ANCHOR_DATE"]
            out["old_EVIDENCE"] = old_best["EVIDENCE"]
        else:
            out["old_NOTE_ID"] = ""
            out["old_NOTE_DATE"] = ""
            out["old_NOTE_TYPE"] = ""
            out["old_VALUE"] = ""
            out["old_SECTION"] = ""
            out["old_STAGE_USED"] = ""
            out["old_ANCHOR_DATE"] = ""
            out["old_EVIDENCE"] = ""

        if new_best is not None:
            out["new_NOTE_ID"] = new_best["NOTE_ID"]
            out["new_NOTE_DATE"] = new_best["NOTE_DATE"]
            out["new_NOTE_TYPE"] = new_best["NOTE_TYPE"]
            out["new_VALUE"] = new_best["VALUE"]
            out["new_SECTION"] = new_best["SECTION"]
            out["new_STAGE_USED"] = new_best["STAGE_USED"]
            out["new_ANCHOR_DATE"] = new_best["ANCHOR_DATE"]
            out["new_EVIDENCE"] = new_best["EVIDENCE"]
        else:
            out["new_NOTE_ID"] = ""
            out["new_NOTE_DATE"] = ""
            out["new_NOTE_TYPE"] = ""
            out["new_VALUE"] = ""
            out["new_SECTION"] = ""
            out["new_STAGE_USED"] = ""
            out["new_ANCHOR_DATE"] = ""
            out["new_EVIDENCE"] = ""

        rows.append(out)

    out_df = pd.DataFrame(rows)

    print("\nQA SUMMARY")
    print("Total gold smoking rows:", len(merged[merged["SmokingStatus_gold"].astype(str).str.strip() != ""]))
    print("Changed smoking predictions:", len(out_df))

    if len(out_df) > 0:
        print("\nChange vs gold counts:")
        print(out_df["change_vs_gold"].value_counts(dropna=False))
    else:
        print("No changed predictions found.")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    print("\nSaved QA file:")
    print(OUT_FILE)
    print("\nEdit these paths first if needed:")
    print(" OLD_MASTER_FILE =", OLD_MASTER_FILE)
    print(" NEW_MASTER_FILE =", NEW_MASTER_FILE)
    print(" OLD_EVID_FILE   =", OLD_EVID_FILE)
    print(" NEW_EVID_FILE   =", NEW_EVID_FILE)


if __name__ == "__main__":
    main()
