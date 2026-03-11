#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_pbs_confusion.py

PBS QA script:
- Merges master and gold on MRN
- Prints TP / TN / FP / FN counts in terminal
- Writes a QA CSV with:
    variable, case_type, gold, pred, snippet
- Does NOT include MRN in the output file
- Uses pbs_only_evidence.csv for snippets when available

Python 3.6.8 compatible.
"""

import os
import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
EVID_FILE = "_outputs/pbs_only_evidence.csv"
OUTPUT_QA = "_outputs/pbs_qa_cases.csv"

MRN = "MRN"

PBS_VARS = [
    "PastBreastSurgery",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "PBS_Other"
]

MISSING_STRINGS = {"", "nan", "none", "null", "na", "NA", "None", "Null"}


# ---------------------------------------------------
# Safe CSV reader
# ---------------------------------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", dtype=str)
    except Exception:
        return pd.read_csv(path, encoding="latin1", dtype=str)


# ---------------------------------------------------
# Basic cleaning
# ---------------------------------------------------

def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s in MISSING_STRINGS:
        return ""
    return s


def normalize_binary_value(x):
    s = clean_cell(x).lower()

    if not s:
        return pd.NA

    if s in ["1", "true", "t", "yes", "y"]:
        return 1

    if s in ["0", "false", "f", "no", "n"]:
        return 0

    return pd.NA


def shorten_text(x, max_len=400):
    s = clean_cell(x)
    if not s:
        return ""
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "..."


# ---------------------------------------------------
# Evidence ranking
# ---------------------------------------------------

def rule_rank(rule_decision):
    s = clean_cell(rule_decision).lower()

    if s == "accept_pre_recon":
        return 0
    if s == "accept_pre_recon_strict_history":
        return 1
    if s == "accept_post_recon_historical":
        return 2

    if s == "reject_contralateral":
        return 10
    if s == "reject_unknown_laterality_unilateral":
        return 11
    if s == "reject_post_recon_not_historical":
        return 12
    if s == "reject_pre_recon_no_strict_history":
        return 13
    if s == "reject_negative_history":
        return 14
    if s == "reject_unknown_recon_laterality":
        return 15

    return 99


def confidence_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return -1.0


def build_best_evidence_map(evid_df):
    """
    Returns:
        best_map[(mrn, field)] = snippet string
    """
    best_map = {}

    if evid_df is None or len(evid_df) == 0:
        return best_map

    use_cols = list(evid_df.columns)

    for col in ["MRN", "FIELD", "EVIDENCE", "RULE_DECISION", "CONFIDENCE"]:
        if col not in use_cols:
            return best_map

    tmp = evid_df.copy()

    tmp["MRN"] = tmp["MRN"].astype(str).str.strip()
    tmp["FIELD"] = tmp["FIELD"].astype(str).str.strip()
    tmp["RULE_DECISION"] = tmp["RULE_DECISION"].astype(str).str.strip()
    tmp["EVIDENCE"] = tmp["EVIDENCE"].astype(str).str.strip()
    tmp["CONFIDENCE_NUM"] = tmp["CONFIDENCE"].apply(confidence_float)
    tmp["RULE_RANK"] = tmp["RULE_DECISION"].apply(rule_rank)

    for (mrn, field), g in tmp.groupby(["MRN", "FIELD"], dropna=False):
        if not mrn or not field:
            continue

        g = g.sort_values(
            by=["RULE_RANK", "CONFIDENCE_NUM"],
            ascending=[True, False]
        )

        best_row = g.iloc[0]
        snippet = best_row["EVIDENCE"]

        if clean_cell(best_row["RULE_DECISION"]):
            snippet = "[{0}] {1}".format(best_row["RULE_DECISION"], snippet)

        best_map[(mrn, field)] = shorten_text(snippet, max_len=500)

    return best_map


# ---------------------------------------------------
# Confusion logic
# ---------------------------------------------------

def classify_case(pred, gold):
    if pd.isna(pred) or pd.isna(gold):
        return None

    if pred == 1 and gold == 1:
        return "TP"
    if pred == 0 and gold == 0:
        return "TN"
    if pred == 1 and gold == 0:
        return "FP"
    if pred == 0 and gold == 1:
        return "FN"

    return None


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    print("Loading files...")

    master = safe_read_csv(MASTER_FILE)
    gold = safe_read_csv(GOLD_FILE)

    print("Master rows:", len(master))
    print("Gold rows:", len(gold))

    if not os.path.exists(EVID_FILE):
        print("Evidence file not found:", EVID_FILE)
        evid = pd.DataFrame()
    else:
        evid = safe_read_csv(EVID_FILE)
        print("Evidence rows:", len(evid))

    if MRN not in master.columns:
        raise RuntimeError("Master missing MRN column")
    if MRN not in gold.columns:
        raise RuntimeError("Gold missing MRN column")

    master[MRN] = master[MRN].astype(str).str.strip()
    gold[MRN] = gold[MRN].astype(str).str.strip()

    master = master[master[MRN] != ""].copy()
    gold = gold[gold[MRN] != ""].copy()

    master = master.drop_duplicates(subset=[MRN])
    gold = gold.drop_duplicates(subset=[MRN])

    merged = pd.merge(
        master,
        gold,
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    print("Merged rows:", len(merged))

    best_evidence_map = build_best_evidence_map(evid)

    qa_rows = []

    print("\nPBS confusion counts\n")

    for var in PBS_VARS:
        pred_col = var + "_pred"
        gold_col = var + "_gold"

        if pred_col not in merged.columns or gold_col not in merged.columns:
            print("Skipping variable:", var)
            continue

        pred_norm = merged[pred_col].apply(normalize_binary_value)
        gold_norm = merged[gold_col].apply(normalize_binary_value)

        valid_mask = gold_norm.notna()
        pred_norm = pred_norm[valid_mask]
        gold_norm = gold_norm[valid_mask]
        sub = merged.loc[valid_mask, [MRN]].copy()
        sub["pred"] = pred_norm.values
        sub["gold"] = gold_norm.values

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for _, row in sub.iterrows():
            mrn = clean_cell(row[MRN])
            pred = row["pred"]
            goldv = row["gold"]

            case_type = classify_case(pred, goldv)
            if case_type is None:
                continue

            if case_type == "TP":
                tp += 1
            elif case_type == "TN":
                tn += 1
            elif case_type == "FP":
                fp += 1
            elif case_type == "FN":
                fn += 1

            snippet = best_evidence_map.get((mrn, var), "")

            qa_rows.append({
                "variable": var,
                "case_type": case_type,
                "gold": int(goldv),
                "pred": int(pred),
                "snippet": snippet
            })

        total = tp + tn + fp + fn
        acc = float(tp + tn) / float(total) if total > 0 else 0.0

        print(var)
        print("  total:", total)
        print("  TP:", tp)
        print("  TN:", tn)
        print("  FP:", fp)
        print("  FN:", fn)
        print("  accuracy:", round(acc, 6))
        print("")

    qa_df = pd.DataFrame(qa_rows)

    # helpful ordering for manual review
    case_order = {
        "FN": 0,
        "FP": 1,
        "TP": 2,
        "TN": 3
    }

    if len(qa_df) > 0:
        qa_df["_case_order"] = qa_df["case_type"].map(case_order).fillna(9)
        qa_df = qa_df.sort_values(
            by=["variable", "_case_order", "gold", "pred"],
            ascending=[True, True, False, False]
        ).drop(columns=["_case_order"])

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    qa_df.to_csv(OUTPUT_QA, index=False)

    print("QA file written to:", OUTPUT_QA)
    print("\nDone.")


if __name__ == "__main__":
    main()
