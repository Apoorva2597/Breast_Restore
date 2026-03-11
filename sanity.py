#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qa_pbs_lumpectomy_targeted.py

Targeted QA script for PBS_Lumpectomy.

What it does:
- merges master and gold on MRN
- focuses only on PBS_Lumpectomy
- prints TP / TN / FP / FN counts in terminal
- writes ONLY lumpectomy mismatches (FP + FN) to a CSV
- includes the best evidence snippet from pbs_only_evidence.csv
- does NOT include MRN in the output file

Output CSV columns:
- case_type
- gold
- pred
- note_type
- note_date
- rule_decision
- snippet

Python 3.6.8 compatible.
"""

import os
import pandas as pd

MASTER_FILE = "_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"
GOLD_FILE = "gold_cleaned_for_cedar.csv"
EVID_FILE = "_outputs/pbs_only_evidence.csv"
OUTPUT_QA = "_outputs/pbs_lumpectomy_targeted_qa.csv"

MRN = "MRN"
FIELD = "PBS_Lumpectomy"

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


def shorten_text(x, max_len=700):
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
    if s == "accept_pre_recon_historical":
        return 1
    if s == "accept_pre_recon_lumpectomy":
        return 2
    if s == "accept_post_recon_historical":
        return 3
    if s == "accept_post_recon_inferred_laterality":
        return 4
    if s == "accept_inferred_laterality":
        return 5

    if s == "reject_post_recon_not_historical":
        return 20
    if s == "reject_unknown_laterality_unilateral":
        return 21
    if s == "reject_unknown_recon_laterality":
        return 22
    if s == "reject_contralateral":
        return 23
    if s == "reject_pre_recon_no_history":
        return 24
    if s == "reject_negative_history":
        return 25
    if s == "reject_missing_date_diff":
        return 26
    if s == "extractor_failed":
        return 98

    return 99


def confidence_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return -1.0


def build_best_evidence_map(evid_df):
    """
    Returns:
        best_map[mrn] = {
            "snippet": ...,
            "rule_decision": ...,
            "note_type": ...,
            "note_date": ...
        }
    """
    best_map = {}

    if evid_df is None or len(evid_df) == 0:
        return best_map

    required_cols = [
        "MRN", "FIELD", "EVIDENCE", "RULE_DECISION",
        "CONFIDENCE", "NOTE_TYPE", "NOTE_DATE"
    ]
    for col in required_cols:
        if col not in evid_df.columns:
            return best_map

    tmp = evid_df.copy()
    tmp["MRN"] = tmp["MRN"].astype(str).str.strip()
    tmp["FIELD"] = tmp["FIELD"].astype(str).str.strip()

    tmp = tmp[tmp["FIELD"] == FIELD].copy()
    if len(tmp) == 0:
        return best_map

    tmp["RULE_DECISION"] = tmp["RULE_DECISION"].astype(str).str.strip()
    tmp["EVIDENCE"] = tmp["EVIDENCE"].astype(str).str.strip()
    tmp["NOTE_TYPE"] = tmp["NOTE_TYPE"].astype(str).str.strip()
    tmp["NOTE_DATE"] = tmp["NOTE_DATE"].astype(str).str.strip()
    tmp["CONFIDENCE_NUM"] = tmp["CONFIDENCE"].apply(confidence_float)
    tmp["RULE_RANK"] = tmp["RULE_DECISION"].apply(rule_rank)

    for mrn, g in tmp.groupby("MRN", dropna=False):
        mrn = clean_cell(mrn)
        if not mrn:
            continue

        g = g.sort_values(
            by=["RULE_RANK", "CONFIDENCE_NUM"],
            ascending=[True, False]
        )

        best_row = g.iloc[0]

        snippet = clean_cell(best_row["EVIDENCE"])
        rule_decision = clean_cell(best_row["RULE_DECISION"])
        note_type = clean_cell(best_row["NOTE_TYPE"])
        note_date = clean_cell(best_row["NOTE_DATE"])

        if rule_decision:
            snippet = "[{0}] {1}".format(rule_decision, snippet)

        best_map[mrn] = {
            "snippet": shorten_text(snippet, max_len=700),
            "rule_decision": rule_decision,
            "note_type": note_type,
            "note_date": note_date
        }

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

    pred_col = FIELD
    gold_col = FIELD

    if pred_col not in master.columns:
        raise RuntimeError("Master missing field: {0}".format(FIELD))
    if gold_col not in gold.columns:
        raise RuntimeError("Gold missing field: {0}".format(FIELD))

    merged = pd.merge(
        master[[MRN, pred_col]],
        gold[[MRN, gold_col]],
        on=MRN,
        how="inner",
        suffixes=("_pred", "_gold")
    )

    print("Merged rows:", len(merged))

    pred_norm = merged[pred_col + "_pred"].apply(normalize_binary_value)
    gold_norm = merged[pred_col + "_gold"].apply(normalize_binary_value)

    valid_mask = gold_norm.notna()

    sub = merged.loc[valid_mask, [MRN]].copy()
    sub["pred"] = pred_norm[valid_mask].values
    sub["gold"] = gold_norm[valid_mask].values

    best_evidence_map = build_best_evidence_map(evid)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    qa_rows = []

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

        if case_type in ("FP", "FN"):
            ev = best_evidence_map.get(mrn, {})
            qa_rows.append({
                "case_type": case_type,
                "gold": int(goldv),
                "pred": int(pred),
                "note_type": clean_cell(ev.get("note_type", "")),
                "note_date": clean_cell(ev.get("note_date", "")),
                "rule_decision": clean_cell(ev.get("rule_decision", "")),
                "snippet": clean_cell(ev.get("snippet", ""))
            })

    total = tp + tn + fp + fn
    acc = float(tp + tn) / float(total) if total > 0 else 0.0

    print("\nTargeted QA for {0}\n".format(FIELD))
    print("total:", total)
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("accuracy:", round(acc, 6))

    qa_df = pd.DataFrame(qa_rows)

    case_order = {
        "FN": 0,
        "FP": 1
    }

    if len(qa_df) > 0:
        qa_df["_case_order"] = qa_df["case_type"].map(case_order).fillna(9)
        qa_df = qa_df.sort_values(
            by=["_case_order", "rule_decision", "note_date"],
            ascending=[True, True, True]
        ).drop(columns=["_case_order"])

    if not os.path.exists("_outputs"):
        os.makedirs("_outputs")

    qa_df.to_csv(OUTPUT_QA, index=False)

    print("\nTargeted mismatch file written to:", OUTPUT_QA)
    print("Done.")


if __name__ == "__main__":
    main()
