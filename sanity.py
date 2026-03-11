#!/usr/bin/env python3
# qa_review_all_fields.py
#
# Creates ONE QA CSV with:
# - one row per case
# - gold/pred columns for selected fields
# - match flags
# - best evidence columns per field
#
# IMPORTANT:
# - Does NOT include MRN in final output
# - Uses master_abstraction_rule_FINAL_NO_GOLD.csv as prediction file
# - You must set GOLD_PATH correctly
#
# Python 3.6.8 compatible

import os
import re
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

PRED_PATH = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_PATH = "{0}/_outputs/master_abstraction_gold.csv".format(BASE_DIR)   # <-- CHANGE THIS IF NEEDED
EVID_PATH = "{0}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv".format(BASE_DIR)

OUT_PATH = "{0}/_outputs/qa_exports/QA_ALL_FIELDS_REVIEW.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

TARGET_FIELDS = [
    "Indication_Left",
    "Indication_Right",
    "LymphNode",
    "Recon_Type",
    "Recon_Classification",
]

MAX_SNIPPET_LEN = 500
MISMATCH_ONLY = True   # True = only rows with at least one mismatch; False = all rows


def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                error_bad_lines=False,
                warn_bad_lines=True
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                on_bad_lines="skip"
            )
        except TypeError:
            return pd.read_csv(
                path,
                **common_kwargs,
                encoding="latin-1",
                error_bad_lines=False,
                warn_bad_lines=True
            )


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    found = None
    for k in key_variants:
        if k in df.columns:
            found = k
            break
    if found is None:
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:60]))
    if found != MERGE_KEY:
        df = df.rename(columns={found: MERGE_KEY})
    df[MERGE_KEY] = df[MERGE_KEY].fillna("").astype(str).str.strip()
    return df


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s


def norm_compare(x):
    s = clean_cell(x)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def truncate_text(x, n=MAX_SNIPPET_LEN):
    s = clean_cell(x)
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return s
    return s[:n] + " ..."


def choose_best_evidence_for_field(evid_df, field):
    tmp = evid_df[evid_df["FIELD"].fillna("").astype(str).str.strip() == field].copy()
    if tmp.empty:
        return pd.DataFrame(columns=[
            MERGE_KEY,
            "evidence_value",
            "evidence_snippet",
            "evidence_note_type",
            "evidence_note_date",
            "evidence_section",
            "evidence_confidence",
            "evidence_note_id"
        ])

    for c in ["CONFIDENCE", "NOTE_TYPE", "SECTION", "NOTE_DATE", "EVIDENCE", "VALUE", "NOTE_ID"]:
        if c not in tmp.columns:
            tmp[c] = ""

    tmp["CONFIDENCE_NUM"] = pd.to_numeric(tmp["CONFIDENCE"], errors="coerce").fillna(0.0)
    tmp["NOTE_TYPE_LOW"] = tmp["NOTE_TYPE"].fillna("").astype(str).str.lower()
    tmp["SECTION_UP"] = tmp["SECTION"].fillna("").astype(str).str.upper()

    tmp["OP_BONUS"] = tmp["NOTE_TYPE_LOW"].apply(
        lambda s: 1 if ("op" in s or "operative" in s or "operation" in s) else 0
    )

    low_value_sections = set([
        "PAST MEDICAL HISTORY", "PAST SURGICAL HISTORY", "SURGICAL HISTORY",
        "HISTORY", "PMH", "PSH"
    ])
    tmp["SECTION_PENALTY"] = tmp["SECTION_UP"].apply(
        lambda s: -1 if s in low_value_sections else 0
    )

    tmp = tmp.sort_values(
        by=[MERGE_KEY, "CONFIDENCE_NUM", "OP_BONUS", "SECTION_PENALTY", "NOTE_DATE"],
        ascending=[True, False, False, False, False]
    )

    tmp = tmp.drop_duplicates(subset=[MERGE_KEY], keep="first").copy()

    tmp["evidence_snippet"] = tmp["EVIDENCE"].apply(truncate_text)

    out = pd.DataFrame()
    out[MERGE_KEY] = tmp[MERGE_KEY]
    out["evidence_value"] = tmp["VALUE"]
    out["evidence_snippet"] = tmp["evidence_snippet"]
    out["evidence_note_type"] = tmp["NOTE_TYPE"]
    out["evidence_note_date"] = tmp["NOTE_DATE"]
    out["evidence_section"] = tmp["SECTION"]
    out["evidence_confidence"] = tmp["CONFIDENCE"]
    out["evidence_note_id"] = tmp["NOTE_ID"]

    return out


def main():
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError("Prediction file not found: {0}".format(PRED_PATH))
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError("Gold file not found: {0}".format(GOLD_PATH))
    if not os.path.exists(EVID_PATH):
        raise FileNotFoundError("Evidence file not found: {0}".format(EVID_PATH))

    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pred = clean_cols(read_csv_robust(PRED_PATH))
    gold = clean_cols(read_csv_robust(GOLD_PATH))
    evid = clean_cols(read_csv_robust(EVID_PATH))

    pred = normalize_mrn(pred)
    gold = normalize_mrn(gold)
    evid = normalize_mrn(evid)

    missing_in_pred = [f for f in TARGET_FIELDS if f not in pred.columns]
    missing_in_gold = [f for f in TARGET_FIELDS if f not in gold.columns]

    if missing_in_pred:
        raise RuntimeError("Missing target fields in pred: {0}".format(missing_in_pred))
    if missing_in_gold:
        raise RuntimeError("Missing target fields in gold: {0}".format(missing_in_gold))

    # Start from gold so all reviewed cases stay in output
    final_df = gold[[MERGE_KEY]].copy()

    for field in TARGET_FIELDS:
        gold_sub = gold[[MERGE_KEY, field]].copy().rename(columns={field: field + "_gold"})
        pred_sub = pred[[MERGE_KEY, field]].copy().rename(columns={field: field + "_pred"})

        merged = gold_sub.merge(pred_sub, on=MERGE_KEY, how="left")

        merged[field + "_gold_norm"] = merged[field + "_gold"].apply(norm_compare)
        merged[field + "_pred_norm"] = merged[field + "_pred"].apply(norm_compare)
        merged[field + "_match"] = merged[field + "_gold_norm"] == merged[field + "_pred_norm"]

        merged[field + "_error_type"] = ""
        merged.loc[
            (merged[field + "_gold_norm"] == "") & (merged[field + "_pred_norm"] != ""),
            field + "_error_type"
        ] = "false_positive"
        merged.loc[
            (merged[field + "_gold_norm"] != "") & (merged[field + "_pred_norm"] == ""),
            field + "_error_type"
        ] = "false_negative"
        merged.loc[
            (merged[field + "_gold_norm"] != "") &
            (merged[field + "_pred_norm"] != "") &
            (merged[field + "_gold_norm"] != merged[field + "_pred_norm"]),
            field + "_error_type"
        ] = "wrong_value"

        best_evid = choose_best_evidence_for_field(evid, field)

        evid_rename = {
            "evidence_value": field + "_evidence_value",
            "evidence_snippet": field + "_evidence_snippet",
            "evidence_note_type": field + "_evidence_note_type",
            "evidence_note_date": field + "_evidence_note_date",
            "evidence_section": field + "_evidence_section",
            "evidence_confidence": field + "_evidence_confidence",
            "evidence_note_id": field + "_evidence_note_id",
        }
        best_evid = best_evid.rename(columns=evid_rename)

        merged = merged.merge(best_evid, on=MERGE_KEY, how="left")

        keep_cols = [
            MERGE_KEY,
            field + "_gold",
            field + "_pred",
            field + "_match",
            field + "_error_type",
            field + "_evidence_value",
            field + "_evidence_snippet",
            field + "_evidence_note_type",
            field + "_evidence_note_date",
            field + "_evidence_section",
            field + "_evidence_confidence",
            field + "_evidence_note_id",
        ]

        merged = merged[keep_cols].copy()
        final_df = final_df.merge(merged, on=MERGE_KEY, how="left")

    # any mismatch across chosen fields
    match_cols = [f + "_match" for f in TARGET_FIELDS]
    final_df["any_mismatch"] = False
    for c in match_cols:
        final_df["any_mismatch"] = final_df["any_mismatch"] | (final_df[c] == False)

    if MISMATCH_ONLY:
        final_df = final_df[final_df["any_mismatch"] == True].copy()

    # drop MRN before export
    if MERGE_KEY in final_df.columns:
        final_df = final_df.drop(columns=[MERGE_KEY])

    final_df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(final_df))
    print("Columns:", len(final_df.columns))


if __name__ == "__main__":
    main()
