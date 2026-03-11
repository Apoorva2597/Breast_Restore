#!/usr/bin/env python3
# qa_error_review_export.py
#
# Exports QA-ready error review CSVs for selected fields with:
# - gold
# - pred
# - evidence snippets
# - note metadata
#
# IMPORTANT:
# - Does NOT include MRN in output
# - Creates one QA CSV per target field
# - Pulls only mismatch rows by default
#
# Python 3.6.8 compatible

import os
import re
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_PRED_PATH = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_PATH = "{0}/gold_cleaned_for_cedar".format(BASE_DIR)   
EVID_PATH = "{0}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUT_DIR = "{0}/_outputs/qa_exports".format(BASE_DIR)

MERGE_KEY = "MRN"

TARGET_FIELDS = [
    "Indication_Left",
    "Indication_Right",
    "LymphNode",
    "Recon_Type",
    "Recon_Classification",
]

MAX_SNIPPET_LEN = 500
MAX_ROWS_PER_FIELD = 50      # set to None to export all mismatches
MISMATCH_ONLY = True         # True = only errors; False = all rows


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
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:50]))
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
    s = clean_cell(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def truncate_text(x, n=MAX_SNIPPET_LEN):
    s = clean_cell(x)
    s = re.sub(r"\s+", " ", s)
    if len(s) <= n:
        return s
    return s[:n] + " ..."


def choose_best_evidence_for_field(evid_df, field):
    """
    For each MRN + FIELD, keep best evidence row.
    Priority:
    1. higher confidence
    2. op/operative note
    3. non-history section
    4. latest non-empty note date string tie-break by lexical sort
    """
    tmp = evid_df[evid_df["FIELD"].astype(str).str.strip() == field].copy()
    if tmp.empty:
        return tmp

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
    tmp["SECTION_PENALTY"] = tmp["SECTION_UP"].apply(lambda s: -1 if s in low_value_sections else 0)

    tmp = tmp.sort_values(
        by=[MERGE_KEY, "CONFIDENCE_NUM", "OP_BONUS", "SECTION_PENALTY", "NOTE_DATE"],
        ascending=[True, False, False, False, False]
    )

    tmp = tmp.drop_duplicates(subset=[MERGE_KEY], keep="first")
    return tmp


def main():
    if not os.path.exists(MASTER_PRED_PATH):
        raise FileNotFoundError("Pred master not found: {0}".format(MASTER_PRED_PATH))
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError("Gold file not found: {0}".format(GOLD_PATH))
    if not os.path.exists(EVID_PATH):
        raise FileNotFoundError("Evidence file not found: {0}".format(EVID_PATH))

    os.makedirs(OUT_DIR, exist_ok=True)

    pred = clean_cols(read_csv_robust(MASTER_PRED_PATH))
    gold = clean_cols(read_csv_robust(GOLD_PATH))
    evid = clean_cols(read_csv_robust(EVID_PATH))

    pred = normalize_mrn(pred)
    gold = normalize_mrn(gold)
    evid = normalize_mrn(evid)

    missing_in_pred = [f for f in TARGET_FIELDS if f not in pred.columns]
    missing_in_gold = [f for f in TARGET_FIELDS if f not in gold.columns]

    if missing_in_pred:
        raise RuntimeError("Missing target fields in pred file: {0}".format(missing_in_pred))
    if missing_in_gold:
        raise RuntimeError("Missing target fields in gold file: {0}".format(missing_in_gold))

    for field in TARGET_FIELDS:
        pred_sub = pred[[MERGE_KEY, field]].copy().rename(columns={field: "pred"})
        gold_sub = gold[[MERGE_KEY, field]].copy().rename(columns={field: "gold"})

        merged = gold_sub.merge(pred_sub, on=MERGE_KEY, how="inner")

        merged["gold_norm"] = merged["gold"].apply(norm_compare)
        merged["pred_norm"] = merged["pred"].apply(norm_compare)
        merged["is_match"] = merged["gold_norm"] == merged["pred_norm"]

        if MISMATCH_ONLY:
            merged = merged[merged["is_match"] == False].copy()

        best_evid = choose_best_evidence_for_field(evid, field)

        keep_cols = [MERGE_KEY]
        for c in ["VALUE", "EVIDENCE", "NOTE_TYPE", "NOTE_DATE", "SECTION", "CONFIDENCE", "NOTE_ID"]:
            if c in best_evid.columns:
                keep_cols.append(c)

        best_evid = best_evid[keep_cols].copy()

        rename_map = {
            "VALUE": "evidence_value",
            "EVIDENCE": "evidence_snippet",
            "NOTE_TYPE": "evidence_note_type",
            "NOTE_DATE": "evidence_note_date",
            "SECTION": "evidence_section",
            "CONFIDENCE": "evidence_confidence",
            "NOTE_ID": "evidence_note_id"
        }
        best_evid = best_evid.rename(columns=rename_map)

        out = merged.merge(best_evid, on=MERGE_KEY, how="left")

        if "evidence_snippet" in out.columns:
            out["evidence_snippet"] = out["evidence_snippet"].apply(truncate_text)

        # helpful QA labels
        out["error_type_guess"] = ""
        out.loc[(out["gold_norm"] == "") & (out["pred_norm"] != ""), "error_type_guess"] = "false_positive"
        out.loc[(out["gold_norm"] != "") & (out["pred_norm"] == ""), "error_type_guess"] = "false_negative"
        out.loc[
            (out["gold_norm"] != "") &
            (out["pred_norm"] != "") &
            (out["gold_norm"] != out["pred_norm"]),
            "error_type_guess"
        ] = "wrong_value"

        # remove MRN before save
        if MAX_ROWS_PER_FIELD is not None and len(out) > MAX_ROWS_PER_FIELD:
            out = out.head(MAX_ROWS_PER_FIELD).copy()

        final_cols = [
            "gold",
            "pred",
            "error_type_guess",
            "evidence_value",
            "evidence_snippet",
            "evidence_note_type",
            "evidence_note_date",
            "evidence_section",
            "evidence_confidence",
            "evidence_note_id",
        ]

        for c in final_cols:
            if c not in out.columns:
                out[c] = ""

        out = out[final_cols].copy()

        out_path = os.path.join(OUT_DIR, "QA_{0}.csv".format(field))
        out.to_csv(out_path, index=False)

        print("Saved:", out_path, "| rows:", len(out))

    print("\nDone.")
    print("Review these files in:", OUT_DIR)


if __name__ == "__main__":
    main()
