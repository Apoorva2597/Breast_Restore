#!/usr/bin/env python3
# qa_review_two_fields.py
#
# Creates ONE QA CSV for:
# - LymphNode
# - Indication_Left
#
# Output includes:
# - gold / pred
# - match flag
# - error type
# - best evidence snippet + note metadata
#
# IMPORTANT:
# - Does NOT include MRN in final output
# - Uses master_abstraction_rule_FINAL_NO_GOLD.csv as prediction file
# - You must set GOLD_PATH correctly if needed
#
# Python 3.6.8 compatible

import os
import re
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

PRED_PATH = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_PATH = "{0}/_outputs/master_abstraction_gold.csv".format(BASE_DIR)   # <-- change if needed
EVID_PATH = "{0}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv".format(BASE_DIR)

OUT_PATH = "{0}/_outputs/qa_exports/QA_LymphNode_IndicationLeft.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

TARGET_FIELDS = [
    "LymphNode",
    "Indication_Left",
]

MAX_SNIPPET_LEN = 600
MISMATCH_ONLY = True   # True = only rows with at least one mismatch


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


def evidence_rank(row):
    conf = pd.to_numeric(row.get("CONFIDENCE", ""), errors="coerce")
    if pd.isna(conf):
        conf = 0.0

    note_type = clean_cell(row.get("NOTE_TYPE", "")).lower()
    section = clean_cell(row.get("SECTION", "")).upper()
    note_date = clean_cell(row.get("NOTE_DATE", ""))
    evidence = clean_cell(row.get("EVIDENCE", "")).lower()

    score = float(conf)

    if "clinic" in note_type or "progress" in note_type or "oncology" in note_type or "follow up" in note_type or "follow-up" in note_type:
        score += 0.20
    if "op note" in note_type or "operative" in note_type or "operation" in note_type or "brief op" in note_type:
        score += 0.10

    if section in {"PAST MEDICAL HISTORY", "PAST SURGICAL HISTORY", "SURGICAL HISTORY", "PMH", "PSH", "HISTORY"}:
        score -= 0.15

    if re.search(r"\b(plan|planned|possible|consider|may need|might need|if positive|if needed|pending)\b", evidence):
        score -= 0.25

    if re.search(r"\b(s/p|status post|underwent|history of|hx of|prior|previous|performed|completed)\b", evidence):
        score += 0.10

    if note_date:
        score += 0.01

    return score


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

    tmp["_rank"] = tmp.apply(evidence_rank, axis=1)
    tmp = tmp.sort_values(by=[MERGE_KEY, "_rank"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=[MERGE_KEY], keep="first").copy()

    out = pd.DataFrame()
    out[MERGE_KEY] = tmp[MERGE_KEY]
    out["evidence_value"] = tmp["VALUE"]
    out["evidence_snippet"] = tmp["EVIDENCE"].apply(truncate_text)
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

        best_evid = best_evid.rename(columns={
            "evidence_value": field + "_evidence_value",
            "evidence_snippet": field + "_evidence_snippet",
            "evidence_note_type": field + "_evidence_note_type",
            "evidence_note_date": field + "_evidence_note_date",
            "evidence_section": field + "_evidence_section",
            "evidence_confidence": field + "_evidence_confidence",
            "evidence_note_id": field + "_evidence_note_id",
        })

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

    match_cols = [f + "_match" for f in TARGET_FIELDS]
    final_df["any_mismatch"] = False
    for c in match_cols:
        final_df["any_mismatch"] = final_df["any_mismatch"] | (final_df[c] == False)

    if MISMATCH_ONLY:
        final_df = final_df[final_df["any_mismatch"] == True].copy()

    if MERGE_KEY in final_df.columns:
        final_df = final_df.drop(columns=[MERGE_KEY])

    final_df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Rows:", len(final_df))
    print("Columns:", len(final_df.columns))


if __name__ == "__main__":
    main()
