#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_complications_qa_batch3.py

Purpose:
- Build a single-file QA sample for the 3 complication variables
  that still need final focused review
- Use existing master + complications evidence
- Output ONE CSV
- Final QA file contains NO MRN column

Sampling:
- 15 predicted-positive rows per field
- 15 predicted-negative rows per field

Focused fields:
- Stage1_MinorComp
- Stage2_MinorComp
- Stage2_Revision

Output:
- /home/apokol/Breast_Restore/_outputs/complications_qa_batch3.csv

Python 3.6.8 compatible.
"""

import os
import random
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_with_stage2_preds_complications.csv".format(BASE_DIR)
EVID_FILE = "{0}/_outputs/complications_patch_evidence.csv".format(BASE_DIR)
OUTPUT_QA = "{0}/_outputs/complications_qa_batch3.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

FIELDS = [
    "Stage1_MinorComp",
    "Stage2_MinorComp",
    "Stage2_Revision",
]

POSITIVES_N = 15
NEGATIVES_N = 15
RANDOM_SEED = 42


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


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


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
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def to_binary_01(x):
    s = clean_cell(x).lower()
    if s in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    return 0


def truncate_text(x, limit=350):
    s = clean_cell(x)
    if len(s) <= limit:
        return s
    return s[:limit] + " ...[TRUNCATED]"


def make_deid_case_id(field, bucket, idx):
    return "{0}_{1}_{2:03d}".format(field, bucket, idx)


def confidence_to_float(x):
    s = clean_cell(x)
    try:
        return float(s)
    except Exception:
        return 0.0


def status_rank(x):
    s = clean_cell(x).lower()
    if s == "history":
        return 3
    if s == "performed":
        return 3
    if s == "denied":
        return 2
    if s == "planned":
        return 1
    return 0


def get_best_evidence_map(evid_df, field_name):
    """
    Keep best evidence row per MRN for one field.
    Preference:
    1) higher status rank
    2) higher confidence
    3) non-empty evidence
    """
    best = {}

    subset = evid_df[evid_df["FIELD"].astype(str).str.strip() == field_name].copy()

    for _, row in subset.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        conf = confidence_to_float(row.get("CONFIDENCE", "0"))
        s_rank = status_rank(row.get("STATUS", ""))
        evidence = clean_cell(row.get("EVIDENCE", ""))
        score = (s_rank, conf, 1 if evidence else 0)

        existing = best.get(mrn)
        if existing is None or score > existing["_score"]:
            saved = dict(row)
            saved["_score"] = score
            best[mrn] = saved

    return best


def build_rows_for_field(master_df, best_evidence, field_name, positives_n, negatives_n, rng):
    out = []

    if field_name not in master_df.columns:
        return out

    tmp = master_df.copy()
    tmp["_bin_"] = tmp[field_name].apply(to_binary_01)

    pos_df = tmp[tmp["_bin_"] == 1].copy()
    neg_df = tmp[tmp["_bin_"] == 0].copy()

    pos_idxs = list(pos_df.index)
    neg_idxs = list(neg_df.index)

    rng.shuffle(pos_idxs)
    rng.shuffle(neg_idxs)

    pos_idxs = pos_idxs[:min(positives_n, len(pos_idxs))]
    neg_idxs = neg_idxs[:min(negatives_n, len(neg_idxs))]

    for i, idx in enumerate(pos_idxs, 1):
        row = pos_df.loc[idx]
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        ev = best_evidence.get(mrn, {})

        out.append({
            "qa_case_id": make_deid_case_id(field_name, "POS", i),
            "field": field_name,
            "sample_bucket": "predicted_positive",
            "predicted_value": 1,
            "review_label": "",
            "error_type": "",
            "review_notes": "",
            "confidence": clean_cell(ev.get("CONFIDENCE", "")),
            "status": clean_cell(ev.get("STATUS", "")),
            "section": clean_cell(ev.get("SECTION", "")),
            "note_type": clean_cell(ev.get("NOTE_TYPE", "")),
            "note_date": clean_cell(ev.get("NOTE_DATE", "")),
            "stage_assigned": clean_cell(ev.get("STAGE_ASSIGNED", "")),
            "evidence_snippet": truncate_text(ev.get("EVIDENCE", ""), 350),
        })

    for i, idx in enumerate(neg_idxs, 1):
        row = neg_df.loc[idx]
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        ev = best_evidence.get(mrn, {})

        out.append({
            "qa_case_id": make_deid_case_id(field_name, "NEG", i),
            "field": field_name,
            "sample_bucket": "predicted_negative",
            "predicted_value": 0,
            "review_label": "",
            "error_type": "",
            "review_notes": "",
            "confidence": clean_cell(ev.get("CONFIDENCE", "")),
            "status": clean_cell(ev.get("STATUS", "")),
            "section": clean_cell(ev.get("SECTION", "")),
            "note_type": clean_cell(ev.get("NOTE_TYPE", "")),
            "note_date": clean_cell(ev.get("NOTE_DATE", "")),
            "stage_assigned": clean_cell(ev.get("STAGE_ASSIGNED", "")),
            "evidence_snippet": truncate_text(ev.get("EVIDENCE", ""), 350),
        })

    return out


def main():
    random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError("Master file not found: {0}".format(MASTER_FILE))
    if not os.path.exists(EVID_FILE):
        raise FileNotFoundError("Evidence file not found: {0}".format(EVID_FILE))

    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    print("Loading evidence...")
    evid = clean_cols(read_csv_robust(EVID_FILE))
    evid = normalize_mrn(evid)

    qa_rows = []

    for field_name in FIELDS:
        if field_name not in master.columns:
            print("WARNING: field missing in master: {0}".format(field_name))
            master[field_name] = 0

        best_evidence = get_best_evidence_map(evid, field_name)
        rows = build_rows_for_field(master, best_evidence, field_name, POSITIVES_N, NEGATIVES_N, rng)
        qa_rows.extend(rows)
        print("{0}: {1} rows".format(field_name, len(rows)))

    qa_df = pd.DataFrame(qa_rows)

    desired_cols = [
        "qa_case_id",
        "field",
        "sample_bucket",
        "predicted_value",
        "review_label",
        "error_type",
        "review_notes",
        "confidence",
        "status",
        "section",
        "note_type",
        "note_date",
        "stage_assigned",
        "evidence_snippet",
    ]

    for c in desired_cols:
        if c not in qa_df.columns:
            qa_df[c] = ""

    qa_df = qa_df[desired_cols].copy()

    leak_cols = [c for c in qa_df.columns if c.strip().upper() in {"MRN", "PAT_MRN", "PATIENT_MRN"}]
    if leak_cols:
        qa_df = qa_df.drop(columns=leak_cols)

    os.makedirs(os.path.dirname(OUTPUT_QA), exist_ok=True)
    qa_df.to_csv(OUTPUT_QA, index=False)

    print("\nDONE.")
    print("- QA file: {0}".format(OUTPUT_QA))
    print("- Total rows: {0}".format(len(qa_df)))
    print("- MRN removed from final output.")


if __name__ == "__main__":
    main()
