#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_comorbidity_qa_single_file.py

Purpose:
- Build a single-file QA sample for comorbidity review
- Use existing master + comorbidity evidence
- Output ONE CSV
- Final QA file contains NO MRN column

Focus fields:
- Obesity
- Diabetes
- Hypertension
- CardiacDisease
- VenousThromboembolism
- Steroid

Sampling:
- positives_per_field: number of predicted-positive rows per field
- negatives_per_field: number of predicted-negative rows per field

Output:
- /home/apokol/Breast_Restore/_outputs/comorbidity_qa_single_file.csv

Python 3.6.8 compatible.
"""

import os
import random
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
EVID_FILE = "{0}/_outputs/comorbidity_only_evidence.csv".format(BASE_DIR)
OUTPUT_QA = "{0}/_outputs/comorbidity_qa_single_file.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

COMORBIDITY_FIELDS = [
    "Obesity",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
]

# adjust if needed
POSITIVES_PER_FIELD = 25
NEGATIVES_PER_FIELD = 25
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


def truncate_text(x, limit=500):
    s = clean_cell(x)
    if len(s) <= limit:
        return s
    return s[:limit] + " ...[TRUNCATED]"


def make_deid_case_id(field, bucket, idx):
    field_short = field.replace("VenousThromboembolism", "VTE")
    field_short = field_short.replace("CardiacDisease", "Cardiac")
    return "{0}_{1}_{2:03d}".format(field_short, bucket, idx)


def get_best_evidence_map(evid_df):
    """
    Keep best accepted evidence row per MRN+FIELD.
    Preference:
    1) higher confidence
    2) non-empty evidence
    """
    best = {}

    for _, row in evid_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        field = clean_cell(row.get("FIELD", ""))
        if not mrn or not field:
            continue
        if field not in COMORBIDITY_FIELDS:
            continue

        key = (mrn, field)

        conf_raw = clean_cell(row.get("CONFIDENCE", "0"))
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0

        evidence = clean_cell(row.get("EVIDENCE", ""))
        score = (conf, 1 if evidence else 0)

        existing = best.get(key)
        if existing is None or score > existing["_score"]:
            saved = dict(row)
            saved["_score"] = score
            best[key] = saved

    return best


def build_positive_rows(master_df, best_evidence, field, n, rng):
    out = []

    eligible = master_df[master_df[field].apply(to_binary_01) == 1].copy()
    if len(eligible) == 0:
        return out

    idxs = list(eligible.index)
    rng.shuffle(idxs)
    idxs = idxs[:min(n, len(idxs))]

    for i, idx in enumerate(idxs, 1):
        row = eligible.loc[idx]
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        ev = best_evidence.get((mrn, field), {})

        out.append({
            "qa_case_id": make_deid_case_id(field, "POS", i),
            "field": field,
            "sample_bucket": "predicted_positive",
            "predicted_value": 1,
            "review_label": "",
            "error_type": "",
            "review_notes": "",
            "confidence": clean_cell(ev.get("CONFIDENCE", "")),
            "status": clean_cell(ev.get("STATUS", "")),
            "rule_decision": clean_cell(ev.get("RULE_DECISION", "")),
            "section": clean_cell(ev.get("SECTION", "")),
            "note_type": clean_cell(ev.get("NOTE_TYPE", "")),
            "note_date": clean_cell(ev.get("NOTE_DATE", "")),
            "evidence_snippet": truncate_text(ev.get("EVIDENCE", ""), 700),
        })

    return out


def build_negative_rows(master_df, best_evidence, field, n, rng):
    out = []

    eligible = master_df[master_df[field].apply(to_binary_01) == 0].copy()
    if len(eligible) == 0:
        return out

    idxs = list(eligible.index)
    rng.shuffle(idxs)
    idxs = idxs[:min(n, len(idxs))]

    for i, idx in enumerate(idxs, 1):
        row = eligible.loc[idx]
        mrn = clean_cell(row.get(MERGE_KEY, ""))

        # most negatives will not have evidence, which is okay
        ev = best_evidence.get((mrn, field), {})

        out.append({
            "qa_case_id": make_deid_case_id(field, "NEG", i),
            "field": field,
            "sample_bucket": "predicted_negative",
            "predicted_value": 0,
            "review_label": "",
            "error_type": "",
            "review_notes": "",
            "confidence": clean_cell(ev.get("CONFIDENCE", "")),
            "status": clean_cell(ev.get("STATUS", "")),
            "rule_decision": clean_cell(ev.get("RULE_DECISION", "")),
            "section": clean_cell(ev.get("SECTION", "")),
            "note_type": clean_cell(ev.get("NOTE_TYPE", "")),
            "note_date": clean_cell(ev.get("NOTE_DATE", "")),
            "evidence_snippet": truncate_text(ev.get("EVIDENCE", ""), 700),
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

    for field in COMORBIDITY_FIELDS:
        if field not in master.columns:
            print("WARNING: field missing in master: {0}".format(field))
            master[field] = 0

    best_evidence = get_best_evidence_map(evid)

    qa_rows = []

    for field in COMORBIDITY_FIELDS:
        pos_rows = build_positive_rows(master, best_evidence, field, POSITIVES_PER_FIELD, rng)
        neg_rows = build_negative_rows(master, best_evidence, field, NEGATIVES_PER_FIELD, rng)

        qa_rows.extend(pos_rows)
        qa_rows.extend(neg_rows)

        print("{0}: +{1} / -{2}".format(field, len(pos_rows), len(neg_rows)))

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
        "rule_decision",
        "section",
        "note_type",
        "note_date",
        "evidence_snippet",
    ]

    for c in desired_cols:
        if c not in qa_df.columns:
            qa_df[c] = ""

    qa_df = qa_df[desired_cols].copy()

    # explicitly ensure no MRN leaks into output
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
