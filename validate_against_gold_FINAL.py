#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import glob
import pandas as pd


def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def normalize_id(x):
    return "" if x is None else str(x).strip()


def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def looks_like_stage_summary(csv_path):
    """
    Accept only files that actually look like your patient_stage_summary.csv
    (has STAGE1/2 fields or HAS_STAGE fields).
    """
    try:
        df = normalize_cols(read_csv_robust(csv_path, nrows=5, dtype=str, low_memory=False))
    except Exception:
        return False

    cols = set(df.columns)
    must_have_any = {
        "STAGE1_DATE", "STAGE2_DATE", "HAS_STAGE1", "HAS_STAGE2",
        "STAGE1_NOTE_ID", "STAGE2_NOTE_ID", "STAGE1_HITS", "STAGE2_HITS"
    }
    return len(cols.intersection(must_have_any)) > 0


def find_prediction_csv(root):
    # 1) Hard-prefer the exact file you showed
    preferred = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    if os.path.isfile(preferred):
        return os.path.abspath(preferred)

    # 2) Otherwise search, but only accept true stage-summary-like files
    candidates = []
    candidates += glob.glob(os.path.join(root, "_outputs", "*patient*stage*summary*.csv"))
    candidates += glob.glob(os.path.join(root, "_outputs", "*stage*summary*.csv"))
    candidates += glob.glob(os.path.join(root, "**", "*patient*stage*summary*.csv"), recursive=True)
    candidates += glob.glob(os.path.join(root, "**", "*stage*summary*.csv"), recursive=True)

    # Exclude obvious non-pred outputs
    bad_substrings = ["validation", "metrics", "audit", "overlap", "scorable", "ppv", "npv", "sensitivity", "specificity"]
    filtered = []
    for p in candidates:
        lp = p.lower()
        if any(b in lp for b in bad_substrings):
            continue
        if os.path.isfile(p):
            filtered.append(os.path.abspath(p))

    # Keep only those that actually look like stage summaries
    filtered = [p for p in filtered if looks_like_stage_summary(p)]

    if not filtered:
        return None

    filtered.sort(key=lambda x: len(x))
    return filtered[0]


def find_op_notes_csv(root):
    candidates = []
    candidates += glob.glob(os.path.join(root, "_staging_inputs", "*Operation Notes*.csv"))
    candidates += glob.glob(os.path.join(root, "**", "*Operation Notes*.csv"), recursive=True)
    files = [os.path.abspath(c) for c in candidates if os.path.isfile(c)]
    if not files:
        return None
    files.sort(key=lambda x: len(x))
    return files[0]


def main():
    root = os.path.abspath(".")

    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    pred_path = find_prediction_csv(root)
    op_path = find_op_notes_csv(root)

    if not os.path.isfile(gold_path):
        raise IOError("Gold file not found: {}".format(gold_path))
    if not pred_path:
        raise IOError("Could not find patient_stage_summary.csv (expected at _outputs/patient_stage_summary.csv).")
    if not op_path:
        raise IOError("Operation Notes CSV not found (needed to map MRN <-> encrypted id).")

    print("Using:")
    print("  Gold:", gold_path)
    print("  Pred:", pred_path)
    print("  Op  :", op_path)
    print("")

    gold = normalize_cols(read_csv_robust(gold_path, dtype=str, low_memory=False))
    pred = normalize_cols(read_csv_robust(pred_path, dtype=str, low_memory=False))
    op = normalize_cols(read_csv_robust(op_path, dtype=str, low_memory=False))

    # --- Map MRN <-> ENCRYPTED_PAT_ID using op notes
    op_mrn_col = pick_first_existing(op, ["MRN", "mrn"])
    op_encpat_col = pick_first_existing(op, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])

    if not op_mrn_col or not op_encpat_col:
        raise ValueError("Op notes must contain MRN and ENCRYPTED_PAT_ID (or equivalent). Found: {}".format(list(op.columns)))

    op[op_mrn_col] = op[op_mrn_col].map(normalize_id)
    op[op_encpat_col] = op[op_encpat_col].map(normalize_id)

    id_map = op[[op_encpat_col, op_mrn_col]].dropna().drop_duplicates()
    id_map.columns = ["ENCRYPTED_PAT_ID", "MRN"]

    # --- Gold
    gold_mrn_col = pick_first_existing(gold, ["MRN", "mrn"])
    if not gold_mrn_col:
        raise ValueError("Gold missing MRN column.")
    gold[gold_mrn_col] = gold[gold_mrn_col].map(normalize_id)

    gold_stage2_col = pick_first_existing(gold, ["Stage2_Applicable", "STAGE2_APPLICABLE"])
    if not gold_stage2_col:
        raise ValueError("Gold missing Stage2_Applicable (or STAGE2_APPLICABLE).")
    gold["GOLD_HAS_STAGE2"] = gold[gold_stage2_col].map(to01).astype(int)

    # --- Pred: ensure MRN exists (your patient_stage_summary starts with ENCRYPTED_PAT_ID)
    pred_encpat_col = pick_first_existing(pred, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    pred_mrn_col = pick_first_existing(pred, ["MRN", "mrn"])

    if pred_mrn_col:
        pred["MRN"] = pred[pred_mrn_col].map(normalize_id)
        print("Pred join key: using MRN column =", pred_mrn_col)
    elif pred_encpat_col:
        pred = pred.rename(columns={pred_encpat_col: "ENCRYPTED_PAT_ID"})
        pred["ENCRYPTED_PAT_ID"] = pred["ENCRYPTED_PAT_ID"].map(normalize_id)
        pred = pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")
        print("Pred join key: using ENCRYPTED_PAT_ID column =", pred_encpat_col, "-> mapped to MRN via op notes")
    else:
        raise ValueError(
            "Prediction summary missing usable ID column. "
            "Need MRN or ENCRYPTED_PAT_ID. Found columns: {}".format(list(pred.columns))
        )

    pred["MRN"] = pred["MRN"].fillna("").map(normalize_id)

    # --- Pred Stage2 signal
    if "HAS_STAGE2" in pred.columns:
        pred["PRED_HAS_STAGE2"] = pred["HAS_STAGE2"].map(to01).astype(int)
        print("Pred stage2 signal: using HAS_STAGE2")
    elif "STAGE2_DATE" in pred.columns:
        pred["PRED_HAS_STAGE2"] = pred["STAGE2_DATE"].notna().astype(int)
        print("Pred stage2 signal: using STAGE2_DATE presence")
    else:
        stage2_note_col = pick_first_existing(pred, ["STAGE2_NOTE_ID", "STAGE2_NOTEID"])
        stage2_hits_col = pick_first_existing(pred, ["STAGE2_HITS"])
        if stage2_note_col:
            pred["PRED_HAS_STAGE2"] = pred[stage2_note_col].notna().astype(int)
            print("Pred stage2 signal: using {} presence".format(stage2_note_col))
        elif stage2_hits_col:
            pred["PRED_HAS_STAGE2"] = pred[stage2_hits_col].fillna("0").map(
                lambda x: 1 if str(x).strip() not in ["0", "0.0", ""] else 0
            ).astype(int)
            print("Pred stage2 signal: using {} > 0".format(stage2_hits_col))
        else:
            raise ValueError("Prediction file missing stage2 signal columns (HAS_STAGE2 / STAGE2_DATE / STAGE2_NOTE_ID / STAGE2_HITS).")

    # --- Merge and metrics
    merged = gold.merge(pred, left_on=gold_mrn_col, right_on="MRN", how="left")
    merged["PRED_HAS_STAGE2"] = merged["PRED_HAS_STAGE2"].fillna(0).astype(int)

    tp = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fp = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 1)).sum())
    fn = int(((merged["GOLD_HAS_STAGE2"] == 1) & (merged["PRED_HAS_STAGE2"] == 0)).sum())
    tn = int(((merged["GOLD_HAS_STAGE2"] == 0) & (merged["PRED_HAS_STAGE2"] == 0)).sum())

    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    mism_path = os.path.join(out_dir, "validation_mismatches.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics.txt")

    merged.to_csv(merged_path, index=False)
    merged[merged["GOLD_HAS_STAGE2"] != merged["PRED_HAS_STAGE2"]].to_csv(mism_path, index=False)

    with open(metrics_path, "w") as f:
        f.write("TP: {}\nFP: {}\nFN: {}\nTN: {}\n".format(tp, fp, fn, tn))
        f.write("Precision: {:.4f}\n".format(precision))
        f.write("Recall: {:.4f}\n".format(recall))
        f.write("F1: {:.4f}\n".format(f1))

    print("")
    print("Validation complete.")
    print("TP={} FP={} FN={} TN={}".format(tp, fp, fn, tn))
    print("Precision={:.3f} Recall={:.3f} F1={:.3f}".format(precision, recall, f1))
    print("")
    print("Wrote:")
    print("  ", merged_path)
    print("  ", mism_path)
    print("  ", metrics_path)


if __name__ == "__main__":
    main()
