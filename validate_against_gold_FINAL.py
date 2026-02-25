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
        "STAGE1_NOTE_ID", "STAGE2_NOTE_ID", "STAGE1_HITS", "STAGE2_HITS",
        # also accept the outcome-pred file if user points it here accidentally
        "Stage2_Reoperation_pred", "Stage2_Rehospitalization_pred", "Stage2_MajorComp_pred",
        "Stage2_Failure_pred", "Stage2_Revision_pred",
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


def find_stage2_outcomes_pred_csv(root):
    """
    Prefer the outputs file produced by build_stage2_outcomes_from_encounters.py
    """
    preferred = os.path.join(root, "_outputs", "stage2_outcomes_pred.csv")
    if os.path.isfile(preferred):
        return os.path.abspath(preferred)

    # fallback search
    candidates = []
    candidates += glob.glob(os.path.join(root, "_outputs", "*stage2*outcomes*pred*.csv"))
    candidates += glob.glob(os.path.join(root, "**", "*stage2*outcomes*pred*.csv"), recursive=True)
    files = [os.path.abspath(c) for c in candidates if os.path.isfile(c)]
    if not files:
        return None
    files.sort(key=lambda x: len(x))
    return files[0]


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_binary_metrics(df, gold_col, pred_col):
    """
    df must already have gold_col and pred_col as 0/1 ints (no NaN).
    """
    tp = int(((df[gold_col] == 1) & (df[pred_col] == 1)).sum())
    fp = int(((df[gold_col] == 0) & (df[pred_col] == 1)).sum())
    fn = int(((df[gold_col] == 1) & (df[pred_col] == 0)).sum())
    tn = int(((df[gold_col] == 0) & (df[pred_col] == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc
    }


def main():
    root = os.path.abspath(".")

    gold_path = os.path.join(root, "gold_cleaned_for_cedar.csv")
    stage_pred_path = find_prediction_csv(root)  # patient_stage_summary.csv (stage2 anchor validation)
    out_pred_path = find_stage2_outcomes_pred_csv(root)  # stage2_outcomes_pred.csv (outcome validation)
    op_path = find_op_notes_csv(root)

    if not os.path.isfile(gold_path):
        raise IOError("Gold file not found: {}".format(gold_path))
    if not stage_pred_path:
        raise IOError("Could not find patient_stage_summary.csv (expected at _outputs/patient_stage_summary.csv).")
    if not out_pred_path:
        raise IOError("Could not find stage2_outcomes_pred.csv (expected at _outputs/stage2_outcomes_pred.csv).")
    if not op_path:
        raise IOError("Operation Notes CSV not found (needed to map MRN <-> encrypted id).")

    print("Using:")
    print("  Gold        :", gold_path)
    print("  Stage Pred  :", stage_pred_path)
    print("  Outcome Pred:", out_pred_path)
    print("  Op Notes    :", op_path)
    print("")

    gold = normalize_cols(read_csv_robust(gold_path, dtype=str, low_memory=False))
    stage_pred = normalize_cols(read_csv_robust(stage_pred_path, dtype=str, low_memory=False))
    out_pred = normalize_cols(read_csv_robust(out_pred_path, dtype=str, low_memory=False))
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

    # --- Gold MRN
    gold_mrn_col = pick_first_existing(gold, ["MRN", "mrn"])
    if not gold_mrn_col:
        raise ValueError("Gold missing MRN column.")
    gold[gold_mrn_col] = gold[gold_mrn_col].map(normalize_id)

    # --- Gold Stage2 applicable
    gold_stage2_app_col = pick_first_existing(gold, ["Stage2_Applicable", "STAGE2_APPLICABLE"])
    if not gold_stage2_app_col:
        raise ValueError("Gold missing Stage2_Applicable (or STAGE2_APPLICABLE).")
    gold["GOLD_HAS_STAGE2"] = gold[gold_stage2_app_col].map(to01).astype(int)

    # --- Gold outcome cols (these exist in your gold file)
    # We validate these only where GOLD_HAS_STAGE2 == 1 (per definition)
    gold_outcome_map = [
        ("Stage2_Reoperation", "GOLD_Stage2_Reoperation"),
        ("Stage2_Rehospitalization", "GOLD_Stage2_Rehospitalization"),
        ("Stage2_MajorComp", "GOLD_Stage2_MajorComp"),
        ("Stage2_Failure", "GOLD_Stage2_Failure"),
        ("Stage2_Revision", "GOLD_Stage2_Revision"),
    ]
    for src, dst in gold_outcome_map:
        if src not in gold.columns:
            raise ValueError("Gold missing expected outcome column: {}".format(src))
        gold[dst] = gold[src].map(to01).astype(int)

    # --- Stage Pred: ensure MRN exists
    stage_pred_encpat_col = pick_first_existing(stage_pred, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    stage_pred_mrn_col = pick_first_existing(stage_pred, ["MRN", "mrn"])

    if stage_pred_mrn_col:
        stage_pred["MRN"] = stage_pred[stage_pred_mrn_col].map(normalize_id)
        print("Stage Pred join key: using MRN column =", stage_pred_mrn_col)
    elif stage_pred_encpat_col:
        stage_pred = stage_pred.rename(columns={stage_pred_encpat_col: "ENCRYPTED_PAT_ID"})
        stage_pred["ENCRYPTED_PAT_ID"] = stage_pred["ENCRYPTED_PAT_ID"].map(normalize_id)
        stage_pred = stage_pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")
        print("Stage Pred join key: using ENCRYPTED_PAT_ID column =", stage_pred_encpat_col, "-> mapped to MRN via op notes")
    else:
        raise ValueError("Stage prediction summary missing usable ID column (MRN or ENCRYPTED_PAT_ID). Found columns: {}".format(list(stage_pred.columns)))

    stage_pred["MRN"] = stage_pred["MRN"].fillna("").map(normalize_id)

    # --- Stage Pred Stage2 signal
    if "HAS_STAGE2" in stage_pred.columns:
        stage_pred["PRED_HAS_STAGE2"] = stage_pred["HAS_STAGE2"].map(to01).astype(int)
        print("Stage Pred stage2 signal: using HAS_STAGE2")
    elif "STAGE2_DATE" in stage_pred.columns:
        stage_pred["PRED_HAS_STAGE2"] = stage_pred["STAGE2_DATE"].notna().astype(int)
        print("Stage Pred stage2 signal: using STAGE2_DATE presence")
    else:
        stage2_note_col = pick_first_existing(stage_pred, ["STAGE2_NOTE_ID", "STAGE2_NOTEID"])
        stage2_hits_col = pick_first_existing(stage_pred, ["STAGE2_HITS"])
        if stage2_note_col:
            stage_pred["PRED_HAS_STAGE2"] = stage_pred[stage2_note_col].notna().astype(int)
            print("Stage Pred stage2 signal: using {} presence".format(stage2_note_col))
        elif stage2_hits_col:
            stage_pred["PRED_HAS_STAGE2"] = stage_pred[stage2_hits_col].fillna("0").map(
                lambda x: 1 if str(x).strip() not in ["0", "0.0", ""] else 0
            ).astype(int)
            print("Stage Pred stage2 signal: using {} > 0".format(stage2_hits_col))
        else:
            raise ValueError("Stage prediction file missing stage2 signal columns (HAS_STAGE2 / STAGE2_DATE / STAGE2_NOTE_ID / STAGE2_HITS).")

    # --- Outcome Pred: ensure MRN exists
    out_pred_encpat_col = pick_first_existing(out_pred, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    out_pred_mrn_col = pick_first_existing(out_pred, ["MRN", "mrn"])

    if out_pred_mrn_col:
        out_pred["MRN"] = out_pred[out_pred_mrn_col].map(normalize_id)
        print("Outcome Pred join key: using MRN column =", out_pred_mrn_col)
    elif out_pred_encpat_col:
        out_pred = out_pred.rename(columns={out_pred_encpat_col: "ENCRYPTED_PAT_ID"})
        out_pred["ENCRYPTED_PAT_ID"] = out_pred["ENCRYPTED_PAT_ID"].map(normalize_id)
        out_pred = out_pred.merge(id_map, on="ENCRYPTED_PAT_ID", how="left")
        print("Outcome Pred join key: using ENCRYPTED_PAT_ID column =", out_pred_encpat_col, "-> mapped to MRN via op notes")
    else:
        raise ValueError("Outcome prediction file missing usable ID column (MRN or ENCRYPTED_PAT_ID). Found columns: {}".format(list(out_pred.columns)))

    out_pred["MRN"] = out_pred["MRN"].fillna("").map(normalize_id)

    # --- Outcome Pred columns we will validate
    pred_outcome_cols = [
        "Stage2_Reoperation_pred",
        "Stage2_Rehospitalization_pred",
        "Stage2_MajorComp_pred",
        "Stage2_Failure_pred",
        "Stage2_Revision_pred",
    ]
    for c in pred_outcome_cols:
        if c not in out_pred.columns:
            raise ValueError("Outcome prediction file missing expected column: {}. Found: {}".format(c, list(out_pred.columns)))

    # Convert pred outcome cols to 0/1
    for c in pred_outcome_cols:
        out_pred[c] = out_pred[c].map(to01).astype(int)

    # --- Merge (gold + stage pred + outcome pred)
    merged = gold.merge(stage_pred[["MRN", "PRED_HAS_STAGE2"]], left_on=gold_mrn_col, right_on="MRN", how="left", suffixes=("", "_stagepred"))
    merged = merged.merge(out_pred, left_on=gold_mrn_col, right_on="MRN", how="left", suffixes=("", "_outpred"))

    merged["PRED_HAS_STAGE2"] = merged["PRED_HAS_STAGE2"].fillna(0).astype(int)

    # If outcome preds missing (no match), treat as 0 for validation purposes
    for c in pred_outcome_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    # --- Stage2 anchor metrics (as before)
    stage_metrics = compute_binary_metrics(merged, "GOLD_HAS_STAGE2", "PRED_HAS_STAGE2")

    # --- Outcome metrics: ONLY where GOLD_HAS_STAGE2 == 1 (per abstraction rules)
    eval_df = merged[merged["GOLD_HAS_STAGE2"] == 1].copy()

    outcome_pairs = [
        ("GOLD_Stage2_Reoperation", "Stage2_Reoperation_pred"),
        ("GOLD_Stage2_Rehospitalization", "Stage2_Rehospitalization_pred"),
        ("GOLD_Stage2_MajorComp", "Stage2_MajorComp_pred"),
        ("GOLD_Stage2_Failure", "Stage2_Failure_pred"),
        ("GOLD_Stage2_Revision", "Stage2_Revision_pred"),
    ]

    outcome_metrics = []
    for gcol, pcol in outcome_pairs:
        m = compute_binary_metrics(eval_df, gcol, pcol)
        m["gold_col"] = gcol
        m["pred_col"] = pcol
        m["n_eval"] = int(len(eval_df))
        outcome_metrics.append(m)

    # --- Write outputs
    out_dir = os.path.join(root, "_outputs")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    merged_path = os.path.join(out_dir, "validation_merged.csv")
    mism_path = os.path.join(out_dir, "validation_mismatches.csv")
    metrics_path = os.path.join(out_dir, "validation_metrics.txt")
    outcome_metrics_path = os.path.join(out_dir, "validation_outcome_metrics.csv")

    merged.to_csv(merged_path, index=False)

    # For mismatches: include stage2 anchor mismatch OR any outcome mismatch (within stage2-applicable)
    outcome_mismatch_mask = False
    for gcol, pcol in outcome_pairs:
        outcome_mismatch_mask = outcome_mismatch_mask | ((merged["GOLD_HAS_STAGE2"] == 1) & (merged[gcol] != merged[pcol]))
    stage_mismatch_mask = (merged["GOLD_HAS_STAGE2"] != merged["PRED_HAS_STAGE2"])
    mism_df = merged[stage_mismatch_mask | outcome_mismatch_mask].copy()
    mism_df.to_csv(mism_path, index=False)

    # metrics txt
    with open(metrics_path, "w") as f:
        f.write("=== Stage2 Anchor (Applicable) ===\n")
        f.write("TP: {TP}\nFP: {FP}\nFN: {FN}\nTN: {TN}\n".format(**stage_metrics))
        f.write("Precision: {:.4f}\n".format(stage_metrics["precision"]))
        f.write("Recall: {:.4f}\n".format(stage_metrics["recall"]))
        f.write("F1: {:.4f}\n".format(stage_metrics["f1"]))
        f.write("Accuracy: {:.4f}\n\n".format(stage_metrics["accuracy"]))

        f.write("=== Stage2 Outcomes (evaluated only where GOLD_HAS_STAGE2==1) ===\n")
        for om in outcome_metrics:
            f.write("\n{} vs {}\n".format(om["gold_col"], om["pred_col"]))
            f.write("n_eval: {}\n".format(om["n_eval"]))
            f.write("TP: {TP} FP: {FP} FN: {FN} TN: {TN}\n".format(**om))
            f.write("Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Acc: {:.4f}\n".format(
                om["precision"], om["recall"], om["f1"], om["accuracy"]
            ))

    # outcome metrics csv (easy to paste into slides/email)
    pd.DataFrame(outcome_metrics).to_csv(outcome_metrics_path, index=False)

    # --- Console summary
    print("")
    print("Validation complete.")
    print("Stage2 Anchor:")
    print("  TP={TP} FP={FP} FN={FN} TN={TN}".format(**stage_metrics))
    print("  Precision={:.3f} Recall={:.3f} F1={:.3f}".format(stage_metrics["precision"], stage_metrics["recall"], stage_metrics["f1"]))
    print("")
    print("Stage2 Outcomes (Gold Stage2 only):")
    for om in outcome_metrics:
        print("  {}: P={:.3f} R={:.3f} F1={:.3f} (TP={} FP={} FN={} TN={})".format(
            om["pred_col"], om["precision"], om["recall"], om["f1"], om["TP"], om["FP"], om["FN"], om["TN"]
        ))

    print("")
    print("Wrote:")
    print("  ", merged_path)
    print("  ", mism_path)
    print("  ", metrics_path)
    print("  ", outcome_metrics_path)


if __name__ == "__main__":
    main()
