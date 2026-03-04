#!/usr/bin/env python3
# validate_stage2_anchor_WITH_METRICS.py
# Python 3.6.8 compatible
# Writes validation evidence artifacts (json/csv/txt) for Stage2 Anchor evaluation.

import os
import json
import math
import datetime
import pandas as pd

# =========================
# CONFIG (EDIT PATHS)
# =========================
GOLD_PATH       = "/home/apokol/Breast_Restore/gold_cleaned_for_cedar.csv"
STAGE_PRED_PATH = "/home/apokol/Breast_Restore/_outputs/patient_stage_summary.csv"

# Output evidence directory/files
OUT_DIR  = "/home/apokol/Breast_Restore/_outputs"
OUT_JSON = os.path.join(OUT_DIR, "stage2_validation_metrics.json")
OUT_CSV  = os.path.join(OUT_DIR, "stage2_validation_metrics.csv")
OUT_TXT  = os.path.join(OUT_DIR, "stage2_validation_report.txt")

# Column in GOLD that indicates true Stage2 (edit if your gold uses a different name)
GOLD_STAGE2_COL_CANDIDATES = ["HAS_STAGE2", "Stage2", "STAGE2", "STAGE2_ANCHOR", "Stage2_Applicable"]

# Column in STAGE_PRED that indicates predicted Stage2 (edit if needed)
PRED_STAGE2_COL_CANDIDATES = ["HAS_STAGE2", "Stage2", "STAGE2", "PRED_STAGE2"]

# Preferred patient key order
KEY_CANDIDATES = ["ENCRYPTED_PAT_ID", "MRN", "PatientID", "PATIENT_ID", "PAT_ID"]

# =========================
# IO helpers
# =========================
def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas
        return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)

def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df

def find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def normalize_key(df, key_col):
    df[key_col] = df[key_col].astype(str).str.strip()
    return df

def to_binary(series):
    # Accept: 1/0, True/False, yes/no, Y/N, etc.
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"]).astype(int)

# =========================
# Metric helpers
# =========================
def safe_div(num, den):
    return float(num) / float(den) if den else float("nan")

def compute_metrics(TP, FP, FN, TN):
    total = TP + FP + FN + TN

    accuracy   = safe_div(TP + TN, total)
    precision  = safe_div(TP, TP + FP)               # PPV
    recall     = safe_div(TP, TP + FN)               # Sensitivity / TPR
    specificity= safe_div(TN, TN + FP)               # TNR
    npv        = safe_div(TN, TN + FN)

    f1         = safe_div(2.0 * precision * recall, precision + recall)
    fpr        = safe_div(FP, FP + TN)               # Type I error rate
    fnr        = safe_div(FN, FN + TP)               # Type II error rate
    bal_acc    = safe_div(recall + specificity, 2.0)

    # MCC
    mcc_num = (TP * TN) - (FP * FN)
    mcc_den = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    mcc = safe_div(mcc_num, math.sqrt(mcc_den)) if mcc_den else float("nan")

    return {
        "Accuracy": accuracy,
        "Precision_PPV": precision,
        "Recall_Sensitivity": recall,
        "Specificity_TNR": specificity,
        "NPV": npv,
        "F1": f1,
        "FPR_TypeI": fpr,
        "FNR_TypeII": fnr,
        "Balanced_Accuracy": bal_acc,
        "MCC": mcc,
    }

def fmt(x, d=3):
    if x != x:  # NaN
        return "NA"
    return ("{0:." + str(d) + "f}").format(x)

def write_csv(path, counts, metrics):
    lines = ["Metric,Value"]
    for k, v in counts.items():
        lines.append("{},{}".format(k, v))
    for k, v in metrics.items():
        lines.append("{},{}".format(k, fmt(v, 3)))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def write_report(path, run_meta, join_key, gold_y_col, pred_y_col, counts, metrics):
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    txt = []
    txt.append("STAGE 2 ANCHOR VALIDATION REPORT")
    txt.append("=" * 34)
    txt.append("Generated: {}".format(run_meta["generated_at"]))
    txt.append("Gold file: {}".format(run_meta["gold_path"]))
    txt.append("Pred file: {}".format(run_meta["pred_path"]))
    txt.append("Join key used: {}".format(join_key))
    txt.append("Gold label col: {}".format(gold_y_col))
    txt.append("Pred label col: {}".format(pred_y_col))
    txt.append("")
    txt.append("Confusion Matrix (positive = Stage 2 present)")
    txt.append("-" * 43)
    txt.append("TP (true positives):  {}  -> Stage 2 correctly detected".format(TP))
    txt.append("FP (false positives): {}  -> Stage 2 incorrectly flagged (Type I error)".format(FP))
    txt.append("FN (false negatives): {}  -> Stage 2 missed (Type II error)".format(FN))
    txt.append("TN (true negatives):  {}  -> correctly identified as no Stage 2".format(TN))
    txt.append("Total: {}".format(counts["Total"]))
    txt.append("")
    txt.append("Key Metrics")
    txt.append("-" * 11)
    txt.append("Accuracy:             {}".format(fmt(metrics["Accuracy"])))
    txt.append("Precision / PPV:      {}".format(fmt(metrics["Precision_PPV"])))
    txt.append("Recall / Sensitivity: {}".format(fmt(metrics["Recall_Sensitivity"])))
    txt.append("Specificity / TNR:    {}".format(fmt(metrics["Specificity_TNR"])))
    txt.append("NPV:                  {}".format(fmt(metrics["NPV"])))
    txt.append("F1:                   {}".format(fmt(metrics["F1"])))
    txt.append("Balanced Accuracy:    {}".format(fmt(metrics["Balanced_Accuracy"])))
    txt.append("MCC:                  {}".format(fmt(metrics["MCC"])))
    txt.append("")
    txt.append("Error Rates (mapped to hypothesis testing language)")
    txt.append("-" * 48)
    txt.append("Type I error rate (FPR):  {}  = FP/(FP+TN)".format(fmt(metrics["FPR_TypeI"])))
    txt.append("Type II error rate (FNR): {}  = FN/(FN+TP)".format(fmt(metrics["FNR_TypeII"])))
    txt.append("")
    txt.append("Interpretation for your pipeline:")
    txt.append("- Type I (FP): adds manual review burden and may create incorrect Stage-2 anchors.")
    txt.append("- Type II (FN): higher clinical/analytic risk because true Stage-2 patients are missed,")
    txt.append("  weakening downstream anchoring and complication abstraction.")
    txt.append("")
    txt.append("End of report.")
    txt.append("")

    with open(path, "w") as f:
        f.write("\n".join(txt))

# =========================
# Main
# =========================
def main():
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError("Missing GOLD_PATH: {}".format(GOLD_PATH))
    if not os.path.exists(STAGE_PRED_PATH):
        raise FileNotFoundError("Missing STAGE_PRED_PATH: {}".format(STAGE_PRED_PATH))

    print("Loading gold...")
    gold = clean_cols(read_csv_robust(GOLD_PATH))

    print("Loading stage predictions...")
    pred = clean_cols(read_csv_robust(STAGE_PRED_PATH))

    # pick join key
    gold_key = find_first_existing_col(gold, KEY_CANDIDATES)
    pred_key = find_first_existing_col(pred, KEY_CANDIDATES)

    if not gold_key:
        raise ValueError("Gold missing a usable key. Found columns: {}".format(list(gold.columns)))
    if not pred_key:
        raise ValueError("Stage prediction file missing a usable key. Found columns: {}".format(list(pred.columns)))

    JOIN_KEY = gold_key
    if pred_key != JOIN_KEY:
        pred = pred.rename(columns={pred_key: JOIN_KEY})

    gold = normalize_key(gold, JOIN_KEY)
    pred = normalize_key(pred, JOIN_KEY)

    # pick label cols
    gold_y_col = find_first_existing_col(gold, GOLD_STAGE2_COL_CANDIDATES)
    pred_y_col = find_first_existing_col(pred, PRED_STAGE2_COL_CANDIDATES)

    if not gold_y_col:
        raise ValueError(
            "Gold missing Stage2 truth column. Tried {}. Found columns: {}".format(
                GOLD_STAGE2_COL_CANDIDATES, list(gold.columns)
            )
        )
    if not pred_y_col:
        raise ValueError(
            "Pred missing Stage2 prediction column. Tried {}. Found columns: {}".format(
                PRED_STAGE2_COL_CANDIDATES, list(pred.columns)
            )
        )

    # merge
    merged = gold[[JOIN_KEY, gold_y_col]].merge(
        pred[[JOIN_KEY, pred_y_col]],
        on=JOIN_KEY,
        how="left",
        suffixes=("_GOLD", "_PRED"),
    )

    # convert to binary
    y_true = to_binary(merged[gold_y_col])
    y_pred = to_binary(merged[pred_y_col].fillna("0"))

    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())

    # compute metrics
    metrics = compute_metrics(TP, FP, FN, TN)

    # print (keeps your existing output + adds accuracy etc.)
    print("\nValidation complete.")
    print("Stage2 Anchor:")
    print("  TP={} FP={} FN={} TN={}".format(TP, FP, FN, TN))
    print("  Accuracy={} Precision(PPV)={} Recall(Sens)={} Specificity(TNR)={} F1={}".format(
        fmt(metrics["Accuracy"]), fmt(metrics["Precision_PPV"]), fmt(metrics["Recall_Sensitivity"]),
        fmt(metrics["Specificity_TNR"]), fmt(metrics["F1"])
    ))

    # write evidence artifacts
    os.makedirs(OUT_DIR, exist_ok=True)

    counts = {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Total": TP + FP + FN + TN
    }

    run_meta = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gold_path": GOLD_PATH,
        "pred_path": STAGE_PRED_PATH,
        "output_files": {"json": OUT_JSON, "csv": OUT_CSV, "txt": OUT_TXT},
    }

    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "meta": run_meta,
                "schema": {"positive_class": "Stage2 present"},
                "columns_used": {"join_key": JOIN_KEY, "gold_label": gold_y_col, "pred_label": pred_y_col},
                "counts": counts,
                "metrics": metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    write_csv(OUT_CSV, counts, metrics)
    write_report(OUT_TXT, run_meta, JOIN_KEY, gold_y_col, pred_y_col, counts, metrics)

    print("\nWrote evidence:")
    print("  {}".format(OUT_JSON))
    print("  {}".format(OUT_CSV))
    print("  {}".format(OUT_TXT))


if __name__ == "__main__":
    main()
