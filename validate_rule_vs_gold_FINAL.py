#!/usr/bin/env python3
# validate_rule_vs_gold_FINAL_NO_GOLD.py
#
# Validates rule-based master output (built WITHOUT gold) vs gold labels.
# - Robust to older pandas (no on_bad_lines required)
# - Robust to column name spacing/underscore differences
# - Robust to merge suffixes (_gold / _pred), preventing KeyError like 'Age'
#
# Outputs:
#   1) /home/apokol/Breast_Restore/_outputs/rule_vs_gold_validation_report_FINAL_NO_GOLD.csv
#   2) /home/apokol/Breast_Restore/_outputs/rule_vs_gold_confusions_FINAL_NO_GOLD.csv
#   3) /home/apokol/Breast_Restore/_outputs/rule_vs_gold_mismatches_FINAL_NO_GOLD.csv

import os
import re
from typing import Dict, Tuple, List

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

GOLD_PATH = f"{BASE_DIR}/gold_cleaned_for_cedar.csv"
MASTER_PATH = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

OUT_REPORT = f"{BASE_DIR}/_outputs/rule_vs_gold_validation_report_FINAL_NO_GOLD.csv"
OUT_CONFUSIONS = f"{BASE_DIR}/_outputs/rule_vs_gold_confusions_FINAL_NO_GOLD.csv"
OUT_MISMATCHES = f"{BASE_DIR}/_outputs/rule_vs_gold_mismatches_FINAL_NO_GOLD.csv"

MERGE_KEY = "MRN"


# -----------------------
# Robust CSV read (old pandas safe)
# -----------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(
                path, **common_kwargs, encoding="latin-1",
                error_bad_lines=False, warn_bad_lines=True
            )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(
                path, **common_kwargs, encoding="latin-1",
                error_bad_lines=False, warn_bad_lines=True
            )


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df: pd.DataFrame) -> pd.DataFrame:
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"MRN column not found. Columns seen: {list(df.columns)[:60]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


# -----------------------
# Column canonicalization
# -----------------------
def canon(s: str) -> str:
    """
    Canonicalize column names so that:
      'PBS_Breast Reduction' == 'PBS_Breast_Reduction' == 'pbs breast   reduction'
    """
    s = str(s).strip()
    s = s.replace("\ufeff", "")
    s = s.lower()
    s = re.sub(r"[\.]", "", s)
    s = re.sub(r"[\s]+", "_", s)       # spaces -> _
    s = re.sub(r"[_]+", "_", s)        # collapse
    return s


def build_canon_map(cols: List[str]) -> Dict[str, str]:
    """
    Returns: canonical_name -> original_column_name (first occurrence)
    """
    out = {}
    for c in cols:
        k = canon(c)
        if k not in out:
            out[k] = c
    return out


# -----------------------
# Conversions
# -----------------------
TRUE_TOKENS = {"1", "true", "yes", "y", "t"}
FALSE_TOKENS = {"0", "false", "no", "n", "f"}

def to_binary_series(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t in TRUE_TOKENS:
            return 1
        if t in FALSE_TOKENS:
            return 0
        # treat empty as NA
        if t == "" or t == "nan":
            return pd.NA
        return pd.NA
    return s.apply(conv)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int, int, int, int, int]:
    """
    Returns TP, FP, FN, TN, N (N compared where both non-missing)
    """
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    n = int(mask.sum())
    return tp, fp, fn, tn, n


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    spec = safe_div(tn, tn + fp)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec}


def numeric_metrics(g: pd.Series, p: pd.Series) -> Dict[str, float]:
    """
    Computes MAE and pct-within thresholds on rows where both are numeric.
    """
    gnum = pd.to_numeric(g, errors="coerce")
    pnum = pd.to_numeric(p, errors="coerce")
    mask = (~gnum.isna()) & (~pnum.isna())
    if int(mask.sum()) == 0:
        return {"n_compared": 0, "mae": pd.NA, "pct_within_1": pd.NA, "pct_within_2": pd.NA}

    diff = (pnum[mask] - gnum[mask]).abs()
    mae = float(diff.mean())
    pct1 = float((diff <= 1.0).mean())
    pct2 = float((diff <= 2.0).mean())
    return {"n_compared": int(mask.sum()), "mae": mae, "pct_within_1": pct1, "pct_within_2": pct2}


def categorical_accuracy(g: pd.Series, p: pd.Series) -> Dict[str, float]:
    """
    Accuracy on rows where both non-missing after trimming / lowercasing.
    """
    g2 = g.astype(str).str.strip()
    p2 = p.astype(str).str.strip()
    # treat empty strings as NA
    g2 = g2.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    p2 = p2.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    mask = (~g2.isna()) & (~p2.isna())
    if int(mask.sum()) == 0:
        return {"n_compared": 0, "accuracy": pd.NA}
    acc = float((g2[mask].str.lower() == p2[mask].str.lower()).mean())
    return {"n_compared": int(mask.sum()), "accuracy": acc}


# -----------------------
# Variables to validate
# -----------------------
NUMERIC_VARS = ["Age", "BMI"]
CATEGORICAL_VARS = [
    "Race", "Ethnicity", "SmokingStatus",
    "Mastectomy_Laterality", "Indication_Left", "Indication_Right", "LymphNode",
    "Recon_Laterality", "Recon_Type", "Recon_Classification", "Recon_Timing",
]
BINARY_VARS = [
    "Diabetes", "Obesity", "Hypertension", "CardiacDisease", "VenousThromboembolism", "Steroid",
    "PastBreastSurgery",
    "PBS_Lumpectomy", "PBS_Breast Reduction", "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other",
    "Radiation", "Radiation_Before", "Radiation_After",
    "Chemo", "Chemo_Before", "Chemo_After",
    "Stage1_MinorComp", "Stage1_Reoperation", "Stage1_Rehospitalization", "Stage1_MajorComp",
    "Stage1_Failure", "Stage1_Revision",
    "Stage2_MinorComp", "Stage2_Reoperation", "Stage2_Rehospitalization", "Stage2_MajorComp",
    "Stage2_Failure", "Stage2_Revision", "Stage2_Applicable",
]


def main():
    print("Loading gold + predictions...")
    gold = normalize_mrn(clean_cols(read_csv_robust(GOLD_PATH)))
    pred = normalize_mrn(clean_cols(read_csv_robust(MASTER_PATH)))

    # Canon maps so minor naming differences don't break matching
    gold_map = build_canon_map(list(gold.columns))
    pred_map = build_canon_map(list(pred.columns))

    # Merge with explicit suffixes so we never lose columns
    merged = gold.merge(pred, on=MERGE_KEY, how="inner", suffixes=("_gold", "_pred"))
    print(f"Joined rows: {len(merged)} (inner join on {MERGE_KEY})")

    # Helper to fetch correct merged col name for a logical variable
    def merged_cols_for(var: str) -> Tuple[str, str]:
        """
        Returns (gold_col_in_merged, pred_col_in_merged)
        """
        # Find original column names in each df using canonical matching
        key = canon(var)
        gold_orig = gold_map.get(key)
        pred_orig = pred_map.get(key)

        if gold_orig is None or pred_orig is None:
            return ("", "")

        # After merge, if both had same name, they become gold_orig + _gold / _pred
        # If names differed, they remain unsuffixed.
        gold_in_merged = gold_orig + "_gold" if (gold_orig in gold.columns and gold_orig in pred.columns) else gold_orig
        pred_in_merged = pred_orig + "_pred" if (pred_orig in pred.columns and pred_orig in gold.columns) else pred_orig

        # Fallback if pandas chose suffixing anyway (rare edge): try direct presence
        if gold_in_merged not in merged.columns:
            if gold_orig + "_gold" in merged.columns:
                gold_in_merged = gold_orig + "_gold"
            elif gold_orig in merged.columns:
                gold_in_merged = gold_orig

        if pred_in_merged not in merged.columns:
            if pred_orig + "_pred" in merged.columns:
                pred_in_merged = pred_orig + "_pred"
            elif pred_orig in merged.columns:
                pred_in_merged = pred_orig

        return (gold_in_merged, pred_in_merged)

    report_rows = []
    confusion_rows = []
    mismatch_rows = []

    # Small safety check: if many shared columns are identical row-by-row, warn
    shared_keys = set(gold_map.keys()) & set(pred_map.keys())
    check_vars = [k for k in shared_keys if k not in {canon(MERGE_KEY)}]
    identical_count = 0
    checked = 0
    for k in check_vars[:25]:  # sample
        gorig = gold_map[k]
        porig = pred_map[k]
        if gorig in gold.columns and porig in pred.columns:
            # compare on MRN inner set
            gtmp = gold[[MERGE_KEY, gorig]].merge(pred[[MERGE_KEY, porig]], on=MERGE_KEY, how="inner")
            checked += 1
            # identical if all equal after string normalize
            gser = gtmp[gorig].astype(str).str.strip().str.lower()
            pser = gtmp[porig].astype(str).str.strip().str.lower()
            if (gser == pser).mean() > 0.98:
                identical_count += 1
    if checked and identical_count >= max(8, int(0.5 * checked)):
        print("WARNING: Many shared columns look identical between gold and master for the joined set.")
        print("         If master was initialized/copied from gold, validation may be inflated.")

    # --- Numeric ---
    for var in NUMERIC_VARS:
        gcol, pcol = merged_cols_for(var)
        if not gcol or not pcol or gcol not in merged.columns or pcol not in merged.columns:
            report_rows.append({
                "variable": var, "type": "numeric", "status": "MISSING_COLUMN",
                "gold_col": gcol, "pred_col": pcol
            })
            continue

        met = numeric_metrics(merged[gcol], merged[pcol])
        n_gold_nonmiss = int(pd.to_numeric(merged[gcol], errors="coerce").notna().sum())
        n_pred_nonmiss = int(pd.to_numeric(merged[pcol], errors="coerce").notna().sum())
        report_rows.append({
            "variable": var,
            "type": "numeric",
            "status": "OK",
            "gold_col": gcol,
            "pred_col": pcol,
            "n_rows_joined": len(merged),
            "n_gold_nonmissing": n_gold_nonmiss,
            "n_pred_nonmissing": n_pred_nonmiss,
            "n_compared": met["n_compared"],
            "mae": met["mae"],
            "pct_within_1": met["pct_within_1"],
            "pct_within_2": met["pct_within_2"],
        })

        # mismatches sample (largest diffs)
        gnum = pd.to_numeric(merged[gcol], errors="coerce")
        pnum = pd.to_numeric(merged[pcol], errors="coerce")
        mask = (~gnum.isna()) & (~pnum.isna())
        if int(mask.sum()) > 0:
            diffs = (pnum[mask] - gnum[mask]).abs()
            top = diffs.sort_values(ascending=False).head(40).index
            for idx in top:
                mismatch_rows.append({
                    "variable": var,
                    MERGE_KEY: merged.loc[idx, MERGE_KEY],
                    "gold": merged.loc[idx, gcol],
                    "pred": merged.loc[idx, pcol],
                    "abs_diff": float(abs(float(pnum.loc[idx]) - float(gnum.loc[idx])))
                })

    # --- Categorical ---
    for var in CATEGORICAL_VARS:
        gcol, pcol = merged_cols_for(var)
        if not gcol or not pcol or gcol not in merged.columns or pcol not in merged.columns:
            report_rows.append({
                "variable": var, "type": "categorical", "status": "MISSING_COLUMN",
                "gold_col": gcol, "pred_col": pcol
            })
            continue

        met = categorical_accuracy(merged[gcol], merged[pcol])
        n_gold_nonmiss = int(merged[gcol].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA}).notna().sum())
        n_pred_nonmiss = int(merged[pcol].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA}).notna().sum())

        report_rows.append({
            "variable": var,
            "type": "categorical",
            "status": "OK",
            "gold_col": gcol,
            "pred_col": pcol,
            "n_rows_joined": len(merged),
            "n_gold_nonmissing": n_gold_nonmiss,
            "n_pred_nonmissing": n_pred_nonmiss,
            "n_compared": met["n_compared"],
            "accuracy": met["accuracy"],
        })

        # mismatches sample
        g2 = merged[gcol].astype(str).str.strip()
        p2 = merged[pcol].astype(str).str.strip()
        g2 = g2.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        p2 = p2.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
        mask = (~g2.isna()) & (~p2.isna()) & (g2.str.lower() != p2.str.lower())
        bad = merged[mask].head(80)
        for _, r in bad.iterrows():
            mismatch_rows.append({
                "variable": var,
                MERGE_KEY: r[MERGE_KEY],
                "gold": r[gcol],
                "pred": r[pcol],
                "abs_diff": ""
            })

    # --- Binary ---
    for var in BINARY_VARS:
        gcol, pcol = merged_cols_for(var)
        if not gcol or not pcol or gcol not in merged.columns or pcol not in merged.columns:
            report_rows.append({
                "variable": var, "type": "binary", "status": "MISSING_COLUMN",
                "gold_col": gcol, "pred_col": pcol
            })
            continue

        yt = to_binary_series(merged[gcol])
        yp = to_binary_series(merged[pcol])
        tp, fp, fn, tn, n = confusion_counts(yt, yp)
        met = metrics_from_counts(tp, fp, fn, tn)

        n_gold_nonmiss = int(yt.notna().sum())
        n_pred_nonmiss = int(yp.notna().sum())

        report_rows.append({
            "variable": var,
            "type": "binary",
            "status": "OK",
            "gold_col": gcol,
            "pred_col": pcol,
            "n_rows_joined": len(merged),
            "n_gold_nonmissing": n_gold_nonmiss,
            "n_pred_nonmissing": n_pred_nonmiss,
            "n_compared": n,
            "accuracy": met["accuracy"],
            "precision": met["precision"],
            "recall": met["recall"],
            "f1": met["f1"],
            "specificity": met["specificity"],
        })

        confusion_rows.append({
            "variable": var, "gold_col": gcol, "pred_col": pcol,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "N_compared": n
        })

        # mismatches sample
        mask = (~yt.isna()) & (~yp.isna()) & (yt.astype(int) != yp.astype(int))
        bad = merged.loc[mask, [MERGE_KEY, gcol, pcol]].head(120)
        for _, r in bad.iterrows():
            mismatch_rows.append({
                "variable": var,
                MERGE_KEY: r[MERGE_KEY],
                "gold": r[gcol],
                "pred": r[pcol],
                "abs_diff": ""
            })

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    pd.DataFrame(report_rows).to_csv(OUT_REPORT, index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT_CONFUSIONS, index=False)
    pd.DataFrame(mismatch_rows).to_csv(OUT_MISMATCHES, index=False)

    print("\nDONE.")
    print(f"- Report:      {OUT_REPORT}")
    print(f"- Confusions:  {OUT_CONFUSIONS}")
    print(f"- Mismatches:  {OUT_MISMATCHES}")
    print("\nQuick view:")
    print(f"  head -n 30 {OUT_REPORT}")
    print(f"  head -n 30 {OUT_CONFUSIONS}")
    print(f"  head -n 30 {OUT_MISMATCHES}")


if __name__ == "__main__":
    main()
