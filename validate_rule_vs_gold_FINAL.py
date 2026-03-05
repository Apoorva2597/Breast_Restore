#!/usr/bin/env python3
# validate_rule_vs_gold_FINAL.py
#
# Robust validation: RULE-based master output (NO_GOLD) vs GOLD.
# Never crashes on missing columns; skips with status instead.
#
# Outputs:
#   /home/apokol/Breast_Restore/_outputs/rule_vs_gold_validation_report_FINAL.csv
#   /home/apokol/Breast_Restore/_outputs/rule_vs_gold_confusions_FINAL.csv
#   /home/apokol/Breast_Restore/_outputs/rule_vs_gold_mismatches_FINAL.csv

import os
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

BASE_DIR = "/home/apokol/Breast_Restore"
GOLD_PATH = f"{BASE_DIR}/gold_cleaned_for_cedar.csv"
PRED_PATH = f"{BASE_DIR}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv"

OUT_REPORT = f"{BASE_DIR}/_outputs/rule_vs_gold_validation_report_FINAL.csv"
OUT_CONF   = f"{BASE_DIR}/_outputs/rule_vs_gold_confusions_FINAL.csv"
OUT_MISM   = f"{BASE_DIR}/_outputs/rule_vs_gold_mismatches_FINAL.csv"

MERGE_KEY = "MRN"


# -------------------------
# Robust CSV read (pandas)
# -------------------------
def read_csv_robust(path: str) -> pd.DataFrame:
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        # older pandas fallback (pandas<1.3)
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)


def norm_col(c: str) -> str:
    c = str(c).replace("\ufeff", "").strip()
    c = re.sub(r"\s+", " ", c)  # collapse whitespace
    return c


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [norm_col(c) for c in df.columns]
    return df


def normalize_mrn(df: pd.DataFrame) -> pd.DataFrame:
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError(f"MRN column not found. Seen columns (first 60): {list(df.columns)[:60]}")
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


# -------------------------
# Column matching
# -------------------------
def keyify(c: str) -> str:
    # lowercase + remove spaces/underscores/dots/hyphens
    return re.sub(r"[\s_\.\-]+", "", norm_col(c).lower())


def build_col_index(df: pd.DataFrame) -> Dict[str, str]:
    return {keyify(c): c for c in df.columns}


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact first
    for c in candidates:
        if c in df.columns:
            return c
    # normalized fallback
    idx = build_col_index(df)
    for c in candidates:
        k = keyify(c)
        if k in idx:
            return idx[k]
    return None


def resolve_merged_column(merged: pd.DataFrame, base_col: str, side_suffix: str) -> Optional[str]:
    """
    Given an original column name (as found in gold or pred), find the actual column name in merged.
    side_suffix is either "_gold" or "_pred".
    We check:
      1) base_col + side_suffix (common when both sides had same name)
      2) base_col (when it existed only on one side)
      3) normalized match (handles whitespace/underscore differences)
    """
    if base_col is None:
        return None

    cand1 = f"{base_col}{side_suffix}"
    if cand1 in merged.columns:
        return cand1
    if base_col in merged.columns:
        return base_col

    # normalized fallback
    idx = build_col_index(merged)
    k1 = keyify(cand1)
    if k1 in idx:
        return idx[k1]
    k2 = keyify(base_col)
    if k2 in idx:
        return idx[k2]

    return None


# -------------------------
# Value normalization
# -------------------------
def to_na(x: Any) -> Any:
    if x is None:
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return pd.NA
    return x


def to_binary_series(s: pd.Series) -> pd.Series:
    def conv(x):
        x = to_na(x)
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t in {"1", "true", "yes", "y"}:
            return 1
        if t in {"0", "false", "no", "n"}:
            return 0
        if t in {"t"}:
            return 1
        if t in {"f"}:
            return 0
        return pd.NA
    return s.apply(conv)


def to_cat_series(s: pd.Series) -> pd.Series:
    def conv(x):
        x = to_na(x)
        if pd.isna(x):
            return pd.NA
        return str(x).strip().lower()
    return s.apply(conv)


def to_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.apply(lambda x: to_na(x)), errors="coerce")


# -------------------------
# Metrics
# -------------------------
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Tuple[int, int, int, int, int]:
    mask = (~y_true.isna()) & (~y_pred.isna())
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    n  = int(mask.sum())
    return tp, fp, fn, tn, n


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -------------------------
# Variable specs
# -------------------------
VAR_SPECS = [
    {"var": "Age", "type": "numeric",
     "gold": ["Age", "4. Age", "Age_DOS"], "pred": ["Age", "4. Age", "Age_DOS"]},
    {"var": "BMI", "type": "numeric",
     "gold": ["BMI", "5. BMI"], "pred": ["BMI", "5. BMI"]},

    {"var": "Race", "type": "categorical",
     "gold": ["Race", "2. Race"], "pred": ["Race", "2. Race"]},
    {"var": "Ethnicity", "type": "categorical",
     "gold": ["Ethnicity", "3. Ethnicity"], "pred": ["Ethnicity", "3. Ethnicity"]},
    {"var": "SmokingStatus", "type": "categorical",
     "gold": ["SmokingStatus", "7. SmokingStatus"], "pred": ["SmokingStatus", "7. SmokingStatus"]},

    {"var": "Diabetes", "type": "binary",
     "gold": ["Diabetes", "8. Diabetes"], "pred": ["Diabetes", "8. Diabetes", "DiabetesMellitus"]},
    {"var": "Obesity", "type": "binary",
     "gold": ["Obesity", "9. Obesity"], "pred": ["Obesity", "9. Obesity"]},
    {"var": "Hypertension", "type": "binary",
     "gold": ["Hypertension", "10. Hypertension"], "pred": ["Hypertension", "10. Hypertension", "HTN"]},
    {"var": "CardiacDisease", "type": "binary",
     "gold": ["CardiacDisease", "11. CardiacDisease"], "pred": ["CardiacDisease", "11. CardiacDisease"]},
    {"var": "VenousThromboembolism", "type": "binary",
     "gold": ["VenousThromboembolism", "12. VenousThromboembolism"], "pred": ["VenousThromboembolism", "12. VenousThromboembolism", "VTE"]},
    {"var": "Steroid", "type": "binary",
     "gold": ["Steroid", "13. Steroid"], "pred": ["Steroid", "13. Steroid", "SteroidUse"]},

    {"var": "PastBreastSurgery", "type": "binary",
     "gold": ["PastBreastSurgery", "14. PastBreastSurgery"], "pred": ["PastBreastSurgery", "14. PastBreastSurgery"]},
    {"var": "PBS_Lumpectomy", "type": "binary",
     "gold": ["PBS_Lumpectomy", "15. PBS_Lumpectomy"], "pred": ["PBS_Lumpectomy", "15. PBS_Lumpectomy"]},
    {"var": "PBS_Breast Reduction", "type": "binary",
     "gold": ["PBS_Breast Reduction", "16. PBS_Breast Reduction"], "pred": ["PBS_Breast Reduction", "16. PBS_Breast Reduction"]},
    {"var": "PBS_Mastopexy", "type": "binary",
     "gold": ["PBS_Mastopexy", "17. PBS_Mastopexy"], "pred": ["PBS_Mastopexy", "17. PBS_Mastopexy"]},
    {"var": "PBS_Augmentation", "type": "binary",
     "gold": ["PBS_Augmentation", "18. PBS_Augmentation"], "pred": ["PBS_Augmentation", "18. PBS_Augmentation"]},
    {"var": "PBS_Other", "type": "binary",
     "gold": ["PBS_Other", "19. PBS_Other"], "pred": ["PBS_Other", "19. PBS_Other"]},

    {"var": "Mastectomy_Laterality", "type": "categorical",
     "gold": ["Mastectomy_Laterality", "20. Mastectomy_Laterality"], "pred": ["Mastectomy_Laterality", "20. Mastectomy_Laterality"]},

    {"var": "Indication_Left", "type": "categorical",
     "gold": ["Indication_Left", "21. Indication_Left"], "pred": ["Indication_Left", "21. Indication_Left"]},
    {"var": "Indication_Right", "type": "categorical",
     "gold": ["Indication_Right", "22. Indication_Right"], "pred": ["Indication_Right", "22. Indication_Right"]},
    {"var": "LymphNode", "type": "categorical",
     "gold": ["LymphNode", "23. LymphNode"], "pred": ["LymphNode", "23. LymphNode"]},

    {"var": "Radiation", "type": "binary",
     "gold": ["Radiation", "24. Radiation"], "pred": ["Radiation", "24. Radiation"]},
    {"var": "Radiation_Before", "type": "binary",
     "gold": ["Radiation_Before", "25. Radiation_Before"], "pred": ["Radiation_Before", "25. Radiation_Before"]},
    {"var": "Radiation_After", "type": "binary",
     "gold": ["Radiation_After", "26. Radiation_After"], "pred": ["Radiation_After", "26. Radiation_After"]},

    {"var": "Chemo", "type": "binary",
     "gold": ["Chemo", "27. Chemo"], "pred": ["Chemo", "27. Chemo"]},
    {"var": "Chemo_Before", "type": "binary",
     "gold": ["Chemo_Before", "28. Chemo_Before"], "pred": ["Chemo_Before", "28. Chemo_Before"]},
    {"var": "Chemo_After", "type": "binary",
     "gold": ["Chemo_After", "29. Chemo_After"], "pred": ["Chemo_After", "29. Chemo_After"]},

    {"var": "Recon_Laterality", "type": "categorical",
     "gold": ["Recon_Laterality", "30. Recon_Laterality"], "pred": ["Recon_Laterality", "30. Recon_Laterality"]},
    {"var": "Recon_Type", "type": "categorical",
     "gold": ["Recon_Type", "31. Recon_Type"], "pred": ["Recon_Type", "31. Recon_Type"]},
    {"var": "Recon_Classification", "type": "categorical",
     "gold": ["Recon_Classification", "32. Recon_Classification"], "pred": ["Recon_Classification", "32. Recon_Classification"]},
    {"var": "Recon_Timing", "type": "categorical",
     "gold": ["Recon_Timing", "33. Recon_Timing"], "pred": ["Recon_Timing", "33. Recon_Timing"]},

    # Stage outcomes often exist; skip if missing on either side
    {"var": "Stage1_Revision", "type": "binary",
     "gold": ["Stage1_Revision", "39. Stage1_Revision"], "pred": ["Stage1_Revision", "39. Stage1_Revision"]},
    {"var": "Stage2_Applicable", "type": "binary",
     "gold": ["Stage2_Applicable", "Stage2_Applicable "], "pred": ["Stage2_Applicable", "Stage2_Applicable "]},
]


def main():
    print("Loading gold + predictions...")
    gold = normalize_mrn(clean_cols(read_csv_robust(GOLD_PATH)))
    pred = normalize_mrn(clean_cols(read_csv_robust(PRED_PATH)))

    merged = gold.merge(pred, on=MERGE_KEY, how="inner", suffixes=("_gold", "_pred"))
    print(f"Joined rows: {len(merged)} (inner join on MRN)")

    report_rows: List[Dict[str, Any]] = []
    conf_rows: List[Dict[str, Any]] = []
    mism_rows: List[Dict[str, Any]] = []

    for spec in VAR_SPECS:
        v = spec["var"]
        vtype = spec["type"]

        gcol = find_col(gold, spec["gold"])
        pcol = find_col(pred, spec["pred"])

        mg = resolve_merged_column(merged, gcol, "_gold") if gcol else None
        mp = resolve_merged_column(merged, pcol, "_pred") if pcol else None

        if (mg is None) or (mp is None):
            report_rows.append({
                "variable": v, "type": vtype, "status": "SKIP_missing_column",
                "gold_col": gcol or "", "pred_col": pcol or "",
                "merged_gold_col": mg or "", "merged_pred_col": mp or "",
                "n_gold_nonmissing": "", "n_compared": "", "coverage_pred_given_gold": "",
                "mae": "", "pct_within_2.0": "", "pct_within_1.0": "",
                "accuracy": "", "precision": "", "recall": "", "f1": ""
            })
            continue

        if vtype == "numeric":
            g = to_num_series(merged[mg])
            p = to_num_series(merged[mp])

            n_gold = int((~g.isna()).sum())
            mask = (~g.isna()) & (~p.isna())
            n_cmp = int(mask.sum())
            coverage = safe_div(n_cmp, n_gold)

            mae = float((g[mask] - p[mask]).abs().mean()) if n_cmp else 0.0
            within_2 = float(((g[mask] - p[mask]).abs() <= 2.0).mean()) if n_cmp else 0.0
            within_1 = float(((g[mask] - p[mask]).abs() <= 1.0).mean()) if n_cmp else 0.0

            report_rows.append({
                "variable": v, "type": "numeric", "status": "OK",
                "gold_col": gcol, "pred_col": pcol,
                "merged_gold_col": mg, "merged_pred_col": mp,
                "n_gold_nonmissing": n_gold, "n_compared": n_cmp,
                "coverage_pred_given_gold": round(coverage, 4),
                "mae": round(mae, 6),
                "pct_within_2.0": round(within_2, 4),
                "pct_within_1.0": round(within_1, 4),
                "accuracy": "", "precision": "", "recall": "", "f1": ""
            })

            if n_cmp:
                err = (g - p).abs()
                big = mask & (err >= 2.0)
                for _, r in merged[big].head(300).iterrows():
                    mism_rows.append({
                        "variable": v,
                        "MRN": r[MERGE_KEY],
                        "gold_value": r.get(mg, ""),
                        "pred_value": r.get(mp, ""),
                        "note": "abs_error>=2"
                    })

        elif vtype == "categorical":
            g = to_cat_series(merged[mg])
            p = to_cat_series(merged[mp])

            n_gold = int((~g.isna()).sum())
            mask = (~g.isna()) & (~p.isna())
            n_cmp = int(mask.sum())
            coverage = safe_div(n_cmp, n_gold)

            acc = float((g[mask] == p[mask]).mean()) if n_cmp else 0.0

            report_rows.append({
                "variable": v, "type": "categorical", "status": "OK",
                "gold_col": gcol, "pred_col": pcol,
                "merged_gold_col": mg, "merged_pred_col": mp,
                "n_gold_nonmissing": n_gold, "n_compared": n_cmp,
                "coverage_pred_given_gold": round(coverage, 4),
                "mae": "", "pct_within_2.0": "", "pct_within_1.0": "",
                "accuracy": round(acc, 4), "precision": "", "recall": "", "f1": ""
            })

            bad = mask & (g != p)
            for _, r in merged[bad].head(500).iterrows():
                mism_rows.append({
                    "variable": v,
                    "MRN": r[MERGE_KEY],
                    "gold_value": r.get(mg, ""),
                    "pred_value": r.get(mp, ""),
                    "note": "categorical_mismatch"
                })

        elif vtype == "binary":
            g = to_binary_series(merged[mg])
            p = to_binary_series(merged[mp])

            n_gold = int((~g.isna()).sum())
            tp, fp, fn, tn, n_cmp = confusion_counts(g, p)
            coverage = safe_div(n_cmp, n_gold)
            met = metrics_from_counts(tp, fp, fn, tn)

            report_rows.append({
                "variable": v, "type": "binary", "status": "OK",
                "gold_col": gcol, "pred_col": pcol,
                "merged_gold_col": mg, "merged_pred_col": mp,
                "n_gold_nonmissing": n_gold, "n_compared": n_cmp,
                "coverage_pred_given_gold": round(coverage, 4),
                "mae": "", "pct_within_2.0": "", "pct_within_1.0": "",
                "accuracy": round(met["accuracy"], 4),
                "precision": round(met["precision"], 4),
                "recall": round(met["recall"], 4),
                "f1": round(met["f1"], 4),
            })

            conf_rows.append({
                "variable": v, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_compared": n_cmp
            })

            mask = (~g.isna()) & (~p.isna())
            bad = mask & (g != p)
            for _, r in merged[bad].head(600).iterrows():
                mism_rows.append({
                    "variable": v,
                    "MRN": r[MERGE_KEY],
                    "gold_value": r.get(mg, ""),
                    "pred_value": r.get(mp, ""),
                    "note": "binary_mismatch"
                })

        else:
            report_rows.append({
                "variable": v, "type": vtype, "status": "SKIP_unknown_type",
                "gold_col": gcol or "", "pred_col": pcol or "",
                "merged_gold_col": mg or "", "merged_pred_col": mp or "",
                "n_gold_nonmissing": "", "n_compared": "", "coverage_pred_given_gold": "",
                "mae": "", "pct_within_2.0": "", "pct_within_1.0": "",
                "accuracy": "", "precision": "", "recall": "", "f1": ""
            })

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    pd.DataFrame(report_rows).to_csv(OUT_REPORT, index=False)
    pd.DataFrame(conf_rows).to_csv(OUT_CONF, index=False)
    pd.DataFrame(mism_rows).to_csv(OUT_MISM, index=False)

    print("\nDONE.")
    print(f"- Report:     {OUT_REPORT}")
    print(f"- Confusions: {OUT_CONF}")
    print(f"- Mismatches: {OUT_MISM}")
    print("\nRun:")
    print("  python validate_rule_vs_gold_FINAL.py")


if __name__ == "__main__":
    main()
