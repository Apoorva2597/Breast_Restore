#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_auroc_logistic.py

Computes AUROC for primary complication outcomes using logistic regression
as a calibration layer on top of rule-based predictions.

METHOD:
    For each primary outcome:
    1. Features: binary rule predictions for related variables +
                 max confidence score from evidence file per patient
    2. Target: gold-standard manual abstraction label
    3. Training: 5-fold stratified cross-validation (out-of-fold probabilities)
    4. AUROC: computed from out-of-fold predicted probabilities
    5. 95% CI: bootstrap (1000 iterations)

This is NOT replacing the rule-based pipeline. The logistic regression is a
calibration layer that converts rule outputs into ranked probability scores,
enabling ROC analysis. All clinical NLP extraction is done by the rules.

INPUTS (all read-only, nothing modified):
    _outputs/validation_merged_patient_level.csv  (paired pred+gold per patient)
    _outputs/complications_patch_evidence.csv      (confidence scores)
    _outputs/rule_hit_evidence_FINAL_NO_GOLD.csv   (confidence scores)

OUTPUTS (new files only):
    _outputs/auroc_results.csv          (AUROC + 95% CI per outcome)
    _outputs/auroc_roc_curve_data.csv   (TPR/FPR points for Figure 2)
    _outputs/auroc_patient_probs.csv    (per-patient predicted probabilities)

Python 3.6.8 + scikit-learn compatible.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

BASE_DIR   = "/home/apokol/Breast_Restore"
MERGED     = "{0}/_outputs/validation_merged_patient_level.csv".format(BASE_DIR)
COMP_EVID  = "{0}/_outputs/complications_patch_evidence.csv".format(BASE_DIR)
RULE_EVID  = "{0}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv".format(BASE_DIR)
GOLD_FILE  = "{0}/gold_cleaned_for_cedar.csv".format(BASE_DIR)

OUT_AUROC  = "{0}/_outputs/auroc_results.csv".format(BASE_DIR)
OUT_ROC    = "{0}/_outputs/auroc_roc_curve_data.csv".format(BASE_DIR)
OUT_PROBS  = "{0}/_outputs/auroc_patient_probs.csv".format(BASE_DIR)

MRN        = "MRN"
N_SPLITS   = 5
N_BOOTSTRAP= 1000
RANDOM_STATE = 42

# ============================================================
# Primary outcomes and their feature sets
# ============================================================
# For each outcome we use:
#   - its own binary rule prediction
#   - closely related binary rule predictions
#   - max confidence score from evidence for that outcome
#   - overall complication signal confidence

OUTCOME_CONFIGS = {
    "AnyComp": {
        "gold_col":  None,  # derived: MinorComp OR MajorComp
        "rule_features": [
            "Stage1_MinorComp_pred",
            "Stage1_MajorComp_pred",
            "Stage1_Reoperation_pred",
            "Stage1_Rehospitalization_pred",
            "Stage1_Failure_pred",
            "Stage1_Revision_pred",
        ],
        "evidence_fields": [
            "Stage1_MinorComp", "Stage1_MajorComp",
            "Stage1_Reoperation", "Stage1_Rehospitalization",
            "Stage1_Failure", "RAW_STAGE1_ComplicationSignal",
        ],
    },
    "MajorComp": {
        "gold_col":  "Stage1_MajorComp_gold",
        "rule_features": [
            "Stage1_MajorComp_pred",
            "Stage1_Reoperation_pred",
            "Stage1_Rehospitalization_pred",
            "Stage1_Failure_pred",
        ],
        "evidence_fields": [
            "Stage1_MajorComp", "Stage1_Reoperation",
            "Stage1_Rehospitalization", "RAW_STAGE1_ComplicationSignal",
        ],
    },
    "Reoperation": {
        "gold_col":  "Stage1_Reoperation_gold",
        "rule_features": [
            "Stage1_Reoperation_pred",
            "Stage1_MajorComp_pred",
            "Stage1_Failure_pred",
        ],
        "evidence_fields": [
            "Stage1_Reoperation", "Stage1_MajorComp",
            "RAW_STAGE1_ComplicationSignal",
        ],
    },
    "Rehospitalization": {
        "gold_col":  "Stage1_Rehospitalization_gold",
        "rule_features": [
            "Stage1_Rehospitalization_pred",
            "Stage1_MajorComp_pred",
            "Stage1_Reoperation_pred",
        ],
        "evidence_fields": [
            "Stage1_Rehospitalization", "Stage1_MajorComp",
            "RAW_STAGE1_ComplicationSignal",
        ],
    },
    "Failure": {
        "gold_col":  "Stage1_Failure_gold",
        "rule_features": [
            "Stage1_Failure_pred",
            "Stage1_MajorComp_pred",
            "Stage1_Reoperation_pred",
        ],
        "evidence_fields": [
            "Stage1_Failure", "Stage1_MajorComp",
            "RAW_STAGE1_ComplicationSignal",
        ],
    },
}

# ============================================================
# Utilities
# ============================================================

def safe_read(path):
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def to_bin(series):
    def conv(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ["1", "true", "yes", "t", "y"]:
            return 1.0
        if s in ["0", "false", "no", "f", "n"]:
            return 0.0
        return np.nan
    return series.apply(conv)


def to_float(series):
    return pd.to_numeric(series, errors="coerce")


def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=1000, random_state=42):
    rng    = np.random.RandomState(random_state)
    aucs   = []
    n      = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        yt  = y_true[idx]
        yp  = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    lo   = float(np.percentile(aucs, 2.5))
    hi   = float(np.percentile(aucs, 97.5))
    return round(lo, 3), round(hi, 3)


# ============================================================
# Build max-confidence feature per patient per outcome field
# ============================================================

def build_confidence_features(comp_evid, rule_evid, mrns):
    """
    For each MRN and each evidence field, compute the max confidence score
    across all notes. Returns a DataFrame: MRN x field -> max_confidence.
    """
    all_evid = pd.concat([comp_evid, rule_evid], ignore_index=True, sort=False)

    # Keep only rows for MRNs in our gold set
    all_evid[MRN] = all_evid[MRN].astype(str).str.strip()
    all_evid       = all_evid[all_evid[MRN].isin(set(mrns))]

    if "CONFIDENCE" not in all_evid.columns or "FIELD" not in all_evid.columns:
        print("  WARNING: CONFIDENCE or FIELD column missing from evidence.")
        return pd.DataFrame({MRN: mrns})

    all_evid["CONFIDENCE_NUM"] = pd.to_numeric(all_evid["CONFIDENCE"], errors="coerce")
    all_evid = all_evid.dropna(subset=["CONFIDENCE_NUM", "FIELD"])

    # Pivot: max confidence per MRN per field
    pivot = (
        all_evid.groupby([MRN, "FIELD"])["CONFIDENCE_NUM"]
        .max()
        .reset_index()
        .pivot(index=MRN, columns="FIELD", values="CONFIDENCE_NUM")
        .reset_index()
    )
    pivot.columns.name = None
    pivot.columns = [MRN] + ["CONF_" + c for c in pivot.columns if c != MRN]

    # Merge with full MRN list (fill missing with 0)
    base = pd.DataFrame({MRN: mrns})
    result = base.merge(pivot, on=MRN, how="left").fillna(0.0)
    return result


# ============================================================
# Cross-validated probability estimation
# ============================================================

def cross_val_predict_proba(X, y, n_splits=5, random_state=42):
    """
    5-fold stratified CV. Returns out-of-fold predicted probabilities.
    Uses logistic regression with L2 regularization.
    """
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    probs   = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr        = y[train_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)

        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",  # handles class imbalance
            solver="lbfgs"
        )
        clf.fit(X_tr, y_tr)

        pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        probs[val_idx] = clf.predict_proba(X_val)[:, pos_idx]

    return probs


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("compute_auroc_logistic.py")
    print("Logistic regression calibration layer for AUROC")
    print("=" * 60)

    # Load merged patient-level data
    print("\nLoading merged patient-level data...")
    merged = clean_cols(safe_read(MERGED))
    merged[MRN] = merged[MRN].astype(str).str.strip()
    print("  Rows: {0}".format(len(merged)))

    # Derive AnyComp gold and pred
    mc_gold = to_bin(merged["Stage1_MinorComp_gold"])
    mj_gold = to_bin(merged["Stage1_MajorComp_gold"])
    mc_pred = to_bin(merged["Stage1_MinorComp_pred"])
    mj_pred = to_bin(merged["Stage1_MajorComp_pred"])

    merged["AnyComp_gold"] = ((mc_gold == 1) | (mj_gold == 1)).astype(float)
    merged["AnyComp_pred"] = ((mc_pred == 1) | (mj_pred == 1)).astype(float)

    # Only use rows where Stage1 outcomes are labeled
    stage1_mask = mc_gold.notna()
    merged = merged[stage1_mask].copy().reset_index(drop=True)
    print("  Stage1-labeled rows: {0}".format(len(merged)))

    mrns = merged[MRN].tolist()

    # Load evidence files for confidence scores
    print("\nLoading evidence files for confidence scores...")
    comp_evid = clean_cols(safe_read(COMP_EVID))
    rule_evid = clean_cols(safe_read(RULE_EVID))
    comp_evid[MRN] = comp_evid[MRN].astype(str).str.strip()
    rule_evid[MRN] = rule_evid[MRN].astype(str).str.strip()
    print("  Complication evidence rows: {0}".format(len(comp_evid)))
    print("  Rule evidence rows: {0}".format(len(rule_evid)))

    # Build confidence feature matrix
    print("\nBuilding max-confidence features per patient...")
    conf_df = build_confidence_features(comp_evid, rule_evid, mrns)
    conf_df[MRN] = conf_df[MRN].astype(str).str.strip()

    # Merge confidence features into merged df
    merged = merged.merge(conf_df, on=MRN, how="left")
    conf_cols = [c for c in merged.columns if c.startswith("CONF_")]
    merged[conf_cols] = merged[conf_cols].fillna(0.0)
    print("  Confidence feature columns: {0}".format(len(conf_cols)))

    # --------------------------------------------------------
    # Run per outcome
    # --------------------------------------------------------
    auroc_results  = []
    roc_curve_rows = []
    prob_df        = merged[[MRN]].copy()

    for outcome_name, cfg in OUTCOME_CONFIGS.items():
        print("\n--- Outcome: {0} ---".format(outcome_name))

        # Gold labels
        if outcome_name == "AnyComp":
            gold_col = "AnyComp_gold"
        else:
            gold_col = cfg["gold_col"]

        if gold_col not in merged.columns:
            print("  SKIP: gold column {0} not found".format(gold_col))
            continue

        y = to_bin(merged[gold_col]).values
        valid_mask = ~np.isnan(y)
        y = y[valid_mask].astype(int)

        if len(np.unique(y)) < 2:
            print("  SKIP: only one class in gold labels")
            continue

        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        print("  Gold: n={0}  pos={1}  neg={2}".format(len(y), n_pos, n_neg))

        # Build feature matrix
        feature_cols = []

        # Binary rule predictions
        for col in cfg["rule_features"]:
            if col in merged.columns:
                feature_cols.append(col)
            elif col == "AnyComp_pred" and "AnyComp_pred" in merged.columns:
                feature_cols.append(col)

        # Confidence scores for related fields
        for field in cfg["evidence_fields"]:
            conf_col = "CONF_" + field
            if conf_col in merged.columns:
                feature_cols.append(conf_col)

        feature_cols = list(dict.fromkeys(feature_cols))  # dedupe, preserve order

        if not feature_cols:
            print("  SKIP: no feature columns found")
            continue

        print("  Features ({0}): {1}".format(len(feature_cols), feature_cols))

        # Extract feature matrix for valid rows
        X_df = merged[feature_cols].copy()
        for col in feature_cols:
            X_df[col] = to_bin(X_df[col]) if col in cfg["rule_features"] else to_float(X_df[col])
            X_df[col] = X_df[col].fillna(0.0)

        X = X_df.values[valid_mask].astype(float)

        # Cross-validated probabilities
        print("  Running {0}-fold stratified CV...".format(N_SPLITS))
        try:
            probs = cross_val_predict_proba(X, y, n_splits=N_SPLITS,
                                             random_state=RANDOM_STATE)
        except Exception as e:
            print("  ERROR in CV: {0}".format(repr(e)))
            continue

        # AUROC
        auroc = roc_auc_score(y, probs)
        ci_lo, ci_hi = bootstrap_auroc_ci(y, probs, N_BOOTSTRAP, RANDOM_STATE)

        print("  AUROC = {0:.3f} (95% CI: {1:.3f}-{2:.3f})".format(auroc, ci_lo, ci_hi))

        auroc_results.append({
            "outcome":   outcome_name,
            "n":         int(len(y)),
            "n_pos":     n_pos,
            "n_neg":     n_neg,
            "AUROC":     round(auroc, 3),
            "CI_lo":     ci_lo,
            "CI_hi":     ci_hi,
            "CI_95":     "{0:.3f}-{1:.3f}".format(ci_lo, ci_hi),
            "n_features": len(feature_cols),
        })

        # ROC curve points
        fpr, tpr, thresholds = roc_curve(y, probs)
        for i in range(len(fpr)):
            roc_curve_rows.append({
                "outcome": outcome_name,
                "FPR":     round(float(fpr[i]), 4),
                "TPR":     round(float(tpr[i]), 4),
                "threshold": round(float(thresholds[i]), 4) if i < len(thresholds) else 1.0,
            })

        # Store probabilities
        full_probs = np.zeros(len(merged))
        idx_valid  = np.where(valid_mask)[0]
        for i, p in zip(idx_valid, probs):
            full_probs[i] = p
        prob_df[outcome_name + "_prob"] = full_probs

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_AUROC), exist_ok=True)

    auroc_df = pd.DataFrame(auroc_results)
    auroc_df.to_csv(OUT_AUROC, index=False)

    roc_df = pd.DataFrame(roc_curve_rows)
    roc_df.to_csv(OUT_ROC, index=False)

    prob_df.to_csv(OUT_PROBS, index=False)

    print("\n" + "=" * 60)
    print("DONE.")
    print()
    print("=== AUROC SUMMARY (for Table 3) ===")
    print()
    for r in auroc_results:
        print("{outcome:<22} AUROC={AUROC:.3f} (95% CI {CI_95})  n={n}  pos={n_pos}".format(**r))

    print()
    print("Outputs:")
    print("  AUROC results:    {0}".format(OUT_AUROC))
    print("  ROC curve data:   {0}".format(OUT_ROC))
    print("  Patient probs:    {0}".format(OUT_PROBS))
    print("=" * 60)


if __name__ == "__main__":
    main()
