#!/usr/bin/env python3
# qa_smoking_mismatches.py
#
# Purpose:
#   QA script for SmokingStatus mismatches only.
#
# What it does:
#   1. Loads gold + predicted/master file
#   2. Compares SmokingStatus
#   3. Pulls smoking evidence rows for mismatched MRNs
#   4. Summarizes which stage/status produced the wrong calls
#
# Outputs:
#   _outputs/qa_smoking_mismatch_summary.csv
#   _outputs/qa_smoking_mismatch_stage_summary.csv
#   _outputs/qa_smoking_mismatch_status_summary.csv
#   _outputs/qa_smoking_mismatch_details.csv
#
# Python 3.6.8 compatible

import os
import pandas as pd


# --------------------------------------------------
# EDIT THESE PATHS
# --------------------------------------------------
BASE_DIR = "/home/apokol/Breast_Restore"

# gold file with true SmokingStatus
GOLD_FILE = os.path.join(BASE_DIR, "_outputs", "master_abstraction_rule_FINAL.csv")

# prediction/master file you just updated
PRED_FILE = os.path.join(BASE_DIR, "_outputs", "master_abstraction_rule_FINAL_NO_GOLD.csv")

# evidence file from updater
EVID_FILE = os.path.join(BASE_DIR, "_outputs", "bmi_smoking_only_evidence.csv")

OUT_DIR = os.path.join(BASE_DIR, "_outputs")
MERGE_KEY = "MRN"
FIELD_NAME = "SmokingStatus"


# --------------------------------------------------
# helpers
# --------------------------------------------------
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
    if s.lower() in {"", "nan", "none", "null", "na", "<na>"}:
        return ""
    return s


def normalize_mrn(df):
    key_variants = ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]
    for k in key_variants:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found. Seen columns: {0}".format(list(df.columns)[:50]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def norm_smoking(x):
    s = clean_cell(x).lower()
    if not s:
        return ""

    mapping = {
        "current": "Current",
        "former": "Former",
        "never": "Never",
        "current smoker": "Current",
        "former smoker": "Former",
        "never smoker": "Never",
    }
    return mapping.get(s, clean_cell(x))


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading files...")
    gold = normalize_mrn(clean_cols(read_csv_robust(GOLD_FILE)))
    pred = normalize_mrn(clean_cols(read_csv_robust(PRED_FILE)))
    evid = normalize_mrn(clean_cols(read_csv_robust(EVID_FILE)))

    if FIELD_NAME not in gold.columns:
        raise RuntimeError("Gold file missing SmokingStatus column.")
    if FIELD_NAME not in pred.columns:
        raise RuntimeError("Pred file missing SmokingStatus column.")

    gold_sub = gold[[MERGE_KEY, FIELD_NAME]].copy()
    gold_sub = gold_sub.rename(columns={FIELD_NAME: "SmokingStatus_gold"})
    gold_sub["SmokingStatus_gold"] = gold_sub["SmokingStatus_gold"].apply(norm_smoking)

    pred_sub = pred[[MERGE_KEY, FIELD_NAME]].copy()
    pred_sub = pred_sub.rename(columns={FIELD_NAME: "SmokingStatus_pred"})
    pred_sub["SmokingStatus_pred"] = pred_sub["SmokingStatus_pred"].apply(norm_smoking)

    merged = gold_sub.merge(pred_sub, on=MERGE_KEY, how="outer")

    merged["gold_present"] = merged["SmokingStatus_gold"].apply(lambda x: 1 if clean_cell(x) else 0)
    merged["pred_present"] = merged["SmokingStatus_pred"].apply(lambda x: 1 if clean_cell(x) else 0)

    compare_df = merged[(merged["gold_present"] == 1) | (merged["pred_present"] == 1)].copy()
    compare_df["match"] = (
        compare_df["SmokingStatus_gold"].fillna("").astype(str).str.strip()
        == compare_df["SmokingStatus_pred"].fillna("").astype(str).str.strip()
    ).astype(int)

    mismatches = compare_df[compare_df["match"] == 0].copy()

    print("Total compared:", len(compare_df))
    print("Smoking mismatches:", len(mismatches))

    # keep only smoking evidence
    evid["FIELD"] = evid["FIELD"].apply(clean_cell)
    smoke_evid = evid[evid["FIELD"] == FIELD_NAME].copy()

    # only evidence for mismatched MRNs
    mismatch_mrns = set(mismatches[MERGE_KEY].astype(str).str.strip().tolist())
    smoke_evid = smoke_evid[smoke_evid[MERGE_KEY].astype(str).str.strip().isin(mismatch_mrns)].copy()

    # summarize stages
    if len(smoke_evid) > 0:
        stage_summary = (
            smoke_evid.groupby(["STAGE_USED", "VALUE"])
            .size()
            .reset_index(name="n_rows")
            .sort_values(["n_rows", "STAGE_USED", "VALUE"], ascending=[False, True, True])
        )
        status_summary = (
            smoke_evid.groupby(["STATUS", "VALUE"])
            .size()
            .reset_index(name="n_rows")
            .sort_values(["n_rows", "STATUS", "VALUE"], ascending=[False, True, True])
        )
    else:
        stage_summary = pd.DataFrame(columns=["STAGE_USED", "VALUE", "n_rows"])
        status_summary = pd.DataFrame(columns=["STATUS", "VALUE", "n_rows"])

    # build detailed QA table
    detail = mismatches.merge(
        smoke_evid,
        on=MERGE_KEY,
        how="left"
    )

    detail_cols = [
        MERGE_KEY,
        "SmokingStatus_gold",
        "SmokingStatus_pred",
        "NOTE_DATE",
        "NOTE_TYPE",
        "VALUE",
        "STATUS",
        "STAGE_USED",
        "WINDOW_USED",
        "SECTION",
        "CONFIDENCE",
        "ANCHOR_TYPE",
        "ANCHOR_DATE",
        "EVIDENCE",
    ]
    detail = detail[[c for c in detail_cols if c in detail.columns]].copy()
    detail = detail.sort_values(
        by=[MERGE_KEY, "NOTE_DATE", "STAGE_USED", "STATUS"],
        ascending=[True, True, True, True]
    )

    # one-row summary per mismatch MRN
    mismatch_summary = mismatches[[MERGE_KEY, "SmokingStatus_gold", "SmokingStatus_pred"]].copy()
    mismatch_summary["has_smoking_evidence_rows"] = mismatch_summary[MERGE_KEY].isin(
        set(smoke_evid[MERGE_KEY].astype(str).str.strip().tolist())
    ).astype(int)

    # best guess: final deciding evidence row = patient override if present, else fallback, else latest evidence row
    final_guess_rows = []
    if len(smoke_evid) > 0:
        priority_map = {
            "patient_level_structured_override": 0,
            "fallback_full_note": 1,
            "historical_preop": 2,
            "pm14": 3,
            "pm7": 4,
            "day0": 5,
        }

        smoke_evid2 = smoke_evid.copy()
        smoke_evid2["_stage_pri"] = smoke_evid2["STAGE_USED"].apply(lambda x: priority_map.get(clean_cell(x), 99))
        smoke_evid2["_conf"] = pd.to_numeric(smoke_evid2["CONFIDENCE"], errors="coerce").fillna(0.0)
        smoke_evid2 = smoke_evid2.sort_values(
            by=[MERGE_KEY, "_stage_pri", "_conf", "NOTE_DATE"],
            ascending=[True, True, False, False]
        )
        final_guess_rows = smoke_evid2.groupby(MERGE_KEY, as_index=False).first()

        final_guess_rows = final_guess_rows[[
            MERGE_KEY, "VALUE", "STATUS", "STAGE_USED", "NOTE_DATE", "NOTE_TYPE", "CONFIDENCE", "EVIDENCE"
        ]].copy()
        final_guess_rows = final_guess_rows.rename(columns={
            "VALUE": "qa_best_evidence_value",
            "STATUS": "qa_best_evidence_status",
            "STAGE_USED": "qa_best_evidence_stage",
            "NOTE_DATE": "qa_best_evidence_note_date",
            "NOTE_TYPE": "qa_best_evidence_note_type",
            "CONFIDENCE": "qa_best_evidence_confidence",
            "EVIDENCE": "qa_best_evidence_text",
        })
    else:
        final_guess_rows = pd.DataFrame(columns=[
            MERGE_KEY,
            "qa_best_evidence_value",
            "qa_best_evidence_status",
            "qa_best_evidence_stage",
            "qa_best_evidence_note_date",
            "qa_best_evidence_note_type",
            "qa_best_evidence_confidence",
            "qa_best_evidence_text",
        ])

    mismatch_summary = mismatch_summary.merge(final_guess_rows, on=MERGE_KEY, how="left")

    # save
    mismatch_summary_path = os.path.join(OUT_DIR, "qa_smoking_mismatch_summary.csv")
    stage_summary_path = os.path.join(OUT_DIR, "qa_smoking_mismatch_stage_summary.csv")
    status_summary_path = os.path.join(OUT_DIR, "qa_smoking_mismatch_status_summary.csv")
    detail_path = os.path.join(OUT_DIR, "qa_smoking_mismatch_details.csv")

    mismatch_summary.to_csv(mismatch_summary_path, index=False)
    stage_summary.to_csv(stage_summary_path, index=False)
    status_summary.to_csv(status_summary_path, index=False)
    detail.to_csv(detail_path, index=False)

    print("\nSaved:")
    print(" ", mismatch_summary_path)
    print(" ", stage_summary_path)
    print(" ", status_summary_path)
    print(" ", detail_path)

    print("\nTop mismatch stage summary:")
    if len(stage_summary) > 0:
        print(stage_summary.head(20).to_string(index=False))
    else:
        print("No smoking evidence rows found for mismatches.")

    print("\nTop mismatch status summary:")
    if len(status_summary) > 0:
        print(status_summary.head(20).to_string(index=False))
    else:
        print("No smoking evidence rows found for mismatches.")

    print("\nDone.")


if __name__ == "__main__":
    main()
