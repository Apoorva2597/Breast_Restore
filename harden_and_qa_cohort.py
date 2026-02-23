#!/usr/bin/env python3
# harden_and_qa_cohort.py
# Python 3.6.8 compatible
#
# Purpose:
#   Make the cohort "presentation-grade":
#     - no blanks for key fields (use explicit UNK categories)
#     - consistent categorical values
#     - binary fields represented as 1/0/UNK (no silent false negatives)
#     - numeric sanity checks
#     - QA flags for contradictions/outliers
#
# Inputs:
#   /home/apokol/Breast_Restore/cohort_all_patient_level_final_gold_order.csv
#
# Outputs:
#   cohort_all_patient_level_final_gold_order.HARDENED.csv
#   qa_perfection_flags.csv
#   qa_perfection_summary.txt

from __future__ import print_function
import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_COHORT = os.path.join(SCRIPT_DIR, "cohort_all_patient_level_final_gold_order.csv")

OUT_COHORT = os.path.join(SCRIPT_DIR, "cohort_all_patient_level_final_gold_order.HARDENED.csv")
OUT_FLAGS  = os.path.join(SCRIPT_DIR, "qa_perfection_flags.csv")
OUT_SUMMARY = os.path.join(SCRIPT_DIR, "qa_perfection_summary.txt")

UNK = "UNK"

# ----------------------------
# Helpers
# ----------------------------
def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", dtype=object)
    finally:
        try:
            f.close()
        except Exception:
            pass

def is_blank(x):
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    t = str(x).strip()
    if t == "":
        return True
    if t.lower() in ("nan", "none", "null", "na", "n/a", ".", "-", "--"):
        return True
    return False

def norm_str(x):
    if is_blank(x):
        return ""
    return str(x).strip()

def to_float(x):
    if is_blank(x):
        return None
    t = str(x).strip()
    # strip trailing .0 common in CSV numeric artifacts
    if re.match(r"^[0-9]+\.0$", t):
        t = t[:-2]
    try:
        return float(t)
    except Exception:
        return None

def normalize_mrn(x):
    t = norm_str(x)
    if t.endswith(".0"):
        t = t[:-2]
    return t

def binarize_to_1_0_unk(x):
    """
    Map common tokens to 1/0/UNK (as strings so we don't lose UNK).
    """
    if is_blank(x):
        return UNK
    t = str(x).strip().lower()
    if t in ("1", "true", "t", "yes", "y", "positive", "pos", "present", "performed"):
        return "1"
    if t in ("0", "false", "f", "no", "n", "negative", "neg", "absent", "denied"):
        return "0"
    # If it's numeric-ish:
    if re.match(r"^[0-9]+(\.[0-9]+)?$", t):
        try:
            return "1" if float(t) >= 1 else "0"
        except Exception:
            return UNK
    return UNK

def pct(n, d):
    if d == 0:
        return "NA"
    return "{:.1f}%".format(100.0 * float(n) / float(d))

# ----------------------------
# Config: which columns we harden
# ----------------------------
KEY_ID_COLS = ["patient_id", "MRN"]

# If your cohort uses these names (it does), we’ll harden them:
KEY_DEMO_COLS = [
    "Race",
    "Ethnicity",
    "Age_DOS",
    "BMI",
    "SmokingStatus",
]

KEY_RECON_COLS = [
    "Mastectomy_Performed",
    "Mastectomy_Type",
    "Mastectomy_Laterality",
    "Recon_Type_op_enc",
    "Recon_has_expander_op_enc",
    "Recon_has_implant_op_enc",
    "Recon_has_flap_op_enc",
]

# Add comorbs you care about (based on what you showed):
KEY_COMORB_COLS = [
    "CardiacDisease",
    "DiabetesMellitus",
    "Hypertension",
    "CancerHistoryOther",
    "SteroidUse",
]

# Stage1/Stage2 outcomes in your cohort:
KEY_STAGE1_COLS = [
    "Stage1_MinorComp_pred",
    "Stage1_MajorComp_pred",
    "Stage1_Reoperation_pred",
    "Stage1_Rehospitalization_pred",
]

KEY_STAGE2_COLS = [
    "stage2_confirmed_flag",
    "stage2_date_final",
    "Stage2_MinorComp",
    "Stage2_MajorComp",
    "Stage2_Reoperation",
    "Stage2_Rehospitalization",
    "Stage2_Failure",
    "Stage2_Revision",
]

# Categorical normalization maps (conservative; doesn't invent categories)
RACE_MAP = {
    "white or caucasian": "White",
    "white": "White",
    "black or african american": "Black",
    "black": "Black",
    "asian": "Asian",
    "chinese": "Asian",
    "other": "Other",
    "choose not to disclose": "Declined",
    "unknown": UNK
}

ETH_MAP = {
    "non-hispanic": "Non-Hispanic",
    "hispanic": "Hispanic",
    "choose not to disclose": "Declined",
    "unknown": UNK
}

SMOKE_MAP = {
    "never": "never",
    "former": "former",
    "current": "current",
    "unknown": UNK
}

MAST_TYPE_MAP = {
    "simple": "simple",
    "skin-sparing": "skin-sparing",
    "nipple-sparing": "nipple-sparing",
    "modified radical": "modified radical",
    "radical": "radical",
}

LAT_MAP = {
    "left": "left",
    "right": "right",
    "bilateral": "bilateral",
}

def norm_cat(x, mapping):
    if is_blank(x):
        return UNK
    t = str(x).strip().lower()
    # exact map
    if t in mapping:
        return mapping[t]
    # partial cleanup
    t2 = re.sub(r"\s+", " ", t)
    if t2 in mapping:
        return mapping[t2]
    # keep original (but trimmed) if not crazy
    return str(x).strip()

# ----------------------------
# Main
# ----------------------------
def main():
    if not os.path.exists(IN_COHORT):
        raise RuntimeError("Missing input: {}".format(IN_COHORT))

    df = read_csv_safe(IN_COHORT)
    n = len(df)

    # Ensure patient_id exists
    if "patient_id" not in df.columns:
        raise RuntimeError("Expected column 'patient_id' not found in cohort.")

    # Standardize empty strings to NaN-like for detection (but we will fill later)
    # We’ll operate column-wise to avoid blowing up memory.
    flags = []

    # ---- Harden demographics ----
    if "Race" in df.columns:
        df["Race"] = df["Race"].map(lambda x: norm_cat(x, RACE_MAP))
    if "Ethnicity" in df.columns:
        df["Ethnicity"] = df["Ethnicity"].map(lambda x: norm_cat(x, ETH_MAP))
    if "SmokingStatus" in df.columns:
        df["SmokingStatus"] = df["SmokingStatus"].map(lambda x: norm_cat(x, SMOKE_MAP))

    # Age_DOS numeric sanity (do NOT delete, just flag)
    if "Age_DOS" in df.columns:
        age_num = df["Age_DOS"].map(to_float)
        # Fill missing with UNK (string), but keep numeric checks from age_num
        df["Age_DOS"] = df["Age_DOS"].map(lambda x: UNK if is_blank(x) else str(x).strip())
        bad_age = age_num.map(lambda v: (v is not None) and (v < 0 or v > 120))
        if bad_age.any():
            for idx in df.index[bad_age]:
                flags.append({
                    "patient_id": df.at[idx, "patient_id"],
                    "issue": "Age_DOS_out_of_range",
                    "detail": "Age_DOS={}".format(df.at[idx, "Age_DOS"])
                })

    # BMI numeric sanity
    if "BMI" in df.columns:
        bmi_num = df["BMI"].map(to_float)
        df["BMI"] = df["BMI"].map(lambda x: UNK if is_blank(x) else str(x).strip())
        bad_bmi = bmi_num.map(lambda v: (v is not None) and (v < 10 or v > 80))
        if bad_bmi.any():
            for idx in df.index[bad_bmi]:
                flags.append({
                    "patient_id": df.at[idx, "patient_id"],
                    "issue": "BMI_out_of_range",
                    "detail": "BMI={}".format(df.at[idx, "BMI"])
                })

    # ---- Harden recon fields ----
    if "Mastectomy_Performed" in df.columns:
        df["Mastectomy_Performed"] = df["Mastectomy_Performed"].map(binarize_to_1_0_unk)

    if "Mastectomy_Type" in df.columns:
        df["Mastectomy_Type"] = df["Mastectomy_Type"].map(lambda x: norm_cat(x, MAST_TYPE_MAP))

    if "Mastectomy_Laterality" in df.columns:
        df["Mastectomy_Laterality"] = df["Mastectomy_Laterality"].map(lambda x: norm_cat(x, LAT_MAP))

    # Recon flags to 1/0/UNK (these were already 0/1 but keep consistent)
    for c in ["Recon_has_expander_op_enc", "Recon_has_implant_op_enc", "Recon_has_flap_op_enc"]:
        if c in df.columns:
            df[c] = df[c].map(binarize_to_1_0_unk)

    # ---- Harden comorbs (1/0/UNK) ----
    for c in KEY_COMORB_COLS:
        if c in df.columns:
            df[c] = df[c].map(binarize_to_1_0_unk)

    # ---- Harden Stage1 (1/0/UNK) ----
    for c in KEY_STAGE1_COLS:
        if c in df.columns:
            df[c] = df[c].map(binarize_to_1_0_unk)

    # ---- Harden Stage2 ----
    if "stage2_confirmed_flag" in df.columns:
        df["stage2_confirmed_flag"] = df["stage2_confirmed_flag"].map(binarize_to_1_0_unk)

    if "stage2_date_final" in df.columns:
        # Date stays as-is but blanks become UNK so not empty
        df["stage2_date_final"] = df["stage2_date_final"].map(lambda x: UNK if is_blank(x) else str(x).strip())

    for c in ["Stage2_MinorComp", "Stage2_MajorComp", "Stage2_Reoperation", "Stage2_Rehospitalization", "Stage2_Failure", "Stage2_Revision"]:
        if c in df.columns:
            df[c] = df[c].map(binarize_to_1_0_unk)

    # ----------------------------
    # Contradiction checks (flag only)
    # ----------------------------
    # Mastectomy performed but type missing/UNK
    if "Mastectomy_Performed" in df.columns and "Mastectomy_Type" in df.columns:
        mask = (df["Mastectomy_Performed"] == "1") & (df["Mastectomy_Type"].map(lambda x: x == UNK))
        for idx in df.index[mask]:
            flags.append({
                "patient_id": df.at[idx, "patient_id"],
                "issue": "Mastectomy_Performed_but_Type_UNK",
                "detail": ""
            })

    # Stage2 confirmed but date UNK
    if "stage2_confirmed_flag" in df.columns and "stage2_date_final" in df.columns:
        mask = (df["stage2_confirmed_flag"] == "1") & (df["stage2_date_final"] == UNK)
        for idx in df.index[mask]:
            flags.append({
                "patient_id": df.at[idx, "patient_id"],
                "issue": "Stage2_confirmed_but_date_UNK",
                "detail": ""
            })

    # ----------------------------
    # Write outputs
    # ----------------------------
    df.to_csv(OUT_COHORT, index=False, encoding="utf-8")

    flags_df = pd.DataFrame(flags, columns=["patient_id", "issue", "detail"])
    flags_df.to_csv(OUT_FLAGS, index=False, encoding="utf-8")

    # Summary text
    with open(OUT_SUMMARY, "w") as f:
        f.write("QA + Hardening Summary\n")
        f.write("======================\n\n")
        f.write("Input: {}\n".format(IN_COHORT))
        f.write("Output hardened: {}\n".format(OUT_COHORT))
        f.write("Output flags: {}\n\n".format(OUT_FLAGS))

        f.write("Rows: {}\n".format(n))
        f.write("Unique patient_id: {}\n\n".format(df["patient_id"].nunique()))

        def report_nonblank(col):
            if col not in df.columns:
                return
            nonblank = int((df[col].map(lambda x: not is_blank(x))).sum())
            f.write("{:<28} nonblank {}/{} ({})\n".format(col + ":", nonblank, n, pct(nonblank, n)))

        f.write("Key field completeness (after hardening):\n")
        for col in (KEY_DEMO_COLS + KEY_RECON_COLS + KEY_COMORB_COLS + KEY_STAGE1_COLS + KEY_STAGE2_COLS):
            report_nonblank(col)

        f.write("\nFlag counts:\n")
        if len(flags_df) == 0:
            f.write("  None\n")
        else:
            vc = flags_df["issue"].value_counts()
            for k, v in vc.items():
                f.write("  {:<40} {}\n".format(k, int(v)))

    print("\nWrote:", OUT_COHORT)
    print("Wrote:", OUT_FLAGS)
    print("Wrote:", OUT_SUMMARY)
    print("\nDone.\n")

if __name__ == "__main__":
    main()
