# stage_reconstruction_from_op_encounters.py
# Python 3.6+ (pandas required)
#
# Goal:
#   Use OPERATION_ENCOUNTERS.csv (structured encounters) to derive:
#     - Stage 1 reconstruction date (index recon)
#     - Stage 2 reconstruction date (if expander->implant exchange / planned stage 2)
#     - Reconstruction pathway classification (2-stage vs single-stage implant vs autologous)
#
# Notes:
# - We stage using *procedure strings + dates*, NOT note NLP.
# - We treat "pc-revision of reconstructed breast" as *potential Stage 2*,
#   BUT only label it Stage 2 if there's evidence of expander-based pathway
#   (e.g., tissue expander placement).
#
# Outputs:
#   1) patient_recon_staging.csv  (patient-level staging)
#   2) qa_recon_staging_summary.txt (quick QA counts)

import re
import sys
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
OP_ENCOUNTERS_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"
OUT_PATIENT_CSV = "patient_recon_staging.csv"
OUT_QA_TXT = "qa_recon_staging_summary.txt"

# Column names seen in your headers (adjust here if needed)
COL_PATIENT = "ENCRYPTED_PAT_ID"     # patient identifier
COL_MRN = "MRN"
COL_OP_DATE = "OPERATION_DATE"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"

# If OPERATION_DATE is sometimes blank, you can fall back to DISCHARGE_DATE_DT
COL_ALT_DATE = "DISCHARGE_DATE_DT"

# -------------------------
# Regex patterns on PROCEDURE text (normalized)
# -------------------------
# Helper: use high-recall phrase matches, but keep the logic conservative.
PAT = {
    # oncologic surgery (not recon staging)
    "MASTECTOMY": re.compile(r"\bmastectomy\b", re.I),

    # stage 1 recon signals
    "EXPANDER_PLACEMENT": re.compile(r"\btissue\s*expand", re.I),  # "tissue expander placement..."
    "IMPLANT_IMMEDIATE": re.compile(r"\bimplant\b.*\bsame\s+day\b.*\bmastectomy\b|\bsame\s+day\b.*\bimplant\b", re.I),
    "IMPLANT_DELAYED": re.compile(r"\bimplant\b.*\bsep(arate)?\s+day\b|\bsep\s+day\b.*\bimplant\b", re.I),

    # autologous (single-stage recon)
    "AUTOLOGOUS_FREE_FLAP": re.compile(r"\bfree\s+flap\b|\bdiep\b|\btr?am\b|\bsiea\b|\bgap\s+flap\b", re.I),
    "LAT_DORSI": re.compile(r"\blatissimus\s+dorsi\b", re.I),

    # stage 2 / exchange-type language (but only counts as Stage 2 if expander pathway exists)
    "REVISION_RECON_BREAST": re.compile(r"\brevision\b.*\breconstruct(ed|ion)\b.*\bbreast\b|\brevision\s+of\s+reconstructed\s+breast\b", re.I),

    # refinements (not staging)
    "NIPPLE_AREOLA_RECON": re.compile(r"\bnipple\b|\bareola\b", re.I),

    # direct implant replacement (often not planned stage2 unless preceded by expander)
    "IMPLANT_REPLACEMENT": re.compile(r"\breplac(e|ement)\b.*\bimplant\b|\bimplant\b.*\breplac(e|ement)\b", re.I),
}


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_date_series(df, primary_col, fallback_col=None):
    # Parse primary date; if missing, use fallback
    dt = pd.to_datetime(df[primary_col], errors="coerce") if primary_col in df.columns else pd.Series([pd.NaT] * len(df))
    if fallback_col and fallback_col in df.columns:
        fb = pd.to_datetime(df[fallback_col], errors="coerce")
        dt = dt.fillna(fb)
    return dt


def classify_row(proc_text):
    """
    Returns a set of tags for this encounter row based on procedure text.
    """
    tags = set()
    t = proc_text

    if PAT["MASTECTOMY"].search(t):
        tags.add("MASTECTOMY")

    if PAT["EXPANDER_PLACEMENT"].search(t):
        tags.add("STAGE1_EXPANDER")

    if PAT["IMPLANT_IMMEDIATE"].search(t):
        tags.add("STAGE1_IMPLANT_IMMEDIATE")

    if PAT["IMPLANT_DELAYED"].search(t):
        tags.add("STAGE1_IMPLANT_DELAYED")

    if PAT["AUTOLOGOUS_FREE_FLAP"].search(t) or PAT["LAT_DORSI"].search(t):
        tags.add("STAGE1_AUTOLOGOUS")

    if PAT["REVISION_RECON_BREAST"].search(t):
        tags.add("POSSIBLE_STAGE2_REVISION")

    if PAT["NIPPLE_AREOLA_RECON"].search(t):
        tags.add("REFINEMENT_NIPPLE_AREOLA")

    if PAT["IMPLANT_REPLACEMENT"].search(t):
        tags.add("IMPLANT_REPLACEMENT")

    return tags


def derive_patient_staging(sub):
    """
    sub: patient-level dataframe of encounters (already has parsed date + proc_norm)
    Returns a dict with staging outputs.
    """
    # Keep only rows with a usable date
    s = sub.dropna(subset=["op_date"]).copy()
    s = s.sort_values("op_date")

    # Gather tags per row
    s["tags"] = s["proc_norm"].apply(classify_row)

    # Determine if any expander placement exists (key for 2-stage)
    has_expander = s["tags"].apply(lambda z: "STAGE1_EXPANDER" in z).any()

    # Stage 1 candidates:
    # - expander placement
    # - immediate implant
    # - delayed implant
    # - autologous
    stage1_mask = s["tags"].apply(lambda z: any(k in z for k in [
        "STAGE1_EXPANDER", "STAGE1_IMPLANT_IMMEDIATE", "STAGE1_IMPLANT_DELAYED", "STAGE1_AUTOLOGOUS"
    ]))
    stage1_rows = s[stage1_mask].copy()

    stage1_date = None
    stage1_proc = None
    stage1_type = None  # expander / implant / autologous

    if not stage1_rows.empty:
        # Choose earliest reconstruction-like procedure as stage 1
        r1 = stage1_rows.iloc[0]
        stage1_date = r1["op_date"]
        stage1_proc = r1["proc_norm"]
        tags1 = r1["tags"]

        if "STAGE1_AUTOLOGOUS" in tags1:
            stage1_type = "autologous"
        elif "STAGE1_EXPANDER" in tags1:
            stage1_type = "expander"
        elif "STAGE1_IMPLANT_IMMEDIATE" in tags1 or "STAGE1_IMPLANT_DELAYED" in tags1:
            stage1_type = "implant"
        else:
            stage1_type = "unknown"

    # Stage 2 candidates:
    # Only call Stage 2 if:
    #   - there is expander placement somewhere (has_expander True)
    #   - and there exists a later "revision of reconstructed breast" after stage1_date
    stage2_date = None
    stage2_proc = None

    if has_expander and stage1_date is not None:
        s2_mask = s["tags"].apply(lambda z: "POSSIBLE_STAGE2_REVISION" in z)
        s2_rows = s[s2_mask & (s["op_date"] > stage1_date)].copy()
        if not s2_rows.empty:
            r2 = s2_rows.iloc[0]  # earliest qualifying stage2
            stage2_date = r2["op_date"]
            stage2_proc = r2["proc_norm"]

    # Pathway classification
    pathway = "unknown"
    if stage1_type == "autologous":
        pathway = "single_stage_autologous"
    elif has_expander:
        # expander anywhere implies 2-stage intent; stage2 may or may not be captured
        pathway = "two_stage_expander_implant"
    elif stage1_type == "implant":
        pathway = "single_stage_implant"

    return {
        "patient_id": str(sub["patient_id"].iloc[0]),
        "mrn": str(sub["mrn"].iloc[0]) if "mrn" in sub.columns else "",
        "has_expander": bool(has_expander),
        "pathway": pathway,
        "stage1_date": stage1_date.strftime("%Y-%m-%d") if pd.notnull(stage1_date) else None,
        "stage1_proc": stage1_proc,
        "stage2_date": stage2_date.strftime("%Y-%m-%d") if pd.notnull(stage2_date) else None,
        "stage2_proc": stage2_proc,
        "n_encounter_rows": int(len(sub)),
        "n_rows_with_dates": int(sub["op_date"].notnull().sum()),
    }


def main():
    df = read_csv_fallback(OP_ENCOUNTERS_CSV)

    # Basic column validation
    for c in [COL_PATIENT, COL_PROC]:
        if c not in df.columns:
            raise RuntimeError("Missing required column in OP encounters file: {}".format(c))

    # Build working columns
    df["patient_id"] = df[COL_PATIENT].fillna("").astype(str)
    df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
    df["proc_norm"] = df[COL_PROC].apply(norm_text)

    # Dates
    df["op_date"] = parse_date_series(df, COL_OP_DATE, fallback_col=COL_ALT_DATE)

    # Drop rows with no patient_id
    df = df[df["patient_id"].str.len() > 0].copy()

    # Patient-level staging
    rows = []
    for pid, sub in df.groupby("patient_id", sort=True):
        rows.append(derive_patient_staging(sub))

    out = pd.DataFrame(rows)

    # Write patient-level output
    out.to_csv(OUT_PATIENT_CSV, index=False)

    # QA summary
    total_patients = int(out.shape[0])
    with_stage1 = int(out["stage1_date"].notnull().sum())
    with_stage2 = int(out["stage2_date"].notnull().sum())
    two_stage = int((out["pathway"] == "two_stage_expander_implant").sum())
    single_implant = int((out["pathway"] == "single_stage_implant").sum())
    single_auto = int((out["pathway"] == "single_stage_autologous").sum())
    unknown = int((out["pathway"] == "unknown").sum())

    lines = []
    lines.append("=== Reconstruction staging QA ===")
    lines.append("Total patients in OP encounters: {}".format(total_patients))
    lines.append("Patients with Stage 1 identified: {} ({:.1f}%)".format(
        with_stage1, (100.0 * with_stage1 / total_patients) if total_patients else 0.0
    ))
    lines.append("Patients with Stage 2 identified: {} ({:.1f}%)".format(
        with_stage2, (100.0 * with_stage2 / total_patients) if total_patients else 0.0
    ))
    lines.append("")
    lines.append("Pathway counts:")
    lines.append("  two_stage_expander_implant: {}".format(two_stage))
    lines.append("  single_stage_implant:       {}".format(single_implant))
    lines.append("  single_stage_autologous:    {}".format(single_auto))
    lines.append("  unknown:                    {}".format(unknown))
    lines.append("")
    lines.append("Wrote: {}".format(OUT_PATIENT_CSV))

    with open(OUT_QA_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("Wrote: {}".format(OUT_QA_TXT))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
