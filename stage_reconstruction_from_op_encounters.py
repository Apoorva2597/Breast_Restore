# stage_reconstruction_from_op_encounters.py
# Python 3.6+ (pandas required)
#
# Goal:
#   Use OPERATION_ENCOUNTERS.csv (structured encounters) to derive:
#     - Stage 1 reconstruction date (index recon)
#     - Stage 2 reconstruction date (if TE->implant exchange / delayed implant)
#     - Reconstruction pathway classification (2-stage vs single-stage implant vs autologous)
#
# Notes:
# - We stage using *procedure strings + dates*, NOT note NLP.
# - IMPORTANT FIX:
#     "tissue expandr placmnt in breast reconst inc subseq expansions"
#   is NOT Stage 1 placement; it indicates expander follow-up/expansion management.
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

# Column names
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_OP_DATE = "OPERATION_DATE"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"
COL_ALT_DATE = "DISCHARGE_DATE_DT"


# -------------------------
# Regex patterns on PROCEDURE text (normalized)
# -------------------------
PAT = {
    # oncologic
    "MASTECTOMY": re.compile(r"\bmastectomy\b", re.I),

    # ---- Expander signals ----
    # Follow-up expansion management / subseq expansions (NOT Stage 1)
    "EXPANDER_SUBSEQ": re.compile(
        r"\bsubseq(uent|)\b.*\bexpansions?\b|\binc\b.*\bsubseq\b.*\bexpansions?\b|\bsubse? q\b.*\bexpans",
        re.I
    ),

    # True Stage 1 TE placement: require placement/insert language
    # (we keep it high-recall but avoid catching "subseq expansions")
    "EXPANDER_PLACEMENT": re.compile(
        r"\b(tissue\s*expand|tiss\s*expand|tissue\s*expander|expandr)\b.*\b(plac(e|m)(ent|t)|insert|insertion|placement)\b|"
        r"\b(plac(e|m)(ent|t)|insert|insertion|placement)\b.*\b(tissue\s*expand|tiss\s*expand|tissue\s*expander|expandr)\b",
        re.I
    ),

    # ---- Implant signals ----
    "IMPLANT_IMMEDIATE": re.compile(
        r"\binsertion\b.*\bimplant\b.*\bsame\s+day\b.*\bmastectomy\b|"
        r"\bimplant\b.*\bsame\s+day\b.*\bmastectomy\b|"
        r"\bimmediate\b.*\bimplant\b",
        re.I
    ),

    # Stage 2 / delayed implant / exchange:
    "IMPLANT_SEP_DAY": re.compile(
        r"\bimplant\b.*\bsep(arate)?\s+day\b|"
        r"\bsep\s+day\b.*\bimplant\b|"
        r"\bon\s+sep(arate)?\s+day\b.*\bimplant\b|"
        r"\bdelayed\b.*\bimplant\b",
        re.I
    ),

    # Exchange language: expander-to-implant exchange, implant exchange, etc.
    "EXCHANGE_OR_REPLACEMENT": re.compile(
        r"\bexchange\b.*\b(implant|expander)\b|"
        r"\b(implant|expander)\b.*\bexchange\b|"
        r"\breplac(e|ement)\b.*\bimplant\b|"
        r"\bimplant\b.*\breplac(e|ement)\b|"
        r"\bremove\b.*\bexpander\b.*\bimplant\b|\bremove\b.*\btissue\s*expander\b.*\bimplant\b",
        re.I
    ),

    # autologous
    "AUTOLOGOUS_FREE_FLAP": re.compile(r"\bfree\s+flap\b|\bdiep\b|\btr?am\b|\bsiea\b|\bgap\s+flap\b", re.I),
    "LAT_DORSI": re.compile(r"\blatissimus\s+dorsi\b", re.I),

    # revision/refinement (NOT staging)
    "REVISION_RECON_BREAST": re.compile(
        r"\brevision\b.*\breconstruct(ed|ion)\b.*\bbreast\b|\brevision\s+of\s+reconstructed\s+breast\b",
        re.I
    ),
    "NIPPLE_AREOLA_RECON": re.compile(r"\bnipple\b|\bareola\b", re.I),
}


# CPT hints (optional)
# 11970 = TE->implant exchange (classic Stage 2)
# 19342 = delayed insertion / replacement (can be stage2-ish; keep as hint only, requires expander pathway)
STAGE2_CPT_HINTS = set(["11970", "19342"])


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
    dt = pd.to_datetime(df[primary_col], errors="coerce") if primary_col in df.columns else pd.Series([pd.NaT] * len(df))
    if fallback_col and fallback_col in df.columns:
        fb = pd.to_datetime(df[fallback_col], errors="coerce")
        dt = dt.fillna(fb)
    return dt


def classify_row(proc_text, cpt_code):
    """
    Returns a set of tags for this encounter row based on procedure text and CPT.
    """
    tags = set()
    t = proc_text
    cpt = (str(cpt_code).strip() if cpt_code is not None else "")

    # Expander follow-up (NOT stage1 placement)
    if PAT["EXPANDER_SUBSEQ"].search(t):
        tags.add("EXPANDER_FOLLOWUP")

    # Stage 1: expander placement (only if not subseq)
    if PAT["EXPANDER_PLACEMENT"].search(t) and ("EXPANDER_FOLLOWUP" not in tags):
        tags.add("STAGE1_EXPANDER")

    # Stage 1: immediate implant
    if PAT["IMPLANT_IMMEDIATE"].search(t):
        tags.add("STAGE1_IMPLANT_IMMEDIATE")

    # Stage 1: autologous
    if PAT["AUTOLOGOUS_FREE_FLAP"].search(t) or PAT["LAT_DORSI"].search(t):
        tags.add("STAGE1_AUTOLOGOUS")

    # Stage 2 signals
    if PAT["IMPLANT_SEP_DAY"].search(t):
        tags.add("STAGE2_IMPLANT_SEP_DAY")

    if PAT["EXCHANGE_OR_REPLACEMENT"].search(t):
        tags.add("STAGE2_EXCHANGE_REPLACEMENT")

    if cpt in STAGE2_CPT_HINTS:
        tags.add("STAGE2_CPT_HINT")

    # QA-only tags
    if PAT["REVISION_RECON_BREAST"].search(t):
        tags.add("REVISION_RECON")

    if PAT["NIPPLE_AREOLA_RECON"].search(t):
        tags.add("REFINEMENT_NIPPLE_AREOLA")

    if PAT["MASTECTOMY"].search(t):
        tags.add("MASTECTOMY")

    return tags


def derive_patient_staging(sub):
    """
    sub: patient-level dataframe of encounters (already has parsed date + proc_norm + cpt)
    """
    s = sub.dropna(subset=["op_date"]).copy()
    s = s.sort_values("op_date")

    s["tags"] = s.apply(lambda r: classify_row(r["proc_norm"], r.get("cpt", "")), axis=1)

    has_expander = s["tags"].apply(lambda z: "STAGE1_EXPANDER" in z).any()
    has_expander_followup = s["tags"].apply(lambda z: "EXPANDER_FOLLOWUP" in z).any()

    # ----------------
    # Stage 1 selection
    # ----------------
    stage1_date = None
    stage1_proc = None
    stage1_type = None  # expander / implant / autologous

    if has_expander:
        exp_rows = s[s["tags"].apply(lambda z: "STAGE1_EXPANDER" in z)]
        if not exp_rows.empty:
            r1 = exp_rows.iloc[0]
            stage1_date = r1["op_date"]
            stage1_proc = r1["proc_norm"]
            stage1_type = "expander"

    if stage1_date is None:
        stage1_mask = s["tags"].apply(lambda z: any(k in z for k in ["STAGE1_IMPLANT_IMMEDIATE", "STAGE1_AUTOLOGOUS"]))
        stage1_rows = s[stage1_mask].copy()
        if not stage1_rows.empty:
            r1 = stage1_rows.iloc[0]
            stage1_date = r1["op_date"]
            stage1_proc = r1["proc_norm"]
            if "STAGE1_AUTOLOGOUS" in r1["tags"]:
                stage1_type = "autologous"
            elif "STAGE1_IMPLANT_IMMEDIATE" in r1["tags"]:
                stage1_type = "implant"
            else:
                stage1_type = "unknown"

    # ----------------
    # Stage 2 selection (ONLY if expander pathway)
    # ----------------
    stage2_date = None
    stage2_proc = None
    stage2_reason = None

    if has_expander and stage1_date is not None:
        s2_mask = s["tags"].apply(lambda z: any(k in z for k in [
            "STAGE2_IMPLANT_SEP_DAY",
            "STAGE2_EXCHANGE_REPLACEMENT",
            "STAGE2_CPT_HINT",
        ]))
        s2_rows = s[s2_mask & (s["op_date"] > stage1_date)].copy()
        if not s2_rows.empty:
            r2 = s2_rows.iloc[0]
            stage2_date = r2["op_date"]
            stage2_proc = r2["proc_norm"]
            # simple reason label for QA
            if "STAGE2_IMPLANT_SEP_DAY" in r2["tags"]:
                stage2_reason = "implant_sep_day"
            elif "STAGE2_EXCHANGE_REPLACEMENT" in r2["tags"]:
                stage2_reason = "exchange_replacement"
            elif "STAGE2_CPT_HINT" in r2["tags"]:
                stage2_reason = "cpt_hint"
            else:
                stage2_reason = "unknown"

    # ----------------
    # Pathway classification
    # ----------------
    pathway = "unknown"
    if stage1_type == "autologous":
        pathway = "single_stage_autologous"
    elif has_expander:
        pathway = "two_stage_expander_implant"
    elif stage1_type == "implant":
        pathway = "single_stage_implant"

    return {
        "patient_id": str(sub["patient_id"].iloc[0]),
        "mrn": str(sub["mrn"].iloc[0]) if "mrn" in sub.columns else "",
        "has_expander": bool(has_expander),
        "has_expander_followup": bool(has_expander_followup),
        "pathway": pathway,
        "stage1_date": stage1_date.strftime("%Y-%m-%d") if pd.notnull(stage1_date) else None,
        "stage1_proc": stage1_proc,
        "stage2_date": stage2_date.strftime("%Y-%m-%d") if pd.notnull(stage2_date) else None,
        "stage2_proc": stage2_proc,
        "stage2_reason": stage2_reason,
        "n_encounter_rows": int(len(sub)),
        "n_rows_with_dates": int(sub["op_date"].notnull().sum()),
    }


def main():
    df = read_csv_fallback(OP_ENCOUNTERS_CSV)

    for c in [COL_PATIENT, COL_PROC]:
        if c not in df.columns:
            raise RuntimeError("Missing required column in OP encounters file: {}".format(c))

    df["patient_id"] = df[COL_PATIENT].fillna("").astype(str)
    df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
    df["proc_norm"] = df[COL_PROC].apply(norm_text)
    df["cpt"] = df[COL_CPT].fillna("").astype(str) if COL_CPT in df.columns else ""
    df["op_date"] = parse_date_series(df, COL_OP_DATE, fallback_col=COL_ALT_DATE)

    df = df[df["patient_id"].str.len() > 0].copy()

    rows = []
    for _, sub in df.groupby("patient_id", sort=True):
        rows.append(derive_patient_staging(sub))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATIENT_CSV, index=False)

    # QA summary
    total_patients = int(out.shape[0])
    with_stage1 = int(out["stage1_date"].notnull().sum())
    with_stage2 = int(out["stage2_date"].notnull().sum())
    two_stage = int((out["pathway"] == "two_stage_expander_implant").sum())
    single_implant = int((out["pathway"] == "single_stage_implant").sum())
    single_auto = int((out["pathway"] == "single_stage_autologous").sum())
    unknown = int((out["pathway"] == "unknown").sum())

    exp_follow = int(out["has_expander_followup"].sum())

    # stage2 reason breakdown
    s2_reason_counts = out["stage2_reason"].fillna("NONE").value_counts()

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
    lines.append("Expander follow-up (subseq expansions) present: {} patients".format(exp_follow))
    lines.append("")
    lines.append("Stage2 reason breakdown (patients):")
    for k, v in s2_reason_counts.items():
        lines.append("  {}: {}".format(k, int(v)))
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
