# sanity_check_index_dates.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   Sanity-check whether our "index" (Stage 1) date selection is plausible.
#   This does NOT change staging. It audits what the current staging logic is doing.
#
# Reads:
#   OPERATION_ENCOUNTERS.csv (structured encounters)
#
# Writes:
#   1) qa_index_sanity_summary.txt
#   2) qa_index_patient_flags.csv
#   3) qa_index_event_timeline_samples.csv   (small per-patient timeline, limited rows)
#
# Key idea:
#   "Index" == stage1_date as derived by CURRENT staging logic:
#     - If expander exists -> earliest STAGE1_EXPANDER row date
#     - else -> earliest immediate implant or autologous date
#
# We flag patients where:
#   - expander pathway but no stage1_date found (should be rare)
#   - any Stage2-tag row exists but ALL are <= stage1_date (date ordering problem)
#   - large number of distinct op dates but no Stage2-tag rows (pattern/coverage problem)
#   - stage2-tag rows exist pre-date-filter but index date is later than the first stage2-tag date
#
# NOTE: This script prints aggregate counts only; the CSV contains patient_ids.
#       No note_text is used; only structured encounter procedure strings.

import re
import sys
import pandas as pd


# -------------------------
# CONFIG
# -------------------------
OP_ENCOUNTERS_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv"

OUT_SUMMARY_TXT = "qa_index_sanity_summary.txt"
OUT_FLAGS_CSV = "qa_index_patient_flags.csv"
OUT_TIMELINES_CSV = "qa_index_event_timeline_samples.csv"

# Column names (from your headers)
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_OP_DATE = "OPERATION_DATE"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"
COL_ALT_DATE = "DISCHARGE_DATE_DT"

# Limit how many timeline rows we write per patient (to keep file small)
MAX_TIMELINE_ROWS_PER_PATIENT = 12


# -------------------------
# Regex patterns (COPIED from your current staging file)
# -------------------------
PAT = {
    "MASTECTOMY": re.compile(r"\bmastectomy\b", re.I),

    "EXPANDER_PLACEMENT": re.compile(
        r"\b(tissue\s*expander|tissue\s*expand|tiss\s*expand|expandr|expander)\b"
        r".{0,80}\b(plac(e|m)(ent|mnt|mt)|plcmnt|placmnt|placement|insert(ion)?|impla?n?t|implnt)\b"
        r"|"
        r"\b(plac(e|m)(ent|mnt|mt)|plcmnt|placmnt|placement|insert(ion)?|impla?n?t|implnt)\b"
        r".{0,80}\b(tissue\s*expander|tissue\s*expand|tiss\s*expand|expandr|expander)\b",
        re.I
    ),

    "EXPANDER_INCLUDES_SUBSEQ_EXPANSIONS": re.compile(
        r"\bsubseq(uent)?\b.{0,40}\bexpansions?\b|\binc\b.{0,20}\bsubseq\b.{0,40}\bexpansions?\b",
        re.I
    ),

    "IMPLANT_IMMEDIATE": re.compile(
        r"\binsertion\b.*\b(implant|implnt)\b.*\bsame\s+day\b.*\bmastectomy\b|"
        r"\b(implant|implnt)\b.*\bsame\s+day\b.*\bmastectomy\b|"
        r"\bimmediate\b.*\b(implant|implnt)\b",
        re.I
    ),

    "IMPLANT_SEP_DAY": re.compile(
        r"\b(implant|implnt)\b.*\bsep(arate)?\s+day\b|"
        r"\bsep\s+day\b.*\b(implant|implnt)\b|"
        r"\bon\s+sep(arate)?\s+day\b.*\b(implant|implnt)\b|"
        r"\bdelayed\b.*\b(implant|implnt)\b",
        re.I
    ),

    "EXCHANGE_OR_REPLACEMENT": re.compile(
        r"\bexchange\b.*\b(implant|implnt|expander|expandr)\b|"
        r"\b(implant|implnt|expander|expandr)\b.*\bexchange\b|"
        r"\brepl(a)?cmnt\b.*\b(implant|implnt)\b|"
        r"\b(implant|implnt)\b.*\brepl(a)?cmnt\b|"
        r"\bremove(d)?\b.*\b(tissue\s*expander|expander|expandr)\b.*\b(implant|implnt)\b|"
        r"\bexplant\b.*\b(tissue\s*expander|expander|expandr)\b.*\b(implant|implnt)\b",
        re.I
    ),

    "AUTOLOGOUS_FREE_FLAP": re.compile(r"\bfree\s+flap\b|\bdiep\b|\btr?am\b|\bsiea\b|\bgap\s+flap\b", re.I),
    "LAT_DORSI": re.compile(r"\blatissimus\s+dorsi\b", re.I),

    "REVISION_RECON_BREAST": re.compile(
        r"\brevision\b.*\breconstruct(ed|ion)\b.*\bbreast\b|\brevision\s+of\s+reconstructed\s+breast\b",
        re.I
    ),
    "NIPPLE_AREOLA_RECON": re.compile(r"\bnipple\b|\bareola\b", re.I),
}

STAGE2_CPT_HINTS = set(["11970", "19342"])


# -------------------------
# Helpers (aligned with your staging)
# -------------------------
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
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_date_series(df, primary_col, fallback_col=None):
    dt = pd.to_datetime(df[primary_col], errors="coerce") if primary_col in df.columns else pd.Series([pd.NaT] * len(df))
    if fallback_col and fallback_col in df.columns:
        fb = pd.to_datetime(df[fallback_col], errors="coerce")
        dt = dt.fillna(fb)
    return dt


def classify_row(proc_text, cpt_code):
    tags = set()
    t = proc_text
    cpt = (str(cpt_code).strip() if cpt_code is not None else "")

    # Stage 1
    if PAT["EXPANDER_PLACEMENT"].search(t):
        tags.add("STAGE1_EXPANDER")
        if PAT["EXPANDER_INCLUDES_SUBSEQ_EXPANSIONS"].search(t):
            tags.add("EXPANDER_DESC_INCLUDES_SUBSEQ_EXPANSIONS")

    if PAT["IMPLANT_IMMEDIATE"].search(t):
        tags.add("STAGE1_IMPLANT_IMMEDIATE")

    if PAT["AUTOLOGOUS_FREE_FLAP"].search(t) or PAT["LAT_DORSI"].search(t):
        tags.add("STAGE1_AUTOLOGOUS")

    # Stage 2 tags (candidate signals)
    if PAT["IMPLANT_SEP_DAY"].search(t):
        tags.add("STAGE2_IMPLANT_SEP_DAY")

    if PAT["EXCHANGE_OR_REPLACEMENT"].search(t):
        tags.add("STAGE2_EXCHANGE_REPLACEMENT")

    if cpt in STAGE2_CPT_HINTS:
        tags.add("STAGE2_CPT_HINT")

    # QA-only
    if PAT["MASTECTOMY"].search(t):
        tags.add("MASTECTOMY")
    if PAT["REVISION_RECON_BREAST"].search(t):
        tags.add("REVISION_RECON")
    if PAT["NIPPLE_AREOLA_RECON"].search(t):
        tags.add("REFINEMENT_NIPPLE_AREOLA")

    return tags


def choose_index_stage1(s):
    """
    s: patient dataframe sorted by op_date, with tags
    Returns (stage1_date, stage1_proc, stage1_type, has_expander)
    matching your CURRENT staging logic.
    """
    has_expander = s["tags"].apply(lambda z: "STAGE1_EXPANDER" in z).any()

    stage1_date = None
    stage1_proc = None
    stage1_type = None

    if has_expander:
        exp_rows = s[s["tags"].apply(lambda z: "STAGE1_EXPANDER" in z)]
        if not exp_rows.empty:
            r1 = exp_rows.iloc[0]
            stage1_date = r1["op_date"]
            stage1_proc = r1["proc_norm"]
            stage1_type = "expander"

    if stage1_date is None:
        mask = s["tags"].apply(lambda z: any(k in z for k in ["STAGE1_IMPLANT_IMMEDIATE", "STAGE1_AUTOLOGOUS"]))
        rows = s[mask]
        if not rows.empty:
            r1 = rows.iloc[0]
            stage1_date = r1["op_date"]
            stage1_proc = r1["proc_norm"]
            if "STAGE1_AUTOLOGOUS" in r1["tags"]:
                stage1_type = "autologous"
            elif "STAGE1_IMPLANT_IMMEDIATE" in r1["tags"]:
                stage1_type = "implant"
            else:
                stage1_type = "unknown"

    return stage1_date, stage1_proc, stage1_type, has_expander


def main():
    df = read_csv_fallback(OP_ENCOUNTERS_CSV)

    # Validate required columns
    for c in [COL_PATIENT, COL_PROC]:
        if c not in df.columns:
            raise RuntimeError("Missing required column in OP encounters file: {}".format(c))

    df["patient_id"] = df[COL_PATIENT].fillna("").astype(str)
    df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
    df["proc_norm"] = df[COL_PROC].apply(norm_text)
    df["cpt"] = df[COL_CPT].fillna("").astype(str) if COL_CPT in df.columns else ""
    df["op_date"] = parse_date_series(df, COL_OP_DATE, fallback_col=COL_ALT_DATE)

    # Drop blank patient_id
    df = df[df["patient_id"].str.len() > 0].copy()

    # Keep only dated rows for timeline logic (this matches how staging drops NaT)
    df_dated = df.dropna(subset=["op_date"]).copy()

    # Add tags
    df_dated["tags"] = df_dated.apply(lambda r: classify_row(r["proc_norm"], r.get("cpt", "")), axis=1)

    # Outputs
    flag_rows = []
    timeline_rows = []

    # Aggregate counters
    n_patients = 0
    n_expander = 0
    n_expander_no_index = 0

    n_any_stage2_tag = 0
    n_stage2_only_before_or_on_index = 0
    n_stage2_exists_but_index_is_after_first_stage2 = 0

    n_expander_multi_dates = 0
    n_expander_multi_dates_no_stage2_tag = 0

    # Iterate patients
    for pid, sub in df_dated.groupby("patient_id", sort=True):
        n_patients += 1
        s = sub.sort_values("op_date").copy()

        stage1_date, stage1_proc, stage1_type, has_expander = choose_index_stage1(s)
        if has_expander:
            n_expander += 1

        # Distinct op dates
        distinct_dates = int(s["op_date"].dt.date.nunique())

        # Any stage2-tag row regardless of date filter
        stage2_tag_mask = s["tags"].apply(lambda z: any(k in z for k in ["STAGE2_IMPLANT_SEP_DAY", "STAGE2_EXCHANGE_REPLACEMENT", "STAGE2_CPT_HINT"]))
        has_any_stage2 = bool(stage2_tag_mask.any())
        if has_any_stage2:
            n_any_stage2_tag += 1

        # First stage2-tag date
        first_stage2_date = None
        if has_any_stage2:
            first_stage2_date = s.loc[stage2_tag_mask, "op_date"].iloc[0]

        # Basic flags
        flags = []

        if has_expander and stage1_date is None:
            flags.append("EXPANDER_BUT_NO_INDEX_STAGE1")
            n_expander_no_index += 1

        # If stage2 tags exist but all are <= index, that’s a red flag
        if has_any_stage2 and stage1_date is not None:
            s2_after = s.loc[stage2_tag_mask & (s["op_date"] > stage1_date)]
            if s2_after.empty:
                flags.append("STAGE2_TAGS_EXIST_BUT_NONE_AFTER_INDEX")
                n_stage2_only_before_or_on_index += 1

        # If first stage2 tag occurs before the chosen index, index may be wrong
        if has_any_stage2 and stage1_date is not None and first_stage2_date is not None:
            if first_stage2_date < stage1_date:
                flags.append("FIRST_STAGE2_DATE_BEFORE_INDEX")
                n_stage2_exists_but_index_is_after_first_stage2 += 1

        # If expander patient has multiple distinct dates but no stage2 tags, coverage likely missing
        if has_expander and distinct_dates >= 2:
            n_expander_multi_dates += 1
            if not has_any_stage2:
                flags.append("EXPANDER_MULTI_DATES_NO_STAGE2_TAGS")
                n_expander_multi_dates_no_stage2_tag += 1

        # Compute simple deltas (days)
        delta_index_to_first_stage2_days = None
        if stage1_date is not None and first_stage2_date is not None:
            delta_index_to_first_stage2_days = int((first_stage2_date - stage1_date).days)

        # Store patient flags row (even if empty, keep a row so we can filter later)
        flag_rows.append({
            "patient_id": pid,
            "mrn": str(s["mrn"].iloc[0]) if "mrn" in s.columns else "",
            "has_expander": bool(has_expander),
            "stage1_type": stage1_type,
            "index_stage1_date": stage1_date.strftime("%Y-%m-%d") if pd.notnull(stage1_date) else None,
            "index_stage1_proc": stage1_proc,
            "distinct_op_dates": distinct_dates,
            "has_any_stage2_tag": bool(has_any_stage2),
            "first_stage2_tag_date": first_stage2_date.strftime("%Y-%m-%d") if pd.notnull(first_stage2_date) else None,
            "delta_days_index_to_first_stage2": delta_index_to_first_stage2_days,
            "flags": "|".join(flags) if flags else ""
        })

        # Write a small timeline sample for flagged patients only (keeps file manageable)
        if flags:
            # Take first few events + any stage2-tag events
            s_small = s.copy()
            s_small["is_stage2_tag"] = stage2_tag_mask

            # Keep: first N rows, plus any stage2-tag rows (dedupe by index)
            head_idx = list(s_small.head(MAX_TIMELINE_ROWS_PER_PATIENT).index)
            s2_idx = list(s_small[s_small["is_stage2_tag"]].index)
            keep_idx = list(dict.fromkeys(head_idx + s2_idx))  # preserve order, unique

            s_keep = s_small.loc[keep_idx].copy()
            if len(s_keep) > MAX_TIMELINE_ROWS_PER_PATIENT:
                s_keep = s_keep.head(MAX_TIMELINE_ROWS_PER_PATIENT)

            for _, r in s_keep.iterrows():
                timeline_rows.append({
                    "patient_id": pid,
                    "op_date": r["op_date"].strftime("%Y-%m-%d") if pd.notnull(r["op_date"]) else None,
                    "cpt": r.get("cpt", ""),
                    "proc_norm": r.get("proc_norm", ""),
                    "tags": ",".join(sorted(list(r.get("tags", set())))),
                    "is_index_stage1": bool(stage1_date is not None and pd.notnull(r["op_date"]) and r["op_date"] == stage1_date and r.get("proc_norm", "") == stage1_proc),
                    "is_stage2_tag": bool(r.get("is_stage2_tag", False)),
                    "patient_flags": "|".join(flags)
                })

    # Build outputs
    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(OUT_FLAGS_CSV, index=False)

    timelines_df = pd.DataFrame(timeline_rows)
    timelines_df.to_csv(OUT_TIMELINES_CSV, index=False)

    # Summary text
    def pct(a, b):
        return (100.0 * a / b) if b else 0.0

    lines = []
    lines.append("=== Index (Stage 1) Sanity Check Summary ===")
    lines.append("Total patients (with >=1 parsed date): {}".format(n_patients))
    lines.append("")
    lines.append("Expander pathway patients: {} ({:.1f}%)".format(n_expander, pct(n_expander, n_patients)))
    lines.append("Expander patients with NO index (stage1_date) found: {} ({:.1f}% of expander)".format(
        n_expander_no_index, pct(n_expander_no_index, n_expander)
    ))
    lines.append("")
    lines.append("Patients with ANY Stage2-tag row (ignoring date filter): {} ({:.1f}%)".format(
        n_any_stage2_tag, pct(n_any_stage2_tag, n_patients)
    ))
    lines.append("Patients where Stage2-tags exist but NONE are AFTER index: {} ({:.1f}% of stage2-tag patients)".format(
        n_stage2_only_before_or_on_index, pct(n_stage2_only_before_or_on_index, n_any_stage2_tag)
    ))
    lines.append("Patients where FIRST Stage2-tag date is BEFORE index: {} ({:.1f}% of stage2-tag patients)".format(
        n_stage2_exists_but_index_is_after_first_stage2, pct(n_stage2_exists_but_index_is_after_first_stage2, n_any_stage2_tag)
    ))
    lines.append("")
    lines.append("Expander patients with >=2 distinct op dates: {} ({:.1f}% of expander)".format(
        n_expander_multi_dates, pct(n_expander_multi_dates, n_expander)
    ))
    lines.append("...of those, expander patients with NO stage2-tag rows: {} ({:.1f}% of multi-date expander)".format(
        n_expander_multi_dates_no_stage2_tag, pct(n_expander_multi_dates_no_stage2_tag, n_expander_multi_dates)
    ))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_FLAGS_CSV))
    lines.append("  - {}".format(OUT_TIMELINES_CSV))
    lines.append("  - {}".format(OUT_SUMMARY_TXT))

    with open(OUT_SUMMARY_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
