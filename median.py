# qa_stage2_from_notes_for_expander_patients.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   For expander-pathway patients (from patient_recon_staging.csv),
#   scan available notes (Operation/Inpatient/Clinic) for Stage-2 / exchange language.
#
# Outputs (safe-ish by design, but still contains snippets from notes):
#   1) qa_stage2_note_hits_expanders.csv
#   2) qa_stage2_patient_summary_expanders.csv
#   3) qa_stage2_keyword_counts.csv
#
# NOTE:
#   This script DOES output short snippets. Store/share carefully.

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT IF NEEDED)
# -------------------------
BASE_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

# Uses your staging output to define expander patients (recommended)
PATIENT_STAGING_CSV = "patient_recon_staging.csv"  # in current working directory by default

# Notes files
OP_NOTES_CSV = BASE_DIR + "/HPI11526 Operation Notes.csv"
INPATIENT_NOTES_CSV = BASE_DIR + "/HPI11526 Inpatient Notes.csv"
CLINIC_NOTES_CSV = BASE_DIR + "/HPI11526 Clinic Notes.csv"

# Toggle which notes to scan
SCAN_OPERATION_NOTES = True
SCAN_INPATIENT_NOTES = True
SCAN_CLINIC_NOTES = True

# Notes schema (from your headers)
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_NOTE_DATE = "NOTE_DATE_OF_SERVICE"   # if missing, try OPERATION_DATE / ADMIT_DATE / HOSP_ADMSN_TIME etc.
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_ID = "NOTE_ID"
COL_LINE = "LINE"
COL_TEXT = "NOTE_TEXT"

# If notes are split by LINE, we'll group by NOTE_ID within patient/date/type.
GROUP_LINES_INTO_NOTE = True

# Snippet window around match
SNIPPET_CHARS = 160

# Filter: only consider hits AFTER index stage1_date (if available)
# (Set False if you want to see ALL hits regardless of timing.)
FILTER_TO_AFTER_INDEX = True

# Safety: to reduce output size
MAX_HITS_PER_PATIENT = 200  # prevents runaway output if a patient has many repetitive mentions


# -------------------------
# Stage-2 / exchange lexicon
# -------------------------
# High recall, but we keep patterns clinically grounded.
# Tip: We track which pattern matched for QA.
STAGE2_PATTERNS = [
    ("EXCHANGE_IMPLANT", r"\bexchang(e|ed|ing)\b.{0,60}\b(implant|implnt)\b"),
    ("EXCHANGE_EXPANDER", r"\bexchang(e|ed|ing)\b.{0,60}\b(expander|expandr|tissue\s*expander)\b"),
    ("EXPANDER_TO_IMPLANT", r"\b(expander|expandr|tissue\s*expander)\b.{0,80}\bto\b.{0,20}\b(implant|implnt)\b"),
    ("REMOVE_EXPANDER_IMPLANT", r"\b(remov(e|ed|al)|explant(ed|ation)?)\b.{0,80}\b(expander|expandr|tissue\s*expander)\b.{0,120}\b(implant|implnt)\b"),
    ("PERMANENT_IMPLANT", r"\b(permanent|final)\b.{0,40}\b(implant|implnt)\b"),
    ("SECOND_STAGE", r"\b(second\s*stage|stage\s*(ii|2)|2nd\s*stage)\b"),
    ("IMPLANT_PLACEMENT", r"\b(implant|implnt)\b.{0,40}\b(placement|placed|insert(ed|ion)|insertion)\b"),
    ("TISSUE_EXPANDER_PRESENT", r"\b(tissue\s*expander|expandr|expander)\b"),  # context helper; low-specificity
    ("CAPSULECTOMY_CAPSULOTOMY", r"\b(capsulectomy|capsulotomy)\b"),
    ("RECON_EXCHANGE_GENERIC", r"\bexchange\b.{0,80}\b(recon|reconstruct)\b"),
]

# Compile regex once
COMPILED = [(name, re.compile(pat, re.I | re.S)) for name, pat in STAGE2_PATTERNS]


# -------------------------
# Helpers
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


def coerce_dt(series):
    return pd.to_datetime(series, errors="coerce")


def pick_note_date_col(df):
    """
    Try NOTE_DATE_OF_SERVICE first; if not present, pick the first plausible date column.
    """
    candidates = [
        COL_NOTE_DATE,
        "OPERATION_DATE",
        "ADMIT_DATE",
        "HOSP_ADMSN_TIME",
        "DISCHARGE_DATE_DT",
        "CHECKOUT_TIME",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_note_level_df(df, source_name):
    """
    Convert a line-level notes table into a note-level table:
      - group by patient_id + note_id + note_date + note_type
      - concatenate lines (in line order if LINE is numeric-ish)
    """
    out_cols = ["source", "patient_id", "note_id", "note_date", "note_type", "note_text"]

    # Validate columns
    missing = [c for c in [COL_PATIENT, COL_TEXT] if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns in {}: {}".format(source_name, missing))

    date_col = pick_note_date_col(df)
    if not date_col:
        raise RuntimeError("No usable note date column found in {}.".format(source_name))

    df2 = df.copy()
    df2["patient_id"] = df2[COL_PATIENT].fillna("").astype(str)
    df2["note_text_line"] = df2[COL_TEXT].apply(norm_text)
    df2["note_type"] = df2[COL_NOTE_TYPE].fillna("").astype(str) if COL_NOTE_TYPE in df2.columns else ""
    df2["note_id"] = df2[COL_NOTE_ID].fillna("").astype(str) if COL_NOTE_ID in df2.columns else ""
    df2["note_date"] = coerce_dt(df2[date_col])

    # Drop empty patient_id or empty text
    df2 = df2[(df2["patient_id"].str.len() > 0) & (df2["note_text_line"].str.len() > 0)].copy()

    if not GROUP_LINES_INTO_NOTE:
        df2["source"] = source_name
        df2 = df2.rename(columns={"note_text_line": "note_text"})
        return df2[["source", "patient_id", "note_id", "note_date", "note_type", "note_text"]]

    group_keys = ["patient_id", "note_id", "note_date", "note_type"]
    if COL_LINE in df2.columns:
        # Try numeric sort for LINE
        tmp = df2.copy()
        tmp["_line_num"] = pd.to_numeric(tmp[COL_LINE], errors="coerce")
        tmp = tmp.sort_values(["patient_id", "note_id", "note_date", "note_type", "_line_num"])
        grouped = tmp.groupby(group_keys, dropna=False)["note_text_line"].apply(lambda x: " ".join([t for t in x if t]))
    else:
        grouped = df2.groupby(group_keys, dropna=False)["note_text_line"].apply(lambda x: " ".join([t for t in x if t]))

    note_df = grouped.reset_index().rename(columns={"note_text_line": "note_text"})
    note_df["source"] = source_name
    note_df = note_df[["source", "patient_id", "note_id", "note_date", "note_type", "note_text"]]
    return note_df


def find_matches(text):
    """
    Return list of dicts for matches: pattern_name, start, end, snippet
    """
    matches = []
    if not text:
        return matches

    for name, rx in COMPILED:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            left = max(0, s - SNIPPET_CHARS)
            right = min(len(text), e + SNIPPET_CHARS)
            snippet = text[left:right]
            matches.append({
                "pattern": name,
                "match_start": s,
                "match_end": e,
                "snippet": snippet
            })
    return matches


def main():
    # ---- Load expander patients from staging output ----
    try:
        stg = read_csv_fallback(PATIENT_STAGING_CSV)
    except Exception as e:
        raise RuntimeError("Could not read {}: {}".format(PATIENT_STAGING_CSV, e))

    if "patient_id" not in stg.columns:
        raise RuntimeError("patient_recon_staging.csv must contain 'patient_id' column.")

    # Use expander pathway patients
    # (Your staging uses has_expander boolean + pathway label)
    if "has_expander" in stg.columns:
        expander_ids = stg[stg["has_expander"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    else:
        # Fallback: pathway label
        if "pathway" not in stg.columns:
            raise RuntimeError("Staging file missing 'has_expander' and 'pathway'. Cannot define expander cohort.")
        expander_ids = stg[stg["pathway"].astype(str) == "two_stage_expander_implant"].copy()

    expander_ids["patient_id"] = expander_ids["patient_id"].astype(str)
    expander_set = set(expander_ids["patient_id"].tolist())

    # Index date (Stage 1 date)
    if "stage1_date" in expander_ids.columns:
        expander_ids["index_date"] = pd.to_datetime(expander_ids["stage1_date"], errors="coerce")
    else:
        expander_ids["index_date"] = pd.NaT

    index_map = dict(zip(expander_ids["patient_id"], expander_ids["index_date"]))

    print("Expander patients loaded from {}: {}".format(PATIENT_STAGING_CSV, len(expander_set)))

    # ---- Load and unify notes ----
    note_frames = []

    if SCAN_OPERATION_NOTES:
        op_raw = read_csv_fallback(OP_NOTES_CSV)
        op_notes = build_note_level_df(op_raw, "operation_notes")
        note_frames.append(op_notes)

    if SCAN_INPATIENT_NOTES:
        ip_raw = read_csv_fallback(INPATIENT_NOTES_CSV)
        ip_notes = build_note_level_df(ip_raw, "inpatient_notes")
        note_frames.append(ip_notes)

    if SCAN_CLINIC_NOTES:
        cl_raw = read_csv_fallback(CLINIC_NOTES_CSV)
        cl_notes = build_note_level_df(cl_raw, "clinic_notes")
        note_frames.append(cl_notes)

    if not note_frames:
        raise RuntimeError("No note sources enabled. Set SCAN_* = True for at least one notes file.")

    notes = pd.concat(note_frames, ignore_index=True)

    # Keep only expander patients
    notes = notes[notes["patient_id"].isin(expander_set)].copy()
    notes["note_text"] = notes["note_text"].fillna("").astype(str)

    # If filtering to AFTER index, attach index date
    notes["index_date"] = notes["patient_id"].map(index_map)

    if FILTER_TO_AFTER_INDEX:
        # Keep notes with note_date >= index_date (or keep if index_date missing)
        keep = (notes["index_date"].isna()) | (notes["note_date"].isna()) | (notes["note_date"] >= notes["index_date"])
        notes = notes[keep].copy()

    print("Total notes for expander patients (post index filter={}): {}".format(FILTER_TO_AFTER_INDEX, notes.shape[0]))

    # ---- Scan notes for matches ----
    hits_rows = []
    per_patient_hit_counts = {}

    # Iterate row-wise (works fine at moderate scale; avoids exploding memory)
    for i, r in notes.iterrows():
        pid = r["patient_id"]
        txt = r["note_text"]

        # cap
        c = per_patient_hit_counts.get(pid, 0)
        if c >= MAX_HITS_PER_PATIENT:
            continue

        matches = find_matches(txt)
        if not matches:
            continue

        for m in matches:
            hits_rows.append({
                "patient_id": pid,
                "index_date": r["index_date"].strftime("%Y-%m-%d") if pd.notnull(r["index_date"]) else "",
                "source": r["source"],
                "note_date": r["note_date"].strftime("%Y-%m-%d") if pd.notnull(r["note_date"]) else "",
                "note_type": r["note_type"],
                "note_id": r["note_id"],
                "pattern": m["pattern"],
                "snippet": m["snippet"],
            })

        per_patient_hit_counts[pid] = c + 1

    hits = pd.DataFrame(hits_rows)
    if hits.empty:
        print("No Stage-2 language hits found in the scanned notes.")
        # Still write empty outputs for consistency
        hits.to_csv("qa_stage2_note_hits_expanders.csv", index=False)
        pd.DataFrame(columns=["patient_id", "index_date", "earliest_hit_date", "n_hit_notes", "n_hit_rows"]).to_csv(
            "qa_stage2_patient_summary_expanders.csv", index=False
        )
        pd.DataFrame(columns=["pattern", "n_hits"]).to_csv("qa_stage2_keyword_counts.csv", index=False)
        return

    # ---- Patient-level summary ----
    # earliest hit date per patient (based on note_date)
    tmp = hits.copy()
    tmp["note_date_dt"] = pd.to_datetime(tmp["note_date"], errors="coerce")

    patient_summary = tmp.groupby("patient_id").agg(
        index_date=("index_date", "first"),
        earliest_hit_date=("note_date_dt", "min"),
        n_hit_rows=("pattern", "size"),
        n_hit_notes=("note_id", pd.Series.nunique),
    ).reset_index()

    patient_summary["earliest_hit_date"] = patient_summary["earliest_hit_date"].dt.strftime("%Y-%m-%d")

    # ---- Keyword counts ----
    keyword_counts = hits["pattern"].value_counts().reset_index()
    keyword_counts.columns = ["pattern", "n_hits"]

    # ---- Write outputs ----
    hits.to_csv("qa_stage2_note_hits_expanders.csv", index=False)
    patient_summary.to_csv("qa_stage2_patient_summary_expanders.csv", index=False)
    keyword_counts.to_csv("qa_stage2_keyword_counts.csv", index=False)

    # ---- Console summary ----
    n_patients_with_hits = patient_summary.shape[0]
    print("")
    print("=== Stage-2 Notes Scan Summary (Expanders) ===")
    print("Expander patients total: {}".format(len(expander_set)))
    print("Patients with >=1 Stage-2 language hit: {} ({:.1f}%)".format(
        n_patients_with_hits,
        100.0 * n_patients_with_hits / float(len(expander_set)) if expander_set else 0.0
    ))
    print("Total hit rows: {}".format(hits.shape[0]))
    print("")
    print("Top patterns:")
    print(keyword_counts.head(12).to_string(index=False))
    print("")
    print("Wrote:")
    print(" - qa_stage2_note_hits_expanders.csv")
    print(" - qa_stage2_patient_summary_expanders.csv")
    print(" - qa_stage2_keyword_counts.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
