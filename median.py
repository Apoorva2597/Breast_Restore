# qa_all_encounters_and_notes_stage2_audit.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   Give Clinic/Inpatient encounters + notes the same chance we gave OP notes.
#   1) Inventory ALL CPT codes and procedure strings per encounter file
#   2) Find Stage-2-relevant CPTs (e.g., 11970, 19342) in any encounter file
#   3) Keyword-scan PROCEDURE / REASON_FOR_VISIT in encounter files for Stage-2 language
#   4) Keyword-scan NOTE_TEXT in note files for Stage-2 language (like the OP note scan)
#
# Outputs (CSV):
#   - qa_all_enc_cpt_inventory_by_file.csv
#   - qa_all_enc_proc_inventory_by_file.csv
#   - qa_all_enc_stage2_cpt_hits.csv
#   - qa_all_enc_stage2_keyword_hits.csv
#   - qa_all_notes_stage2_keyword_hits.csv
#   - qa_all_notes_keyword_counts.csv
#
# Notes:
#   - This script is designed to be "audit-first": broad coverage, minimal assumptions.
#   - Uses chunked reading for the big note files.

import re
import sys
import pandas as pd


# -------------------------
# CONFIG: paths
# -------------------------
BASE = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/"

ENCOUNTER_FILES = [
    ("clinic_encounters",   BASE + "HPI11526 Clinic Encounters.csv"),
    ("inpatient_encounters",BASE + "HPI11526 Inpatient Encounters.csv"),
    ("operation_encounters",BASE + "HPI11526 Operation Encounters.csv"),
]

NOTE_FILES = [
    ("clinic_notes",   BASE + "HPI11526 Clinic Notes.csv"),
    ("inpatient_notes",BASE + "HPI11526 Inpatient Notes.csv"),
    ("operation_notes",BASE + "HPI11526 Operation Notes.csv"),
]

# Encounter columns (from your headers)
COL_PAT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_CPT = "CPT_CODE"
COL_PROC = "PROCEDURE"
COL_REASON = "REASON_FOR_VISIT"          # present in clinic/inpatient encounters; may be absent elsewhere
COL_DATE_CANDIDATES = [
    "OPERATION_DATE",
    "RECONSTRUCTION_DATE",
    "ADMIT_DATE",
    "HOSP_ADMSN_TIME",
    "CHECKOUT_TIME",
    "DISCHARGE_DATE_DT",
    "HOSP_DISCHRG_TIME",
]

# Note columns (common)
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_DATE_CANDIDATES = ["NOTE_DATE_OF_SERVICE", "DATE_OF_SERVICE", "NOTE_DATE", "NOTE_DATETIME"]


# -------------------------
# Stage-2 CPT list (start conservative; expand later if needed)
# -------------------------
STAGE2_CPTS = set([
    "11970",   # replacement of tissue expander with permanent prosthesis
    "19342",   # delayed insertion / replacement (can be used for exchange-ish scenarios)
])

# Also include these as "implant-related" context (not strictly stage2)
IMPLANT_CONTEXT_CPTS = set([
    "19340",   # immediate insertion of breast prosthesis (common)
    "19357",   # tissue expander placement (stage1-ish)
])

ALL_CPT_OF_INTEREST = STAGE2_CPTS.union(IMPLANT_CONTEXT_CPTS)

# -------------------------
# Keyword patterns (same idea as OP note scan)
# -------------------------
PATTERNS = {
    "TISSUE_EXPANDER_PRESENT": re.compile(r"\btissue\s+expand(er|ers)\b|\bexpander\b", re.I),
    "EXCHANGE_IMPLANT": re.compile(r"\bexchange\b.*\bimplant\b|\bimplant\b.*\bexchange\b", re.I),
    "EXCHANGE_EXPANDER": re.compile(r"\bexchange\b.*\bexpander\b|\bexpander\b.*\bexchange\b", re.I),
    "EXPANDER_TO_IMPLANT": re.compile(r"\bexpander\b.*\bto\b.*\bimplant\b|\bimplant\b.*\bafter\b.*\bexpander\b", re.I),
    "REMOVE_EXPANDER_IMPLANT": re.compile(r"\bremove(d)?\b.*\bexpander\b.*\bimplant\b|\bexplant\b.*\bexpander\b.*\bimplant\b", re.I),
    "PERMANENT_IMPLANT": re.compile(r"\bpermanent\b.*\bimplant\b|\bfinal\b.*\bimplant\b", re.I),
    "SECOND_STAGE": re.compile(r"\bsecond\s+stage\b|\bstage\s*2\b|\bstage\s+ii\b", re.I),
    "CAPSULECTOMY_CAPSULOTOMY": re.compile(r"\bcapsulectomy\b|\bcapsulotomy\b", re.I),
    "IMPLANT_PLACEMENT": re.compile(r"\bimplant\b.*\bplacement\b|\bplace\b.*\bimplant\b", re.I),
}

# Treat these as “stage2-ish evidence” patterns in keyword scans
STAGE2_KEYWORDS = [
    "EXCHANGE_IMPLANT",
    "EXPANDER_TO_IMPLANT",
    "REMOVE_EXPANDER_IMPLANT",
    "SECOND_STAGE",
    "PERMANENT_IMPLANT",
    "EXCHANGE_EXPANDER",
    "CAPSULECTOMY_CAPSULOTOMY",
    "IMPLANT_PLACEMENT",
]


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_text(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_first_date(df, candidates):
    for c in candidates:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notnull().any():
                return dt, c
    return pd.Series([pd.NaT] * len(df)), None


def cpt_norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    # normalize common weirdness: "19342.0" -> "19342"
    s = re.sub(r"\.0$", "", s)
    s = s.replace(" ", "")
    return s


def keyword_hits(text):
    """
    Return list of pattern names matched in the given text.
    """
    hits = []
    t = text
    for name in STAGE2_KEYWORDS:
        if PATTERNS[name].search(t):
            hits.append(name)
    return hits


def audit_encounters():
    cpt_rows = []
    proc_rows = []
    stage2_cpt_hits = []
    stage2_kw_hits = []

    for file_tag, path in ENCOUNTER_FILES:
        df = read_csv_fallback(path)

        # minimal required columns
        if COL_PAT not in df.columns or COL_PROC not in df.columns:
            print("WARN: {} missing required columns; skipping. path={}".format(file_tag, path))
            continue

        # normalize fields
        df["file_tag"] = file_tag
        df["patient_id"] = df[COL_PAT].fillna("").astype(str)
        df["mrn"] = df[COL_MRN].fillna("").astype(str) if COL_MRN in df.columns else ""
        df["cpt_norm"] = df[COL_CPT].apply(cpt_norm) if COL_CPT in df.columns else ""
        df["proc_norm"] = df[COL_PROC].apply(norm_text)
        if COL_REASON in df.columns:
            df["reason_norm"] = df[COL_REASON].apply(norm_text)
        else:
            df["reason_norm"] = ""

        # date (best effort)
        dt, dt_col = pick_first_date(df, COL_DATE_CANDIDATES)
        df["enc_date"] = dt
        df["enc_date_source_col"] = dt_col if dt_col else ""

        # ----- CPT inventory (ALL CPTs) -----
        if COL_CPT in df.columns:
            tmp = df[df["cpt_norm"] != ""].copy()
            inv = (tmp.groupby("cpt_norm")
                      .agg(encounter_rows=("cpt_norm", "size"),
                           unique_patients=("patient_id", pd.Series.nunique),
                           unique_procedures=("proc_norm", pd.Series.nunique))
                      .reset_index()
                      .rename(columns={"cpt_norm": "cpt"}))
            inv["file_tag"] = file_tag
            cpt_rows.append(inv)

        # ----- Procedure inventory (ALL procedure strings) -----
        invp = (df.groupby("proc_norm")
                  .agg(encounter_rows=("proc_norm", "size"),
                       unique_patients=("patient_id", pd.Series.nunique),
                       unique_cpt_codes=("cpt_norm", lambda x: (x != "").sum()),
                       unique_cpt_values=("cpt_norm", pd.Series.nunique))
                  .reset_index()
                  .rename(columns={"proc_norm": "procedure"}))
        invp["file_tag"] = file_tag
        proc_rows.append(invp)

        # ----- Stage2 CPT hits -----
        if COL_CPT in df.columns:
            hit = df[df["cpt_norm"].isin(STAGE2_CPTS)].copy()
            if not hit.empty:
                keep = ["file_tag", "patient_id", "mrn", "cpt_norm", "proc_norm", "reason_norm", "enc_date", "enc_date_source_col"]
                hit = hit[keep].rename(columns={"cpt_norm": "cpt", "proc_norm": "procedure", "reason_norm": "reason_for_visit"})
                stage2_cpt_hits.append(hit)

        # ----- Stage2 keyword hits in encounters (procedure + reason_for_visit) -----
        df["kw_text"] = (df["proc_norm"].fillna("") + " || " + df["reason_norm"].fillna("")).apply(norm_text)
        df["kw_hits"] = df["kw_text"].apply(keyword_hits)
        hit2 = df[df["kw_hits"].apply(lambda x: len(x) > 0)].copy()
        if not hit2.empty:
            keep = ["file_tag", "patient_id", "mrn", "cpt_norm", "proc_norm", "reason_norm", "enc_date", "enc_date_source_col", "kw_hits"]
            hit2 = hit2[keep].rename(columns={"cpt_norm": "cpt", "proc_norm": "procedure", "reason_norm": "reason_for_visit"})
            # explode kw_hits to one row per hit
            hit2 = hit2.explode("kw_hits").rename(columns={"kw_hits": "pattern"})
            stage2_kw_hits.append(hit2)

    # write outputs
    if cpt_rows:
        out_cpt = pd.concat(cpt_rows, ignore_index=True)
        out_cpt.to_csv("qa_all_enc_cpt_inventory_by_file.csv", index=False)
    else:
        out_cpt = pd.DataFrame()

    if proc_rows:
        out_proc = pd.concat(proc_rows, ignore_index=True)
        out_proc.to_csv("qa_all_enc_proc_inventory_by_file.csv", index=False)
    else:
        out_proc = pd.DataFrame()

    if stage2_cpt_hits:
        out_s2c = pd.concat(stage2_cpt_hits, ignore_index=True)
        out_s2c.to_csv("qa_all_enc_stage2_cpt_hits.csv", index=False)
    else:
        out_s2c = pd.DataFrame()
        out_s2c.to_csv("qa_all_enc_stage2_cpt_hits.csv", index=False)

    if stage2_kw_hits:
        out_s2k = pd.concat(stage2_kw_hits, ignore_index=True)
        out_s2k.to_csv("qa_all_enc_stage2_keyword_hits.csv", index=False)
    else:
        out_s2k = pd.DataFrame()
        out_s2k.to_csv("qa_all_enc_stage2_keyword_hits.csv", index=False)

    return out_cpt, out_proc, out_s2c, out_s2k


def audit_notes():
    # We will scan NOTE_TEXT for patterns, chunked
    hits_all = []
    keyword_counts = {k: 0 for k in PATTERNS.keys()}
    total_hit_rows = 0

    for file_tag, path in NOTE_FILES:
        # chunked read (notes files can be huge)
        try:
            reader = pd.read_csv(path, encoding="utf-8", engine="python", chunksize=50000)
        except UnicodeDecodeError:
            reader = pd.read_csv(path, encoding="cp1252", engine="python", chunksize=50000)

        for chunk in reader:
            if COL_NOTE_TEXT not in chunk.columns or COL_PAT not in chunk.columns:
                continue

            chunk["file_tag"] = file_tag
            chunk["patient_id"] = chunk[COL_PAT].fillna("").astype(str)
            chunk["mrn"] = chunk[COL_MRN].fillna("").astype(str) if COL_MRN in chunk.columns else ""
            chunk["note_text_norm"] = chunk[COL_NOTE_TEXT].apply(norm_text)

            # date best-effort
            dt, dt_col = pick_first_date(chunk, COL_NOTE_DATE_CANDIDATES)
            chunk["note_date"] = dt
            chunk["note_date_source_col"] = dt_col if dt_col else ""

            # find which patterns match
            def get_hit_patterns(t):
                hits = []
                for name in STAGE2_KEYWORDS:
                    if PATTERNS[name].search(t):
                        hits.append(name)
                return hits

            chunk["hit_patterns"] = chunk["note_text_norm"].apply(get_hit_patterns)
            hit = chunk[chunk["hit_patterns"].apply(lambda x: len(x) > 0)].copy()
            if hit.empty:
                continue

            total_hit_rows += int(hit.shape[0])

            # update counts
            for plist in hit["hit_patterns"].tolist():
                for p in plist:
                    keyword_counts[p] += 1

            # keep a compact hit table (avoid dumping full notes unless you want it)
            keep_cols = ["file_tag", "patient_id", "mrn", "note_date", "note_date_source_col"]
            # If NOTE_ID exists, keep it
            if "NOTE_ID" in hit.columns:
                keep_cols.append("NOTE_ID")
            if "NOTE_TYPE" in hit.columns:
                keep_cols.append("NOTE_TYPE")
            # keep a short snippet for inspection
            hit["snippet"] = hit["note_text_norm"].str.slice(0, 300)

            hit = hit[keep_cols + ["hit_patterns", "snippet"]].explode("hit_patterns").rename(columns={"hit_patterns": "pattern"})
            hits_all.append(hit)

    if hits_all:
        out_hits = pd.concat(hits_all, ignore_index=True)
        out_hits.to_csv("qa_all_notes_stage2_keyword_hits.csv", index=False)
    else:
        out_hits = pd.DataFrame()
        out_hits.to_csv("qa_all_notes_stage2_keyword_hits.csv", index=False)

    kc = pd.DataFrame([{"pattern": k, "n_hits": int(v)} for k, v in keyword_counts.items()])
    kc = kc[kc["pattern"].isin(STAGE2_KEYWORDS)].sort_values("n_hits", ascending=False)
    kc.to_csv("qa_all_notes_keyword_counts.csv", index=False)

    return out_hits, kc, total_hit_rows


def main():
    print("=== Auditing ENCOUNTERS (Clinic/Inpatient/Operation) ===")
    out_cpt, out_proc, out_s2c, out_s2k = audit_encounters()

    print("Wrote:")
    print(" - qa_all_enc_cpt_inventory_by_file.csv")
    print(" - qa_all_enc_proc_inventory_by_file.csv")
    print(" - qa_all_enc_stage2_cpt_hits.csv")
    print(" - qa_all_enc_stage2_keyword_hits.csv")
    print("")

    # Quick console summary: stage2 CPTs by file
    if not out_s2c.empty:
        s = out_s2c.groupby(["file_tag", "cpt"]).agg(
            encounter_rows=("cpt", "size"),
            unique_patients=("patient_id", pd.Series.nunique),
        ).reset_index()
        print("Stage2 CPT hits summary (encounters):")
        print(s.to_string(index=False))
    else:
        print("Stage2 CPT hits summary (encounters): NONE")

    print("\n=== Auditing NOTES (Clinic/Inpatient/Operation) ===")
    out_hits, kc, total_hit_rows = audit_notes()

    print("Wrote:")
    print(" - qa_all_notes_stage2_keyword_hits.csv")
    print(" - qa_all_notes_keyword_counts.csv")
    print("")
    print("Notes hit rows (any stage2-ish pattern): {}".format(int(total_hit_rows)))
    print("Top note patterns:")
    if kc is not None and not kc.empty:
        print(kc.head(15).to_string(index=False))
    else:
        print("NONE")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
