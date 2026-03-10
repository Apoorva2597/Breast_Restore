#!/usr/bin/env python3
# qa_smoking_targeted_patch_test.py
#
# Purpose:
#   Fast, safe patch-testing script for remaining SmokingStatus mismatches.
#   This does NOT overwrite the master file.
#
# What it does:
#   1) Loads the current merged/master file (the 231-match baseline)
#   2) Loads smoking mismatch categories
#   3) Restricts to selected mismatch categories only
#   4) Reconstructs all notes for those MRNs only
#   5) Applies focused structured smoking patch logic on full notes
#   6) Produces a row-level comparison file showing whether the patch:
#        - improves
#        - worsens
#        - changes but stays wrong
#        - no_change
#
# Recommended use:
#   Keep your 231-match updater as baseline.
#   Use this script to test ideas before editing update_bmi_smoking_only.py.
#
# Outputs:
#   _outputs/qa_smoking_patch_test_results.csv
#   _outputs/qa_smoking_patch_test_summary.csv
#
# Python 3.6.8 compatible

import os
import re
from glob import glob
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
MERGE_KEY = "MRN"

MASTER_FILE = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
MISMATCH_FILE = "{0}/_outputs/qa_smoking_mismatches_categorized.csv".format(BASE_DIR)

OUT_RESULTS = "{0}/_outputs/qa_smoking_patch_test_results.csv".format(BASE_DIR)
OUT_SUMMARY = "{0}/_outputs/qa_smoking_patch_test_summary.csv".format(BASE_DIR)

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

# Restrict patch testing to the categories you care about.
TARGET_MISMATCH_CATEGORIES = set([
    "no_evidence_row",
    "recent_quit_or_current_narrative_missed",
    "never_misread_as_former",
    "former_misread_as_current",
    "former_template_misread_as_current",
])

# -----------------------
# Utilities
# -----------------------
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
    if s.lower() in {"", "nan", "none", "null", "na"}:
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
        raise RuntimeError("MRN column not found. Columns seen: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df

def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(
            options, list(df.columns)[:60]
        ))
    return None

def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def parse_date_safe(x):
    s = clean_cell(x)
    if not s:
        return None
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%d-%b-%Y %H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None

def safe_float(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days

def note_type_bucket(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())

    if "brief op" in s:
        return "brief_op"
    if "operative" in s or "operation" in s or "op note" in s or "oper report" in s:
        return "operation"
    if "anesthesia" in s:
        return "anesthesia"
    if "pre-op" in s or "preop" in s:
        return "preop"
    if "h&p" in s or "history and physical" in s:
        return "hp"
    if "progress" in s:
        return "progress"
    if "clinic" in s or "office" in s:
        return "clinic"
    if "consult" in s:
        return "consult"
    return "other"

def note_on_or_before_recon(note_dt, recon_dt):
    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return False
    return dd <= 0

def text_window(text, start, end, width=180):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\n", " ").replace("\r", " ").strip()

# -----------------------
# Load note data
# -----------------------
def load_and_reconstruct_notes(target_mrns):
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))

    if not note_files:
        raise FileNotFoundError("No HPI11526 * Notes.csv files found via NOTE_GLOBS.")

    all_notes_rows = []

    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)

        df = df[df[MERGE_KEY].astype(str).str.strip().isin(target_mrns)].copy()
        if len(df) == 0:
            continue

        note_text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(
            df,
            ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"],
            required=False
        )

        df[note_text_col] = df[note_text_col].fillna("").astype(str)
        df[note_id_col] = df[note_id_col].fillna("").astype(str)
        if line_col:
            df[line_col] = df[line_col].fillna("").astype(str)
        if note_type_col:
            df[note_type_col] = df[note_type_col].fillna("").astype(str)
        if date_col:
            df[date_col] = df[date_col].fillna("").astype(str)

        df["_SOURCE_FILE_"] = os.path.basename(fp)

        keep_cols = [MERGE_KEY, note_id_col, note_text_col, "_SOURCE_FILE_"]
        if line_col:
            keep_cols.append(line_col)
        if note_type_col:
            keep_cols.append(note_type_col)
        if date_col:
            keep_cols.append(date_col)

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={
            note_id_col: "NOTE_ID",
            note_text_col: "NOTE_TEXT",
        })

        if line_col and line_col != "LINE":
            tmp = tmp.rename(columns={line_col: "LINE"})
        if note_type_col and note_type_col != "NOTE_TYPE":
            tmp = tmp.rename(columns={note_type_col: "NOTE_TYPE"})
        if date_col and date_col != "NOTE_DATE_OF_SERVICE":
            tmp = tmp.rename(columns={date_col: "NOTE_DATE_OF_SERVICE"})

        if "LINE" not in tmp.columns:
            tmp["LINE"] = ""
        if "NOTE_TYPE" not in tmp.columns:
            tmp["NOTE_TYPE"] = ""
        if "NOTE_DATE_OF_SERVICE" not in tmp.columns:
            tmp["NOTE_DATE_OF_SERVICE"] = ""

        all_notes_rows.append(tmp)

    if not all_notes_rows:
        return pd.DataFrame(columns=[MERGE_KEY, "NOTE_ID", "NOTE_TYPE", "NOTE_DATE", "SOURCE_FILE", "NOTE_TEXT"])

    notes_raw = pd.concat(all_notes_rows, ignore_index=True)

    def join_note(group):
        tmp = group.copy()
        tmp["_LINE_NUM_"] = tmp["LINE"].apply(to_int_safe)
        tmp = tmp.sort_values(by=["_LINE_NUM_"], na_position="last")
        return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()

    reconstructed = []
    grouped = notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False)

    for (mrn, nid), g in grouped:
        mrn = str(mrn).strip()
        nid = str(nid).strip()
        if not nid:
            continue

        full_text = join_note(g)
        if not full_text:
            continue

        if g["NOTE_TYPE"].astype(str).str.strip().any():
            note_type = g["NOTE_TYPE"].astype(str).iloc[0]
        else:
            note_type = g["_SOURCE_FILE_"].astype(str).iloc[0]

        if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any():
            note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0]
        else:
            note_date = ""

        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })

    return pd.DataFrame(reconstructed)

# -----------------------
# Focused patch logic
# -----------------------
STRUCT_CURRENT = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(current every day smoker|current some day smoker|current smoker|current)\b",
    re.IGNORECASE
)
STRUCT_FORMER = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(former smoker|former)\b",
    re.IGNORECASE
)
STRUCT_NEVER = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(never smoker|never)\b",
    re.IGNORECASE
)
SMOKELESS_NEVER = re.compile(
    r"\bsmokeless tobacco\s*[:\-]?\s*never used\b",
    re.IGNORECASE
)
STRUCT_NOT_ON_FILE = re.compile(
    r"\bsmoking status\s*[:\-]?\s*not on file\b",
    re.IGNORECASE
)
COMMENT_CURRENT = re.compile(
    r"\bcomment\s*[:\-]?\s*(?:states?\s+)?(?:she|he|pt|patient)\s+smokes?\b",
    re.IGNORECASE
)

PACKS_DAY = re.compile(
    r"\bpacks?/day\s*[:\-]?\s*[0-9]+(?:\.[0-9]+)?\b",
    re.IGNORECASE
)
PACK_YEARS = re.compile(
    r"\b[0-9]+(?:\.\d+)?\s*pack[- ]years?\b",
    re.IGNORECASE
)
TYPES_CIG = re.compile(
    r"\btypes?\s*:\s*cigarettes\b",
    re.IGNORECASE
)

QUIT_DATE = re.compile(
    r"\bquit date\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)
LAST_ATTEMPT = re.compile(
    r"\blast attempt to quit\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)
YEARS_SINCE = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.\d+)?)\b",
    re.IGNORECASE
)

NEG_NEVER_PATTERNS = [
    re.compile(r"\bdoes not smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco use\b", re.IGNORECASE),
    re.compile(r"\bdenies use of tobacco products\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bnever smoker\b", re.IGNORECASE),
    re.compile(r"\bnever smoked\b", re.IGNORECASE),
    re.compile(r"\bnonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]smoker\b", re.IGNORECASE),
]

COUNSELING_ONLY = re.compile(
    r"\b(avoid tobacco use|avoid smoking|encouraged to avoid tobacco use|counseled to avoid tobacco use)\b",
    re.IGNORECASE
)

def parse_quit_date(raw):
    s = clean_cell(raw)
    if not s:
        return None

    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass

    m = re.match(r"^([0-9]{1,2})/([0-9]{4})$", s)
    if m:
        try:
            return datetime(int(m.group(2)), int(m.group(1)), 1)
        except Exception:
            pass

    m = re.match(r"^((?:19|20)[0-9]{2})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), 1, 1)
        except Exception:
            pass

    return None

def smoking_value_priority(val):
    v = clean_cell(val)
    if v == "Current":
        return 0
    if v == "Former":
        return 1
    if v == "Never":
        return 2
    return 9

def make_candidate(row, value, rule_name, confidence, start, end):
    note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
    return {
        MERGE_KEY: clean_cell(row.get(MERGE_KEY, "")),
        "NOTE_ID": clean_cell(row.get("NOTE_ID", "")),
        "NOTE_DATE": clean_cell(row.get("NOTE_DATE", "")),
        "NOTE_TYPE": clean_cell(row.get("NOTE_TYPE", "")),
        "SOURCE_FILE": clean_cell(row.get("SOURCE_FILE", "")),
        "VALUE": value,
        "RULE": rule_name,
        "CONFIDENCE": confidence,
        "MATCH_START": start,
        "EVIDENCE": text_window(clean_cell(row.get("NOTE_TEXT", "")), start, end, 180),
        "_NOTE_DT": note_dt,
        "_BUCKET": note_type_bucket(row.get("NOTE_TYPE", ""), row.get("SOURCE_FILE", "")),
    }

def candidate_rank(cand):
    rule = clean_cell(cand.get("RULE", ""))
    note_dt = cand.get("_NOTE_DT")
    bucket = clean_cell(cand.get("_BUCKET", ""))
    value = clean_cell(cand.get("VALUE", ""))
    conf = safe_float(cand.get("CONFIDENCE", 0.0), 0.0)

    if rule == "structured_current":
        rule_pri = 0
    elif rule == "recent_quit_current":
        rule_pri = 1
    elif rule == "comment_current":
        rule_pri = 2
    elif rule == "quantified_current":
        rule_pri = 3
    elif rule == "structured_former":
        rule_pri = 4
    elif rule == "former_from_quit":
        rule_pri = 5
    elif rule == "structured_never":
        rule_pri = 6
    elif rule == "narrative_never":
        rule_pri = 7
    else:
        rule_pri = 99

    if bucket == "hp":
        bucket_pri = 0
    elif bucket == "preop":
        bucket_pri = 1
    elif bucket == "clinic":
        bucket_pri = 2
    elif bucket == "progress":
        bucket_pri = 3
    else:
        bucket_pri = 9

    dt_key = note_dt if note_dt is not None else datetime(1900, 1, 1)

    return (
        rule_pri,
        smoking_value_priority(value),
        bucket_pri,
        -conf,
        dt_key
    )

def extract_patch_candidates_from_note(row):
    text = clean_cell(row.get("NOTE_TEXT", ""))
    if not text:
        return []

    note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
    cands = []

    m_not_on_file = STRUCT_NOT_ON_FILE.search(text)

    m = STRUCT_CURRENT.search(text)
    if m is not None:
        cands.append(make_candidate(row, "Current", "structured_current", 1.000, m.start(), m.end()))

    m = STRUCT_FORMER.search(text)
    if m is not None:
        qd = QUIT_DATE.search(text)
        la = LAST_ATTEMPT.search(text)
        ys = YEARS_SINCE.search(text)

        if qd is not None:
            qd_dt = parse_quit_date(qd.group(1))
            if note_dt is not None and qd_dt is not None:
                dd = days_between(note_dt, qd_dt)
                if dd is not None and dd >= 0 and dd <= 90:
                    cands.append(make_candidate(row, "Current", "recent_quit_current", 0.999, qd.start(), qd.end()))
                else:
                    cands.append(make_candidate(row, "Former", "structured_former", 0.999, m.start(), m.end()))
            else:
                cands.append(make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))
        elif la is not None:
            la_dt = parse_quit_date(la.group(1))
            if note_dt is not None and la_dt is not None:
                dd = days_between(note_dt, la_dt)
                if dd is not None and dd >= 0 and dd <= 90:
                    cands.append(make_candidate(row, "Current", "recent_quit_current", 0.999, la.start(), la.end()))
                else:
                    cands.append(make_candidate(row, "Former", "structured_former", 0.999, m.start(), m.end()))
            else:
                cands.append(make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))
        elif ys is not None:
            yrs = safe_float(ys.group(1), default=999.0)
            if yrs < 0.25:
                cands.append(make_candidate(row, "Current", "recent_quit_current", 0.998, ys.start(), ys.end()))
            else:
                cands.append(make_candidate(row, "Former", "structured_former", 0.998, m.start(), m.end()))
        else:
            cands.append(make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))

    m = STRUCT_NEVER.search(text)
    if m is not None:
        cands.append(make_candidate(row, "Never", "structured_never", 0.997, m.start(), m.end()))

    # Only helpful when paired with smoking-status never; never use alone.
    m_smokeless = SMOKELESS_NEVER.search(text)
    if m_smokeless is not None and STRUCT_NEVER.search(text) is not None:
        cands.append(make_candidate(row, "Never", "structured_never", 0.996, m_smokeless.start(), m_smokeless.end()))

    m = COMMENT_CURRENT.search(text)
    if m is not None:
        cands.append(make_candidate(row, "Current", "comment_current", 0.995, m.start(), m.end()))

    if STRUCT_CURRENT.search(text) is None and STRUCT_FORMER.search(text) is None and STRUCT_NEVER.search(text) is None:
        m_pack = PACKS_DAY.search(text)
        m_types = TYPES_CIG.search(text)
        if m_pack is not None and m_types is not None:
            cands.append(make_candidate(row, "Current", "quantified_current", 0.992, m_pack.start(), m_pack.end()))

    # Narrative Never only when not just counseling and not contradicted by stronger structured current/former.
    if COUNSELING_ONLY.search(text) is None and m_not_on_file is None:
        for rx in NEG_NEVER_PATTERNS:
            mm = rx.search(text)
            if mm is not None:
                cands.append(make_candidate(row, "Never", "narrative_never", 0.985, mm.start(), mm.end()))
                break

    return cands

def choose_best_patch_for_patient(patient_notes, recon_dt):
    all_cands = []

    for _, row in patient_notes.iterrows():
        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
        if note_dt is None:
            continue
        if not note_on_or_before_recon(note_dt, recon_dt):
            continue

        cands = extract_patch_candidates_from_note(row)
        if cands:
            all_cands.extend(cands)

    if not all_cands:
        return None, 0

    all_cands = sorted(all_cands, key=candidate_rank)
    return all_cands[0], len(all_cands)

# -----------------------
# Main
# -----------------------
def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)

    if "SmokingStatus" not in master.columns:
        raise RuntimeError("SmokingStatus column not found in master.")

    print("Master rows: {0}".format(len(master)))

    print("Loading mismatch file...")
    mm = clean_cols(read_csv_robust(MISMATCH_FILE))
    mm = normalize_mrn(mm)

    required_mm_cols = ["Gold", "Pred", "Mismatch_Category"]
    for c in required_mm_cols:
        if c not in mm.columns:
            raise RuntimeError("Required mismatch column missing: {0}".format(c))

    target = mm[mm["Mismatch_Category"].astype(str).str.strip().isin(TARGET_MISMATCH_CATEGORIES)].copy()
    target_mrns = sorted(set(target[MERGE_KEY].astype(str).str.strip().tolist()))

    print("Target mismatch rows: {0}".format(len(target)))
    print("Target MRNs: {0}".format(len(target_mrns)))

    if len(target_mrns) == 0:
        print("No target MRNs found. Exiting.")
        return

    # Pull reconstruction date from mismatch file if present; if not, try common alternatives.
        # Get reconstruction / anchor date from mismatch file if present,
    # otherwise pull it from master.
    recon_col = None
    for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE"]:
        if c in target.columns:
            recon_col = c
            break

    if recon_col is None:
        master_recon_col = None
        for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE"]:
            if c in master.columns:
                master_recon_col = c
                break

        if master_recon_col is None:
            raise RuntimeError(
                "Could not find reconstruction date column in mismatch file or master file."
            )

        target = target.merge(
            master[[MERGE_KEY, master_recon_col]].drop_duplicates(),
            on=MERGE_KEY,
            how="left"
        )
        recon_col = master_recon_col

    print("Using reconstruction date column: {0}".format(recon_col))

    print("Loading and reconstructing notes only for target MRNs...")
    notes_df = load_and_reconstruct_notes(set(target_mrns))
    print("Reconstructed notes for target MRNs: {0}".format(len(notes_df)))

    result_rows = []

    for _, row in target.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        gold = clean_cell(row.get("Gold", ""))
        baseline_pred = clean_cell(row.get("Pred", ""))

        recon_dt = parse_date_safe(row.get(recon_col, ""))
        if recon_dt is None:
            result_rows.append({
                "MRN": mrn,
                "Gold": gold,
                "Baseline_Pred": baseline_pred,
                "Patched_Pred": "",
                "Mismatch_Category": clean_cell(row.get("Mismatch_Category", "")),
                "Patch_Rule": "",
                "Patch_Confidence": "",
                "Patch_Note_Date": "",
                "Patch_Note_Type": "",
                "Patch_Note_ID": "",
                "Patch_Source_File": "",
                "Patch_Evidence": "",
                "Total_Patch_Candidates": 0,
                "Change_Type": "no_recon_date",
            })
            continue

        patient_notes = notes_df[notes_df[MERGE_KEY].astype(str).str.strip() == mrn].copy()
        best_patch, n_patch_cands = choose_best_patch_for_patient(patient_notes, recon_dt)

        if best_patch is None:
            patched_pred = baseline_pred
            change_type = "no_patch_candidate"
            patch_rule = ""
            patch_conf = ""
            patch_note_date = ""
            patch_note_type = ""
            patch_note_id = ""
            patch_source_file = ""
            patch_evidence = ""
        else:
            patched_pred = clean_cell(best_patch.get("VALUE", ""))
            patch_rule = clean_cell(best_patch.get("RULE", ""))
            patch_conf = best_patch.get("CONFIDENCE", "")
            patch_note_date = clean_cell(best_patch.get("NOTE_DATE", ""))
            patch_note_type = clean_cell(best_patch.get("NOTE_TYPE", ""))
            patch_note_id = clean_cell(best_patch.get("NOTE_ID", ""))
            patch_source_file = clean_cell(best_patch.get("SOURCE_FILE", ""))
            patch_evidence = clean_cell(best_patch.get("EVIDENCE", ""))

            baseline_correct = (baseline_pred == gold)
            patched_correct = (patched_pred == gold)

            if patched_pred == baseline_pred:
                change_type = "no_change"
            elif (not baseline_correct) and patched_correct:
                change_type = "improved"
            elif baseline_correct and (not patched_correct):
                change_type = "worsened"
            else:
                change_type = "changed_still_wrong"

        result_rows.append({
            "MRN": mrn,
            "Gold": gold,
            "Baseline_Pred": baseline_pred,
            "Patched_Pred": patched_pred,
            "Mismatch_Category": clean_cell(row.get("Mismatch_Category", "")),
            "Patch_Rule": patch_rule if best_patch is not None else "",
            "Patch_Confidence": patch_conf if best_patch is not None else "",
            "Patch_Note_Date": patch_note_date if best_patch is not None else "",
            "Patch_Note_Type": patch_note_type if best_patch is not None else "",
            "Patch_Note_ID": patch_note_id if best_patch is not None else "",
            "Patch_Source_File": patch_source_file if best_patch is not None else "",
            "Patch_Evidence": patch_evidence if best_patch is not None else "",
            "Total_Patch_Candidates": n_patch_cands,
            "Change_Type": change_type,
        })

    results_df = pd.DataFrame(result_rows)

    summary_rows = []

    if len(results_df) > 0:
        by_change = results_df["Change_Type"].value_counts(dropna=False)
        for k, v in by_change.items():
            summary_rows.append({
                "Summary_Type": "Change_Type",
                "Key": k,
                "Count": int(v),
            })

        by_rule = results_df[results_df["Patch_Rule"].astype(str).str.strip() != ""]["Patch_Rule"].value_counts(dropna=False)
        for k, v in by_rule.items():
            summary_rows.append({
                "Summary_Type": "Patch_Rule",
                "Key": k,
                "Count": int(v),
            })

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(OUT_RESULTS), exist_ok=True)
    results_df.to_csv(OUT_RESULTS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved results to: {0}".format(OUT_RESULTS))
    print("Saved summary to: {0}".format(OUT_SUMMARY))

    if len(results_df) > 0:
        print("\nChange type counts:")
        print(results_df["Change_Type"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()
