#!/usr/bin/env python3
# qa_smoking_targeted_patch_test.py
#
# Fast patch-test script for smoking mismatches only.
# It evaluates targeted smoking fallback logic on the mismatch MRNs
# without rerunning the full build.
#
# Inputs:
#   _outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
#   _outputs/qa_smoking_mismatches_categorized.csv
#   note CSVs from clinic / inpatient / operation notes
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

def safe_float(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

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

def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days

def note_on_or_before_recon(note_dt, recon_dt):
    dd = days_between(note_dt, recon_dt)
    if dd is None:
        return False
    return dd <= 0


# -----------------------
# Notes reconstruction
# -----------------------
def load_and_reconstruct_notes():
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
# Focused targeted smoking logic
# -----------------------
BOX = u"\u25A1"

RX_STRUCT_CURRENT = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(current every day smoker|current some day smoker|current smoker|current)\b",
    re.IGNORECASE
)
RX_STRUCT_FORMER = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(former smoker|former)\b",
    re.IGNORECASE
)
RX_STRUCT_NEVER = re.compile(
    r"\bsmoking status\s*[:\-]?\s*(never smoker|never)\b",
    re.IGNORECASE
)
RX_STRUCT_SMOKELESS_NEVER = re.compile(
    r"\bsmokeless tobacco\s*[:\-]?\s*never used\b",
    re.IGNORECASE
)
RX_COMMENT_CURRENT = re.compile(
    r"\bcomment\s*[:\-]?\s*(?:states?\s+)?(?:she|he|pt|patient)\s+smokes?\b",
    re.IGNORECASE
)

RX_NEVER_TEXT = [
    re.compile(r"\bnever smoker\b", re.IGNORECASE),
    re.compile(r"\bnever smoked\b", re.IGNORECASE),
    re.compile(r"\bnonsmoker\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke\b", re.IGNORECASE),
    re.compile(r"\bdoesn't smoke\b", re.IGNORECASE),
    re.compile(r"\bdoes not smoke or use nicotine\b", re.IGNORECASE),
    re.compile(r"\bdenies tobacco use\b", re.IGNORECASE),
    re.compile(r"\bdenies use of tobacco products\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco\b", re.IGNORECASE),
    re.compile(r"\bno history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bno smoking\b", re.IGNORECASE),
]

RX_FORMER_TEXT = [
    re.compile(r"\bformer smoker\b", re.IGNORECASE),
    re.compile(r"\bex[- ]smoker\b", re.IGNORECASE),
    re.compile(r"\bquit as a teenager\b", re.IGNORECASE),
    re.compile(r"\bremote history of tobacco use\b", re.IGNORECASE),
    re.compile(r"\bformer smoker who quit in [A-Za-z]+\s+(?:19|20)\d{2}\b", re.IGNORECASE),
    re.compile(r"\bquit in [A-Za-z]+\s+(?:19|20)\d{2}\b", re.IGNORECASE),
]

RX_CURRENT_TEXT = [
    re.compile(r"\bcurrent every day smoker\b", re.IGNORECASE),
    re.compile(r"\bcurrent some day smoker\b", re.IGNORECASE),
    re.compile(r"\bcurrent smoker\b", re.IGNORECASE),
    re.compile(r"\bevery day smoker\b", re.IGNORECASE),
    re.compile(r"\bsome day smoker\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+every\s+once\s+in\s+a\s+while\b", re.IGNORECASE),
    re.compile(r"\bsmokes?\s+every\s+once\s+in\s+a\s+while\s+currently\b", re.IGNORECASE),
]

RX_QUIT_DATE = re.compile(
    r"\bquit date\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)
RX_LAST_ATTEMPT = re.compile(
    r"\blast attempt to quit\s*[:\-]?\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}|[0-9]{1,2}/[0-9]{4}|(?:19|20)[0-9]{2})\b",
    re.IGNORECASE
)
RX_YEARS_SINCE = re.compile(
    r"\byears?\s+since\s+quitting\s*[:\-]?\s*([0-9]+(?:\.\d+)?)\b",
    re.IGNORECASE
)
RX_QUIT_YEARS_AGO = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+years?\s+ago\b",
    re.IGNORECASE
)
RX_QUIT_MONTHS_AGO = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+months?\s+ago\b",
    re.IGNORECASE
)
RX_QUIT_WEEKS_AGO = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+weeks?\s+ago\b",
    re.IGNORECASE
)
RX_QUIT_DAYS_AGO = re.compile(
    r"\b(?:quit|stopped)\s+(?:smoking|tobacco)\s+(?:about\s+|approximately\s+|approx\.?\s*)?([0-9]+(?:\.\d+)?)\s+days?\s+ago\b",
    re.IGNORECASE
)

RX_PACKS_DAY = re.compile(
    r"\bpacks?/day\s*[:\-]?\s*[0-9]+(?:\.[0-9]+)?\b",
    re.IGNORECASE
)
RX_TYPES_CIG = re.compile(
    r"\btypes?\s*:\s*cigarettes\b",
    re.IGNORECASE
)

RX_COUNSELING_ONLY = re.compile(
    r"\b(avoid tobacco use|avoid smoking|encouraged to avoid tobacco use|counseled to avoid tobacco use)\b",
    re.IGNORECASE
)

RX_NEGATED_CURRENT = re.compile(
    r"\b(not currently smoking|no current tobacco use|not smoking currently)\b",
    re.IGNORECASE
)


def _find_first(rx, text):
    try:
        return rx.search(text)
    except Exception:
        return None

def _parse_quit_date(raw):
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

def _context(text, start, end, width=180):
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\n", " ").replace("\r", " ").strip()

def _make_candidate(row, value, rule_name, confidence, start, end):
    return {
        "MRN": clean_cell(row.get(MERGE_KEY, "")),
        "NOTE_ID": clean_cell(row.get("NOTE_ID", "")),
        "NOTE_DATE": clean_cell(row.get("NOTE_DATE", "")),
        "NOTE_TYPE": clean_cell(row.get("NOTE_TYPE", "")),
        "SOURCE_FILE": clean_cell(row.get("SOURCE_FILE", "")),
        "VALUE": value,
        "RULE_NAME": rule_name,
        "CONFIDENCE": confidence,
        "MATCH_START": start,
        "EVIDENCE": _context(clean_cell(row.get("NOTE_TEXT", "")), start, end, 180)
    }

def candidate_priority(c):
    rule = clean_cell(c.get("RULE_NAME", ""))
    val = clean_cell(c.get("VALUE", ""))

    if rule == "structured_current":
        rp = 0
    elif rule == "recent_quit_current":
        rp = 1
    elif rule == "comment_current":
        rp = 2
    elif rule == "current_text":
        rp = 3
    elif rule == "quantified_current":
        rp = 4
    elif rule == "structured_former":
        rp = 5
    elif rule == "former_text":
        rp = 6
    elif rule == "structured_never":
        rp = 7
    elif rule == "never_text":
        rp = 8
    else:
        rp = 99

    if val == "Current":
        vp = 0
    elif val == "Former":
        vp = 1
    elif val == "Never":
        vp = 2
    else:
        vp = 9

    return (rp, vp, -safe_float(c.get("CONFIDENCE", 0.0), 0.0))

def extract_candidates_from_note(row):
    text = clean_cell(row.get("NOTE_TEXT", ""))
    note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
    out = []

    if not text:
        return out

    m = _find_first(RX_STRUCT_CURRENT, text)
    if m is not None:
        ctx = _context(text, m.start(), m.end(), 120)
        if RX_NEGATED_CURRENT.search(ctx) is None:
            out.append(_make_candidate(row, "Current", "structured_current", 1.000, m.start(), m.end()))

    m = _find_first(RX_COMMENT_CURRENT, text)
    if m is not None:
        out.append(_make_candidate(row, "Current", "comment_current", 0.997, m.start(), m.end()))

    m = _find_first(RX_STRUCT_FORMER, text)
    if m is not None:
        quit_m = _find_first(RX_QUIT_DATE, text)
        last_m = _find_first(RX_LAST_ATTEMPT, text)
        years_m = _find_first(RX_YEARS_SINCE, text)

        if quit_m is not None:
            quit_dt = _parse_quit_date(quit_m.group(1))
            if note_dt is not None and quit_dt is not None:
                dd = days_between(note_dt, quit_dt)
                if dd is not None and dd >= 0 and dd <= 90:
                    out.append(_make_candidate(row, "Current", "recent_quit_current", 0.999, quit_m.start(), quit_m.end()))
                else:
                    out.append(_make_candidate(row, "Former", "structured_former", 0.998, m.start(), m.end()))
            else:
                out.append(_make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))
        elif last_m is not None:
            quit_dt = _parse_quit_date(last_m.group(1))
            if note_dt is not None and quit_dt is not None:
                dd = days_between(note_dt, quit_dt)
                if dd is not None and dd >= 0 and dd <= 90:
                    out.append(_make_candidate(row, "Current", "recent_quit_current", 0.999, last_m.start(), last_m.end()))
                else:
                    out.append(_make_candidate(row, "Former", "structured_former", 0.998, m.start(), m.end()))
            else:
                out.append(_make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))
        elif years_m is not None:
            yrs = safe_float(years_m.group(1), 999.0)
            if yrs < 0.25:
                out.append(_make_candidate(row, "Current", "recent_quit_current", 0.998, years_m.start(), years_m.end()))
            else:
                out.append(_make_candidate(row, "Former", "structured_former", 0.997, m.start(), m.end()))
        else:
            out.append(_make_candidate(row, "Former", "structured_former", 0.996, m.start(), m.end()))

    m = _find_first(RX_STRUCT_NEVER, text)
    if m is not None:
        out.append(_make_candidate(row, "Never", "structured_never", 0.997, m.start(), m.end()))

    m_smokeless = _find_first(RX_STRUCT_SMOKELESS_NEVER, text)
    if m_smokeless is not None and _find_first(RX_STRUCT_NEVER, text) is not None:
        out.append(_make_candidate(row, "Never", "structured_never", 0.996, m_smokeless.start(), m_smokeless.end()))

    found_current_text = False
    for rx in RX_CURRENT_TEXT:
        m = _find_first(rx, text)
        if m is not None:
            out.append(_make_candidate(row, "Current", "current_text", 0.993, m.start(), m.end()))
            found_current_text = True
            break

    if (not found_current_text) and _find_first(RX_STRUCT_CURRENT, text) is None:
        pack_m = _find_first(RX_PACKS_DAY, text)
        type_m = _find_first(RX_TYPES_CIG, text)
        if pack_m is not None and type_m is not None and _find_first(RX_STRUCT_FORMER, text) is None and _find_first(RX_STRUCT_NEVER, text) is None:
            out.append(_make_candidate(row, "Current", "quantified_current", 0.990, pack_m.start(), pack_m.end()))

    for rx in RX_FORMER_TEXT:
        m = _find_first(rx, text)
        if m is not None:
            out.append(_make_candidate(row, "Former", "former_text", 0.991, m.start(), m.end()))
            break

    for rx in [RX_QUIT_YEARS_AGO, RX_QUIT_MONTHS_AGO, RX_QUIT_WEEKS_AGO, RX_QUIT_DAYS_AGO]:
        m = _find_first(rx, text)
        if m is None:
            continue

        if rx == RX_QUIT_YEARS_AGO:
            out.append(_make_candidate(row, "Former", "former_text", 0.992, m.start(), m.end()))
        elif rx == RX_QUIT_MONTHS_AGO:
            months = safe_float(m.group(1), 999.0)
            if months <= 3.0:
                out.append(_make_candidate(row, "Current", "recent_quit_current", 0.995, m.start(), m.end()))
            else:
                out.append(_make_candidate(row, "Former", "former_text", 0.992, m.start(), m.end()))
        elif rx == RX_QUIT_WEEKS_AGO:
            weeks = safe_float(m.group(1), 999.0)
            if weeks <= 12.0:
                out.append(_make_candidate(row, "Current", "recent_quit_current", 0.995, m.start(), m.end()))
            else:
                out.append(_make_candidate(row, "Former", "former_text", 0.992, m.start(), m.end()))
        else:
            days = safe_float(m.group(1), 999.0)
            if days <= 90.0:
                out.append(_make_candidate(row, "Current", "recent_quit_current", 0.995, m.start(), m.end()))
            else:
                out.append(_make_candidate(row, "Former", "former_text", 0.992, m.start(), m.end()))
        break

    if RX_COUNSELING_ONLY.search(text) is None:
        for rx in RX_NEVER_TEXT:
            m = _find_first(rx, text)
            if m is not None:
                out.append(_make_candidate(row, "Never", "never_text", 0.986, m.start(), m.end()))
                break

    return out

def choose_best_patient_candidate(existing, new):
    if existing is None:
        return new
    if candidate_priority(new) < candidate_priority(existing):
        return new
    return existing


# -----------------------
# Main
# -----------------------
def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)
    print("Master rows: {0}".format(len(master)))

    print("Loading mismatch file...")
    mism = clean_cols(read_csv_robust(MISMATCH_FILE))
    mism = normalize_mrn(mism)
    print("Target mismatch rows: {0}".format(len(mism)))

    if "Gold" not in mism.columns or "Pred" not in mism.columns:
        raise RuntimeError("Mismatch file must contain Gold and Pred columns.")

    target_mrns = sorted(set(mism[MERGE_KEY].astype(str).str.strip().tolist()))
    print("Target MRNs: {0}".format(len(target_mrns)))

    # Get anchor / reconstruction date from mismatch file if present,
    # otherwise from master.
    recon_col = None
    for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE", "Reconstruction_Date", "RECON_DATE"]:
        if c in mism.columns:
            recon_col = c
            break

    if recon_col is None:
        master_recon_col = None
        for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE", "Reconstruction_Date", "RECON_DATE"]:
            if c in master.columns:
                master_recon_col = c
                break

        if master_recon_col is None:
            print("Master columns:")
            print(list(master.columns))
            raise RuntimeError("Could not find reconstruction date column in mismatch file or master file.")

        mism = mism.merge(
            master[[MERGE_KEY, master_recon_col]].drop_duplicates(),
            on=MERGE_KEY,
            how="left"
        )
        recon_col = master_recon_col

    print("Using reconstruction date column: {0}".format(recon_col))

    print("Loading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    notes_df = notes_df[notes_df[MERGE_KEY].astype(str).str.strip().isin(target_mrns)].copy()
    print("Notes for target MRNs: {0}".format(len(notes_df)))

    result_rows = []

    for _, mm in mism.iterrows():
        mrn = clean_cell(mm.get(MERGE_KEY, ""))
        gold = clean_cell(mm.get("Gold", ""))
        pred = clean_cell(mm.get("Pred", ""))
        mismatch_category = clean_cell(mm.get("Mismatch_Category", ""))
        recon_raw = clean_cell(mm.get(recon_col, ""))
        recon_dt = parse_date_safe(recon_raw)

        patient_notes = notes_df[notes_df[MERGE_KEY].astype(str).str.strip() == mrn].copy()

        best_candidate = None
        candidate_count = 0
        notes_scanned = 0

        if recon_dt is not None and len(patient_notes) > 0:
            for _, row in patient_notes.iterrows():
                note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
                if note_dt is None:
                    continue
                if not note_on_or_before_recon(note_dt, recon_dt):
                    continue

                notes_scanned += 1
                cands = extract_candidates_from_note(row)
                if not cands:
                    continue

                for c in cands:
                    candidate_count += 1
                    best_candidate = choose_best_patient_candidate(best_candidate, c)

        new_pred = pred
        best_rule = ""
        best_conf = ""
        best_note_id = ""
        best_note_date = ""
        best_note_type = ""
        best_source_file = ""
        best_evidence = ""
        change_type = "no_change"

        if best_candidate is not None:
            new_pred = clean_cell(best_candidate.get("VALUE", ""))
            best_rule = clean_cell(best_candidate.get("RULE_NAME", ""))
            best_conf = best_candidate.get("CONFIDENCE", "")
            best_note_id = clean_cell(best_candidate.get("NOTE_ID", ""))
            best_note_date = clean_cell(best_candidate.get("NOTE_DATE", ""))
            best_note_type = clean_cell(best_candidate.get("NOTE_TYPE", ""))
            best_source_file = clean_cell(best_candidate.get("SOURCE_FILE", ""))
            best_evidence = clean_cell(best_candidate.get("EVIDENCE", ""))

            was_correct = (pred == gold)
            now_correct = (new_pred == gold)

            if (not was_correct) and now_correct:
                change_type = "improved"
            elif was_correct and (not now_correct):
                change_type = "worsened"
            elif (not was_correct) and (not now_correct) and (new_pred != pred):
                change_type = "changed_but_still_wrong"
            else:
                change_type = "no_change"

        result_rows.append({
            "MRN": mrn,
            "Gold": gold,
            "Original_Pred": pred,
            "Patched_Pred": new_pred,
            "Mismatch_Category": mismatch_category,
            "Recon_Date": recon_raw,
            "Notes_Scanned_PreRecon": notes_scanned,
            "Candidate_Count": candidate_count,
            "Best_Rule": best_rule,
            "Best_Confidence": best_conf,
            "Best_NOTE_ID": best_note_id,
            "Best_NOTE_DATE": best_note_date,
            "Best_NOTE_TYPE": best_note_type,
            "Best_SOURCE_FILE": best_source_file,
            "Best_Evidence": best_evidence,
            "Change_Type": change_type
        })

    results_df = pd.DataFrame(result_rows)

    if len(results_df) > 0:
        orig_correct = int((results_df["Original_Pred"] == results_df["Gold"]).sum())
        patched_correct = int((results_df["Patched_Pred"] == results_df["Gold"]).sum())
        delta = patched_correct - orig_correct

        summary_rows = [{
            "Rows_Evaluated": len(results_df),
            "Original_Correct": orig_correct,
            "Patched_Correct": patched_correct,
            "Net_Change": delta,
            "Improved": int((results_df["Change_Type"] == "improved").sum()),
            "Worsened": int((results_df["Change_Type"] == "worsened").sum()),
            "Changed_But_Still_Wrong": int((results_df["Change_Type"] == "changed_but_still_wrong").sum()),
            "No_Change": int((results_df["Change_Type"] == "no_change").sum())
        }]
    else:
        summary_rows = [{
            "Rows_Evaluated": 0,
            "Original_Correct": 0,
            "Patched_Correct": 0,
            "Net_Change": 0,
            "Improved": 0,
            "Worsened": 0,
            "Changed_But_Still_Wrong": 0,
            "No_Change": 0
        }]

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(OUT_RESULTS), exist_ok=True)
    results_df.to_csv(OUT_RESULTS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(" {0}".format(OUT_RESULTS))
    print(" {0}".format(OUT_SUMMARY))

    if len(summary_df) > 0:
        print("\nPatch summary:")
        print(summary_df.to_string(index=False))

    if len(results_df) > 0:
        print("\nChange_Type counts:")
        print(results_df["Change_Type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
