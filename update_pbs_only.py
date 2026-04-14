#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_pbs_only.py

PBS-only updater for:
- PastBreastSurgery
- PBS_Lumpectomy
- PBS_Breast Reduction
- PBS_Mastopexy
- PBS_Augmentation
- PBS_Other

FIX (this version):
- PBS_Breast Reduction, PBS_Mastopexy, PBS_Augmentation, PBS_Other:
  laterality check skipped entirely on BOTH pre- and post-recon paths.
  These are past cosmetic/surgical history — laterality is irrelevant.
  Only gate: history_ok (and explicit contralateral rejection still applies).
- PBS_Lumpectomy: laterality logic preserved completely unchanged.

Outputs:
1) /home/apokol/Breast_Restore/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv
2) /home/apokol/Breast_Restore/_outputs/pbs_only_evidence.csv

Python 3.6.8 compatible.
"""

import os
import re
from glob import glob
from datetime import datetime

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

MASTER_FILE   = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
OUTPUT_EVID   = "{0}/_outputs/pbs_only_evidence.csv".format(BASE_DIR)

MERGE_KEY = "MRN"

STRUCT_GLOBS = [
    "{0}/**/HPI11526*Clinic Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient encounters.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation encounters.csv".format(BASE_DIR),
]

NOTE_GLOBS = [
    "{0}/**/HPI11526*Clinic Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Inpatient Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*Operation Notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*clinic notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*inpatient notes.csv".format(BASE_DIR),
    "{0}/**/HPI11526*operation notes.csv".format(BASE_DIR),
]

from models import SectionedNote   # noqa: E402
from extractors.pbs import extract_pbs  # noqa: E402

PBS_FIELDS = [
    "PastBreastSurgery",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "PBS_Other",
]

# ============================================================
# Utilities
# ============================================================

def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1",
                               error_bad_lines=False, warn_bad_lines=True)


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def normalize_mrn(df):
    for k in ["MRN", "mrn", "Patient_MRN", "PAT_MRN", "PATIENT_MRN"]:
        if k in df.columns:
            if k != MERGE_KEY:
                df = df.rename(columns={k: MERGE_KEY})
            break
    if MERGE_KEY not in df.columns:
        raise RuntimeError("MRN column not found. Columns: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(
            options, list(df.columns)[:60]))
    return None


def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def parse_date_safe(x):
    s = clean_cell(x)
    if not s:
        return None
    fmts = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%Y %H:%M:%S",
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


# ============================================================
# Regex
# ============================================================

LEFT_RX  = re.compile(r"\b(left|lt)\b|\bleft\s+breast\b|\bleft[- ]sided\b|\(left\)|\(lt\)", re.I)
RIGHT_RX = re.compile(r"\b(right|rt)\b|\bright\s+breast\b|\bright[- ]sided\b|\(right\)|\(rt\)", re.I)
BILAT_RX = re.compile(r"\b(bilateral|bilat|both\s+breasts?)\b", re.I)

HISTORY_CUE_RX = re.compile(
    r"\b(s/p|status\s+post|history\s+of|with\s+a\s+history\s+of|prior|previous|"
    r"remote|previously|underwent|treated\s+with)\b", re.I)

NEGATIVE_HISTORY_RX = re.compile(
    r"\b(no\s+prior\s+breast\s+surgery|no\s+history\s+of\s+breast\s+surgery|"
    r"denies\s+prior\s+breast\s+surgery|never\s+had\s+breast\s+surgery)\b", re.I)

CANCER_CONTEXT_RX = re.compile(
    r"\b(ductal\s+carcinoma|lobular\s+carcinoma|dcis|invasive\s+ductal|breast\s+cancer|"
    r"sentinel\s+lymph\s+node|slnb|alnd|radiation|chemo|xrt)\b", re.I)

YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b", re.I)

AUGMENT_NEGATIVE_CONTEXT_RX = re.compile(
    r"\b(reconstruction|implant[- ]based\s+reconstruction|tissue\s+expander|expander|"
    r"implant\s+exchange|exchange\s+of\s+(?:the\s+)?(?:tissue\s+expanders?|implants?)|"
    r"permanent\s+(?:silicone|saline)\s+breast\s+implants?|breast\s+implant\s+reconstruction|"
    r"post[- ]mastectomy|mastectomy)\b", re.I)

AUGMENT_POSITIVE_CONTEXT_RX = re.compile(
    r"\b(cosmetic|augmentation|history\s+of|prior|previous|previously|s/p|"
    r"submuscular|saline|silicone|(?:19|20)\d{2})\b", re.I)

LUMPECTOMY_STRICT_FP_FILTER_RX = re.compile(
    r"\b(candidate\s+for\s+lumpectomy|not\s+(?:felt\s+to\s+be\s+)?a\s+lumpectomy\s+candidate|"
    r"lumpectomy\s+vs\.?\s+mastectomy|treatment\s+options?.{0,120}\blumpectomy\b|"
    r"discussion\s+of\s+lumpectomy|discussed\s+lumpectomy|recommend(?:ed)?\s+lumpectomy|"
    r"planned\s+lumpectomy|scheduled\s+for\s+lumpectomy)\b", re.I)

# ============================================================
# Laterality helpers
# ============================================================

def normalize_recon_laterality(x):
    s = clean_cell(x).lower()
    if not s:
        return ""
    if "bilat" in s or "bilateral" in s or "both" in s:
        return "bilateral"
    if "left" in s or s == "l":
        return "left"
    if "right" in s or s == "r":
        return "right"
    return ""


def extract_laterality_from_text(text):
    t = clean_cell(text)
    if not t:
        return ""
    has_b = BILAT_RX.search(t) is not None
    has_l = LEFT_RX.search(t) is not None
    has_r = RIGHT_RX.search(t) is not None
    if has_b or (has_l and has_r):
        return "bilateral"
    if has_l:
        return "left"
    if has_r:
        return "right"
    return ""


def infer_laterality_from_field_context(field, text):
    ctx = clean_cell(text)
    if not ctx:
        return ""
    direct = extract_laterality_from_text(ctx)
    if direct:
        return direct
    if field == "PBS_Lumpectomy":
        lc = re.search(r"\bleft\b.{0,100}\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy)\b", ctx, re.I)
        rc = re.search(r"\bright\b.{0,100}\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy)\b", ctx, re.I)
        lr = re.search(r"\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy)\b.{0,100}\bleft\b", ctx, re.I)
        rr = re.search(r"\b(?:breast\s+cancer|dcis|carcinoma|lumpectomy)\b.{0,100}\bright\b", ctx, re.I)
        if (lc or lr) and not (rc or rr):
            return "left"
        if (rc or rr) and not (lc or lr):
            return "right"
    return ""


def laterality_relation(recon_lat, proc_lat, context_text):
    recon_lat = normalize_recon_laterality(recon_lat)
    proc_lat  = normalize_recon_laterality(proc_lat)
    ctx = clean_cell(context_text).lower()
    if recon_lat == "bilateral":
        return "accept"
    if recon_lat in {"left", "right"}:
        if proc_lat == recon_lat:
            return "accept"
        if proc_lat == "bilateral":
            return "accept"
        if proc_lat in {"left", "right"} and proc_lat != recon_lat:
            return "reject_contralateral"
        if "contralateral" in ctx:
            return "reject_contralateral"
        return "unknown_unilateral"
    return "unknown_recon"

# ============================================================
# Context helpers
# ============================================================

def is_historical_context(text):
    return HISTORY_CUE_RX.search(clean_cell(text)) is not None


def has_negative_history(text):
    return NEGATIVE_HISTORY_RX.search(clean_cell(text)) is not None


def has_cancer_context(text):
    return CANCER_CONTEXT_RX.search(clean_cell(text)) is not None


def has_year_context(text):
    return YEAR_RX.search(clean_cell(text)) is not None


def has_strict_lumpectomy_fp_context(text):
    return LUMPECTOMY_STRICT_FP_FILTER_RX.search(clean_cell(text)) is not None


def augmentation_true_history_context(text):
    ctx = clean_cell(text)
    if not ctx:
        return False
    explicit = re.search(
        r"\b(breast\s+augmentation|augmentation\s+mammaplasty|cosmetic\s+augmentation|"
        r"breast\s+implants?\s+for\s+augmentation|previous\s+submuscular\s+(?:saline|silicone)"
        r"\s+breast\s+augmentation)\b", ctx, re.I) is not None
    if explicit:
        return True
    pos = AUGMENT_POSITIVE_CONTEXT_RX.search(ctx) is not None
    neg = AUGMENT_NEGATIVE_CONTEXT_RX.search(ctx) is not None
    return pos and not neg


def field_specific_history_ok(field, combined_context):
    ctx = clean_cell(combined_context)
    if field == "PBS_Lumpectomy":
        if is_historical_context(ctx): return True
        if has_cancer_context(ctx):    return True
        if has_year_context(ctx):      return True
        if re.search(r"\bunderwent\b", ctx, re.I): return True
        return False
    if field == "PBS_Breast Reduction":
        return is_historical_context(ctx)
    if field == "PBS_Mastopexy":
        return is_historical_context(ctx)
    if field == "PBS_Augmentation":
        return augmentation_true_history_context(ctx)
    if field == "PBS_Other":
        return is_historical_context(ctx)
    return False

# ============================================================
# Note type helpers
# ============================================================

def is_operation_note_type(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())
    return any(x in s for x in ["brief op", "op note", "operative", "operation", "oper report"])


def is_clinic_like_note(note_type, source_file):
    s = "{0} {1}".format(clean_cell(note_type).lower(), clean_cell(source_file).lower())
    return any(x in s for x in ["progress", "clinic", "office", "follow up",
                                  "follow-up", "pre-op", "preop", "consult",
                                  "h&p", "history and physical"])

# ============================================================
# Candidate ranking
# ============================================================

def stage_and_rank(note_type, source_file, note_dt, recon_dt, accepted_post_hist):
    dd        = days_between(note_dt, recon_dt)
    is_op     = is_operation_note_type(note_type, source_file)
    is_clinic = is_clinic_like_note(note_type, source_file)
    if dd is None:
        return (9, 9999, 9)
    if dd < 0:
        if is_op:     return (0, abs(dd), 0)
        if is_clinic: return (1, abs(dd), 1)
        return (2, abs(dd), 2)
    if dd >= 0 and accepted_post_hist:
        if is_op:     return (3, abs(dd), 0)
        if is_clinic: return (4, abs(dd), 1)
        return (5, abs(dd), 2)
    return (9, abs(dd), 9)


def candidate_score(c):
    conf     = float(getattr(c, "confidence", 0.0) or 0.0)
    nt       = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    dt_bonus = 0.01 if clean_cell(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + dt_bonus


def choose_better_pbs(existing, new, recon_dt):
    if existing is None:
        return new
    ex_rank = stage_and_rank(
        getattr(existing, "note_type", ""), getattr(existing, "_source_file", ""),
        parse_date_safe(getattr(existing, "note_date", "")), recon_dt,
        getattr(existing, "_accepted_post_hist", False))
    nw_rank = stage_and_rank(
        getattr(new, "note_type", ""), getattr(new, "_source_file", ""),
        parse_date_safe(getattr(new, "note_date", "")), recon_dt,
        getattr(new, "_accepted_post_hist", False))
    if nw_rank < ex_rank: return new
    if ex_rank < nw_rank: return existing
    return new if candidate_score(new) > candidate_score(existing) else existing

# ============================================================
# Note loading
# ============================================================

def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))
    if not note_files:
        raise FileNotFoundError("No HPI11526 Notes CSVs found.")

    all_rows = []
    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        id_col   = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(df, ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE",
                                  "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"], required=False)

        df[text_col] = df[text_col].fillna("").astype(str)
        df[id_col]   = df[id_col].fillna("").astype(str)
        if line_col: df[line_col] = df[line_col].fillna("").astype(str)
        if type_col: df[type_col] = df[type_col].fillna("").astype(str)
        if date_col: df[date_col] = df[date_col].fillna("").astype(str)
        df["_SOURCE_FILE_"] = os.path.basename(fp)

        keep = [MERGE_KEY, id_col, text_col, "_SOURCE_FILE_"]
        if line_col: keep.append(line_col)
        if type_col: keep.append(type_col)
        if date_col: keep.append(date_col)

        tmp = df[keep].copy().rename(columns={id_col: "NOTE_ID", text_col: "NOTE_TEXT"})
        if line_col and line_col != "LINE":       tmp = tmp.rename(columns={line_col: "LINE"})
        if type_col and type_col != "NOTE_TYPE":  tmp = tmp.rename(columns={type_col: "NOTE_TYPE"})
        if date_col and date_col != "NOTE_DATE_OF_SERVICE":
            tmp = tmp.rename(columns={date_col: "NOTE_DATE_OF_SERVICE"})
        for col in ["LINE", "NOTE_TYPE", "NOTE_DATE_OF_SERVICE"]:
            if col not in tmp.columns:
                tmp[col] = ""
        all_rows.append(tmp)

    notes_raw = pd.concat(all_rows, ignore_index=True)

    def join_note(group):
        tmp = group.copy()
        tmp["_LN_"] = tmp["LINE"].apply(to_int_safe)
        tmp = tmp.sort_values("_LN_", na_position="last")
        return "\n".join(tmp["NOTE_TEXT"].tolist()).strip()

    reconstructed = []
    for (mrn, nid), g in notes_raw.groupby([MERGE_KEY, "NOTE_ID"], dropna=False):
        mrn = str(mrn).strip(); nid = str(nid).strip()
        if not nid: continue
        full_text = join_note(g)
        if not full_text: continue
        note_type = g["NOTE_TYPE"].astype(str).iloc[0] if g["NOTE_TYPE"].astype(str).str.strip().any() else g["_SOURCE_FILE_"].astype(str).iloc[0]
        note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0] if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any() else ""
        reconstructed.append({
            MERGE_KEY: mrn, "NOTE_ID": nid, "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date, "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text
        })
    return pd.DataFrame(reconstructed)

# ============================================================
# Structured encounter loading + anchor
# ============================================================

def load_structured_encounters():
    rows = []
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))

    for fp in sorted(set(struct_files)):
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        sn = os.path.basename(fp).lower()

        if "operation encounters" in sn:   src = "operation"; pri = 1
        elif "clinic encounters" in sn:    src = "clinic";    pri = 2
        elif "inpatient encounters" in sn: src = "inpatient"; pri = 3
        else:                              src = "other";      pri = 9

        admit_col  = pick_col(df, ["ADMIT_DATE", "Admit_Date"], required=False)
        recon_col  = pick_col(df, ["RECONSTRUCTION_DATE", "RECONSTRUCTION DATE"], required=False)
        cpt_col    = pick_col(df, ["CPT_CODE", "CPT CODE", "CPT"], required=False)
        proc_col   = pick_col(df, ["PROCEDURE", "Procedure"], required=False)
        reason_col = pick_col(df, ["REASON_FOR_VISIT", "REASON FOR VISIT"], required=False)
        date_col   = pick_col(df, ["OPERATION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT"], required=False)

        out = pd.DataFrame()
        out[MERGE_KEY]                    = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"]              = src
        out["STRUCT_PRIORITY"]            = pri
        out["STRUCT_DATE_RAW"]            = df[date_col].astype(str)  if date_col   else ""
        out["ADMIT_DATE_STRUCT"]          = df[admit_col].astype(str) if admit_col  else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col  else ""
        out["CPT_CODE_STRUCT"]            = df[cpt_col].astype(str)   if cpt_col    else ""
        out["PROCEDURE_STRUCT"]           = df[proc_col].astype(str)  if proc_col   else ""
        out["REASON_FOR_VISIT_STRUCT"]    = df[reason_col].astype(str)if reason_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "ADMIT_DATE_STRUCT", "RECONSTRUCTION_DATE_STRUCT",
            "CPT_CODE_STRUCT", "PROCEDURE_STRUCT", "REASON_FOR_VISIT_STRUCT"])
    return pd.concat(rows, ignore_index=True)


def _is_recon_like_row(row, has_preferred_cpt):
    preferred   = {"19357", "19340", "19342", "19361", "19364", "19367", "S2068"}
    exclude     = {"19325", "19330"}
    fallback    = {"19350", "19380"}
    cpt    = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
    proc   = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()
    reason = clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")).lower()
    if cpt in exclude: return False
    if cpt in preferred: return True
    if (not has_preferred_cpt) and cpt in fallback: return True
    text = proc + " " + reason
    kws = ["tissue expander", "breast recon",
           "implant on same day of mastectomy",
           "insert or replcmnt breast implnt on sep day from mastectomy",
           "latissimus", "diep", "tram", "flap", "free flap",
           "expander placmnt", "reconstruct", "reconstruction"]
    return any(kw in text for kw in kws)


def choose_best_anchor_rows(struct_df):
    best = {}
    if len(struct_df) == 0:
        return best
    src_prio = {"clinic": 1, "operation": 2, "inpatient": 3}
    eligible = struct_df[struct_df["STRUCT_SOURCE"].isin(src_prio)].copy()
    if len(eligible) == 0:
        return best
    preferred = {"19357", "19340", "19342", "19361", "19364", "19367", "S2068"}
    has_pref = {}
    for mrn, g in eligible.groupby(MERGE_KEY):
        has_pref[mrn] = any(clean_cell(v).upper() in preferred
                            for v in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist())
    for _, row in eligible.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or source not in src_prio: continue
        admit_dt = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        if admit_dt is None or recon_dt is None: continue
        if not _is_recon_like_row(row, has_pref.get(mrn, False)): continue
        score = (src_prio[source], recon_dt, admit_dt)
        cur = best.get(mrn)
        if cur is None or score < cur["score"]:
            best[mrn] = {
                "recon_date":       recon_dt.strftime("%Y-%m-%d"),
                "admit_date":       admit_dt.strftime("%Y-%m-%d"),
                "score":            score,
                "source":           source,
                "cpt_code":         clean_cell(row.get("CPT_CODE_STRUCT", "")),
                "procedure":        clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "reason_for_visit": clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")),
            }
    return best

# ============================================================
# Sectionizer
# ============================================================

HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")

def _sectionize(text):
    if not text:
        return {"FULL": ""}
    lines = text.splitlines()
    sections = {}
    current = "FULL"
    sections[current] = []
    for line in lines:
        m = HEADER_RX.match(line)
        if m:
            hdr = m.group(1).strip().upper()
            current = hdr
            if current not in sections:
                sections[current] = []
            continue
        sections[current].append(line)
    out = {}
    for k, v in sections.items():
        joined = "\n".join(v).strip()
        if joined:
            out[k] = joined
    return out if out else {"FULL": text}

# ============================================================
# Main
# ============================================================

def main():
    print("Loading master...")
    master = clean_cols(read_csv_robust(MASTER_FILE))
    master = normalize_mrn(master)
    for c in PBS_FIELDS:
        if c not in master.columns:
            master[c] = pd.NA
    print("Master rows: {0}".format(len(master)))

    print("Loading notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    print("Loading structured encounters...")
    struct_df  = load_structured_encounters()
    anchor_map = choose_best_anchor_rows(struct_df)
    print("Recon anchors found: {0}".format(len(anchor_map)))

    # Reset all PBS columns to 0
    for c in PBS_FIELDS:
        master[c] = 0

    evidence_rows = []
    best_by_mrn   = {}

    for _, row in notes_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        anchor = anchor_map.get(mrn)
        if anchor is None:
            continue

        recon_dt = parse_date_safe(anchor.get("recon_date", ""))
        note_dt  = parse_date_safe(row.get("NOTE_DATE", ""))
        if recon_dt is None or note_dt is None:
            continue

        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        # Recon laterality: master first, then anchor procedure text
        recon_lat = ""
        if "Recon_Laterality" in master.columns:
            recon_lat = normalize_recon_laterality(
                master.loc[mask, "Recon_Laterality"].astype(str).iloc[0])
        if not recon_lat:
            recon_lat = extract_laterality_from_text(
                "{0} {1}".format(anchor.get("procedure", ""),
                                 anchor.get("reason_for_visit", "")))

        snote = SectionedNote(
            sections=_sectionize(row["NOTE_TEXT"]),
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        try:
            cands = extract_pbs(snote)
        except Exception as e:
            evidence_rows.append({
                MERGE_KEY: mrn, "NOTE_ID": row["NOTE_ID"],
                "NOTE_DATE": row["NOTE_DATE"], "NOTE_TYPE": row["NOTE_TYPE"],
                "FIELD": "EXTRACTOR_ERROR", "VALUE": "", "STATUS": "",
                "CONFIDENCE": "", "SECTION": "",
                "RECON_DATE": anchor.get("recon_date", ""),
                "RECON_LATERALITY": recon_lat, "PROC_LATERALITY": "",
                "RULE_DECISION": "extractor_failed", "EVIDENCE": repr(e)
            })
            continue

        if not cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}

        full_note_text = clean_cell(row.get("NOTE_TEXT", ""))

        for c in cands:
            field = clean_cell(getattr(c, "field", ""))
            if field not in {"PBS_Lumpectomy", "PBS_Breast Reduction",
                              "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"}:
                continue

            evid = clean_cell(getattr(c, "evidence", ""))
            if not evid:
                continue

            combined_context = "{0}\n{1}".format(evid, full_note_text)
            proc_lat    = extract_laterality_from_text(combined_context)
            if (not proc_lat) and field == "PBS_Lumpectomy":
                proc_lat = infer_laterality_from_field_context(field, combined_context)

            lat_decision = laterality_relation(recon_lat, proc_lat, combined_context)
            neg_context  = has_negative_history(combined_context)
            day_diff     = days_between(note_dt, recon_dt)
            history_ok   = field_specific_history_ok(field, combined_context)

            accept = False
            reason = ""

            # ----------------------------------------------------------
            # Gate 1: universal rejects
            # ----------------------------------------------------------
            if neg_context:
                accept = False
                reason = "reject_negative_history"

            elif day_diff is None:
                accept = False
                reason = "reject_missing_date_diff"

            elif field == "PBS_Lumpectomy" and has_strict_lumpectomy_fp_context(combined_context):
                accept = False
                reason = "reject_lumpectomy_planning_context"

            # ----------------------------------------------------------
            # Gate 2: non-lumpectomy PBS — skip laterality entirely.
            # These are cosmetic/surgical history fields. The only valid
            # reject here is an explicit contralateral mention, which
            # would be unusual but possible (e.g. contralateral reduction).
            # ----------------------------------------------------------
            elif field != "PBS_Lumpectomy":
                if lat_decision == "reject_contralateral":
                    accept = False
                    reason = "reject_contralateral"
                elif history_ok:
                    accept = True
                    reason = "accept_non_lumpectomy_history"
                else:
                    accept = False
                    reason = "reject_no_history_context"

            # ----------------------------------------------------------
            # Gate 3: PBS_Lumpectomy — full laterality logic (unchanged)
            # ----------------------------------------------------------
            elif day_diff < 0:
                if lat_decision == "accept":
                    if history_ok:
                        accept = True
                        reason = "accept_pre_recon_historical"
                    else:
                        accept = True
                        reason = "accept_pre_recon_lumpectomy"
                elif lat_decision == "reject_contralateral":
                    accept = False
                    reason = "reject_contralateral"
                elif lat_decision == "unknown_unilateral":
                    inferred = infer_laterality_from_field_context(field, combined_context)
                    if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                        accept = True
                        reason = "accept_inferred_laterality"
                    else:
                        accept = False
                        reason = "reject_unknown_laterality_unilateral"
                else:
                    accept = False
                    reason = "reject_unknown_recon_laterality"

            else:
                # PBS_Lumpectomy post-recon
                if not history_ok:
                    accept = False
                    reason = "reject_post_recon_not_historical"
                else:
                    if lat_decision == "accept":
                        accept = True
                        reason = "accept_post_recon_historical"
                    elif lat_decision == "reject_contralateral":
                        accept = False
                        reason = "reject_contralateral"
                    elif lat_decision == "unknown_unilateral":
                        inferred = infer_laterality_from_field_context(field, combined_context)
                        if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                            accept = True
                            reason = "accept_post_recon_inferred_laterality"
                        else:
                            accept = False
                            reason = "reject_unknown_laterality_unilateral"
                    else:
                        if history_ok:
                            accept = True
                            reason = "accept_post_recon_history_no_recon_lat"
                        else:
                            accept = False
                            reason = "reject_unknown_recon_laterality"

            evidence_rows.append({
                MERGE_KEY:          mrn,
                "NOTE_ID":          getattr(c, "note_id",    row["NOTE_ID"]),
                "NOTE_DATE":        getattr(c, "note_date",  row["NOTE_DATE"]),
                "NOTE_TYPE":        getattr(c, "note_type",  row["NOTE_TYPE"]),
                "FIELD":            field,
                "VALUE":            getattr(c, "value",      True),
                "STATUS":           getattr(c, "status",     ""),
                "CONFIDENCE":       getattr(c, "confidence", ""),
                "SECTION":          getattr(c, "section",    ""),
                "RECON_DATE":       anchor.get("recon_date", ""),
                "RECON_LATERALITY": recon_lat,
                "PROC_LATERALITY":  proc_lat,
                "RULE_DECISION":    reason,
                "EVIDENCE":         evid,
            })

            if not accept:
                continue

            setattr(c, "_source_file",       row.get("SOURCE_FILE", ""))
            setattr(c, "_accepted_post_hist", bool(day_diff >= 0 and history_ok))

            existing = best_by_mrn[mrn].get(field)
            best_by_mrn[mrn][field] = choose_better_pbs(existing, c, recon_dt)

    print("Accepted PBS note-based predictions for MRNs: {0}".format(len(best_by_mrn)))

    for mrn, fields in best_by_mrn.items():
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue
        any_positive = False
        for field in ["PBS_Lumpectomy", "PBS_Breast Reduction",
                       "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"]:
            cand = fields.get(field)
            if cand is None:
                continue
            master.loc[mask, field] = 1
            any_positive = True
        master.loc[mask, "PastBreastSurgery"] = 1 if any_positive else 0

    # Final re-derive PastBreastSurgery from subtypes
    subtype_cols = ["PBS_Lumpectomy", "PBS_Breast Reduction",
                    "PBS_Mastopexy", "PBS_Augmentation", "PBS_Other"]
    for idx in master.index:
        any_pos = False
        for c in subtype_cols:
            try:
                if int(float(str(master.at[idx, c]).strip())) == 1:
                    any_pos = True
                    break
            except Exception:
                pass
        master.at[idx, "PastBreastSurgery"] = 1 if any_pos else 0

    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    master.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\nDONE.")
    print("- Updated master: {0}".format(OUTPUT_MASTER))
    print("- PBS evidence:   {0}".format(OUTPUT_EVID))


if __name__ == "__main__":
    main()
