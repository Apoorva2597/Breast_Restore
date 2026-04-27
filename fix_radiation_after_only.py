#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_radiation_after_only.py

STANDALONE fix for Radiation_After over-prediction.

ROOT CAUSE:
    The original pipeline assigns Radiation_After=1 if any radiation mention
    appears in a note dated AFTER the reconstruction date. This causes FPs
    because post-recon clinic notes frequently reference prior radiation
    in historical context (e.g. "she received XRT before reconstruction").

FIX:
    Radiation_After=1 ONLY when:
      (a) The note is dated after reconstruction AND
      (b) The note contains explicit language indicating radiation is
          CURRENTLY ONGOING or PLANNED after reconstruction
          (e.g. "will receive radiation", "currently receiving XRT",
           "scheduled for radiation", "postmastectomy radiation",
           "adjuvant radiation" without clear historical framing)
      AND NOT blocked by historical/completed language
          (e.g. "received radiation before", "prior XRT", "s/p radiation",
           "radiation before reconstruction")

SAFETY:
    - Reads from existing master (NEVER modifies it)
    - Writes to a NEW separate output file
    - Only changes Radiation_After and Radiation columns
    - All other columns copied verbatim from existing master
    - Has its own evidence file

OUTPUTS:
    _outputs/master_abstraction_rule_FINAL_NO_GOLD_RAD_FIXED.csv
    _outputs/radiation_after_fix_evidence.csv

Python 3.6.8 compatible.
"""

import os
import re
from glob import glob
from datetime import datetime

import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"

# INPUT: existing master (read-only)
INPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)

# OUTPUTS: completely new files, nothing shared with existing pipeline
OUTPUT_MASTER = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD_RAD_FIXED.csv".format(BASE_DIR)
OUTPUT_EVID   = "{0}/_outputs/radiation_after_fix_evidence.csv".format(BASE_DIR)

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

# ============================================================
# Radiation patterns
# ============================================================

# Prefilter - must match to even process the note
RAD_PREFILTER = re.compile(
    r"\b(radiation|radiotherapy|xrt|pmrt|imrt|wbrt|sbrt|rads?|gray|gy|fractions?|"
    r"postmastectomy radiation|adjuvant radiation|adjuvant radiotherapy|"
    r"radiation therapy|radiation treatment|chest wall radiation)\b",
    re.IGNORECASE
)

# Strong signals that radiation is PLANNED or ONGOING after reconstruction
RAD_AFTER_POS = [
    # Future/planned
    re.compile(r"\b(will\s+(?:receive|undergo|need|require|start|begin|complete)\s+(?:radiation|radiotherapy|xrt|pmrt|imrt|wbrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(scheduled\s+(?:for|to\s+(?:begin|start|receive))\s+(?:radiation|radiotherapy|xrt|pmrt|imrt|wbrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(plan(?:ned|ning)?\s+(?:to\s+)?(?:receive|undergo|start|begin)?\s*(?:radiation|radiotherapy|xrt|pmrt|imrt|wbrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(recommend(?:ed|ing)?\s+(?:adjuvant\s+)?(?:radiation|radiotherapy|xrt|pmrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(radiation\s+(?:is\s+)?(?:recommended|planned|scheduled|indicated|pending))\b", re.IGNORECASE),
    re.compile(r"\b(referred\s+(?:for|to)\s+(?:radiation|radiotherapy|xrt|rads?|radiation\s+oncology))\b", re.IGNORECASE),
    re.compile(r"\b(radiation\s+oncology\s+(?:referral|consultation|consult|follow\s*up|appointment))\b", re.IGNORECASE),

    # Currently receiving
    re.compile(r"\b(currently\s+(?:receiving|undergoing|on)\s+(?:radiation|radiotherapy|xrt|pmrt|imrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(in\s+the\s+(?:midst|middle)\s+of\s+(?:radiation|radiotherapy|xrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(receiving\s+(?:adjuvant\s+)?(?:radiation|radiotherapy|xrt|pmrt|imrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(on\s+(?:adjuvant\s+)?(?:radiation|radiotherapy|xrt|pmrt|rads?)\s+(?:therapy|treatment))\b", re.IGNORECASE),
    re.compile(r"\b(mid[- ]radiation|mid[- ]xrt|mid[- ]treatment\s+(?:radiation|xrt))\b", re.IGNORECASE),

    # Post-recon PMRT explicit
    re.compile(r"\b(postmastectomy\s+radiation\s+(?:therapy|treatment)?)\b", re.IGNORECASE),
    re.compile(r"\b(post[- ]mastectomy\s+radiation\s+(?:therapy|treatment)?)\b", re.IGNORECASE),
    re.compile(r"\b(pmrt)\b", re.IGNORECASE),

    # Completed after reconstruction (strong signal: "completed radiation" in post-recon note)
    re.compile(r"\b(completed\s+(?:adjuvant\s+)?(?:radiation|radiotherapy|xrt|pmrt|imrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b(finished\s+(?:radiation|radiotherapy|xrt|pmrt|rads?))\b", re.IGNORECASE),

    # Adjuvant XRT phrases
    re.compile(r"\b(adjuvant\s+(?:radiation|radiotherapy|xrt|pmrt|imrt|rads?))\b", re.IGNORECASE),
    re.compile(r"\b((?:radiation|radiotherapy|xrt|rads?)\s+(?:after|following|post)\s+(?:reconstruction|recon|surgery|mastectomy))\b", re.IGNORECASE),
]

# Patterns that indicate radiation was BEFORE reconstruction or purely historical
# If these appear without a strong AFTER signal, block the hit
RAD_BEFORE_BLOCKER = [
    re.compile(r"\b(prior\s+(?:radiation|radiotherapy|xrt|rads?|xrt|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(previous\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(history\s+of\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(h/o\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(s/p\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(status\s+post\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b((?:radiation|radiotherapy|xrt|rads?)\s+(?:before|prior\s+to|preceding)\s+(?:reconstruction|recon|surgery|mastectomy))\b", re.IGNORECASE),
    re.compile(r"\b(received\s+(?:radiation|radiotherapy|xrt|rads?|pmrt)\s+(?:before|prior|previously))\b", re.IGNORECASE),
    re.compile(r"\b(preoperative\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(neoadjuvant\s+(?:radiation|radiotherapy|xrt|rads?|pmrt))\b", re.IGNORECASE),
    re.compile(r"\b(radiation\s+(?:was\s+)?(?:given|administered|delivered|performed)\s+(?:before|prior|previously))\b", re.IGNORECASE),
    re.compile(r"\b(previously\s+(?:irradiated|radiated))\b", re.IGNORECASE),
    re.compile(r"\b(prior\s+(?:irradiation|irradiated))\b", re.IGNORECASE),
    re.compile(r"\b(radiated\s+(?:breast|chest\s+wall|field))\b", re.IGNORECASE),  # "previously radiated field"
]

# Negation
NEG_RX = re.compile(
    r"\b(no|not|denies|denied|without|negative\s+for|does\s+not\s+(?:need|require|plan))\b",
    re.IGNORECASE
)


def window_around(text, start, end, width=250):
    left  = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right]


def has_rad_after_signal(text):
    """Returns (has_signal: bool, evidence: str)"""
    for rx in RAD_AFTER_POS:
        m = rx.search(text)
        if m is None:
            continue
        snippet = window_around(text, m.start(), m.end(), 200)
        # Check negation
        neg_check = snippet[max(0, m.start()-snippet.find(snippet[:50])):].lower() if snippet else ""
        left_ctx = text[max(0, m.start()-120):m.start()]
        if NEG_RX.search(left_ctx):
            continue
        # Check if before-blocker is in nearby context (within 150 chars)
        ctx = window_around(text, m.start(), m.end(), 150)
        blocked = any(b.search(ctx) for b in RAD_BEFORE_BLOCKER)
        if blocked:
            continue
        return True, snippet.replace("\n", " ").strip()
    return False, ""


def has_rad_before_only(text):
    """True if note only has before-recon radiation signals, no after signals."""
    has_after, _ = has_rad_after_signal(text)
    if has_after:
        return False
    return any(b.search(text) for b in RAD_BEFORE_BLOCKER)


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
        raise RuntimeError("MRN not found. Cols: {0}".format(list(df.columns)[:40]))
    df[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
    return df


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Column missing. Tried={0}".format(options))
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
# Load structured encounters (for recon date anchor only)
# ============================================================

PREFERRED_CPTS  = {"19357", "19340", "19342", "19361", "19364", "19367", "S2068"}
EXCLUDE_CPTS    = {"19325", "19330"}
FALLBACK_CPTS   = {"19350", "19380"}
RECON_KEYWORDS  = [
    "tissue expander", "breast recon", "latissimus", "diep", "tram",
    "flap", "free flap", "reconstruct", "reconstruction",
    "implant on same day of mastectomy",
    "insert or replcmnt breast implnt on sep day from mastectomy",
]


def _is_recon_row(row, has_pref):
    cpt    = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
    proc   = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()
    reason = clean_cell(row.get("REASON_FOR_VISIT_STRUCT", "")).lower()
    if cpt in EXCLUDE_CPTS:    return False
    if cpt in PREFERRED_CPTS:  return True
    if (not has_pref) and cpt in FALLBACK_CPTS: return True
    text = proc + " " + reason
    return any(kw in text for kw in RECON_KEYWORDS)


def load_recon_anchor_map():
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

        out = pd.DataFrame()
        out[MERGE_KEY]                    = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"]              = src
        out["STRUCT_PRIORITY"]            = pri
        out["ADMIT_DATE_STRUCT"]          = df[admit_col].astype(str)  if admit_col  else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str)  if recon_col  else ""
        out["CPT_CODE_STRUCT"]            = df[cpt_col].astype(str)    if cpt_col    else ""
        out["PROCEDURE_STRUCT"]           = df[proc_col].astype(str)   if proc_col   else ""
        out["REASON_FOR_VISIT_STRUCT"]    = df[reason_col].astype(str) if reason_col else ""
        rows.append(out)

    if not rows:
        return {}

    struct_df = pd.concat(rows, ignore_index=True)
    src_prio  = {"clinic": 1, "operation": 2, "inpatient": 3}
    eligible  = struct_df[struct_df["STRUCT_SOURCE"].isin(src_prio)].copy()

    has_pref = {}
    for mrn, g in eligible.groupby(MERGE_KEY):
        has_pref[mrn] = any(
            clean_cell(v).upper() in PREFERRED_CPTS
            for v in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist()
        )

    best = {}
    for _, row in eligible.iterrows():
        mrn    = clean_cell(row.get(MERGE_KEY, ""))
        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if not mrn or source not in src_prio:
            continue
        admit_dt = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        if admit_dt is None or recon_dt is None:
            continue
        if not _is_recon_row(row, has_pref.get(mrn, False)):
            continue
        score = (src_prio[source], recon_dt, admit_dt)
        cur   = best.get(mrn)
        if cur is None or score < cur["score"]:
            best[mrn] = {"recon_date": recon_dt.strftime("%Y-%m-%d"), "score": score}

    return {mrn: v["recon_date"] for mrn, v in best.items()}


# ============================================================
# Load notes
# ============================================================

def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))
    if not note_files:
        raise FileNotFoundError("No note CSVs found.")

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
        if line_col and line_col != "LINE":      tmp = tmp.rename(columns={line_col: "LINE"})
        if type_col and type_col != "NOTE_TYPE": tmp = tmp.rename(columns={type_col: "NOTE_TYPE"})
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
# Main
# ============================================================

def main():
    print("=" * 60)
    print("fix_radiation_after_only.py")
    print("Reads existing master, writes new separate output.")
    print("Only changes: Radiation_After, Radiation")
    print("=" * 60)

    # Load existing master (read-only)
    print("\nLoading existing master (read-only)...")
    master = clean_cols(read_csv_robust(INPUT_MASTER))
    master = normalize_mrn(master)
    print("  Rows: {0}".format(len(master)))

    # Make a clean copy — this is our working output
    out = master.copy()

    # Reset Radiation_After to 0 for everyone (we will re-derive it)
    out["Radiation_After"] = 0

    # Load recon anchor dates
    print("\nLoading reconstruction anchor dates...")
    recon_date_map = load_recon_anchor_map()
    print("  Anchors found: {0}".format(len(recon_date_map)))

    # Load notes
    print("\nLoading and reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("  Reconstructed notes: {0}".format(len(notes_df)))

    evidence_rows = []

    # Per-patient: collect post-recon notes with Radiation_After signal
    rad_after_positive = set()

    for _, row in notes_df.iterrows():
        mrn       = clean_cell(row.get(MERGE_KEY, ""))
        note_text = clean_cell(row.get("NOTE_TEXT", ""))
        if not mrn or not note_text:
            continue

        # Prefilter — must mention radiation at all
        if not RAD_PREFILTER.search(note_text):
            continue

        # Must exist in master
        mask = (out[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        # Must have recon anchor date
        recon_date_str = recon_date_map.get(mrn)
        recon_dt       = parse_date_safe(recon_date_str)
        if recon_dt is None:
            continue

        note_dt = parse_date_safe(row.get("NOTE_DATE", ""))
        if note_dt is None:
            continue

        dd = days_between(note_dt, recon_dt)
        if dd is None:
            continue

        # Note must be AFTER reconstruction to be a Radiation_After candidate
        if dd <= 0:
            continue

        # Now apply the tightened logic
        has_signal, evid_snippet = has_rad_after_signal(note_text)

        evidence_rows.append({
            MERGE_KEY:        mrn,
            "NOTE_ID":        row.get("NOTE_ID", ""),
            "NOTE_DATE":      row.get("NOTE_DATE", ""),
            "NOTE_TYPE":      row.get("NOTE_TYPE", ""),
            "DAYS_AFTER_RECON": dd,
            "HAS_RAD_AFTER_SIGNAL": 1 if has_signal else 0,
            "RECON_DATE":     recon_date_str,
            "EVIDENCE":       evid_snippet[:400] if evid_snippet else ""
        })

        if has_signal:
            rad_after_positive.add(mrn)

    print("\nMRNs with confirmed Radiation_After signal: {0}".format(len(rad_after_positive)))

    # Apply to output master
    for mrn in rad_after_positive:
        mask = (out[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue
        out.loc[mask, "Radiation_After"] = 1
        # Ensure Radiation=1 if Radiation_After=1
        out.loc[mask, "Radiation"] = 1

    # Also ensure: if Radiation_Before=1, Radiation=1 (preserve existing logic)
    for idx in out.index:
        rb = clean_cell(out.at[idx, "Radiation_Before"])
        ra = clean_cell(out.at[idx, "Radiation_After"])
        if rb in {"1", "True", "true"} or ra in {"1", "True", "true"}:
            out.at[idx, "Radiation"] = 1

    # Summary
    old_rad_after = (master["Radiation_After"].astype(str).str.strip()
                     .isin({"1", "True", "true"})).sum()
    new_rad_after = (out["Radiation_After"].astype(str).str.strip()
                     .isin({"1", "True", "true"})).sum()

    print("\nRadiation_After change:")
    print("  Before fix: {0} patients".format(int(old_rad_after)))
    print("  After fix:  {0} patients".format(int(new_rad_after)))
    print("  Reduced by: {0} patients".format(int(old_rad_after) - int(new_rad_after)))

    # Write new output — completely separate from original pipeline
    os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
    out.to_csv(OUTPUT_MASTER, index=False)
    pd.DataFrame(evidence_rows).to_csv(OUTPUT_EVID, index=False)

    print("\n" + "=" * 60)
    print("DONE.")
    print("New master (with fix): {0}".format(OUTPUT_MASTER))
    print("Evidence:              {0}".format(OUTPUT_EVID))
    print()
    print("IMPORTANT: Original master is UNCHANGED at:")
    print("  {0}".format(INPUT_MASTER))
    print()
    print("Next step: run validate_radiation_after_fix.py to check results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
