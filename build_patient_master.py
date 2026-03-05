#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master.py  (Python 3.6.8 friendly)

Builds patient-level abstraction table from ORIGINAL HPI11526 files.

Inputs (default directory):
  /home/apokol/my_data_Breast/HPI-11526/HPI11256

Expected filenames (exactly as you showed):
  HPI11526 Clinic Encounters.csv
  HPI11526 Inpatient Encounters.csv
  HPI11526 Operation Encounters.csv
  HPI11526 Clinic Notes.csv
  HPI11526 Inpatient Notes.csv
  HPI11526 Operation Notes.csv

Outputs:
  _outputs/patient_master.csv
  _outputs/mrn_encrypted_map.csv   (only if MRN present)

Notes:
- Uses encounter fields where available (more reliable than NLP).
- Falls back to note text regex for missing items and for Chemo/Radiation/PBS/Steroid signals.
- Handles UnicodeDecodeError by trying utf-8, then latin-1.
"""

from __future__ import print_function

import os
import re
import sys
import argparse
import pandas as pd


# -----------------------------
# Config
# -----------------------------

DEFAULT_DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"

ENCOUNTER_FILES = [
    "HPI11526 Clinic Encounters.csv",
    "HPI11526 Inpatient Encounters.csv",
    "HPI11526 Operation Encounters.csv",
]

NOTE_FILES = [
    "HPI11526 Clinic Notes.csv",
    "HPI11526 Inpatient Notes.csv",
    "HPI11526 Operation Notes.csv",
]

OUT_DIR = "_outputs"
OUT_MASTER = os.path.join(OUT_DIR, "patient_master.csv")
OUT_MAP = os.path.join(OUT_DIR, "mrn_encrypted_map.csv")

PID_STANDARD = "ENCRYPTED_PAT_ID"

# Variables you’re validating right now (keep aligned with validate script)
OUTPUT_COLUMNS = [
    PID_STANDARD,
    "Race",
    "Ethnicity",
    "Age",
    "BMI",
    "Obesity",
    "SmokingStatus",
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PBS_Lumpectomy",
    "PBS_Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "Radiation",
    "Chemo",
]


# -----------------------------
# Helpers
# -----------------------------

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def ensure_outdir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)


def read_csv_flexible(path):
    """
    Read CSV robustly for older datasets / mixed encodings.
    Tries utf-8, then latin-1. Uses pandas default engine for speed.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: python engine sometimes survives weird bytes
    try:
        return pd.read_csv(path, encoding="latin-1", engine="python")
    except Exception as e:
        raise e


def normalize_str(x):
    if pd.isnull(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def to_int_safe(x):
    if pd.isnull(x):
        return None
    try:
        # handle floats like "47.0"
        return int(float(str(x).strip()))
    except Exception:
        return None


def to_float_safe(x):
    if pd.isnull(x):
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def bool01_from_any(x):
    """
    Normalize common boolean-ish representations to 0/1/None.
    """
    if pd.isnull(x):
        return None
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "present", "pos", "+"):
        return 1
    if s in ("0", "false", "f", "no", "n", "absent", "neg", "-"):
        return 0
    # sometimes "Current"/"Former" etc - not boolean
    try:
        v = int(float(s))
        if v in (0, 1):
            return v
    except Exception:
        pass
    return None


def coalesce_first_nonnull(values):
    for v in values:
        if v is not None and not pd.isnull(v):
            return v
    return None


def mode_nonnull(series):
    """
    Most common non-null value; tie breaks by first encountered.
    """
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if len(s) == 0:
        return None
    # Normalize string whitespace for stable mode
    s2 = s.map(lambda x: str(x).strip() if not pd.isnull(x) else x)
    vc = s2.value_counts()
    if vc.empty:
        return None
    return vc.index[0]


def max01(series):
    """
    For 0/1/None series: if any 1 -> 1; else if any 0 -> 0; else None
    """
    if series is None or len(series) == 0:
        return None
    s = series.dropna()
    if len(s) == 0:
        return None
    # convert to ints when possible
    vals = []
    for x in s.values:
        b = bool01_from_any(x)
        if b is not None:
            vals.append(b)
    if not vals:
        return None
    return 1 if 1 in vals else 0


# -----------------------------
# Column discovery
# -----------------------------

def find_column(df, candidates_regex):
    """
    Find a column name in df matching any regex (case-insensitive).
    Returns first match by df column order.
    """
    cols = list(df.columns)
    for c in cols:
        c_low = str(c).strip().lower()
        for rx in candidates_regex:
            if re.search(rx, c_low):
                return c
    return None


def standardize_pid(df):
    """
    Ensure df has ENCRYPTED_PAT_ID. If already present, keep it.
    Otherwise try common variants.
    """
    if PID_STANDARD in df.columns:
        return df

    pid_col = find_column(df, [
        r"encrypted.*pat.*id",
        r"enc.*pat.*id",
        r"patient.*encrypted",
        r"pat.*enc",
        r"encryptedid",
    ])
    if pid_col is None:
        raise ValueError("Could not find ENCRYPTED patient id column in file.")
    df = df.rename(columns={pid_col: PID_STANDARD})
    return df


def pick_text_column(df):
    """
    Notes file: pick best note text column.
    Original datasets vary. Try common names.
    """
    text_col = find_column(df, [
        r"note.*text",
        r"text",
        r"document.*text",
        r"full.*text",
        r"body",
        r"content",
    ])
    return text_col


# -----------------------------
# Regex extractors from notes
# -----------------------------

RX_AGE = re.compile(r"\b(\d{2,3})\s*(?:yo|y/o|year[- ]old)\b", re.I)
RX_BMI = re.compile(r"\bBMI\b[^0-9]{0,15}(\d{2}\.\d|\d{2})\b", re.I)

# Smoking normalization patterns
RX_SMOKE_CURRENT = re.compile(r"\b(current smoker|smokes\b|smoking\b.*current)\b", re.I)
RX_SMOKE_FORMER = re.compile(r"\b(former smoker|quit smoking|stopped smoking|ex-smoker)\b", re.I)
RX_SMOKE_NEVER = re.compile(r"\b(never smoker|never smoked|non-smoker|nonsmoker)\b", re.I)

# Steroid: broad + common meds
RX_STEROID = re.compile(r"\b(steroid|prednisone|prednisolone|methylprednisolone|dexamethasone|hydrocortisone)\b", re.I)

# Chemo / Radiation with negation
RX_CHEMO_POS = re.compile(r"\b(chemotherapy|chemo)\b", re.I)
RX_CHEMO_NEG = re.compile(r"\b(no|not|without|denies)\b[^\.]{0,35}\b(chemotherapy|chemo)\b", re.I)

RX_RAD_POS = re.compile(r"\b(radiation therapy|radiotherapy|xrt|rt\b)\b", re.I)
RX_RAD_NEG = re.compile(r"\b(no|not|without|denies)\b[^\.]{0,35}\b(radiation therapy|radiotherapy|xrt|rt\b)\b", re.I)

# PBS procedures
RX_LUMPECTOMY = re.compile(r"\b(lumpectomy|partial mastectomy)\b", re.I)
RX_REDUCTION = re.compile(r"\b(breast reduction|reduction mammoplasty|mammoplasty)\b", re.I)
RX_MASTOPEXY = re.compile(r"\b(mastopexy|breast lift)\b", re.I)
RX_AUGMENT = re.compile(r"\b(augmentation|breast augmentation|augment)\b", re.I)

# Comorbidity hints (fallback only)
RX_DIABETES = re.compile(r"\bdiabetes\b", re.I)
RX_HTN = re.compile(r"\b(hypertension|htn)\b", re.I)
RX_CARDIAC = re.compile(r"\b(coronary|cad\b|mi\b|myocardial infarction|heart failure|chf\b|angina)\b", re.I)
RX_VTE = re.compile(r"\b(dvt\b|pe\b|pulmonary embol|venous thromboembol|thrombosis)\b", re.I)


def note_signal(text, rx_pos, rx_neg=None):
    """
    Returns 1/0/None from a note:
      - if negation match -> 0
      - elif positive match -> 1
      - else None
    """
    if text is None:
        return None
    if rx_neg is not None and rx_neg.search(text):
        return 0
    if rx_pos.search(text):
        return 1
    return None


def extract_from_notes_group(note_texts):
    """
    note_texts: list of strings for one patient
    returns dict of extracted values (only for fields we want to fill/boost)
    """
    out = {}

    # Try to find first plausible age / BMI
    age_vals = []
    bmi_vals = []
    smoke = []

    steroid_any = 0
    chemo_votes = []
    rad_votes = []
    pbs_lump = 0
    pbs_red = 0
    pbs_mast = 0
    pbs_aug = 0

    diab_any = 0
    htn_any = 0
    card_any = 0
    vte_any = 0

    for t in note_texts:
        if not t:
            continue

        # Age
        m = RX_AGE.search(t)
        if m:
            age_vals.append(to_int_safe(m.group(1)))

        # BMI
        m2 = RX_BMI.search(t)
        if m2:
            bmi_vals.append(to_float_safe(m2.group(1)))

        # Smoking
        if RX_SMOKE_CURRENT.search(t):
            smoke.append("Current")
        elif RX_SMOKE_FORMER.search(t):
            smoke.append("Former")
        elif RX_SMOKE_NEVER.search(t):
            smoke.append("Never")

        # Steroid
        if RX_STEROID.search(t):
            steroid_any = 1

        # Chemo / Rad (vote with negation awareness)
        chemo_votes.append(note_signal(t, RX_CHEMO_POS, RX_CHEMO_NEG))
        rad_votes.append(note_signal(t, RX_RAD_POS, RX_RAD_NEG))

        # PBS
        if RX_LUMPECTOMY.search(t):
            pbs_lump = 1
        if RX_REDUCTION.search(t):
            pbs_red = 1
        if RX_MASTOPEXY.search(t):
            pbs_mast = 1
        if RX_AUGMENT.search(t):
            pbs_aug = 1

        # Comorbidities (weak fallback)
        if RX_DIABETES.search(t):
            diab_any = 1
        if RX_HTN.search(t):
            htn_any = 1
        if RX_CARDIAC.search(t):
            card_any = 1
        if RX_VTE.search(t):
            vte_any = 1

    # consolidate
    age_vals = [a for a in age_vals if a is not None and 0 < a < 120]
    bmi_vals = [b for b in bmi_vals if b is not None and 10.0 < b < 90.0]

    out["Age_note"] = age_vals[0] if age_vals else None
    out["BMI_note"] = round(bmi_vals[0], 1) if bmi_vals else None

    out["SmokingStatus_note"] = smoke[0] if smoke else None
    out["Steroid_note"] = steroid_any if steroid_any == 1 else None

    # chemo/rad: if any explicit 1 -> 1, else if any explicit 0 -> 0, else None
    def consolidate_votes(votes):
        vv = [v for v in votes if v is not None]
        if not vv:
            return None
        return 1 if 1 in vv else 0

    out["Chemo_note"] = consolidate_votes(chemo_votes)
    out["Radiation_note"] = consolidate_votes(rad_votes)

    out["PBS_Lumpectomy_note"] = 1 if pbs_lump else None
    out["PBS_Reduction_note"] = 1 if pbs_red else None
    out["PBS_Mastopexy_note"] = 1 if pbs_mast else None
    out["PBS_Augmentation_note"] = 1 if pbs_aug else None

    out["Diabetes_note"] = 1 if diab_any else None
    out["Hypertension_note"] = 1 if htn_any else None
    out["CardiacDisease_note"] = 1 if card_any else None
    out["VenousThromboembolism_note"] = 1 if vte_any else None

    return out


# -----------------------------
# Main build logic
# -----------------------------

def build_patient_table(enc_df, notes_by_pid):
    """
    enc_df: concatenated encounters (all settings)
    notes_by_pid: dict pid -> list of note texts
    """
    enc_df = standardize_pid(enc_df)

    # identify likely encounter columns (best-effort)
    col_race = find_column(enc_df, [r"^race$", r"race"])
    col_eth = find_column(enc_df, [r"ethnic", r"hispanic"])
    col_age = find_column(enc_df, [r"^age$", r"age"])
    col_bmi = find_column(enc_df, [r"\bbmi\b"])
    col_smoke = find_column(enc_df, [r"smok"])
    col_diab = find_column(enc_df, [r"diabet"])
    col_htn = find_column(enc_df, [r"hypert", r"\bhtn\b"])
    col_card = find_column(enc_df, [r"cardiac", r"\bcad\b", r"heart"])
    col_vte = find_column(enc_df, [r"thrombo", r"\bvte\b", r"\bdvt\b", r"embol"])
    col_steroid = find_column(enc_df, [r"steroid", r"predni", r"cortic"])

    # Sometimes chemo/rad stored as flags
    col_chemo = find_column(enc_df, [r"\bchemo\b", r"chemotherapy"])
    col_rad = find_column(enc_df, [r"radiat", r"\bxrt\b", r"\brt\b"])

    # PBS structured columns (rare)
    col_lump = find_column(enc_df, [r"lumpect"])
    col_red = find_column(enc_df, [r"reduction"])
    col_mast = find_column(enc_df, [r"mastopex", r"breast lift"])
    col_aug = find_column(enc_df, [r"augment"])

    # obesity sometimes present, else derive from BMI
    col_obesity = find_column(enc_df, [r"obes"])

    # group by patient
    grouped = enc_df.groupby(PID_STANDARD, sort=False)

    rows = []
    for pid, g in grouped:
        # Encounter-driven (preferred)
        race = mode_nonnull(g[col_race]) if col_race else None
        eth = mode_nonnull(g[col_eth]) if col_eth else None

        age_enc = None
        if col_age:
            # if multiple, take most recent non-null (last row order) else mode
            vals = [to_int_safe(x) for x in g[col_age].values]
            vals = [v for v in vals if v is not None and 0 < v < 120]
            age_enc = vals[-1] if vals else None

        bmi_enc = None
        if col_bmi:
            vals = [to_float_safe(x) for x in g[col_bmi].values]
            vals = [v for v in vals if v is not None and 10.0 < v < 90.0]
            bmi_enc = round(vals[-1], 1) if vals else None

        smoke_enc = None
        if col_smoke:
            smoke_raw = mode_nonnull(g[col_smoke])
            smoke_enc = normalize_smoking(smoke_raw)

        diab_enc = max01(g[col_diab]) if col_diab else None
        htn_enc = max01(g[col_htn]) if col_htn else None
        card_enc = max01(g[col_card]) if col_card else None
        vte_enc = max01(g[col_vte]) if col_vte else None
        steroid_enc = max01(g[col_steroid]) if col_steroid else None

        chemo_enc = max01(g[col_chemo]) if col_chemo else None
        rad_enc = max01(g[col_rad]) if col_rad else None

        pbs_lump_enc = max01(g[col_lump]) if col_lump else None
        pbs_red_enc = max01(g[col_red]) if col_red else None
        pbs_mast_enc = max01(g[col_mast]) if col_mast else None
        pbs_aug_enc = max01(g[col_aug]) if col_aug else None

        obesity = None
        if col_obesity:
            obesity = max01(g[col_obesity])
        # derive if missing
        if obesity is None and bmi_enc is not None:
            obesity = 1 if bmi_enc >= 30.0 else 0

        # Note-driven fallback/boost
        note_pack = extract_from_notes_group(notes_by_pid.get(pid, []))

        age = coalesce_first_nonnull([age_enc, note_pack.get("Age_note")])
        bmi = coalesce_first_nonnull([bmi_enc, note_pack.get("BMI_note")])

        # If BMI found late, re-derive obesity
        if (obesity is None or pd.isnull(obesity)) and bmi is not None:
            obesity = 1 if float(bmi) >= 30.0 else 0

        smoke = coalesce_first_nonnull([smoke_enc, note_pack.get("SmokingStatus_note")])
        if smoke is None:
            smoke = "Unknown"

        diabetes = coalesce_first_nonnull([diab_enc, note_pack.get("Diabetes_note")])
        hypertension = coalesce_first_nonnull([htn_enc, note_pack.get("Hypertension_note")])
        cardiac = coalesce_first_nonnull([card_enc, note_pack.get("CardiacDisease_note")])
        vte = coalesce_first_nonnull([vte_enc, note_pack.get("VenousThromboembolism_note")])
        steroid = coalesce_first_nonnull([steroid_enc, note_pack.get("Steroid_note")])

        chemo = coalesce_first_nonnull([chemo_enc, note_pack.get("Chemo_note")])
        rad = coalesce_first_nonnull([rad_enc, note_pack.get("Radiation_note")])

        pbs_lump = coalesce_first_nonnull([pbs_lump_enc, note_pack.get("PBS_Lumpectomy_note")])
        pbs_red = coalesce_first_nonnull([pbs_red_enc, note_pack.get("PBS_Reduction_note")])
        pbs_mast = coalesce_first_nonnull([pbs_mast_enc, note_pack.get("PBS_Mastopexy_note")])
        pbs_aug = coalesce_first_nonnull([pbs_aug_enc, note_pack.get("PBS_Augmentation_note")])

        # Final type normalization: force 0/1 where applicable
        def force01(x):
            b = bool01_from_any(x)
            return b if b is not None else 0  # default conservative 0

        # For these binary variables, default to 0 if still None (avoid NA mismatches in validation)
        diabetes = force01(diabetes)
        hypertension = force01(hypertension)
        cardiac = force01(cardiac)
        vte = force01(vte)
        steroid = force01(steroid)
        pbs_lump = force01(pbs_lump)
        pbs_red = force01(pbs_red)
        pbs_mast = force01(pbs_mast)
        pbs_aug = force01(pbs_aug)
        rad = force01(rad)
        chemo = force01(chemo)

        # Race/Ethnicity normalization (light-touch)
        race = normalize_race(race)
        eth = normalize_ethnicity(eth)

        row = {
            PID_STANDARD: pid,
            "Race": race if race is not None else "Unknown",
            "Ethnicity": eth if eth is not None else "Unknown",
            "Age": age if age is not None else "",
            "BMI": bmi if bmi is not None else "",
            "Obesity": force01(obesity) if obesity is not None else 0,
            "SmokingStatus": smoke,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "CardiacDisease": cardiac,
            "VenousThromboembolism": vte,
            "Steroid": steroid,
            "PBS_Lumpectomy": pbs_lump,
            "PBS_Reduction": pbs_red,
            "PBS_Mastopexy": pbs_mast,
            "PBS_Augmentation": pbs_aug,
            "Radiation": rad,
            "Chemo": chemo,
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    # Ensure all expected columns exist
    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = ""

    out = out[OUTPUT_COLUMNS]
    return out


def normalize_smoking(smoke_raw):
    s = normalize_str(smoke_raw)
    if s is None:
        return None
    sl = s.lower()
    if "current" in sl:
        return "Current"
    if "former" in sl or "quit" in sl or "ex-" in sl:
        return "Former"
    if "never" in sl or "non" in sl:
        return "Never"
    # sometimes encoded as 0/1/2 etc
    if sl in ("0", "none"):
        return "Never"
    return s  # keep original if unrecognized


def normalize_race(race_raw):
    s = normalize_str(race_raw)
    if s is None:
        return None
    sl = s.lower()
    if "white" in sl or "cauc" in sl:
        return "Caucasian"
    if "black" in sl or "african" in sl:
        return "Black"
    if "asian" in sl:
        return "Asian"
    if "unknown" in sl or "declin" in sl:
        return "Unknown"
    return s


def normalize_ethnicity(eth_raw):
    s = normalize_str(eth_raw)
    if s is None:
        return None
    sl = s.lower()
    if "non" in sl and "hisp" in sl:
        return "Non-hispanic"
    if "hisp" in sl or "latin" in sl:
        return "Hispanic"
    if "unknown" in sl or "declin" in sl:
        return "Unknown"
    return s


def build_notes_index(data_dir):
    """
    Build dict: ENCRYPTED_PAT_ID -> list of note texts
    Keeps memory reasonable by only storing text (not whole df).
    """
    notes_by_pid = {}

    for fn in NOTE_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            log("WARNING: missing notes file: {}".format(path))
            continue

        log("Reading notes: {}".format(path))
        df = read_csv_flexible(path)
        df = standardize_pid(df)

        text_col = pick_text_column(df)
        if text_col is None:
            log("WARNING: could not find note text column in {}".format(fn))
            continue

        # iterate rows (vectorized grouping can be memory heavy for huge notes)
        for _, row in df[[PID_STANDARD, text_col]].iterrows():
            pid = row[PID_STANDARD]
            txt = row[text_col]
            if pd.isnull(pid) or pd.isnull(txt):
                continue
            pid = str(pid).strip()
            t = str(txt)

            # light cleaning to remove artifacts like <U+0095>
            t = re.sub(r"<U\+\d+>", " ", t)
            t = re.sub(r"\s+", " ", t)

            if pid not in notes_by_pid:
                notes_by_pid[pid] = []
            notes_by_pid[pid].append(t)

    return notes_by_pid


def extract_mrn_mapping(enc_df):
    """
    If MRN exists in encounters, write mapping MRN -> ENCRYPTED_PAT_ID.
    """
    mrn_col = find_column(enc_df, [r"\bmrn\b", r"medical.*record"])
    if mrn_col is None:
        return None

    tmp = enc_df[[PID_STANDARD, mrn_col]].dropna()
    if tmp.empty:
        return None

    tmp[mrn_col] = tmp[mrn_col].map(lambda x: str(x).strip())
    tmp[PID_STANDARD] = tmp[PID_STANDARD].map(lambda x: str(x).strip())

    # Keep first observed mapping per MRN (assumes stable)
    tmp = tmp.drop_duplicates(subset=[mrn_col], keep="first")
    tmp = tmp.rename(columns={mrn_col: "MRN"})
    return tmp[["MRN", PID_STANDARD]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory containing original HPI11526 files.")
    args = parser.parse_args()

    data_dir = args.data_dir

    log("Using data_dir: {}".format(data_dir))
    for fn in ENCOUNTER_FILES + NOTE_FILES:
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p):
            log("WARNING: expected file missing: {}".format(p))

    ensure_outdir()

    # Load encounters
    enc_list = []
    for fn in ENCOUNTER_FILES:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            log("WARNING: missing encounters file: {}".format(path))
            continue
        log("Reading encounters: {}".format(path))
        df = read_csv_flexible(path)
        df = standardize_pid(df)
        enc_list.append(df)

    if not enc_list:
        raise RuntimeError("No encounter files could be loaded. Check paths.")

    enc = pd.concat(enc_list, axis=0, sort=False, ignore_index=True)
    log("Total encounter rows: {}".format(len(enc)))
    log("Unique patients in encounters: {}".format(enc[PID_STANDARD].nunique()))

    # Build notes index
    notes_by_pid = build_notes_index(data_dir)
    log("Patients with notes indexed: {}".format(len(notes_by_pid)))

    # Build patient table
    log("Building patient-level master...")
    master = build_patient_table(enc, notes_by_pid)

    # Save outputs
    master.to_csv(OUT_MASTER, index=False)
    log("Wrote: {}".format(os.path.abspath(OUT_MASTER)))

    # Save MRN mapping if possible
    mrn_map = extract_mrn_mapping(enc)
    if mrn_map is not None and not mrn_map.empty:
        mrn_map.to_csv(OUT_MAP, index=False)
        log("Wrote: {}".format(os.path.abspath(OUT_MAP)))
    else:
        log("No MRN mapping written (MRN column not found or empty).")

    log("Done.")


if __name__ == "__main__":
    main()
