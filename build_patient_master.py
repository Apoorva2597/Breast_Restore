#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_patient_master_from_original.py  (Python 3.6.8 friendly)

Builds a patient-level abstraction table keyed by ENCRYPTED_PAT_ID using ONLY:
  /home/apokol/my_data_Breast/HPI-11526/HPI11256

- Seeds patients + structured demographics from encounter CSVs
- Loads original note CSVs, reconstructs note text (NOTE_ID + LINE)
- Runs rule-based extractors from ./extractors
- Aggregates to patient-level (best candidate per field; boolean OR logic)
- Normalizes Race/Ethnicity/Smoking + binaries to match GOLD conventions
- Writes:
    /home/apokol/Breast_Restore/_outputs/patient_master.csv
    /home/apokol/Breast_Restore/_outputs/rule_hit_evidence.csv
"""

from __future__ import print_function

import os
import re
from glob import glob

import pandas as pd

# -----------------------
# PATHS
# -----------------------
BASE_DIR = "/home/apokol/Breast_Restore"  # repo + outputs
DATA_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"  # ORIGINAL DATA ONLY

OUT_DIR = os.path.join(BASE_DIR, "_outputs")
OUT_MASTER = os.path.join(OUT_DIR, "patient_master.csv")
OUT_EVID = os.path.join(OUT_DIR, "rule_hit_evidence.csv")

# Original files (exact names you showed)
ENCOUNTER_FILES = [
    os.path.join(DATA_DIR, "HPI11526 Clinic Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Inpatient Encounters.csv"),
    os.path.join(DATA_DIR, "HPI11526 Operation Encounters.csv"),
]
NOTE_FILES = [
    os.path.join(DATA_DIR, "HPI11526 Clinic Notes.csv"),
    os.path.join(DATA_DIR, "HPI11526 Inpatient Notes.csv"),
    os.path.join(DATA_DIR, "HPI11526 Operation Notes.csv"),
]

PID = "ENCRYPTED_PAT_ID"
MRN = "MRN"

# -----------------------
# Imports from your repo
# -----------------------
from models import SectionedNote, Candidate  # noqa: E402

from extractors.age import extract_age  # noqa: E402
from extractors.bmi import extract_bmi  # noqa: E402
from extractors.smoking import extract_smoking  # noqa: E402
from extractors.comorbidities import extract_comorbidities  # noqa: E402
from extractors.pbs import extract_pbs  # noqa: E402
from extractors.cancer_treatment import extract_cancer_treatment  # noqa: E402

# -----------------------
# Robust CSV read (Py3.6 / older pandas friendly)
# -----------------------
def read_csv_robust(path):
    """
    Handles:
      - bad lines
      - weird encodings (latin-1 fallback)
      - unicode decode errors
    Compatible with older pandas (no on_bad_lines).
    """
    if not os.path.exists(path):
        raise IOError("Missing file: %s" % path)

    common = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True, **common)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True, **common)
    except Exception:
        # last resort: read as bytes-ish via latin-1
        return pd.read_csv(path, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True, **common)


def clean_cols(df):
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError("Missing required column. Tried=%s Seen=%s" % (options, list(df.columns)[:80]))
    return None


def to_int_safe(x):
    try:
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


# -----------------------
# Lightweight sectionizer
# -----------------------
HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")

def sectionize(text):
    if not text:
        return {"FULL": ""}

    lines = text.splitlines()
    sections = {"FULL": []}
    current = "FULL"

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


def build_sectioned_note(note_text, note_type, note_id, note_date):
    return SectionedNote(
        sections=sectionize(note_text),
        note_type=note_type or "",
        note_id=note_id or "",
        note_date=note_date or ""
    )


# -----------------------
# Normalization to match GOLD conventions
# -----------------------
def norm_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "none", "null"):
        return ""
    return s

def normalize_race(val):
    """
    Target examples in your gold: Caucasian, Black or African American, Other Asian, Chinese, Unknown
    We keep this conservative to avoid wrong mapping.
    """
    v = norm_str(val).lower()
    if not v:
        return ""

    if "cauc" in v or v == "white" or "white" in v:
        return "Caucasian"
    if "black" in v or "african" in v:
        return "Black or African American"
    if "asian" in v:
        # if specific subgroup appears, preserve if present
        if "chinese" in v:
            return "Chinese"
        return "Other Asian"
    if "unknown" in v or "decline" in v or "not to disclose" in v:
        return "Unknown"
    # keep original trimmed if already looks like gold-style
    return str(val).strip()

def normalize_ethnicity(val):
    """
    Target examples in your gold: Non-hispanic, Hispanic, Unknown
    """
    v = norm_str(val).lower()
    if not v:
        return ""
    if "non" in v and "hisp" in v:
        return "Non-hispanic"
    if "hisp" in v:
        return "Hispanic"
    if "unknown" in v or "decline" in v or "not to disclose" in v:
        return "Unknown"
    return str(val).strip()

def normalize_smoking(val):
    """
    Target examples: Never, Former, Current
    """
    v = norm_str(val).lower()
    if not v:
        return ""
    if "never" in v:
        return "Never"
    if "former" in v or "ex" in v or "quit" in v:
        return "Former"
    if "current" in v or "smokes" in v or "every day" in v or "daily" in v:
        return "Current"
    return str(val).strip()

def to_binary01(x):
    """
    Convert common boolean-ish values to 0/1, else blank.
    """
    v = norm_str(x).lower()
    if not v:
        return ""
    if v in ("1", "true", "t", "yes", "y", "positive", "+"):
        return 1
    if v in ("0", "false", "f", "no", "n", "negative", "-"):
        return 0
    # if already numeric-like:
    try:
        fx = float(v)
        if fx == 1.0:
            return 1
        if fx == 0.0:
            return 0
    except Exception:
        pass
    return ""


# -----------------------
# Candidate aggregation
# -----------------------
def cand_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    date_bonus = 0.01 if norm_str(getattr(c, "note_date", "")) else 0.0
    return conf + op_bonus + date_bonus

def choose_best(existing, new):
    if existing is None:
        return new
    return new if cand_score(new) > cand_score(existing) else existing

def merge_boolean(existing, new):
    if existing is None:
        return new
    exv = to_binary01(getattr(existing, "value", ""))
    nwv = to_binary01(getattr(new, "value", ""))
    if nwv == 1 and exv != 1:
        return new
    if exv == 1 and nwv != 1:
        return existing
    return choose_best(existing, new)


# -----------------------
# Output schema (bucket 2)
# -----------------------
MASTER_COLUMNS = [
    PID,
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
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "Radiation",
    "Chemo",
]

BOOLEAN_FIELDS = set([
    "Diabetes",
    "Hypertension",
    "CardiacDisease",
    "VenousThromboembolism",
    "Steroid",
    "PBS_Lumpectomy",
    "PBS_Breast Reduction",
    "PBS_Mastopexy",
    "PBS_Augmentation",
    "Radiation",
    "Chemo",
])

FIELD_MAP = {
    # from extractors -> our master columns
    "Age": "Age",
    "Age_DOS": "Age",
    "BMI": "BMI",
    "SmokingStatus": "SmokingStatus",
    "Diabetes": "Diabetes",
    "DiabetesMellitus": "Diabetes",
    "Hypertension": "Hypertension",
    "CardiacDisease": "CardiacDisease",
    "VTE": "VenousThromboembolism",
    "VenousThromboembolism": "VenousThromboembolism",
    "SteroidUse": "Steroid",
    "Steroid": "Steroid",
    "PBS_Lumpectomy": "PBS_Lumpectomy",
    "PBS_Breast Reduction": "PBS_Breast Reduction",
    "PBS_Mastopexy": "PBS_Mastopexy",
    "PBS_Augmentation": "PBS_Augmentation",
    "Radiation": "Radiation",
    "Chemo": "Chemo",
}


# -----------------------
# Load encounters -> seed patients + demographics
# -----------------------
def load_encounters_seed():
    frames = []
    for fp in ENCOUNTER_FILES:
        df = clean_cols(read_csv_robust(fp))
        # Required ID columns (per your screenshots)
        pid_col = pick_col(df, [PID, "ENCRYPTED_PATID", "ENCRYPTED PAT ID"], required=True)
        mrn_col = pick_col(df, [MRN, "mrn"], required=False)

        race_col = pick_col(df, ["RACE", "Race", "2. Race"], required=False)
        eth_col = pick_col(df, ["ETHNICITY", "Ethnicity", "3. Ethnicity"], required=False)
        age_col = pick_col(df, ["AGE_AT_ENCOUNTER", "AGE AT ENCOUNTER", "Age", "4. Age"], required=False)

        out = pd.DataFrame()
        out[PID] = df[pid_col].astype(str).str.strip()

        if mrn_col:
            out[MRN] = df[mrn_col].astype(str).str.strip()
        else:
            out[MRN] = ""

        out["Race_raw"] = df[race_col].astype(str).str.strip() if race_col else ""
        out["Ethnicity_raw"] = df[eth_col].astype(str).str.strip() if eth_col else ""
        out["Age_raw"] = df[age_col].astype(str).str.strip() if age_col else ""

        frames.append(out)

    all_enc = pd.concat(frames, ignore_index=True)
    all_enc = all_enc[all_enc[PID].astype(str).str.strip() != ""].copy()

    # patient-level: choose most frequent non-empty race/eth + median age (if numeric)
    def mode_nonempty(series):
        vals = [norm_str(x) for x in series.tolist()]
        vals = [x for x in vals if x]
        if not vals:
            return ""
        # mode
        vc = pd.Series(vals).value_counts()
        return str(vc.index[0])

    grp = all_enc.groupby(PID, dropna=False)

    pat = pd.DataFrame({PID: list(grp.groups.keys())})
    pat[PID] = pat[PID].astype(str).str.strip()

    race_mode = grp["Race_raw"].apply(mode_nonempty).reset_index().rename(columns={"Race_raw": "Race"})
    eth_mode = grp["Ethnicity_raw"].apply(mode_nonempty).reset_index().rename(columns={"Ethnicity_raw": "Ethnicity"})

    # age: median of numeric-able values
    def median_age(series):
        nums = []
        for x in series.tolist():
            s = norm_str(x)
            try:
                nums.append(float(s))
            except Exception:
                continue
        if not nums:
            return ""
        return str(int(round(pd.Series(nums).median())))

    age_med = grp["Age_raw"].apply(median_age).reset_index().rename(columns={"Age_raw": "Age"})

    pat = pat.merge(race_mode, on=PID, how="left")
    pat = pat.merge(eth_mode, on=PID, how="left")
    pat = pat.merge(age_med, on=PID, how="left")

    # normalize to gold-style
    pat["Race"] = pat["Race"].apply(normalize_race)
    pat["Ethnicity"] = pat["Ethnicity"].apply(normalize_ethnicity)

    # ensure master columns exist
    for c in MASTER_COLUMNS:
        if c not in pat.columns:
            pat[c] = ""
    pat = pat[MASTER_COLUMNS].copy()
    return pat


# -----------------------
# Load + reconstruct notes (original schema)
# -----------------------
def load_and_reconstruct_notes():
    all_rows = []

    for fp in NOTE_FILES:
        df = clean_cols(read_csv_robust(fp))

        pid_col = pick_col(df, [PID, "ENCRYPTED PAT ID", "ENCRYPTED_PATID"], required=True)
        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"], required=True)
        text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"], required=True)
        line_col = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(df, ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE"], required=False)

        tmp = pd.DataFrame()
        tmp[PID] = df[pid_col].astype(str).str.strip()
        tmp["NOTE_ID"] = df[note_id_col].fillna("").astype(str).str.strip()
        tmp["NOTE_TEXT"] = df[text_col].fillna("").astype(str)
        tmp["LINE"] = df[line_col].fillna("").astype(str) if line_col else ""
        tmp["NOTE_TYPE"] = df[note_type_col].fillna("").astype(str) if note_type_col else os.path.basename(fp)
        tmp["NOTE_DATE"] = df[date_col].fillna("").astype(str) if date_col else ""
        tmp["SOURCE_FILE"] = os.path.basename(fp)

        tmp = tmp[(tmp[PID].astype(str).str.strip() != "") & (tmp["NOTE_ID"].astype(str).str.strip() != "")]
        all_rows.append(tmp)

    notes_raw = pd.concat(all_rows, ignore_index=True)

    # reconstruct per (PID, NOTE_ID)
    reconstructed = []
    g = notes_raw.groupby([PID, "NOTE_ID"], dropna=False)

    for (pid, nid), block in g:
        pid = str(pid).strip()
        nid = str(nid).strip()
        if not pid or not nid:
            continue

        b = block.copy()
        b["_LINE_NUM_"] = b["LINE"].apply(to_int_safe)
        b = b.sort_values(by=["_LINE_NUM_"], na_position="last")

        full_text = "\n".join(b["NOTE_TEXT"].tolist()).strip()
        if not full_text:
            continue

        note_type = ""
        if (b["NOTE_TYPE"].astype(str).str.strip() != "").any():
            note_type = b["NOTE_TYPE"].astype(str).iloc[0]
        else:
            note_type = b["SOURCE_FILE"].astype(str).iloc[0]

        note_date = ""
        if (b["NOTE_DATE"].astype(str).str.strip() != "").any():
            note_date = b["NOTE_DATE"].astype(str).iloc[0]

        reconstructed.append({
            PID: pid,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": b["SOURCE_FILE"].astype(str).iloc[0],
            "NOTE_TEXT": full_text,
        })

    return pd.DataFrame(reconstructed)


# -----------------------
# MAIN
# -----------------------
def main():
    print("Building patient master from ORIGINAL files only...")
    print("DATA_DIR:", DATA_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n1) Seeding patients + structured demographics from encounters...")
    master = load_encounters_seed()
    print("Seeded patients:", len(master))

    print("\n2) Loading + reconstructing notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes:", len(notes_df))

    print("\n3) Running extractors + aggregating to patient-level...")
    extractor_fns = [
        extract_age,
        extract_bmi,
        extract_smoking,
        extract_comorbidities,
        extract_pbs,
        extract_cancer_treatment,
    ]

    best_by_pid = {}   # pid -> logical_field -> Candidate
    evidence_rows = []

    for _, row in notes_df.iterrows():
        pid = str(row[PID]).strip()
        snote = build_sectioned_note(
            note_text=row["NOTE_TEXT"],
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=row["NOTE_DATE"]
        )

        cands = []
        for fn in extractor_fns:
            try:
                cands.extend(fn(snote))
            except Exception as e:
                evidence_rows.append({
                    PID: pid,
                    "NOTE_ID": row["NOTE_ID"],
                    "NOTE_DATE": row["NOTE_DATE"],
                    "NOTE_TYPE": row["NOTE_TYPE"],
                    "FIELD": "EXTRACTOR_ERROR",
                    "VALUE": "",
                    "STATUS": "",
                    "CONFIDENCE": "",
                    "SECTION": "",
                    "EVIDENCE": "%s failed: %s" % (fn.__name__, repr(e)),
                })

        if not cands:
            continue

        if pid not in best_by_pid:
            best_by_pid[pid] = {}

        for c in cands:
            logical = FIELD_MAP.get(str(getattr(c, "field", "")))
            if not logical:
                continue

            # evidence row
            evidence_rows.append({
                PID: pid,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": logical,
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "EVIDENCE": getattr(c, "evidence", ""),
            })

            existing = best_by_pid[pid].get(logical)
            if logical in BOOLEAN_FIELDS:
                best_by_pid[pid][logical] = merge_boolean(existing, c)
            else:
                best_by_pid[pid][logical] = choose_best(existing, c)

    print("Patients with any extracted signals:", len(best_by_pid))

    print("\n4) Writing values into master + normalizing to gold conventions...")
    # fast index
    master[PID] = master[PID].astype(str).str.strip()
    master_idx = {pid: i for i, pid in enumerate(master[PID].tolist())}

    for pid, fields in best_by_pid.items():
        if pid not in master_idx:
            continue
        i = master_idx[pid]

        for logical, cand in fields.items():
            val = getattr(cand, "value", "")

            if logical == "SmokingStatus":
                master.at[i, "SmokingStatus"] = normalize_smoking(val)
                continue

            if logical == "BMI":
                # numeric BMI + obesity
                try:
                    bmi = float(str(val).strip())
                    master.at[i, "BMI"] = round(bmi, 1)
                    master.at[i, "Obesity"] = 1 if bmi >= 30.0 else 0
                except Exception:
                    # leave blank if not parseable
                    pass
                continue

            if logical == "Age":
                # store as int if possible
                try:
                    age = int(float(str(val).strip()))
                    master.at[i, "Age"] = age
                except Exception:
                    # leave existing structured Age if present
                    pass
                continue

            if logical in BOOLEAN_FIELDS:
                b = to_binary01(val)
                if b in (0, 1):
                    master.at[i, logical] = b
                continue

            # fallback: write raw
            if logical in master.columns:
                master.at[i, logical] = val

    # normalize Race/Ethnicity again (in case extractor ever touches them later)
    master["Race"] = master["Race"].apply(normalize_race)
    master["Ethnicity"] = master["Ethnicity"].apply(normalize_ethnicity)

    # ensure binary columns are 0/1 or blank (not "True"/"False")
    for c in BOOLEAN_FIELDS.union(set(["Obesity"])):
        if c in master.columns:
            master[c] = master[c].apply(lambda x: to_binary01(x) if norm_str(x) != "" else "")

    # order + write
    master = master[MASTER_COLUMNS].copy()
    master.to_csv(OUT_MASTER, index=False)

    pd.DataFrame(evidence_rows).to_csv(OUT_EVID, index=False)

    print("\nDONE.")
    print("Master:", OUT_MASTER)
    print("Evidence:", OUT_EVID)
    print("\nRun:")
    print("  python build_patient_master_from_original.py")


if __name__ == "__main__":
    main()
