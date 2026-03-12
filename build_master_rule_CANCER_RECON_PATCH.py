#!/usr/bin/env python3
# build_master_rule_CANCER_RECON_PATCH.py
# Python 3.6.8 compatible
import os
import re
from glob import glob
from datetime import datetime
import pandas as pd

BASE_DIR = "/home/apokol/Breast_Restore"
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
MASTER_PATH = "{0}/_outputs/master_abstraction_rule_FINAL_NO_GOLD.csv".format(BASE_DIR)
EVID_PATH = "{0}/_outputs/rule_hit_evidence_FINAL_NO_GOLD.csv".format(BASE_DIR)
MERGE_KEY = "MRN"
TARGET_FIELDS = [
    "Mastectomy_Laterality",
    "Indication_Left",
    "Indication_Right",
    "LymphNode",
    "Radiation",
    "Radiation_Before",
    "Radiation_After",
    "Chemo",
    "Chemo_Before",
    "Chemo_After",
    "Recon_Laterality",
    "Recon_Type",
    "Recon_Classification",
    "Recon_Timing",
]

from models import SectionedNote, Candidate  # noqa: E402
from extractors.breast_cancer_recon import extract_breast_cancer_recon  # noqa: E402

LEFT_RX = re.compile(r"\b(left|lt)\b", re.IGNORECASE)
RIGHT_RX = re.compile(r"\b(right|rt)\b", re.IGNORECASE)
BILAT_RX = re.compile(r"\b(bilateral|bilat)\b", re.IGNORECASE)
MASTECTOMY_RX = re.compile(
    r"\b(mastectomy|simple\s+mastectomy|total\s+mastectomy|skin[- ]sparing\s+mastectomy|nipple[- ]sparing\s+mastectomy|\bMRM\b)\b",
    re.IGNORECASE,
)
HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")
KEYWORD_PREFILTER = re.compile(
    r"\b(mastectomy|diep|tram|siea|gap|sgap|igap|latissimus|flap|reconstruction|expander|implant|radiation|xrt|pmrt|chemo|chemotherapy|taxol|herceptin|sentinel|axillary|alnd|slnb|prophylactic|carcinoma|dcis|lcis|oncology)\b",
    re.IGNORECASE,
)
FIELD_MAP = {
    "Mastectomy_Laterality": "Mastectomy_Laterality",
    "Mastectomy_Date": "Mastectomy_Date",
    "Indication_Left": "Indication_Left",
    "Indication_Right": "Indication_Right",
    "LymphNode": "LymphNode",
    "Radiation": "Radiation",
    "Chemo": "Chemo",
    "Recon_Laterality": "Recon_Laterality",
    "Recon_Type": "Recon_Type",
    "Recon_Classification": "Recon_Classification",
    "Recon_Timing": "Recon_Timing",
}
BOOLEAN_FIELDS = {"Radiation", "Chemo"}


def read_csv_robust(path):
    common_kwargs = dict(dtype=str, engine="python")
    try:
        return pd.read_csv(path, **common_kwargs, on_bad_lines="skip")
    except TypeError:
        try:
            return pd.read_csv(path, **common_kwargs, error_bad_lines=False, warn_bad_lines=True)
        except UnicodeDecodeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(path, **common_kwargs, encoding="latin-1", error_bad_lines=False, warn_bad_lines=True)


def clean_cols(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


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
        raise RuntimeError("Required column missing. Tried={0}. Seen={1}".format(options, list(df.columns)[:60]))
    return None


def clean_cell(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def to_int_safe(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def parse_date_safe(x):
    s = clean_cell(x)
    if not s:
        return None
    fmts = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%Y %H:%M:%S"]
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


def same_calendar_date(dt1, dt2):
    if dt1 is None or dt2 is None:
        return False
    return dt1.date() == dt2.date()


def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days


def sectionize(text):
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


def build_sectioned_note(note_text, note_type, note_id, note_date):
    return SectionedNote(sections=sectionize(note_text), note_type=note_type or "", note_id=note_id or "", note_date=note_date or "")


def cand_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    op_bonus = 0.05 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    clinic_bonus = 0.03 if ("clinic" in nt or "progress" in nt or "consult" in nt or "oncology" in nt or "follow up" in nt or "follow-up" in nt) else 0.0
    date_bonus = 0.01 if (getattr(c, "note_date", "") or "").strip() else 0.0
    return conf + op_bonus + clinic_bonus + date_bonus


def choose_best(existing, new):
    if existing is None:
        return new
    return new if cand_score(new) > cand_score(existing) else existing


def merge_boolean(existing, new):
    if existing is None:
        return new
    try:
        exv = bool(existing.value)
        nwv = bool(new.value)
    except Exception:
        return choose_best(existing, new)
    if nwv and not exv:
        return new
    if exv and not nwv:
        return existing
    return choose_best(existing, new)


def infer_laterality(text):
    low = clean_cell(text).lower()
    if BILAT_RX.search(low):
        return "BILATERAL"
    has_left = bool(LEFT_RX.search(low))
    has_right = bool(RIGHT_RX.search(low))
    if has_left and has_right:
        return "BILATERAL"
    if has_left:
        return "LEFT"
    if has_right:
        return "RIGHT"
    return None


def infer_recon_type_and_class(text):
    low = clean_cell(text).lower()
    flap_types_found = []
    if "diep" in low:
        flap_types_found.append("DIEP")
    if "tram" in low:
        flap_types_found.append("TRAM")
    if "siea" in low:
        flap_types_found.append("SIEA")
    if ("gluteal artery perforator" in low or "gap flap" in low or re.search(r"\bsgap\b", low) or re.search(r"\bigap\b", low)):
        flap_types_found.append("gluteal artery perforator flap")
    if "latissimus" in low:
        flap_types_found.append("latissimus dorsi")
    has_any_flap = (len(flap_types_found) > 0 or (" flap" in low) or low.startswith("flap") or ("mixed flaps" in low))
    has_direct_to_implant = bool(re.search(r"\bdirect[- ]to[- ]implant\b", low))
    has_expander = ("tissue expander" in low) or ("expander" in low)
    has_implant = ("implant" in low)
    rtype = None
    if "mixed flaps" in low:
        rtype = "mixed flaps"
    elif len(set(flap_types_found)) >= 2:
        rtype = "mixed flaps"
    elif "DIEP" in flap_types_found:
        rtype = "DIEP"
    elif "TRAM" in flap_types_found:
        rtype = "TRAM"
    elif "SIEA" in flap_types_found:
        rtype = "SIEA"
    elif "gluteal artery perforator flap" in flap_types_found:
        rtype = "gluteal artery perforator flap"
    elif "latissimus dorsi" in flap_types_found:
        rtype = "latissimus dorsi"
    elif has_direct_to_implant:
        rtype = "direct-to-implant"
    elif has_expander or (has_implant and not has_any_flap):
        rtype = "expander/implant"
    elif has_any_flap:
        rtype = "other"
    elif has_implant:
        rtype = "expander/implant"
    rclass = None
    if has_any_flap:
        rclass = "autologous"
    elif has_direct_to_implant or has_expander or has_implant:
        rclass = "implant"
    elif rtype == "other":
        rclass = "other"
    return rtype, rclass


def load_existing_master():
    if not os.path.exists(MASTER_PATH):
        raise FileNotFoundError("Existing master file not found: {0}\nRestore your original build first.".format(MASTER_PATH))
    master = clean_cols(read_csv_robust(MASTER_PATH))
    master = normalize_mrn(master)
    for c in TARGET_FIELDS:
        if c not in master.columns:
            master[c] = pd.NA
    return master


def load_and_reconstruct_notes():
    note_files = []
    for g in NOTE_GLOBS:
        note_files.extend(glob(g, recursive=True))
    note_files = sorted(set(note_files))
    if not note_files:
        raise FileNotFoundError("No HPI11526 * Notes.csv files found.")
    all_notes_rows = []
    for fp in note_files:
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        note_text_col = pick_col(df, ["NOTE_TEXT", "NOTE TEXT", "NOTE_TEXT_FULL", "TEXT", "NOTE"])
        note_id_col = pick_col(df, ["NOTE_ID", "NOTE ID"])
        line_col = pick_col(df, ["LINE"], required=False)
        note_type_col = pick_col(df, ["NOTE_TYPE", "NOTE TYPE"], required=False)
        date_col = pick_col(df, ["NOTE_DATE_OF_SERVICE", "NOTE DATE OF SERVICE", "OPERATION_DATE", "ADMIT_DATE", "HOSP_ADMSN_TIME"], required=False)
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
        tmp = tmp.rename(columns={note_id_col: "NOTE_ID", note_text_col: "NOTE_TEXT"})
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
        note_type = g["NOTE_TYPE"].astype(str).iloc[0] if g["NOTE_TYPE"].astype(str).str.strip().any() else g["_SOURCE_FILE_"].astype(str).iloc[0]
        note_date = g["NOTE_DATE_OF_SERVICE"].astype(str).iloc[0] if g["NOTE_DATE_OF_SERVICE"].astype(str).str.strip().any() else ""
        reconstructed.append({
            MERGE_KEY: mrn,
            "NOTE_ID": nid,
            "NOTE_TYPE": note_type,
            "NOTE_DATE": note_date,
            "SOURCE_FILE": g["_SOURCE_FILE_"].astype(str).iloc[0],
            "NOTE_TEXT": full_text,
        })
    return pd.DataFrame(reconstructed)


def load_structured_encounters():
    rows = []
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))
    for fp in sorted(set(struct_files)):
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=[MERGE_KEY])
    return pd.concat(rows, ignore_index=True, sort=False)


def collect_candidates(notes_df):
    by_mrn = {}
    for mrn, g in notes_df.groupby(MERGE_KEY):
        out = {}
        for _, row in g.iterrows():
            text = clean_cell(row.get("NOTE_TEXT"))
            if not text:
                continue
            if not KEYWORD_PREFILTER.search(text):
                continue
            note = build_sectioned_note(text, row.get("NOTE_TYPE", ""), row.get("NOTE_ID", ""), row.get("NOTE_DATE", ""))
            cands = extract_breast_cancer_recon(note)
            for c in cands:
                field = FIELD_MAP.get(c.field)
                if not field:
                    continue
                existing = out.get(field)
                if field in BOOLEAN_FIELDS:
                    out[field] = merge_boolean(existing, c)
                elif field == "LymphNode":
                    if existing is None:
                        out[field] = c
                    elif str(c.value).upper() == "ALND" and str(existing.value).upper() != "ALND":
                        out[field] = c
                    else:
                        out[field] = choose_best(existing, c)
                else:
                    out[field] = choose_best(existing, c)
        by_mrn[mrn] = out
    return by_mrn


def apply_to_master(master, cand_map):
    evidence_rows = []
    for idx in master.index:
        mrn = clean_cell(master.at[idx, MERGE_KEY])
        m = cand_map.get(mrn, {})
        for field in TARGET_FIELDS:
            c = m.get(field)
            if c is None:
                continue
            val = c.value
            if isinstance(val, bool):
                val = True if val else False
            master.at[idx, field] = val
            evidence_rows.append({
                "MRN": mrn,
                "field": field,
                "value": val,
                "evidence": getattr(c, "evidence", ""),
                "section": getattr(c, "section", ""),
                "note_type": getattr(c, "note_type", ""),
                "note_id": getattr(c, "note_id", ""),
                "note_date": getattr(c, "note_date", ""),
                "confidence": getattr(c, "confidence", ""),
            })
    return master, pd.DataFrame(evidence_rows)


def main():
    master = load_existing_master()
    notes_df = load_and_reconstruct_notes()
    _ = load_structured_encounters()
    cand_map = collect_candidates(notes_df)
    master, evid_df = apply_to_master(master, cand_map)
    out_dir = os.path.dirname(MASTER_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    master.to_csv(MASTER_PATH, index=False)
    if os.path.exists(EVID_PATH):
        old = clean_cols(read_csv_robust(EVID_PATH))
        if old is not None and not old.empty:
            evid_df = pd.concat([old, evid_df], ignore_index=True, sort=False)
    evid_dir = os.path.dirname(EVID_PATH)
    if evid_dir and not os.path.exists(evid_dir):
        os.makedirs(evid_dir)
    evid_df.to_csv(EVID_PATH, index=False)
    print("DONE")
    print("Master updated: {0}".format(MASTER_PATH))
    print("Evidence updated: {0}".format(EVID_PATH))


if __name__ == "__main__":
    main()
