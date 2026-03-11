#!/usr/bin/env python3
# build_master_rule_CANCER_RECON_PATCH.py
#
# PATCH-ONLY builder for:
# - Mastectomy_Laterality
# - Indication_Left
# - Indication_Right
# - LymphNode
# - Radiation
# - Radiation_Before
# - Radiation_After
# - Chemo
# - Chemo_Before
# - Chemo_After
# - Recon_Laterality
# - Recon_Type
# - Recon_Classification
# - Recon_Timing
#
# IMPORTANT:
# - Reads the EXISTING original master file
# - Updates ONLY the target variables above
# - Preserves all other columns exactly as they already are
# - Writes back to the SAME original master/evidence paths
#
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


def same_calendar_date(dt1, dt2):
    if dt1 is None or dt2 is None:
        return False
    return dt1.date() == dt2.date()


def days_between(dt1, dt2):
    if dt1 is None or dt2 is None:
        return None
    return (dt1.date() - dt2.date()).days


HEADER_RX = re.compile(r"^\s*([A-Z][A-Z0-9 /&\-]{2,60})\s*:\s*$")


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
    return SectionedNote(
        sections=sectionize(note_text),
        note_type=note_type or "",
        note_id=note_id or "",
        note_date=note_date or ""
    )


def cand_score(c):
    conf = float(getattr(c, "confidence", 0.0) or 0.0)
    nt = str(getattr(c, "note_type", "") or "").lower()
    sec = str(getattr(c, "section", "") or "").upper()
    evid = str(getattr(c, "evidence", "") or "").lower()

    op_bonus = 0.08 if ("op" in nt or "operative" in nt or "operation" in nt) else 0.0
    clinic_bonus = 0.02 if (
        "clinic" in nt or "progress" in nt or "consult" in nt or
        "oncology" in nt or "follow up" in nt or "follow-up" in nt
    ) else 0.0
    date_bonus = 0.01 if (getattr(c, "note_date", "") or "").strip() else 0.0
    section_penalty = -0.10 if sec in {
        "PAST MEDICAL HISTORY", "PAST SURGICAL HISTORY", "SURGICAL HISTORY",
        "HISTORY", "PMH", "PSH", "GYNECOLOGIC HISTORY", "OB HISTORY"
    } else 0.0
    history_penalty = -0.08 if re.search(r"\b(history of|hx of|status post|s/p|prior|previous)\b", evid) else 0.0
    procedure_bonus = 0.05 if re.search(r"\b(procedure|operative|operation|surgery|performed|placement)\b", evid) else 0.0

    return conf + op_bonus + clinic_bonus + date_bonus + section_penalty + history_penalty + procedure_bonus


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


def choose_best_indication(existing, new):
    if existing is None:
        return new

    ex_score = cand_score(existing)
    nw_score = cand_score(new)

    ex_val = clean_cell(getattr(existing, "value", ""))
    nw_val = clean_cell(getattr(new, "value", ""))

    if nw_score > ex_score:
        return new
    if ex_score > nw_score:
        return existing

    rank = {"Therapeutic": 3, "Prophylactic": 2, "None": 1, "": 0}
    if rank.get(nw_val, 0) > rank.get(ex_val, 0):
        return new
    return existing


def choose_best_lymphnode(existing, new):
    if existing is None:
        return new

    ex_val = clean_cell(getattr(existing, "value", ""))
    nw_val = clean_cell(getattr(new, "value", ""))

    rank = {"ALND": 3, "SLNB": 2, "none": 1, "": 0}
    ex_rank = rank.get(ex_val, 0)
    nw_rank = rank.get(nw_val, 0)

    if nw_rank > ex_rank:
        return new
    if ex_rank > nw_rank:
        return existing

    return choose_best(existing, new)


def choose_best_recon(existing, new):
    if existing is None:
        return new

    ex_score = cand_score(existing)
    nw_score = cand_score(new)

    ex_evid = str(getattr(existing, "evidence", "") or "").lower()
    nw_evid = str(getattr(new, "evidence", "") or "").lower()

    revision_rx = re.compile(
        r"\b(revision|fat graft|fat grafting|nipple reconstruction|nipple-areolar|tattoo|"
        r"capsulotomy|capsulectomy|symmetry|symmetrization|scar revision|dog ear|"
        r"lipofilling|liposuction|capsulorrhaphy)\b",
        re.IGNORECASE
    )

    anchor_rx = re.compile(
        r"\b(tissue expander placement|expander placement|implant placement|"
        r"implant-based reconstruction|direct-to-implant|diep flap|tram flap|siea flap|"
        r"latissimus dorsi flap|autologous reconstruction|free flap|immediate reconstruction|"
        r"delayed reconstruction)\b",
        re.IGNORECASE
    )

    ex_revision_only = bool(revision_rx.search(ex_evid)) and not bool(anchor_rx.search(ex_evid))
    nw_revision_only = bool(revision_rx.search(nw_evid)) and not bool(anchor_rx.search(nw_evid))

    if ex_revision_only and not nw_revision_only:
        return new
    if nw_revision_only and not ex_revision_only:
        return existing

    return new if nw_score > ex_score else existing


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

BOOLEAN_FIELDS = {
    "Radiation",
    "Chemo",
}

KEYWORD_PREFILTER = re.compile(
    r"\b("
    r"mastectomy|diep|tram|siea|gap|sgap|igap|latissimus|flap|reconstruction|expander|implant|"
    r"radiation|xrt|pmrt|chemo|chemotherapy|taxol|herceptin|"
    r"sentinel|axillary|alnd|slnb|prophylactic|carcinoma|dcis|lcis|oncology"
    r")\b",
    re.IGNORECASE
)

LEFT_RX = re.compile(r"\b(left|lt)\b", re.IGNORECASE)
RIGHT_RX = re.compile(r"\b(right|rt)\b", re.IGNORECASE)
BILAT_RX = re.compile(r"\b(bilateral|bilat)\b", re.IGNORECASE)

MASTECTOMY_RX = re.compile(
    r"\b(mastectomy|simple\s+mastectomy|total\s+mastectomy|skin[- ]sparing\s+mastectomy|nipple[- ]sparing\s+mastectomy|\bMRM\b)\b",
    re.IGNORECASE
)


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
    if (
        "gluteal artery perforator" in low or
        "gap flap" in low or
        re.search(r"\bsgap\b", low) or
        re.search(r"\bigap\b", low)
    ):
        flap_types_found.append("gluteal artery perforator flap")
    if "latissimus" in low:
        flap_types_found.append("latissimus dorsi")

    has_any_flap = (
        len(flap_types_found) > 0 or
        (" flap" in low) or
        low.startswith("flap") or
        ("mixed flaps" in low)
    )

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
        raise FileNotFoundError(
            "Existing master file not found: {0}\nRestore your original build first.".format(MASTER_PATH)
        )
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


def load_structured_encounters():
    rows = []
    struct_files = []
    for g in STRUCT_GLOBS:
        struct_files.extend(glob(g, recursive=True))

    for fp in sorted(set(struct_files)):
        df = clean_cols(read_csv_robust(fp))
        df = normalize_mrn(df)
        source_name = os.path.basename(fp).lower()

        if "operation encounters" in source_name:
            encounter_source = "operation"
            priority = 1
        elif "clinic encounters" in source_name:
            encounter_source = "clinic"
            priority = 2
        elif "inpatient encounters" in source_name:
            encounter_source = "inpatient"
            priority = 3
        else:
            encounter_source = "other"
            priority = 9

        recon_col = pick_col(df, ["RECONSTRUCTION_DATE", "RECONSTRUCTION DATE"], required=False)
        cpt_col = pick_col(df, ["CPT_CODE", "CPT CODE", "CPT"], required=False)
        proc_col = pick_col(df, ["PROCEDURE", "Procedure"], required=False)
        date_col = pick_col(df, ["OPERATION_DATE", "CHECKOUT_TIME", "DISCHARGE_DATE_DT"], required=False)

        out = pd.DataFrame()
        out[MERGE_KEY] = df[MERGE_KEY].astype(str).str.strip()
        out["STRUCT_SOURCE"] = encounter_source
        out["STRUCT_PRIORITY"] = priority
        out["STRUCT_DATE_RAW"] = df[date_col].astype(str) if date_col else ""
        out["RECONSTRUCTION_DATE_STRUCT"] = df[recon_col].astype(str) if recon_col else ""
        out["CPT_CODE_STRUCT"] = df[cpt_col].astype(str) if cpt_col else ""
        out["PROCEDURE_STRUCT"] = df[proc_col].astype(str) if proc_col else ""
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=[
            MERGE_KEY, "STRUCT_SOURCE", "STRUCT_PRIORITY", "STRUCT_DATE_RAW",
            "RECONSTRUCTION_DATE_STRUCT", "CPT_CODE_STRUCT", "PROCEDURE_STRUCT"
        ])

    return pd.concat(rows, ignore_index=True)


def choose_best_recon_anchor_rows(struct_df):
    recon_best = {}
    if len(struct_df) == 0:
        return recon_best

    source_priority = {
        "operation": 1,
        "clinic": 2,
        "inpatient": 3,
        "other": 9
    }

    preferred_cpts = set([
        "19357", "19340", "19342", "19361", "19364", "19367", "S2068"
    ])
    fallback_allowed_cpts = set([
        "19350", "19380"
    ])
    primary_exclude_cpts = set([
        "19325", "19330"
    ])

    eligible = struct_df.copy()
    has_preferred_cpt = {}

    for mrn, g in eligible.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            cpt = clean_cell(val).upper()
            if cpt in preferred_cpts:
                found = True
                break
        has_preferred_cpt[mrn] = found

    for _, row in eligible.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        recon_date = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))
        struct_date = parse_date_safe(row.get("STRUCT_DATE_RAW", ""))
        cpt_code = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
        procedure = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()

        if recon_date is None:
            continue
        if cpt_code in primary_exclude_cpts:
            continue
        if has_preferred_cpt.get(mrn, False) and cpt_code in fallback_allowed_cpts:
            continue

        is_anchor = False
        if cpt_code in preferred_cpts:
            is_anchor = True
        if (not has_preferred_cpt.get(mrn, False)) and (cpt_code in fallback_allowed_cpts):
            is_anchor = True
        if not is_anchor:
            if (
                ("breast recon" in procedure) or
                ("reconstruction" in procedure) or
                ("diep" in procedure) or
                ("tram" in procedure) or
                ("siea" in procedure) or
                ("latissimus" in procedure) or
                ("flap" in procedure) or
                ("expander" in procedure) or
                ("implant" in procedure) or
                ("direct-to-implant" in procedure)
            ):
                is_anchor = True

        if not is_anchor:
            continue

        score = (
            source_priority.get(source, 9),
            recon_date,
            struct_date if struct_date is not None else recon_date
        )

        current = recon_best.get(mrn)
        if current is None or score < current["score"]:
            lat = infer_laterality(procedure)
            rtype, rclass = infer_recon_type_and_class(procedure)
            recon_best[mrn] = {
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "source": source,
                "cpt_code": cpt_code,
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "recon_laterality": lat,
                "recon_type": rtype,
                "recon_classification": rclass,
                "score": score
            }

    return recon_best


def build_structured_mastectomy_events(struct_df):
    out = {}
    if len(struct_df) == 0:
        return out

    for _, row in struct_df.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        proc = clean_cell(row.get("PROCEDURE_STRUCT", ""))
        if not proc:
            continue
        if not MASTECTOMY_RX.search(proc):
            continue

        event_dt = parse_date_safe(row.get("STRUCT_DATE_RAW", ""))
        if event_dt is None:
            event_dt = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))

        lat = infer_laterality(proc)

        if mrn not in out:
            out[mrn] = []

        out[mrn].append({
            "date": event_dt,
            "laterality": lat,
            "procedure": proc
        })

    return out


def choose_best_mastectomy_event(events, recon_dt):
    if not events:
        return None

    best_same_day = None
    best_prior = None

    for ev in events:
        ev_dt = ev.get("date")
        if ev_dt is None:
            continue
        if recon_dt is not None and same_calendar_date(ev_dt, recon_dt):
            if best_same_day is None:
                best_same_day = ev
        elif recon_dt is not None and ev_dt.date() < recon_dt.date():
            if best_prior is None or ev_dt > best_prior.get("date"):
                best_prior = ev
        elif recon_dt is None:
            if best_prior is None or ev_dt > best_prior.get("date"):
                best_prior = ev

    if best_same_day is not None:
        return best_same_day
    return best_prior


def append_evidence_rows(existing_evid_df, new_rows):
    if existing_evid_df is None:
        return pd.DataFrame(new_rows)

    if len(new_rows) == 0:
        return existing_evid_df

    add_df = pd.DataFrame(new_rows)
    for c in existing_evid_df.columns:
        if c not in add_df.columns:
            add_df[c] = ""
    for c in add_df.columns:
        if c not in existing_evid_df.columns:
            existing_evid_df[c] = ""
    return pd.concat([existing_evid_df, add_df[existing_evid_df.columns]], ignore_index=True)


def load_existing_evidence():
    if os.path.exists(EVID_PATH):
        return clean_cols(read_csv_robust(EVID_PATH))
    return pd.DataFrame(columns=[
        MERGE_KEY,
        "NOTE_ID",
        "NOTE_DATE",
        "NOTE_TYPE",
        "FIELD",
        "VALUE",
        "STATUS",
        "CONFIDENCE",
        "SECTION",
        "EVIDENCE"
    ])


def main():
    print("Loading EXISTING master...")
    master = load_existing_master()
    print("Existing master rows: {0}".format(len(master)))

    print("Loading notes...")
    notes_df = load_and_reconstruct_notes()
    print("Reconstructed notes: {0}".format(len(notes_df)))

    print("Loading structured encounters...")
    struct_df = load_structured_encounters()
    print("Structured encounter rows: {0}".format(len(struct_df)))

    recon_anchor_map = choose_best_recon_anchor_rows(struct_df)
    mastectomy_events_map = build_structured_mastectomy_events(struct_df)

    evidence_rows = []
    best_by_mrn = {}
    therapy_dates = {}

    print("Running patch-only extractor...")

    for _, row in notes_df.iterrows():
        mrn = str(row[MERGE_KEY]).strip()
        note_text = clean_cell(row["NOTE_TEXT"])
        note_date = clean_cell(row["NOTE_DATE"])

        if not note_text:
            continue
        if not KEYWORD_PREFILTER.search(note_text):
            continue

        snote = build_sectioned_note(
            note_text=note_text,
            note_type=row["NOTE_TYPE"],
            note_id=row["NOTE_ID"],
            note_date=note_date
        )

        try:
            all_cands = extract_breast_cancer_recon(snote)
        except Exception as e:
            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": row["NOTE_ID"],
                "NOTE_DATE": row["NOTE_DATE"],
                "NOTE_TYPE": row["NOTE_TYPE"],
                "FIELD": "EXTRACTOR_ERROR",
                "VALUE": "",
                "STATUS": "",
                "CONFIDENCE": "",
                "SECTION": "",
                "EVIDENCE": "extract_breast_cancer_recon failed: {0}".format(repr(e))
            })
            continue

        if not all_cands:
            continue

        if mrn not in best_by_mrn:
            best_by_mrn[mrn] = {}
        if mrn not in therapy_dates:
            therapy_dates[mrn] = {"Radiation": [], "Chemo": [], "Mastectomy_Date": []}

        for c in all_cands:
            logical = FIELD_MAP.get(str(c.field))
            if not logical:
                continue

            evidence_rows.append({
                MERGE_KEY: mrn,
                "NOTE_ID": getattr(c, "note_id", row["NOTE_ID"]),
                "NOTE_DATE": getattr(c, "note_date", row["NOTE_DATE"]),
                "NOTE_TYPE": getattr(c, "note_type", row["NOTE_TYPE"]),
                "FIELD": logical,
                "VALUE": getattr(c, "value", ""),
                "STATUS": getattr(c, "status", ""),
                "CONFIDENCE": getattr(c, "confidence", ""),
                "SECTION": getattr(c, "section", ""),
                "EVIDENCE": getattr(c, "evidence", "")
            })

            if logical == "Radiation":
                dt = parse_date_safe(getattr(c, "note_date", row["NOTE_DATE"]))
                if dt is not None:
                    therapy_dates[mrn]["Radiation"].append(dt)

            if logical == "Chemo":
                dt = parse_date_safe(getattr(c, "note_date", row["NOTE_DATE"]))
                if dt is not None:
                    therapy_dates[mrn]["Chemo"].append(dt)

            if logical == "Mastectomy_Date":
                dt = parse_date_safe(getattr(c, "value", ""))
                if dt is not None:
                    therapy_dates[mrn]["Mastectomy_Date"].append(dt)

            existing = best_by_mrn[mrn].get(logical)

            if logical in BOOLEAN_FIELDS:
                best_by_mrn[mrn][logical] = merge_boolean(existing, c)
            elif logical == "LymphNode":
                best_by_mrn[mrn][logical] = choose_best_lymphnode(existing, c)
            elif logical in {"Indication_Left", "Indication_Right"}:
                best_by_mrn[mrn][logical] = choose_best_indication(existing, c)
            elif logical in {"Recon_Type", "Recon_Classification"}:
                best_by_mrn[mrn][logical] = choose_best_recon(existing, c)
            else:
                best_by_mrn[mrn][logical] = choose_best(existing, c)

    print("Applying updates to EXISTING master only for target fields...")

    all_mrns = master[MERGE_KEY].astype(str).str.strip().tolist()

    for mrn in all_mrns:
        mask = (master[MERGE_KEY].astype(str).str.strip() == mrn)
        if not mask.any():
            continue

        fields = best_by_mrn.get(mrn, {})
        recon_info = recon_anchor_map.get(mrn)
        recon_dt = parse_date_safe((recon_info or {}).get("recon_date", ""))

        for logical, cand in fields.items():
            if logical == "Mastectomy_Date":
                continue

            val = getattr(cand, "value", pd.NA)

            if logical in BOOLEAN_FIELDS:
                try:
                    val = 1 if bool(val) else 0
                except Exception:
                    val = pd.NA

            if logical in TARGET_FIELDS:
                master.loc[mask, logical] = val

        # structured recon anchor should help stabilize recon fields
        if recon_info is not None:
            current_val = clean_cell(master.loc[mask, "Recon_Laterality"].iloc[0])
            if clean_cell(recon_info.get("recon_laterality", "")):
                if not current_val or current_val in {"", "BILATERAL"}:
                    master.loc[mask, "Recon_Laterality"] = recon_info["recon_laterality"]

            current_val = clean_cell(master.loc[mask, "Recon_Type"].iloc[0])
            if clean_cell(recon_info.get("recon_type", "")):
                if not current_val or current_val in {"other"}:
                    master.loc[mask, "Recon_Type"] = recon_info["recon_type"]

            current_val = clean_cell(master.loc[mask, "Recon_Classification"].iloc[0])
            if clean_cell(recon_info.get("recon_classification", "")):
                if not current_val or current_val in {"other"}:
                    master.loc[mask, "Recon_Classification"] = recon_info["recon_classification"]

        best_mast_ev = choose_best_mastectomy_event(mastectomy_events_map.get(mrn, []), recon_dt)
        current_mast_lat = clean_cell(master.loc[mask, "Mastectomy_Laterality"].iloc[0])
        if not current_mast_lat:
            if best_mast_ev is not None and clean_cell(best_mast_ev.get("laterality", "")):
                master.loc[mask, "Mastectomy_Laterality"] = best_mast_ev["laterality"]

        current_lymph = clean_cell(master.loc[mask, "LymphNode"].iloc[0])
        if not current_lymph:
            master.loc[mask, "LymphNode"] = "none"

        timing_val = clean_cell(master.loc[mask, "Recon_Timing"].iloc[0])
        if not timing_val and recon_dt is not None:
            immediate = False
            delayed = False

            for ev in mastectomy_events_map.get(mrn, []):
                ev_dt = ev.get("date")
                if ev_dt is None:
                    continue
                if same_calendar_date(ev_dt, recon_dt):
                    immediate = True
                    break
                if ev_dt.date() < recon_dt.date():
                    delayed = True

            if not immediate:
                for ev_dt in therapy_dates.get(mrn, {}).get("Mastectomy_Date", []):
                    if same_calendar_date(ev_dt, recon_dt):
                        immediate = True
                        break
                    if ev_dt.date() < recon_dt.date():
                        delayed = True

            if immediate:
                master.loc[mask, "Recon_Timing"] = "Immediate"
            elif delayed:
                master.loc[mask, "Recon_Timing"] = "Delayed"

        rad_dates = therapy_dates.get(mrn, {}).get("Radiation", [])
        chemo_dates = therapy_dates.get(mrn, {}).get("Chemo", [])

        rad_before = 0
        rad_after = 0
        if recon_dt is not None:
            for dt in rad_dates:
                dd = days_between(dt, recon_dt)
                if dd is None:
                    continue
                if dd < 0:
                    rad_before = 1
                elif dd > 0:
                    rad_after = 1

        chemo_before = 0
        chemo_after = 0
        if recon_dt is not None:
            for dt in chemo_dates:
                dd = days_between(dt, recon_dt)
                if dd is None:
                    continue
                if dd < 0:
                    chemo_before = 1
                elif dd > 0:
                    chemo_after = 1

        master.loc[mask, "Radiation_Before"] = rad_before
        master.loc[mask, "Radiation_After"] = rad_after
        master.loc[mask, "Chemo_Before"] = chemo_before
        master.loc[mask, "Chemo_After"] = chemo_after

        if rad_before or rad_after:
            master.loc[mask, "Radiation"] = 1
        else:
            current_rad = clean_cell(master.loc[mask, "Radiation"].iloc[0])
            if current_rad in {"1", "True", "true"} or len(rad_dates) > 0:
                master.loc[mask, "Radiation"] = 1
            else:
                master.loc[mask, "Radiation"] = 0

        if chemo_before or chemo_after:
            master.loc[mask, "Chemo"] = 1
        else:
            current_chemo = clean_cell(master.loc[mask, "Chemo"].iloc[0])
            if current_chemo in {"1", "True", "true"} or len(chemo_dates) > 0:
                master.loc[mask, "Chemo"] = 1
            else:
                master.loc[mask, "Chemo"] = 0

    print("Appending evidence without deleting old evidence...")
    old_evid = load_existing_evidence()
    new_evid = append_evidence_rows(old_evid, evidence_rows)

    os.makedirs(os.path.dirname(MASTER_PATH), exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)
    new_evid.to_csv(EVID_PATH, index=False)

    print("\nDONE.")
    print("- Patched existing master: {0}".format(MASTER_PATH))
    print("- Appended evidence: {0}".format(EVID_PATH))
    print("\nRun:")
    print(" python build_master_rule_CANCER_RECON_PATCH.py")


if __name__ == "__main__":
    main()
