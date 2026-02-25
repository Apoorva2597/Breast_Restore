#!/usr/bin/env python3
# export_one_patient_deid_bundle.py
# Python 3.6.8 compatible
#
# Export ONE de-identified patient bundle (notes + encounter timelines).
# Supports:
#   - direct --patient_id (ENCRYPTED_PAT_ID)
#   - OR --mrn (will resolve MRN -> ENCRYPTED_PAT_ID via a crosswalk CSV)
#   - OR pick an exemplar from validation_merged.csv using --outcome + --case_type (FP/FN/TP/TN)

from __future__ import print_function
import os
import re
import argparse
import random
import glob
import pandas as pd


# ----------------------------
# HARD-CODED DEFAULTS (edit only if paths change)
# ----------------------------
DEFAULT_PATIENT_ID = "63B0526207E98425D35E7EA737AB89AA"
DEFAULT_OUT_DIR = "/home/apokol/Breast_Restore/PATIENT_BUNDLES"

# DE-ID note text files (you said: v3 clinic, v4 op, v5 IP)
DEFAULT_DEID_INPUTS = [
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_Clinic_Notes_CTXWIPE_v3.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v4.csv",
    "/home/apokol/Breast_Restore/DEID_FULLTEXT_HPI11526_NOTES_CTXWIPE_v5.csv",
]

# Structured encounter files (has usable dates)
DEFAULT_ENCOUNTER_INPUTS = [
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Encounters.csv",
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Encounters.csv",
]

# MRN -> ENCRYPTED_PAT_ID crosswalk
DEFAULT_MRN_CROSSWALK = "/home/apokol/Breast_Restore/CROSSWALK/CROSSWALK__MRN_to_patient_id__vNEW.csv"

# Validation merged default
DEFAULT_VALIDATION_MERGED = "/home/apokol/Breast_Restore/_outputs/validation_merged.csv"


# ----------------------------
# Column detection helpers
# ----------------------------
def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _norm_col(c):
    return re.sub(r"\s+", "_", _safe_str(c).strip().upper())

def detect_pid_col(columns):
    for c in columns:
        if _norm_col(c) == "ENCRYPTED_PAT_ID":
            return c
    for c in columns:
        lc = _safe_str(c).lower()
        if "encrypt" in lc and "id" in lc:
            return c
    for c in columns:
        lc = _safe_str(c).lower()
        if ("pat" in lc or "patient" in lc) and "id" in lc:
            return c
    return None

def detect_mrn_col(columns):
    preferred = set(["MRN", "PAT_MRN", "PATIENT_MRN", "MEDICAL_RECORD_NUMBER"])
    for c in columns:
        if _norm_col(c) in preferred:
            return c
    for c in columns:
        if "mrn" in _safe_str(c).lower():
            return c
    return None

def detect_note_type_col(columns):
    for c in columns:
        if _norm_col(c) == "NOTE_TYPE":
            return c
    for c in columns:
        lc = _safe_str(c).lower()
        if "note" in lc and "type" in lc:
            return c
    return None

def detect_deid_text_col(columns):
    candidates = []
    for c in columns:
        uc = _safe_str(c).strip().upper()
        if uc in ("NOTE_TEXT_DEID", "NOTE_DEID", "TEXT_DEID", "NOTE_TEXT_DEIDENTIFIED"):
            return c
        if "DEID" in uc or "DE-ID" in uc or "DE_IDENT" in uc:
            candidates.append(c)
    for c in candidates:
        uc = _safe_str(c).upper()
        if "NOTE" in uc and "TEXT" in uc:
            return c
    return candidates[0] if candidates else None

def detect_datetime_col(columns):
    date_like = []
    for c in columns:
        lc = _safe_str(c).lower()
        if any(k in lc for k in ["date", "datetime", "time", "created", "service"]):
            date_like.append(c)

    priority = [
        "service_date", "note_date", "note_datetime", "note_time",
        "encounter_date", "admit_date", "surgery_date", "created_date", "created"
    ]
    for key in priority:
        for c in date_like:
            if key in _safe_str(c).lower():
                return c
    return date_like[0] if date_like else None

def detect_encounter_date_cols(columns):
    norm_map = {}
    for c in columns:
        norm_map[_norm_col(c)] = c

    priority_norm = [
        "RECONSTRUCTION_DATE",
        "OPERATION_DATE",
        "DISCHARGE_DATE_DT",
        "ADMIT_DATE",
        "HOSP_ADMSN_TIME",
        "HOSP_DISCHRG_TIME",
        "CHECKOUT_TIME",
        "ENCOUNTER_DATE",
        "VISIT_DATE",
        "DATE",
    ]

    out = []
    for p in priority_norm:
        if p in norm_map:
            out.append(norm_map[p])

    if not out:
        for c in columns:
            n = _norm_col(c)
            if ("DATE" in n) or ("TIME" in n) or ("DT" in n):
                out.append(c)

    seen = set()
    dedup = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


# ----------------------------
# IO helpers
# ----------------------------
def ensure_out_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=object, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, dtype=object, engine="python", encoding="latin1")

def normalize_note_type(x):
    t = _safe_str(x).strip()
    t = re.sub(r"\s+", " ", t)
    return t if t else "UNKNOWN_NOTE_TYPE"

def try_parse_datetime(series):
    try:
        parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
        non_null = parsed.notnull().sum()
        if non_null < max(3, int(0.05 * len(parsed))):
            return None
        return parsed
    except Exception:
        return None

def to01(v):
    if v is None:
        return 0
    s = _safe_str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except Exception:
        return 0


# ----------------------------
# MRN -> ENCRYPTED_PAT_ID resolution
# ----------------------------
def resolve_patient_id_from_mrn(mrn, crosswalk_path, mrn_col_override=None, pid_col_override=None):
    if not os.path.exists(crosswalk_path):
        raise RuntimeError("MRN crosswalk file not found: {}".format(crosswalk_path))

    df = read_csv_robust(crosswalk_path)

    mrn_col = mrn_col_override if mrn_col_override else detect_mrn_col(df.columns)
    if mrn_col is None:
        raise RuntimeError("Could not detect MRN column in crosswalk: {}".format(crosswalk_path))

    pid_col = pid_col_override if pid_col_override else detect_pid_col(df.columns)
    if pid_col is None:
        raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column in crosswalk: {}".format(crosswalk_path))

    mrn = _safe_str(100036884).strip()
    if not mrn:
        raise RuntimeError("MRN is empty.")

    s = df[mrn_col].astype(str).str.strip()
    match = df[s == mrn]

    if match.empty:
        raise RuntimeError("No ENCRYPTED_PAT_ID found for MRN={} in {}".format(mrn, crosswalk_path))

    pid = _safe_str(match.iloc[0][pid_col]).strip()
    if not pid:
        raise RuntimeError("Matched row had empty ENCRYPTED_PAT_ID for MRN={}".format(mrn))

    return pid, mrn_col, pid_col, len(match)


# ----------------------------
# Stage2 anchor loading (frozen preferred)
# ----------------------------
def find_latest_frozen_stage2_patient_clean(root):
    base = os.path.join(root, "_frozen_stage2")
    if not os.path.isdir(base):
        return None
    cands = sorted(glob.glob(os.path.join(base, "*", "stage2_patient_clean.csv")))
    if not cands:
        return None
    return os.path.abspath(cands[-1])

def find_patient_stage_summary(root):
    p = os.path.join(root, "_outputs", "patient_stage_summary.csv")
    return os.path.abspath(p) if os.path.isfile(p) else None

def parse_date_any(x):
    s = _safe_str(x).strip()
    if not s:
        return None
    try:
        # pandas handles most formats
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if pd.isnull(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None

def load_stage2_anchor_dt(root, patient_id):
    # frozen preferred
    frozen = find_latest_frozen_stage2_patient_clean(root)
    if frozen and os.path.exists(frozen):
        df = read_csv_robust(frozen)
        pid_col = detect_pid_col(df.columns)
        if pid_col and "STAGE2_DATE" in df.columns:
            sub = df[df[pid_col].astype(str).str.strip() == _safe_str(patient_id).strip()]
            if not sub.empty:
                dt = parse_date_any(sub.iloc[0]["STAGE2_DATE"])
                if dt:
                    return dt, "frozen_stage2_patient_clean", frozen
    # fallback outputs
    summ = find_patient_stage_summary(root)
    if summ and os.path.exists(summ):
        df = read_csv_robust(summ)
        pid_col = detect_pid_col(df.columns)
        if pid_col and "STAGE2_DATE" in df.columns:
            sub = df[df[pid_col].astype(str).str.strip() == _safe_str(patient_id).strip()]
            if not sub.empty:
                dt = parse_date_any(sub.iloc[0]["STAGE2_DATE"])
                if dt:
                    return dt, "patient_stage_summary", summ
    return None, "NONE", ""


# ----------------------------
# Validation exemplar picker
# ----------------------------
def pick_case_from_validation(validation_path, outcome, case_type):
    """
    Returns dict with MRN, ENCRYPTED_PAT_ID (if present), and context cols.
    """
    if not os.path.exists(validation_path):
        raise RuntimeError("Validation file not found: {}".format(validation_path))

    df = read_csv_robust(validation_path)

    # Determine columns
    mrn_col = detect_mrn_col(df.columns) or "MRN" if "MRN" in df.columns else None
    pid_col = detect_pid_col(df.columns)

    # Map outcome -> gold/pred cols (your merged file uses these names)
    gold_map = {
        "MinorComp": "GOLD_Stage2_MinorComp",
        "Reoperation": "GOLD_Stage2_Reoperation",
        "Rehospitalization": "GOLD_Stage2_Rehospitalization",
        "MajorComp": "GOLD_Stage2_MajorComp",
        "Failure": "GOLD_Stage2_Failure",
        "Revision": "GOLD_Stage2_Revision",
    }
    pred_map = {
        "MinorComp": "Stage2_MinorComp_pred",
        "Reoperation": "Stage2_Reoperation_pred",
        "Rehospitalization": "Stage2_Rehospitalization_pred",
        "MajorComp": "Stage2_MajorComp_pred",
        "Failure": "Stage2_Failure_pred",
        "Revision": "Stage2_Revision_pred",
    }

    if outcome not in gold_map or outcome not in pred_map:
        raise RuntimeError("Unknown outcome: {}. Use one of: {}".format(outcome, sorted(gold_map.keys())))

    gold_col = gold_map[outcome]
    pred_col = pred_map[outcome]

    if gold_col not in df.columns or pred_col not in df.columns:
        raise RuntimeError("Missing columns in validation file for {}: {} / {}".format(outcome, gold_col, pred_col))

    tmp = df.copy()
    tmp["_g"] = tmp[gold_col].map(to01)
    tmp["_p"] = tmp[pred_col].map(to01)

    ct = case_type.upper().strip()
    if ct == "FP":
        sub = tmp[(tmp["_g"] == 0) & (tmp["_p"] == 1)]
    elif ct == "FN":
        sub = tmp[(tmp["_g"] == 1) & (tmp["_p"] == 0)]
    elif ct == "TP":
        sub = tmp[(tmp["_g"] == 1) & (tmp["_p"] == 1)]
    elif ct == "TN":
        sub = tmp[(tmp["_g"] == 0) & (tmp["_p"] == 0)]
    else:
        raise RuntimeError("case_type must be one of FP/FN/TP/TN")

    if sub.empty:
        raise RuntimeError("No rows found for outcome={} case_type={} in {}".format(outcome, ct, validation_path))

    # Prefer rows that have an MRN (so you can de-id via MRN)
    if mrn_col:
        ok = sub[mrn_col].astype(str).str.strip()
        sub["_mrn_ok"] = (ok != "") & (ok.str.lower() != "nan")
        sub = sub.sort_values(["_mrn_ok"], ascending=False)

    r = sub.iloc[0].to_dict()

    ctx = {
        "outcome": outcome,
        "case_type": ct,
        "gold_col": gold_col,
        "pred_col": pred_col,
        "gold": _safe_str(r.get(gold_col, "")),
        "pred": _safe_str(r.get(pred_col, "")),
        "MRN": _safe_str(r.get(mrn_col, "")) if mrn_col else "",
        "ENCRYPTED_PAT_ID": _safe_str(r.get(pid_col, "")) if pid_col else "",
        "STAGE2_DATE": _safe_str(r.get("STAGE2_DATE", "")),
        "WINDOW_START": _safe_str(r.get("WINDOW_START", "")),
        "WINDOW_END": _safe_str(r.get("WINDOW_END", "")),
    }

    # Include any evidence columns that exist for convenience
    for c in df.columns:
        if "evidence" in _safe_str(c).lower() or "snippet" in _safe_str(c).lower() or "pattern" in _safe_str(c).lower():
            ctx[c] = _safe_str(r.get(c, ""))

    return ctx


# ----------------------------
# Bundle writers
# ----------------------------
def write_notes_bundle(patient_id, rows, out_dir):
    safe_pid = re.sub(r"[^A-Za-z0-9_\-]+", "_", patient_id)
    patient_dir = os.path.join(out_dir, safe_pid)
    ensure_out_dir(patient_dir)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            (r["dt_parsed"] is None, r["dt_parsed"]) if r["dt_parsed"] is not None else (True, None),
            r["source_file"],
            r["row_idx"],
        )
    )

    timeline_path = os.path.join(patient_dir, "timeline.csv")
    timeline_df = pd.DataFrame([{
        "ENCRYPTED_PAT_ID": r["pid"],
        "NOTE_TYPE": r["note_type"],
        "NOTE_DATETIME_RAW": r["dt_raw"],
        "SOURCE_FILE": r["source_file"],
        "ROW_IDX": r["row_idx"],
        "NOTE_TEXT_DEID_LEN": len(_safe_str(r["note_text_deid"]))
    } for r in rows_sorted])
    timeline_df.to_csv(timeline_path, index=False, encoding="utf-8")

    combined_path = os.path.join(patient_dir, "ALL_NOTES_COMBINED.txt")
    with open(combined_path, "w") as f_out:
        f_out.write("DE-ID PATIENT NOTE BUNDLE\n")
        f_out.write("=========================\n\n")
        f_out.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
        f_out.write("TOTAL_NOTES: {}\n".format(len(rows_sorted)))
        f_out.write("\n---\n\n")

        for i, r in enumerate(rows_sorted, start=1):
            note_type = r["note_type"]
            dt = _safe_str(r["dt_raw"]).strip()
            src = r["source_file"]
            idx = r["row_idx"]
            text = _safe_str(r["note_text_deid"])

            note_fname = "note_{:04d}__{}.txt".format(
                i,
                re.sub(r"[^A-Za-z0-9_\-]+", "_", note_type)[:60] or "UNKNOWN"
            )
            note_path = os.path.join(patient_dir, note_fname)
            with open(note_path, "w") as nf:
                nf.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
                nf.write("NOTE_NUMBER: {}\n".format(i))
                nf.write("NOTE_TYPE: {}\n".format(note_type))
                nf.write("NOTE_DATETIME_RAW: {}\n".format(dt))
                nf.write("SOURCE_FILE: {}\n".format(src))
                nf.write("ROW_IDX: {}\n".format(idx))
                nf.write("\n--- NOTE_TEXT_DEID ---\n\n")
                nf.write(text)

            f_out.write("NOTE {:04d}\n".format(i))
            f_out.write("---------\n")
            f_out.write("NOTE_TYPE: {}\n".format(note_type))
            f_out.write("NOTE_DATETIME_RAW: {}\n".format(dt))
            f_out.write("SOURCE_FILE: {}\n".format(src))
            f_out.write("ROW_IDX: {}\n".format(idx))
            f_out.write("\n--- NOTE_TEXT_DEID ---\n\n")
            f_out.write(text)
            f_out.write("\n\n" + ("=" * 80) + "\n\n")

    return patient_dir, timeline_path, combined_path

def write_encounters_timeline(encounter_rows, patient_dir):
    out_path = os.path.join(patient_dir, "encounters_timeline.csv")
    df = pd.DataFrame(encounter_rows)

    if len(df) > 0 and "BEST_EVENT_DT_PARSED" in df.columns:
        # Sort by parsed datetime
        try:
            df["_sort"] = df["BEST_EVENT_DT_PARSED"].apply(lambda x: x if x is not None else pd.NaT)
            df = df.sort_values(["_sort", "SOURCE_FILE", "ROW_IDX"], ascending=True)
            df = df.drop(columns=["_sort"])
        except Exception:
            pass

    preferred = [
        "ENCRYPTED_PAT_ID", "BEST_EVENT_DT_RAW", "BEST_EVENT_DT_PARSED",
        "ANCHOR_SOURCE_COL", "SOURCE_FILE",
        "PAT_ENC_CSN_ID", "ENCRYPTED_CSN", "ENCOUNTER_TYPE",
        "DEPARTMENT", "OP_DEPARTMENT",
        "RECONSTRUCTION_DATE", "OPERATION_DATE", "ADMIT_DATE", "DISCHARGE_DATE_DT",
        "CPT_CODE", "PROCEDURE", "REASON_FOR_VISIT",
        "ROW_IDX"
    ]
    cols = []
    for c in preferred:
        if c in df.columns:
            cols.append(c)
    for c in df.columns:
        if c not in cols:
            cols.append(c)
    df = df[cols] if len(cols) else df

    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def write_anchor_summary_stage2(patient_id, stage2_dt, source_label, source_path, patient_dir):
    out_path = os.path.join(patient_dir, "stage2_anchor_summary.txt")
    with open(out_path, "w") as f:
        f.write("STAGE 2 ANCHOR SUMMARY (STAGE2_DATE-BASED)\n")
        f.write("=========================================\n\n")
        f.write("ENCRYPTED_PAT_ID: {}\n".format(patient_id))
        f.write("SOURCE_LABEL: {}\n".format(source_label))
        f.write("SOURCE_PATH: {}\n\n".format(source_path))

        if stage2_dt is None:
            f.write("STAGE2_DATE (parsed): NONE FOUND\n\n")
            f.write("WARNING: Could not find Stage2 date for this patient in frozen pack or patient_stage_summary.\n")
            return out_path, None

        anchor_30 = stage2_dt + pd.Timedelta(days=30)
        anchor_90 = stage2_dt + pd.Timedelta(days=90)
        anchor_365 = stage2_dt + pd.Timedelta(days=365)

        f.write("STAGE2_DATE (parsed): {}\n\n".format(stage2_dt.isoformat()))
        f.write("WINDOWS (relative to STAGE2_DATE)\n")
        f.write("---------------------------------\n")
        f.write("STAGE2 + 30 days : {}\n".format(anchor_30.date().isoformat()))
        f.write("STAGE2 + 90 days : {}\n".format(anchor_90.date().isoformat()))
        f.write("STAGE2 + 365 days: {}\n\n".format(anchor_365.date().isoformat()))
        f.write("Interpretation:\n")
        f.write("  0–30d   = early\n")
        f.write("  31–90d  = intermediate\n")
        f.write("  91–365d = late\n")
        f.write("  >365d   = very late\n")

    return out_path, stage2_dt

def write_case_context(case_ctx, patient_dir):
    if not case_ctx:
        return ""
    out_path = os.path.join(patient_dir, "case_context.txt")
    with open(out_path, "w") as f:
        f.write("VALIDATION EXEMPLAR CONTEXT\n")
        f.write("===========================\n\n")
        for k in sorted(case_ctx.keys()):
            f.write("{}: {}\n".format(k, _safe_str(case_ctx.get(k, ""))))
    return out_path


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="DE-ID note CSV files (optional; uses defaults if omitted).")
    ap.add_argument("--encounters", nargs="+", default=None, help="Encounter CSV files (optional; uses defaults if omitted).")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory for patient bundles.")

    # Either provide patient_id OR mrn OR exemplar selection
    ap.add_argument("--patient_id", default=None, help="ENCRYPTED_PAT_ID to export.")
    ap.add_argument("--mrn", default=None, help="MRN to resolve to ENCRYPTED_PAT_ID via crosswalk, then export.")
    ap.add_argument("--mrn_crosswalk", default=DEFAULT_MRN_CROSSWALK, help="CSV mapping MRN -> ENCRYPTED_PAT_ID.")
    ap.add_argument("--mrn_col", default=None, help="Optional: MRN column name in crosswalk (override auto-detect).")
    ap.add_argument("--pid_col", default=None, help="Optional: ENCRYPTED_PAT_ID column name in crosswalk (override auto-detect).")

    # Exemplar mode from validation_merged.csv
    ap.add_argument("--validation", default=DEFAULT_VALIDATION_MERGED, help="Path to validation_merged.csv")
    ap.add_argument("--outcome", default=None, help="Outcome name for exemplar mode: MinorComp, Reoperation, Rehospitalization, MajorComp, Failure, Revision")
    ap.add_argument("--case_type", default=None, help="Case type for exemplar mode: FP, FN, TP, TN")

    ap.add_argument("--pick_random", action="store_true", help="Pick a random patient with >= --min_notes (from notes only).")
    ap.add_argument("--min_notes", type=int, default=10, help="Used with --pick_random.")
    ap.add_argument("--max_rows_per_file", type=int, default=None, help="Optional: cap rows per file for fast tests.")
    args = ap.parse_args()

    root = os.path.abspath(".")
    deid_inputs = args.inputs if args.inputs else list(DEFAULT_DEID_INPUTS)
    enc_inputs = args.encounters if args.encounters else list(DEFAULT_ENCOUNTER_INPUTS)

    ensure_out_dir(args.out_dir)

    # -------------
    # Resolve patient selection
    # -------------
    patient_id = None
    selected_mrn = None
    case_ctx = None

    # Exemplar mode: needs both outcome and case_type
    if args.outcome and args.case_type:
        case_ctx = pick_case_from_validation(args.validation, args.outcome, args.case_type)
        selected_mrn = _safe_str(case_ctx.get("MRN", "")).strip()
        selected_pid = _safe_str(case_ctx.get("ENCRYPTED_PAT_ID", "")).strip()

        if selected_mrn:
            resolved_pid, used_mrn_col, used_pid_col, n_matches = resolve_patient_id_from_mrn(
                mrn=selected_mrn,
                crosswalk_path=args.mrn_crosswalk,
                mrn_col_override=args.mrn_col,
                pid_col_override=args.pid_col
            )
            patient_id = resolved_pid
            print("Exemplar pick: outcome={} case_type={} MRN={} -> ENCRYPTED_PAT_ID={} (matches={})".format(
                args.outcome, args.case_type, selected_mrn, patient_id, n_matches
            ))
        elif selected_pid:
            patient_id = selected_pid
            print("Exemplar pick: outcome={} case_type={} ENCRYPTED_PAT_ID={}".format(args.outcome, args.case_type, patient_id))
        else:
            raise RuntimeError("Exemplar row had neither MRN nor ENCRYPTED_PAT_ID. Try another case_type or outcome.")
    elif args.mrn:
        resolved_pid, used_mrn_col, used_pid_col, n_matches = resolve_patient_id_from_mrn(
            mrn=args.mrn,
            crosswalk_path=args.mrn_crosswalk,
            mrn_col_override=args.mrn_col,
            pid_col_override=args.pid_col
        )
        patient_id = resolved_pid
        selected_mrn = args.mrn
        print("Resolved MRN {} -> ENCRYPTED_PAT_ID {} (matches={})".format(args.mrn, patient_id, n_matches))
        print("Crosswalk cols: MRN={}, ENCRYPTED_PAT_ID={}".format(used_mrn_col, used_pid_col))
    elif args.patient_id:
        patient_id = args.patient_id
    else:
        patient_id = DEFAULT_PATIENT_ID

    # ---------
    # NOTES: Collect per-note rows for the patient
    # ---------
    all_note_rows = []
    for path in deid_inputs:
        if not os.path.exists(path):
            raise RuntimeError("Input file not found: {}".format(path))

        df = read_csv_robust(path)

        pid_col = detect_pid_col(df.columns)
        note_type_col = detect_note_type_col(df.columns)
        deid_col = detect_deid_text_col(df.columns)
        dt_col = detect_datetime_col(df.columns)

        if pid_col is None:
            raise RuntimeError("Could not detect ENCRYPTED_PAT_ID column in: {}".format(path))
        if deid_col is None:
            raise RuntimeError(
                "Could not detect a DE-ID text column in: {}\n"
                "Expected something like NOTE_TEXT_DEID (must contain 'deid' in the header).".format(path)
            )
        if "deid" not in _safe_str(deid_col).lower():
            raise RuntimeError("Refusing to proceed: detected text col doesn't look de-identified: {}".format(deid_col))

        dt_parsed_series = None
        if dt_col is not None:
            dt_parsed_series = try_parse_datetime(df[dt_col].astype(str))

        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        for i in range(n_rows):
            pid = _safe_str(df.iloc[i][pid_col]).strip()
            if not pid:
                continue

            if not args.pick_random:
                if pid != patient_id:
                    continue

            note_type = normalize_note_type(df.iloc[i][note_type_col]) if note_type_col else "UNKNOWN_NOTE_TYPE"
            text_deid = _safe_str(df.iloc[i][deid_col])

            dt_raw = _safe_str(df.iloc[i][dt_col]) if dt_col is not None else ""
            dt_parsed = None
            if dt_parsed_series is not None:
                v = dt_parsed_series.iloc[i]
                if pd.notnull(v):
                    dt_parsed = v.to_pydatetime()

            all_note_rows.append({
                "source_file": os.path.basename(path),
                "pid": pid,
                "note_type": note_type,
                "note_text_deid": text_deid,
                "dt_raw": dt_raw,
                "dt_parsed": dt_parsed,
                "row_idx": i
            })

    if not all_note_rows:
        raise RuntimeError("No note rows found across the provided DE-ID inputs.")

    # If pick_random, choose after reading notes
    if args.pick_random:
        counts = {}
        for r in all_note_rows:
            counts[r["pid"]] = counts.get(r["pid"], 0) + 1
        eligible = [pid for pid, n in counts.items() if n >= args.min_notes]
        if eligible:
            patient_id = random.choice(eligible)
        else:
            patient_id = max(counts.keys(), key=lambda k: counts[k])
        print("Picked patient_id:", patient_id, "with notes:", counts.get(patient_id, 0))

    patient_note_rows = [r for r in all_note_rows if r["pid"] == patient_id]
    if not patient_note_rows:
        raise RuntimeError("No note rows found for patient_id: {}".format(patient_id))

    # Write note bundle
    patient_dir, notes_timeline_path, combined_path = write_notes_bundle(patient_id, patient_note_rows, args.out_dir)

    # If exemplar mode, write case context into the bundle
    case_context_path = ""
    if case_ctx:
        case_context_path = write_case_context(case_ctx, patient_dir)

    # ---------
    # STAGE2 anchor summary (from frozen pack preferred)
    # ---------
    stage2_dt, stage2_source_label, stage2_source_path = load_stage2_anchor_dt(root, patient_id)
    anchor_summary_path, stage2_dt = write_anchor_summary_stage2(patient_id, stage2_dt, stage2_source_label, stage2_source_path, patient_dir)

    # ---------
    # ENCOUNTERS: Build structured date timeline
    # ---------
    encounter_rows = []

    for path in enc_inputs:
        if not os.path.exists(path):
            raise RuntimeError("Encounter file not found: {}".format(path))

        df = read_csv_robust(path)
        pid_col = detect_pid_col(df.columns)
        if pid_col is None:
            raise RuntimeError("Could not detect ENCRYPTED_PAT_ID in encounter file: {}".format(path))

        date_cols = detect_encounter_date_cols(df.columns)
        parsed_cols = {}
        for dc in date_cols:
            parsed_cols[dc] = try_parse_datetime(df[dc].astype(str))

        n_rows = len(df)
        if args.max_rows_per_file is not None:
            n_rows = min(n_rows, args.max_rows_per_file)

        for i in range(n_rows):
            pid = _safe_str(df.iloc[i][pid_col]).strip()
            if not pid or pid != patient_id:
                continue

            best_dt = None
            best_raw = ""
            best_source = None

            for dc in date_cols:
                ser = parsed_cols.get(dc, None)
                if ser is None:
                    continue
                v = ser.iloc[i]
                if pd.notnull(v):
                    best_dt = v.to_pydatetime()
                    best_raw = _safe_str(df.iloc[i][dc])
                    best_source = dc
                    break

            row = {
                "ENCRYPTED_PAT_ID": pid,
                "BEST_EVENT_DT_RAW": best_raw,
                "BEST_EVENT_DT_PARSED": best_dt,
                "ANCHOR_SOURCE_COL": best_source if best_source else "",
                "SOURCE_FILE": os.path.basename(path),
                "ROW_IDX": i
            }

            for want in [
                "PAT_ENC_CSN_ID", "ENCRYPTED_CSN", "ENCOUNTER_TYPE", "DEPARTMENT", "OP_DEPARTMENT",
                "RECONSTRUCTION_DATE", "OPERATION_DATE", "ADMIT_DATE", "DISCHARGE_DATE_DT",
                "CPT_CODE", "PROCEDURE", "REASON_FOR_VISIT"
            ]:
                for c in df.columns:
                    if _norm_col(c) == _norm_col(want):
                        row[want] = _safe_str(df.iloc[i][c])
                        break

            encounter_rows.append(row)

    if encounter_rows:
        encounters_path = write_encounters_timeline(encounter_rows, patient_dir)
    else:
        encounters_path = os.path.join(patient_dir, "encounters_timeline.csv")
        pd.DataFrame([]).to_csv(encounters_path, index=False, encoding="utf-8")

    # ---------
    # Print summary
    # ---------
    print("\nEXPORTED PATIENT:")
    print("  ENCRYPTED_PAT_ID:", patient_id)
    print("  MRN (if used)    :", selected_mrn if selected_mrn else "")
    print("  Notes exported   :", len(patient_note_rows))
    print("  Patient dir      :", patient_dir)
    print("  Notes timeline   :", notes_timeline_path)
    print("  Combined notes   :", combined_path)
    print("  Encounters file  :", encounters_path)
    print("  Stage2 anchor    :", stage2_dt.isoformat() if stage2_dt else "NONE")
    print("  Anchor summary   :", anchor_summary_path)
    if case_context_path:
        print("  Case context     :", case_context_path)
    print("Done.")

if __name__ == "__main__":
    main()
