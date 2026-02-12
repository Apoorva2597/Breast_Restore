# stage2_scan_op_notes_expanders.py
 -*- coding: cp1252 -*-
# Python 3.6+ (pandas required)
#
# Purpose:
#   For expander-pathway patients (from patient_recon_staging.csv),
#   scan Operation Notes for Stage 2 evidence (TE -> implant exchange).
#
# Outputs:
#   1) stage2_from_notes_patient_level.csv  (1 best hit per patient)
#   2) stage2_from_notes_row_hits.csv       (all hit rows for QA)
#   3) stage2_from_notes_summary.txt        (summary counts)

from __future__ import print_function
import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
PATIENT_STAGING_CSV = "patient_recon_staging.csv"
OP_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"

OUT_PATIENT_LEVEL = "stage2_from_notes_patient_level.csv"
OUT_ROW_HITS = "stage2_from_notes_row_hits.csv"
OUT_SUMMARY = "stage2_from_notes_summary.txt"

# Note file columns
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_ID = "NOTE_ID"


# -------------------------
# IO helpers
# -------------------------
def read_csv_fallback(path, **kwargs):
    """
    Try utf-8 first, then cp1252.
    Note: This is for READING CSV DATA. Your current error is from the PY file itself,
    so putting this code into a NEW file should avoid that.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python", **kwargs)


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_dt(series):
    return pd.to_datetime(series, errors="coerce")


def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "t"]


def safe_snippet(x, n=240):
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > n:
        return s[:n] + "..."
    return s


# -------------------------
# Stage 2 evidence rules
# -------------------------
# Tier A: definitive
#   - exchange expander to implant
#   - remove/explant expander + place/insert implant (in same note text window)
#
# Tier B: strong
#   - second stage + implant + expander + an action word (exchange/remove/place)
#   - capsulotomy/capsulectomy + implant + expander + action
#
# Tier C: suggestive
#   - second stage + implant/exchange (may be plan/history)
#   - implant + expander context but no action

RX = {
    "EXPANDER": re.compile(r"\b(tissue\s*expander|expander|expandr|te)\b", re.I),
    "IMPLANT": re.compile(r"\b(implant|implnt|permanent\s+implant)\b", re.I),
    "REMOVE": re.compile(r"\b(remove|removed|explant|explanted|take\s+out|taken\s+out)\b", re.I),
    "PLACE": re.compile(r"\b(place|placed|insert|inserted|insertion|implantation)\b", re.I),
    "EXCHANGE": re.compile(r"\b(exchange|exchanged)\b", re.I),
    "SECOND_STAGE": re.compile(r"\b(second\s+stage|stage\s*2|stage\s*ii)\b", re.I),
    "CAPSU": re.compile(r"\b(capsulotomy|capsulectomy)\b", re.I),

    "EXCHANGE_TE_TO_IMPLANT": re.compile(
        r"\bexchange\b.{0,60}\b(tissue\s*expander|expander|expandr|te)\b.{0,140}\b(implant|implnt|permanent\s+implant)\b|"
        r"\bexchange\b.{0,60}\b(implant|implnt|permanent\s+implant)\b.{0,140}\b(tissue\s*expander|expander|expandr|te)\b",
        re.I
    ),
    "REMOVE_EXPANDER_PLACE_IMPLANT": re.compile(
        r"\b(remove|removed|explant|explanted)\b.{0,140}\b(tissue\s*expander|expander|expandr|te)\b.{0,260}\b(place|placed|insert|inserted)\b.{0,100}\b(implant|implnt|permanent\s+implant)\b|"
        r"\b(place|placed|insert|inserted)\b.{0,100}\b(implant|implnt|permanent\s+implant)\b.{0,260}\b(remove|removed|explant|explanted)\b.{0,140}\b(tissue\s*expander|expander|expandr|te)\b",
        re.I
    ),
}


def classify_stage2(text_norm):
    """
    Returns (tier, rule) or (None, None)
    """
    t = text_norm

    # Tier A
    if RX["EXCHANGE_TE_TO_IMPLANT"].search(t):
        return "A", "EXCHANGE_TE_TO_IMPLANT"
    if RX["REMOVE_EXPANDER_PLACE_IMPLANT"].search(t):
        return "A", "REMOVE_EXPANDER_PLACE_IMPLANT"

    has_exp = bool(RX["EXPANDER"].search(t))
    has_imp = bool(RX["IMPLANT"].search(t))
    has_exch = bool(RX["EXCHANGE"].search(t))
    has_rem = bool(RX["REMOVE"].search(t))
    has_place = bool(RX["PLACE"].search(t))
    has_second = bool(RX["SECOND_STAGE"].search(t))
    has_capsu = bool(RX["CAPSU"].search(t))
    has_action = has_exch or has_rem or has_place

    # Tier B
    if has_exp and has_imp and has_exch:
        return "B", "EXCHANGE+EXPANDER+IMPLANT"
    if has_exp and has_imp and has_second and has_action:
        return "B", "SECOND_STAGE+EXPANDER+IMPLANT+ACTION"
    if has_exp and has_imp and has_capsu and has_action:
        return "B", "CAPSU+EXPANDER+IMPLANT+ACTION"

    # Tier C
    if has_second and (has_imp or has_exch):
        return "C", "SECOND_STAGE_SUGGESTIVE"
    if has_exp and has_imp and (not has_action):
        return "C", "EXPANDER+IMPLANT_NO_ACTION"

    return None, None


def tier_rank(tier):
    if tier == "A":
        return 3
    if tier == "B":
        return 2
    if tier == "C":
        return 1
    return 0


# -------------------------
# Main
# -------------------------
def main():
    # Load staging, restrict to expander cohort
    stg = read_csv_fallback(PATIENT_STAGING_CSV)
    for c in ["patient_id", "has_expander", "stage1_date"]:
        if c not in stg.columns:
            raise RuntimeError("Missing column in {}: {}".format(PATIENT_STAGING_CSV, c))

    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)
    exp = stg[stg["has_expander_bool"]].copy()

    if exp.empty:
        raise RuntimeError("No expander patients found (has_expander==True).")

    exp["patient_id"] = exp["patient_id"].astype(str)
    exp_ids = set(exp["patient_id"].tolist())
    exp["stage1_dt"] = parse_dt(exp["stage1_date"])

    stage1_map = dict(zip(exp["patient_id"], exp["stage1_dt"]))
    mrn_map = {}
    if "mrn" in exp.columns:
        mrn_map = dict(zip(exp["patient_id"], exp["mrn"].fillna("").astype(str)))
    else:
        mrn_map = dict((pid, "") for pid in exp_ids)

    # Check note columns exist
    need_note_cols = [COL_PATIENT, COL_MRN, COL_NOTE_TEXT, COL_NOTE_DOS, COL_OP_DATE, COL_NOTE_TYPE, COL_NOTE_ID]
    head = read_csv_fallback(OP_NOTES_CSV, nrows=5)
    for c in need_note_cols:
        if c not in head.columns:
            raise RuntimeError("Missing required note column: {}".format(c))

    chunksize = 150000
    hit_frames = []

    total_note_rows_seen = 0
    total_note_rows_expanders = 0

    for chunk in read_csv_fallback(OP_NOTES_CSV, usecols=need_note_cols, chunksize=chunksize):
        total_note_rows_seen += len(chunk)

        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PATIENT].isin(exp_ids)].copy()
        if chunk.empty:
            continue

        total_note_rows_expanders += len(chunk)

        chunk["text_norm"] = chunk[COL_NOTE_TEXT].apply(norm_text)

        # event date: prefer NOTE_DOS else OP_DATE
        chunk["note_dt"] = parse_dt(chunk[COL_NOTE_DOS])
        chunk["op_dt"] = parse_dt(chunk[COL_OP_DATE])
        chunk["event_dt"] = chunk["note_dt"].fillna(chunk["op_dt"])

        # classify row-wise
        tiers = []
        rules = []
        for txt in chunk["text_norm"].tolist():
            tier, rule = classify_stage2(txt)
            tiers.append(tier)
            rules.append(rule)

        chunk["tier"] = tiers
        chunk["rule"] = rules

        hits = chunk[chunk["tier"].notnull()].copy()
        if hits.empty:
            continue

        hits["stage1_dt"] = hits[COL_PATIENT].map(stage1_map)
        hits["delta_days_vs_stage1"] = (hits["event_dt"] - hits["stage1_dt"]).dt.days
        hits["after_index"] = hits["delta_days_vs_stage1"].apply(lambda x: pd.notnull(x) and x >= 0)

        hits["snippet"] = hits[COL_NOTE_TEXT].apply(safe_snippet)
        hits["tier_rank"] = hits["tier"].apply(tier_rank)

        keep = [
            COL_PATIENT, COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID,
            COL_NOTE_DOS, COL_OP_DATE, "event_dt",
            "tier", "rule", "tier_rank",
            "stage1_dt", "delta_days_vs_stage1", "after_index",
            "snippet"
        ]
        hit_frames.append(hits[keep])

    if hit_frames:
        hits_all = pd.concat(hit_frames, ignore_index=True)
    else:
        hits_all = pd.DataFrame(columns=[
            COL_PATIENT, COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID,
            COL_NOTE_DOS, COL_OP_DATE, "event_dt",
            "tier", "rule", "tier_rank",
            "stage1_dt", "delta_days_vs_stage1", "after_index",
            "snippet"
        ])

    # Choose best hit per patient:
    # Prefer after_index True, then higher tier, then earliest event_dt
    if not hits_all.empty:
        hits_all = hits_all.sort_values(
            by=[COL_PATIENT, "after_index", "tier_rank", "event_dt"],
            ascending=[True, False, False, True]
        )
        best = hits_all.groupby(COL_PATIENT, as_index=False).head(1).copy()
    else:
        best = pd.DataFrame(columns=hits_all.columns)

    # Build patient-level table for all expander patients
    patient_level = pd.DataFrame({"patient_id": list(exp_ids)})
    patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)
    patient_level["mrn_from_staging"] = patient_level["patient_id"].map(mrn_map)

    if not best.empty:
        best = best.rename(columns={COL_PATIENT: "patient_id"})
        patient_level = patient_level.merge(best, on="patient_id", how="left")
    else:
        # add placeholders
        for c in [
            COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_DOS, COL_OP_DATE, "event_dt",
            "tier", "rule", "tier_rank", "delta_days_vs_stage1", "after_index", "snippet"
        ]:
            patient_level[c] = None

    # Rename to clean output names
    patient_level = patient_level.rename(columns={
        COL_MRN: "mrn_from_notes",
        COL_NOTE_TYPE: "best_note_type",
        COL_NOTE_ID: "best_note_id",
        COL_NOTE_DOS: "best_note_dos",
        COL_OP_DATE: "best_note_op_date",
        "event_dt": "stage2_event_dt_best",
        "tier": "stage2_tier_best",
        "rule": "stage2_rule_best",
        "after_index": "stage2_after_index",
        "delta_days_vs_stage1": "stage2_delta_days_from_stage1"
    })

    # Write outputs
    hits_all.to_csv(OUT_ROW_HITS, index=False)
    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False)

    # Summary
    n_exp = len(exp_ids)
    n_any_hit = int(patient_level["stage2_tier_best"].notnull().sum())
    n_after = int(patient_level["stage2_after_index"].fillna(False).astype(bool).sum())
    tier_counts = patient_level["stage2_tier_best"].fillna("NONE").value_counts()

    lines = []
    lines.append("=== Stage 2 from OPERATION NOTES (Expanders) ===")
    lines.append("Expander patients: {}".format(n_exp))
    lines.append("Patients with ANY Stage2 hit (any date): {} ({:.1f}%)".format(
        n_any_hit, (100.0 * n_any_hit / n_exp) if n_exp else 0.0
    ))
    lines.append("Patients with best hit AFTER stage1 index: {} ({:.1f}%)".format(
        n_after, (100.0 * n_after / n_exp) if n_exp else 0.0
    ))
    lines.append("")
    lines.append("Best-hit tier distribution (patients):")
    for k, v in tier_counts.items():
        lines.append("  {}: {}".format(k, int(v)))
    lines.append("")
    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_PATIENT_LEVEL))
    lines.append("  - {}".format(OUT_ROW_HITS))
    lines.append("  - {}".format(OUT_SUMMARY))

    with open(OUT_SUMMARY, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
