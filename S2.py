# -*- coding: utf-8 -*-
# stage2_from_operation_notes_expanders.py
# Python 3.6+ (pandas required)
#
# Purpose:
#   For expander-pathway patients (from patient_recon_staging.csv),
#   scan OPERATION NOTES for Stage 2 evidence (tissue expander -> implant exchange).
#
# Inputs:
#   1) patient_recon_staging.csv
#   2) HPI11526 Operation Notes.csv
#
# Outputs:
#   1) stage2_from_notes_patient_level.csv
#   2) stage2_from_notes_row_hits.csv
#   3) stage2_from_notes_summary.txt

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
PATIENT_STAGING_CSV = "patient_recon_staging.csv"

OP_NOTES_CSV = (
    "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Operation Notes.csv"
)

OUT_PATIENT_LEVEL = "stage2_from_notes_patient_level.csv"
OUT_ROW_HITS = "stage2_from_notes_row_hits.csv"
OUT_SUMMARY = "stage2_from_notes_summary.txt"

# Note file columns (adjust if your headers differ)
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_ID = "NOTE_ID"


# -------------------------
# CSV read helpers
# -------------------------
def read_csv_fallback(path, **kwargs):
    """
    Read CSV with UTF-8 first; fallback to CP1252 if needed.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python", **kwargs)


def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "t"]


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_dt(series):
    return pd.to_datetime(series, errors="coerce")


def make_snippet(x, n=240):
    if x is None:
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > n:
        return s[:n] + "..."
    return s


# -------------------------
# Stage 2 evidence patterns (notes)
# -------------------------
# Tier A (definitive):
#   - "exchange expander to implant"
#   - "remove/explant expander" + "place/insert implant" (close proximity)
#
# Tier B (strong):
#   - exchange + expander + implant (not necessarily explicit removal)
#   - second stage + implant + action + expander
#   - capsulotomy/capsulectomy + implant + action + expander
#
# Tier C (suggestive):
#   - second stage + implant/exchange (plan/history possible)
#   - expander + implant mentioned but no explicit action

RX = {
    "EXPANDER": re.compile(r"\b(tissue\s*expander|expander|expandr|\bte\b)\b", re.I),
    "IMPLANT": re.compile(r"\b(implant|implnt|permanent\s+implant)\b", re.I),
    "REMOVE": re.compile(
        r"\b(remove|removed|explant|explanted|take\s*out|taken\s*out)\b", re.I
    ),
    "PLACE": re.compile(
        r"\b(place|placed|insert|inserted|insertion|implantation)\b", re.I
    ),
    "EXCHANGE": re.compile(r"\b(exchange|exchanged)\b", re.I),
    "SECOND_STAGE": re.compile(r"\b(second\s+stage|stage\s*2|stage\s*ii)\b", re.I),
    "CAPSU": re.compile(r"\b(capsulotomy|capsulectomy)\b", re.I),
    "EXCHANGE_TE_TO_IMPLANT": re.compile(
        r"\bexchange\b.{0,60}\b(tissue\s*expander|expander|expandr|\bte\b)\b.{0,120}\b(implant|implnt|permanent\s+implant)\b|"
        r"\bexchange\b.{0,60}\b(implant|implnt|permanent\s+implant)\b.{0,120}\b(tissue\s*expander|expander|expandr|\bte\b)\b",
        re.I,
    ),
    "REMOVE_EXPANDER_PLACE_IMPLANT": re.compile(
        r"\b(remove|removed|explant|explanted)\b.{0,140}\b(tissue\s*expander|expander|expandr|\bte\b)\b.{0,260}\b(place|placed|insert|inserted)\b.{0,100}\b(implant|implnt|permanent\s+implant)\b|"
        r"\b(place|placed|insert|inserted)\b.{0,100}\b(implant|implnt|permanent\s+implant)\b.{0,260}\b(remove|removed|explant|explanted)\b.{0,140}\b(tissue\s*expander|expander|expandr|\bte\b)\b",
        re.I,
    ),
}


def classify_note_stage2(note_text):
    """
    Return (tier, rule_name) where tier in {"A","B","C"} or (None, None).
    """
    t = note_text

    # Tier A: definitive
    if RX["EXCHANGE_TE_TO_IMPLANT"].search(t):
        return ("A", "EXCHANGE_TE_TO_IMPLANT")
    if RX["REMOVE_EXPANDER_PLACE_IMPLANT"].search(t):
        return ("A", "REMOVE_EXPANDER_PLACE_IMPLANT")

    has_expander = bool(RX["EXPANDER"].search(t))
    has_implant = bool(RX["IMPLANT"].search(t))
    has_exchange = bool(RX["EXCHANGE"].search(t))
    has_remove = bool(RX["REMOVE"].search(t))
    has_place = bool(RX["PLACE"].search(t))
    has_second = bool(RX["SECOND_STAGE"].search(t))
    has_capsu = bool(RX["CAPSU"].search(t))

    # Tier B: strong combos
    if has_exchange and has_implant and has_expander:
        return ("B", "EXCHANGE+IMPLANT+EXPANDER")
    if (
        has_second
        and has_implant
        and (has_exchange or has_remove or has_place)
        and has_expander
    ):
        return ("B", "SECOND_STAGE+IMPLANT+ACTION+EXPANDER")
    if (
        has_capsu
        and has_implant
        and (has_exchange or has_remove or has_place)
        and has_expander
    ):
        return ("B", "CAPSU+IMPLANT+ACTION+EXPANDER")

    # Tier C: suggestive
    if has_second and (has_implant or has_exchange):
        return ("C", "SECOND_STAGE_SUGGESTIVE")
    if has_expander and has_implant and not (has_exchange or has_remove or has_place):
        return ("C", "EXPANDER+IMPLANT_NO_ACTION")

    return (None, None)


def main():
    # -------------------------
    # Load expander cohort from patient_recon_staging.csv
    # -------------------------
    stg = read_csv_fallback(PATIENT_STAGING_CSV)

    required = ["patient_id", "has_expander", "stage1_date"]
    for c in required:
        if c not in stg.columns:
            raise RuntimeError(
                "Missing required column in {}: {}".format(PATIENT_STAGING_CSV, c)
            )

    stg["patient_id"] = stg["patient_id"].fillna("").astype(str)
    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)
    exp = stg[stg["has_expander_bool"]].copy()

    if exp.empty:
        raise RuntimeError(
            "No expander patients found (has_expander == True) in patient_recon_staging.csv."
        )

    exp["stage1_dt"] = parse_dt(exp["stage1_date"])

    exp_ids = set(exp["patient_id"].tolist())
    stage1_map = dict(zip(exp["patient_id"], exp["stage1_dt"]))

    # mrn might exist in staging output; if not, keep blank
    if "mrn" in exp.columns:
        mrn_map = dict(zip(exp["patient_id"], exp["mrn"].fillna("").astype(str)))
    else:
        mrn_map = dict((pid, "") for pid in exp_ids)

    print("Expander patients:", len(exp_ids))

    # -------------------------
    # Validate OP notes columns exist
    # -------------------------
    head = read_csv_fallback(OP_NOTES_CSV, nrows=5)
    needed_note_cols = [
        COL_PATIENT,
        COL_NOTE_TEXT,
        COL_NOTE_DOS,
        COL_OP_DATE,
        COL_NOTE_TYPE,
        COL_NOTE_ID,
        COL_MRN,
    ]
    missing = [c for c in needed_note_cols if c not in head.columns]
    if missing:
        raise RuntimeError(
            "Missing required column(s) in OP notes CSV: {}".format(missing)
        )

    # -------------------------
    # Stream OP notes and collect hits
    # -------------------------
    usecols = needed_note_cols
    chunksize = 200000

    hit_frames = []
    n_rows_seen = 0
    n_rows_in_expanders = 0

    for chunk in read_csv_fallback(OP_NOTES_CSV, usecols=usecols, chunksize=chunksize):
        # standardize patient id
        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)

        n_rows_seen += len(chunk)

        chunk = chunk[chunk[COL_PATIENT].isin(exp_ids)].copy()
        if chunk.empty:
            continue

        n_rows_in_expanders += len(chunk)

        # normalize text
        chunk["note_text_norm"] = chunk[COL_NOTE_TEXT].apply(norm_text)

        # event date: prefer DOS, fallback OPERATION_DATE
        chunk["dos_dt"] = parse_dt(chunk[COL_NOTE_DOS])
        chunk["op_dt"] = parse_dt(chunk[COL_OP_DATE])
        chunk["event_dt"] = chunk["dos_dt"].fillna(chunk["op_dt"])

        # classify tier/rule
        tiers = []
        rules = []
        for txt in chunk["note_text_norm"].tolist():
            tier, rule = classify_note_stage2(txt)
            tiers.append(tier)
            rules.append(rule)

        chunk["tier"] = tiers
        chunk["rule"] = rules

        hits = chunk[chunk["tier"].notnull()].copy()
        if hits.empty:
            continue

        # attach stage1 and delta
        hits["stage1_dt"] = hits[COL_PATIENT].map(stage1_map)
        hits["delta_days_vs_stage1"] = (hits["event_dt"] - hits["stage1_dt"]).dt.days

        # snippet for QA
        hits["snippet"] = hits[COL_NOTE_TEXT].apply(lambda x: make_snippet(x, n=240))

        keep = [
            COL_PATIENT,
            COL_MRN,
            COL_NOTE_TYPE,
            COL_NOTE_ID,
            COL_NOTE_DOS,
            COL_OP_DATE,
            "event_dt",
            "tier",
            "rule",
            "delta_days_vs_stage1",
            "snippet",
        ]
        hit_frames.append(hits[keep])

    if hit_frames:
        hits_all = pd.concat(hit_frames, ignore_index=True)
    else:
        hits_all = pd.DataFrame(
            columns=[
                COL_PATIENT,
                COL_MRN,
                COL_NOTE_TYPE,
                COL_NOTE_ID,
                COL_NOTE_DOS,
                COL_OP_DATE,
                "event_dt",
                "tier",
                "rule",
                "delta_days_vs_stage1",
                "snippet",
            ]
        )

    # -------------------------
    # Patient-level best Stage 2 selection
    # -------------------------
    tier_rank = {"A": 3, "B": 2, "C": 1}
    if not hits_all.empty:
        hits_all["tier_rank"] = hits_all["tier"].map(tier_rank).fillna(0).astype(int)
        hits_all["after_index"] = hits_all["delta_days_vs_stage1"].apply(
            lambda x: (x is not None) and pd.notnull(x) and (x >= 0)
        )

        hits_all = hits_all.sort_values(
            by=[COL_PATIENT, "after_index", "tier_rank", "event_dt"],
            ascending=[True, False, False, True],
        )
        best = hits_all.groupby(COL_PATIENT, as_index=False).head(1).copy()
    else:
        best = pd.DataFrame()

    patient_level = pd.DataFrame({"patient_id": sorted(list(exp_ids))})
    patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)
    patient_level["mrn_from_staging"] = patient_level["patient_id"].map(mrn_map)

    if not best.empty:
        best = best.rename(columns={COL_PATIENT: "patient_id"})
        patient_level = patient_level.merge(best, on="patient_id", how="left")
    else:
        # ensure columns exist
        for c in [
            COL_MRN,
            COL_NOTE_TYPE,
            COL_NOTE_ID,
            COL_NOTE_DOS,
            COL_OP_DATE,
            "event_dt",
            "tier",
            "rule",
            "delta_days_vs_stage1",
            "snippet",
        ]:
            patient_level[c] = None
        patient_level["after_index"] = None
        patient_level["tier_rank"] = None

    patient_level = patient_level.rename(
        columns={
            "event_dt": "stage2_event_dt_best",
            "tier": "stage2_tier_best",
            "rule": "stage2_rule_best",
            "after_index": "stage2_after_index",
            "delta_days_vs_stage1": "stage2_delta_days_from_stage1",
            COL_NOTE_TYPE: "best_note_type",
            COL_NOTE_ID: "best_note_id",
            COL_NOTE_DOS: "best_note_dos",
            COL_OP_DATE: "best_note_op_date",
            COL_MRN: "mrn_from_notes",
        }
    )

    # -------------------------
    # Write outputs
    # -------------------------
    hits_all.to_csv(OUT_ROW_HITS, index=False)
    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False)

    # -------------------------
    # Summary
    # -------------------------
    n_exp = len(exp_ids)
    n_any_hit = int(patient_level["stage2_tier_best"].notnull().sum())
    n_after = int(patient_level["stage2_after_index"].fillna(False).astype(bool).sum())
    tier_counts = patient_level["stage2_tier_best"].fillna("NONE").value_counts()

    lines = []
    lines.append("=== Stage 2 from OPERATION NOTES (Expanders) ===")
    lines.append("Expander patients: {}".format(n_exp))
    lines.append(
        "Rows scanned total (OP notes file, all patients): {}".format(n_rows_seen)
    )
    lines.append(
        "Rows scanned within expander patients: {}".format(n_rows_in_expanders)
    )
    lines.append("")
    lines.append(
        "Patients with ANY Stage2 note hit (any date): {} ({:.1f}%)".format(
            n_any_hit, (100.0 * n_any_hit / n_exp) if n_exp else 0.0
        )
    )
    lines.append(
        "Patients with best Stage2 hit AFTER Stage1 index: {} ({:.1f}%)".format(
            n_after, (100.0 * n_after / n_exp) if n_exp else 0.0
        )
    )
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