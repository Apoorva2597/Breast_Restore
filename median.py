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
# Robust CSV read helpers
# -------------------------
def read_csv_robust(path, **kwargs):
    """
    Reads a CSV robustly across common encodings.
    Priority:
      1) utf-8
      2) cp1252 (Windows)
      3) latin-1 (never fails)
    """
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python", **kwargs)
        except UnicodeDecodeError:
            continue
    # If we somehow get here, raise last error
    return pd.read_csv(path, encoding="latin-1", engine="python", **kwargs)


def norm_text(x):
    if x is None:
        return ""
    s = str(x)

    # Normalize the usual whitespace + weird bytes
    # NBSP (non-breaking space) often appears as U+00A0
    s = s.replace(u"\u00A0", " ")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("_", " ")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_date_series(s):
    return pd.to_datetime(s, errors="coerce")


def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y", "t"]


# -------------------------
# Stage 2 evidence patterns (notes)
# -------------------------
RX = {
    # components
    "EXPANDER": re.compile(r"\b(tissue\s*expander|expander|expandr|te)\b", re.I),
    "IMPLANT": re.compile(r"\b(implant|implnt|permanent\s+implant)\b", re.I),
    "REMOVE": re.compile(r"\b(remove|removed|explant|explanted|take\s*out|taken\s*out)\b", re.I),
    "PLACE": re.compile(r"\b(place|placed|insert|inserted|insertion|implantation)\b", re.I),
    "EXCHANGE": re.compile(r"\b(exchange|exchanged)\b", re.I),
    "SECOND_STAGE": re.compile(r"\b(second\s+stage|stage\s*2|stage\s+ii)\b", re.I),
    "CAPSU": re.compile(r"\b(capsulotomy|capsulectomy)\b", re.I),

    # strong combined phrases (Tier A)
    "EXCHANGE_TE_TO_IMPLANT": re.compile(
        r"\bexchange\b.{0,60}\b(tissue\s*expander|expander|expandr|te)\b.{0,120}\b(implant|implnt|permanent\s+implant)\b|"
        r"\bexchange\b.{0,60}\b(implant|implnt|permanent\s+implant)\b.{0,120}\b(tissue\s*expander|expander|expandr|te)\b",
        re.I
    ),
    "REMOVE_EXPANDER_PLACE_IMPLANT": re.compile(
        r"\b(remove|removed|explant|explanted)\b.{0,120}\b(tissue\s*expander|expander|expandr|te)\b.{0,220}\b(place|placed|insert|inserted)\b.{0,80}\b(implant|implnt|permanent\s+implant)\b|"
        r"\b(place|placed|insert|inserted)\b.{0,80}\b(implant|implnt|permanent\s+implant)\b.{0,220}\b(remove|removed|explant|explanted)\b.{0,120}\b(tissue\s*expander|expander|expandr|te)\b",
        re.I
    ),
}


def classify_note_stage2(note_text):
    """
    Returns (tier, matched_rule) where tier is one of: A, B, C, None
    """
    t = note_text

    # Tier A: definitive
    if RX["EXCHANGE_TE_TO_IMPLANT"].search(t):
        return ("A", "EXCHANGE_TE_TO_IMPLANT")
    if RX["REMOVE_EXPANDER_PLACE_IMPLANT"].search(t):
        return ("A", "REMOVE_EXPANDER_PLACE_IMPLANT")

    # Tier B: strong but not perfect
    has_second = bool(RX["SECOND_STAGE"].search(t))
    has_capsu = bool(RX["CAPSU"].search(t))
    has_implant = bool(RX["IMPLANT"].search(t))
    has_expander = bool(RX["EXPANDER"].search(t))
    has_exchange = bool(RX["EXCHANGE"].search(t))
    has_remove = bool(RX["REMOVE"].search(t))
    has_place = bool(RX["PLACE"].search(t))

    if has_exchange and has_implant and has_expander:
        return ("B", "EXCHANGE+IMPLANT+EXPANDER")
    if has_capsu and has_implant and (has_exchange or has_remove or has_place) and has_expander:
        return ("B", "CAPSU+IMPLANT+(ACTION)+EXPANDER")
    if has_second and has_implant and (has_exchange or has_remove or has_place) and has_expander:
        return ("B", "SECOND_STAGE+IMPLANT+(ACTION)+EXPANDER")

    # Tier C: suggestive
    if has_second and (has_implant or has_exchange):
        return ("C", "SECOND_STAGE_SUGGESTIVE")
    if has_implant and has_expander and not (has_remove or has_exchange or has_place):
        return ("C", "IMPLANT+EXPANDER_CONTEXT_NO_ACTION")

    return (None, None)


def snippet(text, n=240):
    s = "" if text is None else str(text)
    s = s.replace(u"\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "...") if len(s) > n else s


def main():
    # -------------------------
    # Load expander cohort + index dates (stage1_date)
    # -------------------------
    stg = read_csv_robust(PATIENT_STAGING_CSV)

    needed_cols = ["patient_id", "has_expander", "stage1_date"]
    missing = [c for c in needed_cols if c not in stg.columns]
    if missing:
        raise RuntimeError("Missing required columns in {}: {}".format(PATIENT_STAGING_CSV, missing))

    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)
    exp = stg[stg["has_expander_bool"]].copy()

    exp["patient_id"] = exp["patient_id"].fillna("").astype(str)
    exp["stage1_dt"] = parse_date_series(exp["stage1_date"])

    exp_ids = set(exp["patient_id"].tolist())
    if not exp_ids:
        raise RuntimeError("No expander patients found in {} (has_expander==True).".format(PATIENT_STAGING_CSV))

    stage1_map = dict(zip(exp["patient_id"], exp["stage1_dt"]))
    mrn_map = dict(zip(exp["patient_id"], exp.get("mrn", pd.Series([""] * len(exp))).fillna("").astype(str)))

    # -------------------------
    # Validate note columns exist
    # -------------------------
    usecols = [COL_PATIENT, COL_MRN, COL_NOTE_TEXT, COL_NOTE_DOS, COL_OP_DATE, COL_NOTE_TYPE, COL_NOTE_ID]
    head = read_csv_robust(OP_NOTES_CSV, nrows=5)
    for c in usecols:
        if c not in head.columns:
            raise RuntimeError("Missing required note column in OP notes file: {}".format(c))

    # -------------------------
    # Stream operation notes and collect hits
    # -------------------------
    chunksize = 200000  # adjust if memory issues
    hit_chunks = []

    total_rows_expander = 0
    total_hit_rows = 0

    for chunk in read_csv_robust(OP_NOTES_CSV, usecols=usecols, chunksize=chunksize):
        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)

        # keep expander cohort only
        chunk = chunk[chunk[COL_PATIENT].isin(exp_ids)].copy()
        if chunk.empty:
            continue

        total_rows_expander += len(chunk)

        chunk["note_text_norm"] = chunk[COL_NOTE_TEXT].apply(norm_text)

        # date: prefer NOTE_DATE_OF_SERVICE, fallback OPERATION_DATE
        chunk["note_dt"] = parse_date_series(chunk[COL_NOTE_DOS])
        chunk["op_dt"] = parse_date_series(chunk[COL_OP_DATE])
        chunk["event_dt"] = chunk["note_dt"].fillna(chunk["op_dt"])

        # classify
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

        hits["stage1_dt"] = hits[COL_PATIENT].map(stage1_map)
        hits["delta_days_vs_stage1"] = (hits["event_dt"] - hits["stage1_dt"]).dt.days
        hits["snippet"] = hits[COL_NOTE_TEXT].apply(snippet)

        keep = [
            COL_PATIENT, COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID,
            COL_NOTE_DOS, COL_OP_DATE, "event_dt",
            "tier", "rule", "delta_days_vs_stage1", "snippet"
        ]
        hit_chunks.append(hits[keep])
        total_hit_rows += len(hits)

    if hit_chunks:
        hits_all = pd.concat(hit_chunks, ignore_index=True)
    else:
        hits_all = pd.DataFrame(columns=[
            COL_PATIENT, COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID,
            COL_NOTE_DOS, COL_OP_DATE, "event_dt",
            "tier", "rule", "delta_days_vs_stage1", "snippet"
        ])

    # -------------------------
    # Patient-level best Stage 2
    # Prefer: AFTER index, Tier A > B > C, earliest event_dt
    # -------------------------
    tier_rank = {"A": 3, "B": 2, "C": 1}
    hits_all["tier_rank"] = hits_all["tier"].map(tier_rank).fillna(0).astype(int)

    def after_index_flag(x):
        return pd.notnull(x) and (x >= 0)

    hits_all["after_index"] = hits_all["delta_days_vs_stage1"].apply(after_index_flag)

    hits_all = hits_all.sort_values(
        by=[COL_PATIENT, "after_index", "tier_rank", "event_dt"],
        ascending=[True, False, False, True]
    )

    best = hits_all.groupby(COL_PATIENT, as_index=False).head(1).copy()

    patient_level = pd.DataFrame({"patient_id": list(exp_ids)})
    patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)
    patient_level["mrn_from_staging"] = patient_level["patient_id"].map(mrn_map)

    if not best.empty:
        patient_level = patient_level.merge(
            best.rename(columns={COL_PATIENT: "patient_id"}),
            on="patient_id",
            how="left"
        )
    else:
        for c in [COL_MRN, COL_NOTE_TYPE, COL_NOTE_ID, COL_NOTE_DOS, COL_OP_DATE, "event_dt",
                  "tier", "rule", "delta_days_vs_stage1", "snippet", "after_index", "tier_rank"]:
            patient_level[c] = None

    patient_level = patient_level.rename(columns={
        "event_dt": "stage2_event_dt_best",
        "tier": "stage2_tier_best",
        "rule": "stage2_rule_best",
        "after_index": "stage2_after_index",
        "delta_days_vs_stage1": "stage2_delta_days_from_stage1",
        COL_NOTE_TYPE: "best_note_type",
        COL_NOTE_ID: "best_note_id",
        COL_NOTE_DOS: "best_note_dos",
        COL_OP_DATE: "best_note_op_date",
        COL_MRN: "mrn_from_notes"
    })

    # -------------------------
    # Write outputs
    # -------------------------
    hits_all.to_csv(OUT_ROW_HITS, index=False)
    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False)

    # -------------------------
    # Summary
    # -------------------------
    n_exp = len(exp_ids)
    n_pat_with_any_hit = int(patient_level["stage2_tier_best"].notnull().sum())
    n_pat_after_index = int(patient_level["stage2_after_index"].fillna(False).astype(bool).sum())
    tier_counts = patient_level["stage2_tier_best"].fillna("NONE").value_counts()

    lines = []
    lines.append("=== Stage 2 from OPERATION NOTES (Expanders) ===")
    lines.append("Expander patients (from patient_recon_staging.csv): {}".format(n_exp))
    lines.append("Total note rows scanned (expanders only): {}".format(total_rows_expander))
    lines.append("Total hit rows (tier A/B/C): {}".format(total_hit_rows))
    lines.append("Patients with ANY Stage2 note hit (any date): {} ({:.1f}%)".format(
        n_pat_with_any_hit, (100.0 * n_pat_with_any_hit / n_exp) if n_exp else 0.0
    ))
    lines.append("Patients with best Stage2 hit AFTER Stage1 index: {} ({:.1f}%)".format(
        n_pat_after_index, (100.0 * n_pat_after_index / n_exp) if n_exp else 0.0
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

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
