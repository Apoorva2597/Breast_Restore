# stage2_from_operation_notes_expanders.py
# Python 3.6.8+ (pandas required)
#
# Goal:
#   For expander-pathway patients (from patient_recon_staging.csv),
#   scan OPERATION NOTES for Stage 2 evidence (TE -> implant exchange).
#
# Inputs:
#   1) patient_recon_staging.csv
#   2) HPI11526 Operation Notes.csv  (must include NOTE_TEXT and date columns)
#
# Outputs:
#   1) stage2_from_notes_patient_level.csv
#   2) stage2_from_notes_row_hits.csv
#   3) stage2_from_notes_summary.txt
#
# Key design choices (scope-aligned):
#   - Stage2 is ONLY TE->implant exchange (not TE->TE swap).
#   - Prefer high-signal procedure section when present.
#   - If procedure section missing, use NOTE_TYPE as proxy (OP NOTE / BRIEF OP NOTE / OPERATIVE).
#   - Robust file reading for Python 3.6 + messy CSV bytes (0xA0 etc): latin1 + errors=replace.

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

# Operation notes columns (adjust ONLY if your file headers differ)
COL_PATIENT = "ENCRYPTED_PAT_ID"
COL_MRN = "MRN"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_ID = "NOTE_ID"

CHUNKSIZE = 100000  # safe default for big notes files


# -------------------------
# Robust CSV reading (Python 3.6 safe)
# -------------------------
def read_csv_safe(path, **kwargs):
    """
    Read CSV robustly in Python 3.6 by using latin1 + errors=replace.
    Avoids hard failures from 0xA0 (NBSP) or other stray bytes.
    """
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        # IMPORTANT: if chunksize is used, caller should use iter_csv_safe instead.
        if "chunksize" not in kwargs:
            try:
                f.close()
            except Exception:
                pass


def iter_csv_safe(path, **kwargs):
    """
    Chunk iterator version: keep file handle open for the duration of iteration.
    """
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        for chunk in pd.read_csv(f, engine="python", **kwargs):
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass


def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = s.replace("_", " ")
    # normalize NBSP if present (shows up as \xa0 in python strings)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


def to_bool(x):
    s = str(x).strip().lower()
    return s in ["true", "1", "yes", "y"]


def snippet(text, n=320):
    s = "" if text is None else str(text)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "...") if len(s) > n else s


# -------------------------
# Procedure-section extraction (HIGH signal)
# -------------------------
PROC_RE = re.compile(
    r"\b("
    r"PROCEDURES?\s*(?:\(\s*S\s*\))?\s*(?:PERFORMED)?"
    r"|OPERATIVE\s+PROCEDURES?"
    r"|OPERATION\s+(?:PERFORMED)?"
    r"|PROCEDURE\s*(?:\(\s*S\s*\))?"
    r")\b\s*[:\-]?",
    re.I
)

STOP_RE = re.compile(
    r"\b("
    r"INDICATIONS?"
    r"|FINDINGS?"
    r"|ANESTHESIA"
    r"|COMPLICATIONS?"
    r"|DISPOSITION"
    r"|SPECIMENS?"
    r"|ESTIMATED\s+BLOOD\s+LOSS"
    r"|EBL"
    r"|DRAINS?"
    r"|IMPLANTS?"
    r"|PATHOLOGY"
    r"|DICTATED\s+BY"
    r")\b\s*[:\-]?",
    re.I
)


def extract_procedure_segment(note_text, max_len=3500):
    """
    Return a high-signal segment starting at the first procedure-ish heading.
    If not found, return "".
    """
    t = norm_text(note_text)
    m = PROC_RE.search(t)
    if not m:
        return ""

    start = m.start()
    window = t[start:start + max_len]

    # stop at first stop heading after some buffer
    m2 = STOP_RE.search(window[30:])
    if m2:
        end = 30 + m2.start()
        return window[:end].strip()

    return window.strip()


# -------------------------
# Stage 2 evidence patterns (notes)
# -------------------------
# Stage 2 = exchange/remove tissue expander + place/insert implant (or explicit expander->implant exchange)
RX = {
    "EXPANDER": re.compile(r"\b(tissue\s*expander|expander|expandr|\bte\b)\b", re.I),
    "IMPLANT": re.compile(r"\b(implant|implnt|permanent\s+implant)\b", re.I),
    "REMOVE": re.compile(r"\b(remove|removed|explant|explanted|take\s+out|taken\s+out)\b", re.I),
    "PLACE": re.compile(r"\b(place|placed|insert|inserted|insertion|implantation)\b", re.I),
    "EXCHANGE": re.compile(r"\b(exchange|exchanged)\b", re.I),
    "SECOND_STAGE": re.compile(r"\b(second\s+stage|stage\s*2|stage\s+ii)\b", re.I),
    "CAPSU": re.compile(r"\b(capsulotomy|capsulectomy)\b", re.I),

    # Tier A: definitive combined phrases
    "EXCHANGE_TE_TO_IMPLANT": re.compile(
        r"\bexchange\b.{0,80}\b(tissue\s*expander|expander|expandr|\bte\b)\b.{0,220}\b(implant|implnt|permanent\s+implant)\b|"
        r"\bexchange\b.{0,80}\b(implant|implnt|permanent\s+implant)\b.{0,220}\b(tissue\s*expander|expander|expandr|\bte\b)\b",
        re.I
    ),
    "REMOVE_EXPANDER_PLACE_IMPLANT": re.compile(
        r"\b(remove|removed|explant|explanted)\b.{0,200}\b(tissue\s*expander|expander|expandr|\bte\b)\b.{0,380}\b(place|placed|insert|inserted|insertion|implantation)\b.{0,140}\b(implant|implnt|permanent\s+implant)\b|"
        r"\b(place|placed|insert|inserted|insertion|implantation)\b.{0,140}\b(implant|implnt|permanent\s+implant)\b.{0,380}\b(remove|removed|explant|explanted)\b.{0,200}\b(tissue\s*expander|expander|expandr|\bte\b)\b",
        re.I
    ),
}


def is_op_like(note_type):
    nt = "" if note_type is None else str(note_type).upper()
    return ("OP NOTE" in nt) or ("BRIEF OP NOTE" in nt) or ("OPERATIVE" in nt)


def classify_note_stage2(note_text, note_type):
    """
    Returns (tier, rule, using_proc_section)
    Tier in {"A","B","C",None}
    using_proc_section is True if we had a procedure segment OR NOTE_TYPE indicates op-like.
    """
    full = norm_text(note_text)
    proc = extract_procedure_segment(full)

    op_like = is_op_like(note_type)
    using_proc = bool(proc) or op_like

    # If we have procedure segment use it; else fallback to full.
    text = proc if proc else full

    has_expander = bool(RX["EXPANDER"].search(text))
    has_implant = bool(RX["IMPLANT"].search(text))
    has_exchange = bool(RX["EXCHANGE"].search(text))
    has_remove = bool(RX["REMOVE"].search(text))
    has_place = bool(RX["PLACE"].search(text))
    has_second = bool(RX["SECOND_STAGE"].search(text))
    has_capsu = bool(RX["CAPSU"].search(text))

    # Suppress clear TE->TE exchanges (exchange + expander but no implant)
    # This is directly supported by your de-identified examples (ruptured TE exchanged for new TE).
    if has_exchange and has_expander and (not has_implant):
        return (None, "EXCHANGE_EXPANDER_NO_IMPLANT", using_proc)

    # Tier A: definitive phrases (only label A when we're confident it's procedural)
    if RX["EXCHANGE_TE_TO_IMPLANT"].search(text):
        return ("A" if using_proc else "B", "EXCHANGE_TE_TO_IMPLANT", using_proc)

    if RX["REMOVE_EXPANDER_PLACE_IMPLANT"].search(text):
        return ("A" if using_proc else "B", "REMOVE_EXPANDER_PLACE_IMPLANT", using_proc)

    # Tier B: strong combos
    if has_expander and has_implant and has_exchange:
        return ("B" if using_proc else "C", "EXPANDER+IMPLANT+EXCHANGE", using_proc)

    if has_expander and has_implant and (has_remove or has_place) and has_capsu:
        return ("B" if using_proc else "C", "CAPSU+(REMOVE/PLACE)+EXPANDER+IMPLANT", using_proc)

    if has_expander and has_implant and (has_remove or has_place) and has_second:
        return ("B" if using_proc else "C", "SECOND_STAGE+(REMOVE/PLACE)+EXPANDER+IMPLANT", using_proc)

    # Tier C: suggestive (conservative; requires expander context)
    if has_expander and has_implant and has_second:
        return ("C", "SECOND_STAGE_MENTION_EXPANDER_IMPLANT", using_proc)

    if has_expander and has_implant and not (has_remove or has_place or has_exchange):
        return ("C", "EXPANDER+IMPLANT_CONTEXT_NO_ACTION", using_proc)

    return (None, None, using_proc)


def bin_label(delta_days):
    if delta_days is None or pd.isnull(delta_days):
        return None
    try:
        x = int(delta_days)
    except Exception:
        return None
    if x <= 30:
        return "0-30d"
    if x <= 90:
        return "31-90d"
    if x <= 180:
        return "91-180d"
    if x <= 365:
        return "181-365d"
    return ">365d"


def main():
    # -------------------------
    # Load expander cohort
    # -------------------------
    stg = read_csv_safe(PATIENT_STAGING_CSV)
    for col in ["patient_id", "has_expander", "stage1_date"]:
        if col not in stg.columns:
            raise RuntimeError("Missing required column '{}' in {}".format(col, PATIENT_STAGING_CSV))

    stg["patient_id"] = stg["patient_id"].fillna("").astype(str)
    stg["has_expander_bool"] = stg["has_expander"].apply(to_bool)
    exp = stg[stg["has_expander_bool"]].copy()

    exp["stage1_dt"] = to_dt(exp["stage1_date"])
    exp_ids = set(exp["patient_id"].astype(str).tolist())
    if not exp_ids:
        raise RuntimeError("No expander patients found (has_expander==True).")

    stage1_map = dict(zip(exp["patient_id"].astype(str), exp["stage1_dt"]))
    mrn_map = dict(zip(exp["patient_id"].astype(str), exp.get("mrn", pd.Series([""] * len(exp))).astype(str)))

    print("Expander patients:", len(exp_ids))

    # -------------------------
    # Validate note columns exist
    # -------------------------
    head = read_csv_safe(OP_NOTES_CSV, nrows=5)
    required_cols = [COL_PATIENT, COL_NOTE_TEXT, COL_NOTE_DOS, COL_OP_DATE, COL_NOTE_TYPE, COL_NOTE_ID]
    for c in required_cols:
        if c not in head.columns:
            raise RuntimeError("Missing required note column '{}' in {}".format(c, OP_NOTES_CSV))

    usecols = required_cols[:]
    if COL_MRN in head.columns:
        usecols.append(COL_MRN)

    # -------------------------
    # Scan notes in chunks
    # -------------------------
    hit_rows = []
    total_rows_seen = 0
    total_rows_expanders = 0

    # extra diagnostic counters
    suppressed_te_to_te = 0
    any_proc_section_rows = 0
    op_like_rows = 0

    for chunk in iter_csv_safe(OP_NOTES_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows_seen += len(chunk)

        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PATIENT].isin(exp_ids)].copy()
        if chunk.empty:
            continue

        total_rows_expanders += len(chunk)

        # normalize text
        chunk["note_text_norm"] = chunk[COL_NOTE_TEXT].apply(norm_text)

        # date: prefer DOS; fallback OPERATION_DATE
        chunk["note_dt"] = to_dt(chunk[COL_NOTE_DOS])
        chunk["op_dt"] = to_dt(chunk[COL_OP_DATE])
        chunk["event_dt"] = chunk["note_dt"].fillna(chunk["op_dt"])

        # classify row-by-row (needed for proc-section extraction + note_type)
        tiers = []
        rules = []
        proc_found = []
        op_like_flags = []

        note_texts = chunk["note_text_norm"].tolist()
        note_types = chunk[COL_NOTE_TYPE].fillna("").astype(str).tolist()

        for i in range(len(note_texts)):
            txt = note_texts[i]
            nt = note_types[i]

            proc_seg = extract_procedure_segment(txt)
            has_proc = bool(proc_seg)
            proc_found.append(has_proc)

            op_like = is_op_like(nt)
            op_like_flags.append(op_like)

            tier, rule, _using_proc = classify_note_stage2(txt, nt)
            tiers.append(tier)
            rules.append(rule)

            if rule == "EXCHANGE_EXPANDER_NO_IMPLANT":
                suppressed_te_to_te += 1

        chunk["tier"] = tiers
        chunk["rule"] = rules
        chunk["has_proc_section"] = proc_found
        chunk["note_type_oplike"] = op_like_flags

        any_proc_section_rows += int(pd.Series(proc_found).sum())
        op_like_rows += int(pd.Series(op_like_flags).sum())

        hits = chunk[chunk["tier"].notnull()].copy()
        if hits.empty:
            continue

        hits["stage1_dt"] = hits[COL_PATIENT].map(stage1_map)
        hits["delta_days_vs_stage1"] = (hits["event_dt"] - hits["stage1_dt"]).dt.days
        hits["snippet"] = hits[COL_NOTE_TEXT].apply(lambda x: snippet(x, n=320))

        keep_cols = [
            COL_PATIENT,
            (COL_MRN if COL_MRN in hits.columns else None),
            COL_NOTE_TYPE, COL_NOTE_ID,
            COL_NOTE_DOS, COL_OP_DATE,
            "event_dt", "tier", "rule",
            "has_proc_section", "note_type_oplike",
            "delta_days_vs_stage1", "snippet"
        ]
        keep_cols = [c for c in keep_cols if c is not None]
        hit_rows.append(hits[keep_cols])

    if hit_rows:
        hits_all = pd.concat(hit_rows, ignore_index=True)
    else:
        hits_all = pd.DataFrame()

    # -------------------------
    # Patient-level best Stage 2
    # -------------------------
    if hits_all.empty:
        patient_level = pd.DataFrame({"patient_id": list(exp_ids)})
        patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)
        patient_level["mrn_from_staging"] = patient_level["patient_id"].map(mrn_map)
        patient_level["stage2_event_dt_best"] = None
        patient_level["stage2_tier_best"] = None
        patient_level["stage2_rule_best"] = None
        patient_level["stage2_after_index"] = None
        patient_level["stage2_delta_days_from_stage1"] = None
        patient_level["best_note_type"] = None
        patient_level["best_note_id"] = None
        patient_level["best_note_dos"] = None
        patient_level["best_note_op_date"] = None
        patient_level["mrn_from_notes"] = None
        patient_level["snippet"] = None
        patient_level["best_has_proc_section"] = None
        patient_level["best_note_type_oplike"] = None
    else:
        tier_rank = {"A": 3, "B": 2, "C": 1}
        hits_all["tier_rank"] = hits_all["tier"].map(tier_rank).fillna(0).astype(int)

        hits_all["after_index"] = hits_all["delta_days_vs_stage1"].apply(
            lambda x: (x is not None) and pd.notnull(x) and (x >= 0)
        )

        # prefer after_index True, then higher tier, then earliest event_dt
        hits_all = hits_all.sort_values(
            by=[COL_PATIENT, "after_index", "tier_rank", "event_dt"],
            ascending=[True, False, False, True]
        )

        best = hits_all.groupby(COL_PATIENT, as_index=False).head(1).copy()

        patient_level = pd.DataFrame({"patient_id": list(exp_ids)})
        patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)
        patient_level["mrn_from_staging"] = patient_level["patient_id"].map(mrn_map)

        best = best.rename(columns={COL_PATIENT: "patient_id"})
        patient_level = patient_level.merge(best, on="patient_id", how="left")

        rename_map = {
            "event_dt": "stage2_event_dt_best",
            "tier": "stage2_tier_best",
            "rule": "stage2_rule_best",
            "after_index": "stage2_after_index",
            "delta_days_vs_stage1": "stage2_delta_days_from_stage1",
            COL_NOTE_TYPE: "best_note_type",
            COL_NOTE_ID: "best_note_id",
            COL_NOTE_DOS: "best_note_dos",
            COL_OP_DATE: "best_note_op_date",
            "has_proc_section": "best_has_proc_section",
            "note_type_oplike": "best_note_type_oplike",
        }
        if COL_MRN in patient_level.columns:
            rename_map[COL_MRN] = "mrn_from_notes"

        patient_level = patient_level.rename(columns=rename_map)

    # -------------------------
    # Write outputs (UTF-8 for outputs)
    # -------------------------
    if not hits_all.empty:
        hits_all.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # -------------------------
    # Summary
    # -------------------------
    n_exp = len(exp_ids)
    n_pat_with_any_hit = int(patient_level["stage2_tier_best"].notnull().sum())
    n_pat_after_index = int(patient_level["stage2_after_index"].fillna(False).astype(bool).sum())
    tier_counts = patient_level["stage2_tier_best"].fillna("NONE").value_counts()

    # timing bins (only after-index)
    timing = patient_level[patient_level["stage2_after_index"].fillna(False).astype(bool)].copy()
    timing_bins = None
    if not timing.empty and "stage2_delta_days_from_stage1" in timing.columns:
        d = timing["stage2_delta_days_from_stage1"].dropna()
        if not d.empty:
            timing_bins = d.apply(bin_label).value_counts()

    # best-hit had procedure section?
    best_proc_counts = patient_level["best_has_proc_section"].fillna(False).astype(bool).value_counts()

    lines = []
    lines.append("=== Stage 2 from OPERATION NOTES (Expanders) ===")
    lines.append("Python: 3.6-compatible | Read encoding: latin1 (errors=replace) | Write outputs: utf-8")
    lines.append("Classifier: procedure-section aware + NOTE_TYPE fallback + TE->TE exchange suppression")
    lines.append("")
    lines.append("Expander patients (from patient_recon_staging.csv): {}".format(n_exp))
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
    lines.append("Best-hit had procedure section (patients):")
    if True in best_proc_counts.index:
        lines.append("  True: {}".format(int(best_proc_counts.get(True, 0))))
    else:
        lines.append("  True: 0")
    if False in best_proc_counts.index:
        lines.append("  False: {}".format(int(best_proc_counts.get(False, 0))))
    else:
        lines.append("  False: 0")

    lines.append("")
    lines.append("Row diagnostics (expanders only):")
    lines.append("  Total rows scanned (all patients in file): {}".format(total_rows_seen))
    lines.append("  Total rows scanned (expander cohort): {}".format(total_rows_expanders))
    lines.append("  Rows with detected procedure section: {}".format(any_proc_section_rows))
    lines.append("  Rows with OP-like NOTE_TYPE: {}".format(op_like_rows))
    lines.append("  Suppressed TE->TE exchanges (exchange+expander without implant): {}".format(suppressed_te_to_te))

    if timing_bins is not None:
        lines.append("")
        lines.append("Timing of Stage2 (delta days from Stage1) among AFTER-index hits:")
        for k, v in timing_bins.items():
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
