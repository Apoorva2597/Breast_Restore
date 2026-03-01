#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py (Python 3.6.8 compatible)

Stage2 *staging only* + audit outputs.
- NO validation in this script.
- Keeps output file name used by your validation + freeze-pack:
    ./_outputs/patient_stage_summary.csv

Primary goal: increase precision by constraining Stage2 signals to *intraoperative* context
(EBL, drains, specimens, anesthesia / OR language) and down-weight clinic-only hits.

Run from: /home/apokol/Breast_Restore

Inputs (defaults; edit paths below if needed):
- ./_staging_inputs/HPI11526 Operation Notes.csv

Outputs:
- ./_outputs/patient_stage_summary.csv   (required by validate/freeze)
- ./_outputs/stage2_audit_event_hits.csv (row-level hits for quick review)
- ./_outputs/stage2_audit_bucket_counts.csv (bucket counts)

Notes:
- Robust CSV read for common encodings.
- Flexible column detection (NOTE_TEXT / SNIP_01.. / SNIPPET / etc).
- Produces required columns for downstream scripts:
    ENCRYPTED_PAT_ID, HAS_STAGE2, STAGE2_DATE, STAGE2_NOTE_ID, STAGE2_NOTE_TYPE,
    STAGE2_MATCH_PATTERN, STAGE2_HITS
"""

from __future__ import print_function
import os
import re
import pandas as pd


# -------------------------
# IO paths (edit if needed)
# -------------------------
ROOT = os.path.abspath(".")
INPUT_NOTES = os.path.join(ROOT, "_staging_inputs", "HPI11526 Operation Notes.csv")

OUT_DIR = os.path.join(ROOT, "_outputs")
OUT_PATIENT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")

OUT_AUDIT_HITS = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_AUDIT_BUCKETS = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")


# -------------------------
# Helpers
# -------------------------

def read_csv_robust(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise IOError("Failed to read CSV with common encodings: {}".format(path))


def normalize_cols(df):
    df.columns = [str(c).replace(u"\xa0", " ").strip() for c in df.columns]
    return df


def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None


def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        head = s[:-2]
        if head.isdigit():
            return head
    return s


def safe_str(x):
    if x is None:
        return ""
    s = str(x)
    if s.lower() == "nan":
        return ""
    return s


def coalesce_text(row, text_cols):
    parts = []
    for c in text_cols:
        v = safe_str(row.get(c, ""))
        if v:
            parts.append(v)
    return " ".join(parts).strip()


def detect_text_columns(df):
    # Prefer a single full-text column if present, else use SNIP_/SNIPPET-like fields
    preferred = ["NOTE_TEXT", "NoteText", "TEXT", "Text", "SNIPPET", "Snippet", "NOTE", "Note"]
    col = pick_first_existing(df, preferred)
    if col:
        return [col]

    # Otherwise gather SNIP_01.., SNIP_ columns, and any column containing "snip" or "snippet"
    snip_cols = []
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("snip_") or "snippet" in cl or cl == "snip":
            snip_cols.append(c)

    # Fallback: if nothing found, try "HPI" / "CONTENT" style columns
    if not snip_cols:
        for c in df.columns:
            cl = str(c).lower()
            if "content" in cl or "body" in cl or "narrative" in cl:
                snip_cols.append(c)

    return snip_cols


def lower_clean(s):
    s = safe_str(s)
    s = s.replace(u"\xa0", " ")
    return s.lower()


# -------------------------
# Stage2 logic (precision-first)
# -------------------------

# Core procedure language for Stage2 exchange/implant
RE_EXCHANGE = re.compile(
    r"\b(exchange|exchanged|replacement|replace|remove(?:d)?\s+and\s+replace(?:d)?)\b"
    r".{0,80}\b(tissue\s+expander|expander|te)\b"
    r"|"
    r"\b(tissue\s+expander|expander|te)\b.{0,80}\b(exchange|exchanged|replacement|replace)\b",
    re.IGNORECASE | re.DOTALL,
)

RE_IMPLANT = re.compile(
    r"\b(implant|implants)\b"
    r"|"
    r"\b(permanent)\s+(silicone|saline)\s+implant(s)?\b"
    r"|"
    r"\b(silicone|saline)\s+(gel\s+)?implant(s)?\b"
    r"|"
    r"\b(implant)\s+(placement|placed)\b",
    re.IGNORECASE,
)

# Intraoperative context signals (to reduce clinic-note FP)
RE_INTRAOP = re.compile(
    r"\b(estimated\s+blood\s+loss|ebl)\b"
    r"|"
    r"\b(drains?\s+(?:were\s+)?(?:placed|left)\b|\bjp\s+drain\b|\bjackson[-\s]?pratt\b)\b"
    r"|"
    r"\b(specimen(s)?\b|\bwas\s+sent\s+to\s+pathology\b|\bsent\s+to\s+path\b)\b"
    r"|"
    r"\b(anesthesia\b|\bgeneral\s+anesthesia\b|\bmac\b|\blma\b|\bet\s+tube\b|\bintubat(ed|ion)\b)\b"
    r"|"
    r"\b(operating\s+room|or\s+suite|prepped\s+and\s+draped|time\s*out|counts?\s+were\s+correct)\b"
    r"|"
    r"\b(procedure(s)?\b|\boperative\s+report\b|\bpost[-\s]?op(?:erative)?\b)\b",
    re.IGNORECASE,
)

# Explicit “plan to” language (should NOT count as Stage2 anchor)
RE_FUTURE_PLAN = re.compile(
    r"\b(plan(?:s|ned)?\s+to|will\s+(?:schedule|plan|proceed|return)\s+for|scheduled\s+for|anticipate)\b"
    r".{0,120}\b(exchange|implant|expander)\b",
    re.IGNORECASE | re.DOTALL,
)

# Use SOURCE_FILE / NOTE_TYPE to identify clinic notes vs operative docs
def infer_note_type(row, note_type_col, source_col):
    nt = lower_clean(row.get(note_type_col, "")) if note_type_col else ""
    sf = lower_clean(row.get(source_col, "")) if source_col else ""
    blob = (nt + " " + sf).strip()

    if any(k in blob for k in ["operative", "operation", "op note", "brief op", "surgical", "intraop", "or note"]):
        return "OP"
    if any(k in blob for k in ["clinic", "office", "follow-up", "follow up", "hpi", "outpatient"]):
        return "CLINIC"
    # If the file itself is "Operation Notes.csv", default to OP unless strongly clinic
    if "clinic" in blob:
        return "CLINIC"
    return "OP"


def extract_stage2_hits(text):
    """
    Returns:
      hits_count (int),
      match_pattern (str),
      is_future_plan (bool),
      has_exchange (bool),
      has_implant (bool),
      has_intraop (bool)
    """
    t = safe_str(text)
    if not t:
        return 0, "", False, False, False, False

    is_future = True if RE_FUTURE_PLAN.search(t) else False
    has_exchange = True if RE_EXCHANGE.search(t) else False
    has_implant = True if RE_IMPLANT.search(t) else False
    has_intraop = True if RE_INTRAOP.search(t) else False

    # Count "hits" as number of distinct core signals present (exchange + implant + intraop)
    hits = 0
    hits += 1 if has_exchange else 0
    hits += 1 if has_implant else 0
    hits += 1 if has_intraop else 0

    # Pattern label
    if has_exchange and has_implant and has_intraop and (not is_future):
        pat = "EXCHANGE+IMPLANT+INTRAOP"
    elif has_exchange and has_implant and (not is_future):
        pat = "EXCHANGE+IMPLANT"
    elif has_exchange and has_intraop and (not is_future):
        pat = "EXCHANGE+INTRAOP"
    elif has_exchange and (not is_future):
        pat = "EXCHANGE_ONLY"
    elif has_exchange and is_future:
        pat = "FUTURE_PLAN_EXCHANGE"
    elif has_implant and is_future:
        pat = "FUTURE_PLAN_IMPLANT"
    elif has_implant:
        pat = "IMPLANT_ONLY"
    else:
        pat = ""

    return hits, pat, is_future, has_exchange, has_implant, has_intraop


def bucket_row(note_type, is_future, has_exchange, has_implant, has_intraop):
    """
    Buckets:
      - DEFINITIVE_OP: exchange + implant + intraop, not future
      - PROBABLE_OP: exchange + implant (no intraop) but OP, not future
      - POSSIBLE_OP: exchange + intraop (no implant) but OP, not future
      - CLINIC_ONLY: any core hit but note_type=CLINIC
      - FUTURE_PLAN: plan language present
      - OTHER: everything else
    """
    if is_future:
        return "FUTURE_PLAN"
    if note_type == "CLINIC" and (has_exchange or has_implant):
        return "CLINIC_ONLY"
    if (has_exchange and has_implant and has_intraop):
        return "DEFINITIVE_OP"
    if note_type == "OP" and (has_exchange and has_implant):
        return "PROBABLE_OP"
    if note_type == "OP" and (has_exchange and has_intraop):
        return "POSSIBLE_OP"
    if note_type == "OP" and has_exchange:
        return "EXCHANGE_ONLY_OP"
    if note_type == "OP" and has_implant:
        return "IMPLANT_ONLY_OP"
    return "OTHER"


def bucket_score(bucket):
    # Higher wins when picking best row per patient
    order = {
        "DEFINITIVE_OP": 60,
        "PROBABLE_OP": 50,
        "POSSIBLE_OP": 40,
        "EXCHANGE_ONLY_OP": 30,
        "IMPLANT_ONLY_OP": 20,
        "CLINIC_ONLY": 10,
        "FUTURE_PLAN": 0,
        "OTHER": -1,
    }
    return order.get(bucket, -1)


def compute_has_stage2(bucket):
    # Precision-first: only count OP buckets (exclude CLINIC_ONLY and FUTURE_PLAN)
    return 1 if bucket in ["DEFINITIVE_OP", "PROBABLE_OP", "POSSIBLE_OP", "EXCHANGE_ONLY_OP"] else 0


# -------------------------
# Main
# -------------------------

def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    if not os.path.isfile(INPUT_NOTES):
        raise IOError("Input notes CSV not found: {}".format(INPUT_NOTES))

    df = normalize_cols(read_csv_robust(INPUT_NOTES, dtype=str, low_memory=False))

    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Missing ENCRYPTED_PAT_ID in input. Found: {}".format(list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})

    note_id_col = pick_first_existing(df, ["NOTE_ID", "NoteID", "ENC_NOTE_ID", "ID", "NOTEID"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "NoteType", "TYPE", "DOC_TYPE", "DOCUMENT_TYPE"])
    source_col = pick_first_existing(df, ["SOURCE_FILE", "SourceFile", "FILE", "FILENAME"])

    date_col = pick_first_existing(df, ["NOTE_DATE", "DATE", "SERVICE_DATE", "ENC_DATE", "VISIT_DATE"])
    # If you have a known column for op date, add it to the options above.

    text_cols = detect_text_columns(df)
    if not text_cols:
        raise ValueError("Could not detect any text/snippet columns to search. Found: {}".format(list(df.columns)))

    # Build row-level audit hits
    rows = []
    for i, r in df.iterrows():
        enc = normalize_id(r.get("ENCRYPTED_PAT_ID", ""))
        if not enc:
            continue

        note_id = safe_str(r.get(note_id_col, "")) if note_id_col else ""
        note_type = infer_note_type(r, note_type_col, source_col)
        note_date = safe_str(r.get(date_col, "")) if date_col else ""

        text = coalesce_text(r, text_cols)
        hits, pat, is_future, has_exchange, has_implant, has_intraop = extract_stage2_hits(text)
        bucket = bucket_row(note_type, is_future, has_exchange, has_implant, has_intraop)

        if hits > 0 or bucket in ["FUTURE_PLAN", "CLINIC_ONLY", "EXCHANGE_ONLY_OP", "IMPLANT_ONLY_OP"]:
            # Keep only "interesting" rows for audit (reduces file size)
            rows.append({
                "ENCRYPTED_PAT_ID": enc,
                "NOTE_ID": note_id,
                "NOTE_TYPE": note_type,
                "NOTE_DATE": note_date,
                "BUCKET": bucket,
                "HITS": hits,
                "MATCH_PATTERN": pat,
                "HAS_EXCHANGE": int(has_exchange),
                "HAS_IMPLANT": int(has_implant),
                "HAS_INTRAOP": int(has_intraop),
                "IS_FUTURE_PLAN": int(is_future),
                "TEXT_SNIPPET": (text[:450].replace("\n", " ").replace("\r", " ")) if text else ""
            })

    audit = pd.DataFrame(rows)

    # If audit is empty, still emit required patient summary (all zeros)
    if audit.empty:
        patient_ids = df["ENCRYPTED_PAT_ID"].map(normalize_id)
        patient_ids = patient_ids[patient_ids != ""].drop_duplicates()
        out = pd.DataFrame({
            "ENCRYPTED_PAT_ID": patient_ids,
            "HAS_STAGE2": 0,
            "STAGE2_DATE": "",
            "STAGE2_NOTE_ID": "",
            "STAGE2_NOTE_TYPE": "",
            "STAGE2_MATCH_PATTERN": "",
            "STAGE2_HITS": 0
        })
        out.to_csv(OUT_PATIENT_SUMMARY, index=False)

        # Minimal audits
        pd.DataFrame({"BUCKET": [], "COUNT": []}).to_csv(OUT_AUDIT_BUCKETS, index=False)
        pd.DataFrame([]).to_csv(OUT_AUDIT_HITS, index=False)

        print("Staging complete.")
        print("Patients: {}".format(int(out["ENCRYPTED_PAT_ID"].nunique())))
        print("Events: 0")
        return

    # Bucket counts
    bucket_counts = audit.groupby("BUCKET", as_index=False).size()
    bucket_counts = bucket_counts.rename(columns={"size": "COUNT"})

    # Pick best evidence row per patient (score desc, then hits desc)
    audit["_BUCKET_SCORE"] = audit["BUCKET"].map(bucket_score)
    audit["_NOTE_ID_SORT"] = audit["NOTE_ID"].fillna("").astype(str)

    audit_sorted = audit.sort_values(
        by=["ENCRYPTED_PAT_ID", "_BUCKET_SCORE", "HITS", "_NOTE_ID_SORT"],
        ascending=[True, False, False, True]
    )

    best = audit_sorted.groupby("ENCRYPTED_PAT_ID", as_index=False).head(1).copy()
    best["HAS_STAGE2"] = best["BUCKET"].map(compute_has_stage2).astype(int)

    # Map to required output schema for validation + freeze-pack
    out = pd.DataFrame({
        "ENCRYPTED_PAT_ID": best["ENCRYPTED_PAT_ID"],
        "HAS_STAGE2": best["HAS_STAGE2"],
        "STAGE2_DATE": best["NOTE_DATE"].fillna(""),
        "STAGE2_NOTE_ID": best["NOTE_ID"].fillna(""),
        "STAGE2_NOTE_TYPE": best["NOTE_TYPE"].fillna(""),
        "STAGE2_MATCH_PATTERN": best["MATCH_PATTERN"].fillna(""),
        "STAGE2_HITS": best["HITS"].fillna(0).astype(int)
    })

    # Ensure we include patients that had zero "interesting" rows (rare but possible)
    all_pats = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    all_pats = all_pats[all_pats != ""].drop_duplicates()
    out = all_pats.to_frame(name="ENCRYPTED_PAT_ID").merge(out, on="ENCRYPTED_PAT_ID", how="left")
    out["HAS_STAGE2"] = out["HAS_STAGE2"].fillna(0).astype(int)
    out["STAGE2_DATE"] = out["STAGE2_DATE"].fillna("")
    out["STAGE2_NOTE_ID"] = out["STAGE2_NOTE_ID"].fillna("")
    out["STAGE2_NOTE_TYPE"] = out["STAGE2_NOTE_TYPE"].fillna("")
    out["STAGE2_MATCH_PATTERN"] = out["STAGE2_MATCH_PATTERN"].fillna("")
    out["STAGE2_HITS"] = out["STAGE2_HITS"].fillna(0).astype(int)

    # Write outputs
    out.to_csv(OUT_PATIENT_SUMMARY, index=False)
    audit.drop(columns=["_BUCKET_SCORE", "_NOTE_ID_SORT"], errors="ignore").to_csv(OUT_AUDIT_HITS, index=False)
    bucket_counts.to_csv(OUT_AUDIT_BUCKETS, index=False)

    # Console summary
    print("Staging complete.")
    print("Patients: {}".format(int(out["ENCRYPTED_PAT_ID"].nunique())))
    print("Events: {}".format(int(len(audit))))

    # Extra quick sanity counts
    s2 = int((out["HAS_STAGE2"] == 1).sum())
    print("HAS_STAGE2=1: {}".format(s2))
    print("Wrote:")
    print("  {}".format(OUT_PATIENT_SUMMARY))
    print("  {}".format(OUT_AUDIT_HITS))
    print("  {}".format(OUT_AUDIT_BUCKETS))


if __name__ == "__main__":
    main()
