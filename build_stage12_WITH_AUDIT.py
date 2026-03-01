#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_stage12_WITH_AUDIT.py  (Python 3.6.8 compatible)

STAGING + AUDIT (NO validation here)

KEY BEHAVIOR (per your decision):
- KEEP HAS_STAGE2 strict (for stable TN/FP in validation).
- ADD a separate looser planning flag:
    CANDIDATE_PLANNING = 1  (clinic/intention language)
  This does NOT affect validation metrics unless you change HAS_STAGE2.

RUN FROM: /home/apokol/Breast_Restore

INPUTS (auto if present):
- ./_staging_inputs/HPI11526 Operation Notes.csv
- ./_staging_inputs/HPI11526 Clinic Notes.csv   (optional; helps planning candidates)

OUTPUTS:
- ./_outputs/patient_stage_summary.csv                  (required by validator + freeze)
- ./_outputs/stage2_audit_event_hits.csv                (row-level evidence + buckets)
- ./_outputs/stage2_audit_bucket_counts.csv             (bucket counts)
- ./_outputs/stage2_candidate_planning_hits.csv         (planning-only hits)
"""

from __future__ import print_function
import os
import re
import pandas as pd


ROOT = os.path.abspath(".")
STAGING_DIR = os.path.join(ROOT, "_staging_inputs")
OUT_DIR = os.path.join(ROOT, "_outputs")

OUT_PATIENT_SUMMARY = os.path.join(OUT_DIR, "patient_stage_summary.csv")
OUT_AUDIT_HITS = os.path.join(OUT_DIR, "stage2_audit_event_hits.csv")
OUT_AUDIT_BUCKETS = os.path.join(OUT_DIR, "stage2_audit_bucket_counts.csv")
OUT_PLANNING_HITS = os.path.join(OUT_DIR, "stage2_candidate_planning_hits.csv")


# -------------------------
# Robust CSV reader (same style as validator)
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


def lower_clean(x):
    return safe_str(x).replace(u"\xa0", " ").lower()


def detect_text_columns(df):
    preferred = ["NOTE_TEXT", "Text", "TEXT", "NOTE", "NOTE_BODY", "BODY", "SNIPPET", "Snippet"]
    c = pick_first_existing(df, preferred)
    if c:
        return [c]

    snip_cols = []
    for col in df.columns:
        cl = str(col).lower()
        if cl.startswith("snip_") or "snippet" in cl:
            snip_cols.append(col)

    # common older columns
    for c2 in ["SNIP_01", "SNIP_02", "SNIP_03", "SNIP_04", "SNIP_05"]:
        if c2 in df.columns and c2 not in snip_cols:
            snip_cols.append(c2)

    return snip_cols


def coalesce_text(row, text_cols):
    parts = []
    for c in text_cols:
        v = safe_str(row.get(c, ""))
        if v.strip():
            parts.append(v)
    return " ".join(parts).strip()


# -------------------------
# Note-type inference
# -------------------------

def infer_note_type(row, note_type_col, source_col):
    blob = ""
    if note_type_col:
        blob += " " + lower_clean(row.get(note_type_col, ""))
    if source_col:
        blob += " " + lower_clean(row.get(source_col, ""))
    b = blob.strip()

    if any(k in b for k in ["operative", "operation", "op note", "brief op", "surgical", "or note", "operative report"]):
        return "OP"
    if any(k in b for k in ["clinic", "office", "follow-up", "follow up", "outpatient", "visit", "hpi"]):
        return "CLINIC"
    return "OP"


# -------------------------
# Core regex signals
# -------------------------

RE_EXCHANGE = re.compile(
    r"\b(exchange|exchanged|replace|replaced|replacement|remove(?:d)?\s+and\s+replace(?:d)?)\b.{0,90}\b"
    r"(tissue\s+expander|expander|te)\b"
    r"|"
    r"\b(tissue\s+expander|expander|te)\b.{0,90}\b(exchange|exchanged|replace|replaced|replacement)\b",
    re.I | re.S
)

RE_IMPLANT = re.compile(
    r"\b(implant|implants|prosthesis)\b"
    r"|"
    r"\b(permanent)\s+(silicone|saline)\s+implant(s)?\b"
    r"|"
    r"\b(silicone|saline)\s+(gel\s+)?implant(s)?\b",
    re.I
)

RE_INTRAOP = re.compile(
    r"\b(estimated\s+blood\s+loss|ebl)\b"
    r"|"
    r"\b(drains?\s+(?:were\s+)?(?:placed|left)|jp\s+drain|jackson[-\s]?pratt|blake\s+drain)\b"
    r"|"
    r"\b(specimen(s)?|sent\s+to\s+pathology|sent\s+to\s+path)\b"
    r"|"
    r"\b(anesthesia|general\s+anesthesia|mac\b|lma\b|intubat(ed|ion)|endotracheal)\b"
    r"|"
    r"\b(prepped\s+and\s+draped|time\s*out|counts?\s+were\s+correct|operating\s+room|or\s+suite)\b"
    r"|"
    r"\b(procedure(s)?|operative\s+report|post[-\s]?op(?:erative)?)\b",
    re.I
)

RE_FUTURE_PLAN = re.compile(
    r"\b(plan(?:s|ned)?\s+to|will\s+(?:schedule|plan|proceed|return)\s+for|scheduled\s+for|anticipate)\b"
    r".{0,140}\b(exchange|implant|expander|stage\s*2|second\s+stage)\b",
    re.I | re.S
)

RE_NEG_PLAN = re.compile(
    r"\b(no\s+plans?|not\s+planning|never\s+scheduled|declines?\s+surgery|does\s+not\s+want)\b",
    re.I
)

RE_STAGE2_PHRASE = re.compile(
    r"\b(stage\s*2|second\s+stage|expander\s+to\s+implant|tissue\s+expander\s+to\s+implant)\b",
    re.I
)

RE_COUNSEL = re.compile(
    r"\b(discuss(ed|ion)?|counsel(ing)?|options?|risks?\s+and\s+benefits|consider(ing)?|candidate)\b",
    re.I
)

# -------------------------
# Buckets + scoring
# -------------------------

def bucket_score(bucket):
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


def compute_has_stage2_strict(bucket):
    # strict signal used by validator
    return 1 if bucket in ["DEFINITIVE_OP", "PROBABLE_OP", "POSSIBLE_OP", "EXCHANGE_ONLY_OP"] else 0


def compute_candidate_planning(note_type, text, has_exchange, has_implant):
    """
    Separate flag for planning/intention.
    Does NOT change HAS_STAGE2.

    Rule: planning/counsel/scheduled language + stage2 phrase OR exchange/implant concepts,
    excluding explicit negation.
    """
    if not text:
        return 0, ""

    if RE_NEG_PLAN.search(text):
        return 0, ""

    planning_cue = True if (RE_FUTURE_PLAN.search(text) or RE_COUNSEL.search(text)) else False
    stage2_cue = True if (RE_STAGE2_PHRASE.search(text) or has_exchange or has_implant) else False

    if note_type == "CLINIC" and planning_cue and stage2_cue:
        # label
        if RE_FUTURE_PLAN.search(text) and RE_STAGE2_PHRASE.search(text):
            return 1, "CLINIC_PLAN_STAGE2_PHRASE"
        if RE_FUTURE_PLAN.search(text) and has_exchange:
            return 1, "CLINIC_PLAN_EXCHANGE"
        if RE_FUTURE_PLAN.search(text) and has_implant:
            return 1, "CLINIC_PLAN_IMPLANT"
        if RE_COUNSEL.search(text) and RE_STAGE2_PHRASE.search(text):
            return 1, "CLINIC_COUNSEL_STAGE2_PHRASE"
        return 1, "CLINIC_PLANNING_OTHER"

    # allow OP note planning too, but keep separate from HAS_STAGE2
    if note_type == "OP" and planning_cue and stage2_cue:
        return 1, "OP_PLANNING"

    return 0, ""


def derive_bucket(note_type, is_future, has_exchange, has_implant, has_intraop):
    if is_future:
        return "FUTURE_PLAN"
    if note_type == "CLINIC" and (has_exchange or has_implant or has_intraop):
        return "CLINIC_ONLY"
    if has_exchange and has_implant and has_intraop:
        return "DEFINITIVE_OP"
    if note_type == "OP" and has_exchange and has_implant:
        return "PROBABLE_OP"
    if note_type == "OP" and has_exchange and has_intraop:
        return "POSSIBLE_OP"
    if note_type == "OP" and has_exchange:
        return "EXCHANGE_ONLY_OP"
    if note_type == "OP" and has_implant and has_intraop:
        return "IMPLANT_ONLY_OP"
    return "OTHER"


# -------------------------
# Main
# -------------------------

def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Collect inputs (op notes mandatory; clinic optional)
    inputs = []
    op_path = os.path.join(STAGING_DIR, "HPI11526 Operation Notes.csv")
    cl_path = os.path.join(STAGING_DIR, "HPI11526 Clinic Notes.csv")

    if os.path.isfile(op_path):
        inputs.append(op_path)
    else:
        raise IOError("Operation Notes CSV not found: {}".format(op_path))

    if os.path.isfile(cl_path):
        inputs.append(cl_path)

    # Load + concat
    dfs = []
    for p in inputs:
        d = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
        d["_SOURCE_FILE"] = os.path.basename(p)
        dfs.append(d)
    df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    enc_col = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID"])
    if not enc_col:
        raise ValueError("Missing ENCRYPTED_PAT_ID in inputs. Found: {}".format(list(df.columns)))
    if enc_col != "ENCRYPTED_PAT_ID":
        df = df.rename(columns={enc_col: "ENCRYPTED_PAT_ID"})
    df["ENCRYPTED_PAT_ID"] = df["ENCRYPTED_PAT_ID"].map(normalize_id)

    note_id_col = pick_first_existing(df, ["NOTE_ID", "NOTEID", "DOCUMENT_ID", "ID"])
    note_type_col = pick_first_existing(df, ["NOTE_TYPE", "DOC_TYPE", "DOCUMENT_TYPE", "TYPE"])
    date_col = pick_first_existing(df, ["NOTE_DATE", "DATE", "SERVICE_DATE", "DATE_OF_SERVICE", "VISIT_DATE"])

    source_col = "_SOURCE_FILE"

    text_cols = detect_text_columns(df)
    if not text_cols:
        raise ValueError("Could not detect any text/snippet columns. Found: {}".format(list(df.columns)))

    audit_rows = []
    planning_rows = []

    # Per-patient best evidence trackers
    best_strict = {}     # pid -> (score, rowdict)
    best_plan = {}       # pid -> (score, rowdict)

    # Iterate rows
    for _, r in df.iterrows():
        pid = normalize_id(r.get("ENCRYPTED_PAT_ID", ""))
        if not pid:
            continue

        note_id = safe_str(r.get(note_id_col, "")) if note_id_col else ""
        note_date = safe_str(r.get(date_col, "")) if date_col else ""
        note_type = infer_note_type(r, note_type_col, source_col)

        text = coalesce_text(r, text_cols)
        t = text  # already str
        has_exchange = True if RE_EXCHANGE.search(t) else False
        has_implant = True if RE_IMPLANT.search(t) else False
        has_intraop = True if RE_INTRAOP.search(t) else False
        is_future = True if RE_FUTURE_PLAN.search(t) else False

        bucket = derive_bucket(note_type, is_future, has_exchange, has_implant, has_intraop)
        score = bucket_score(bucket)
        has_stage2 = compute_has_stage2_strict(bucket)

        # candidate planning (separate)
        cand_plan, plan_pat = compute_candidate_planning(note_type, t, has_exchange, has_implant)

        # keep audit rows for anything relevant
        if (has_exchange or has_implant or has_intraop or is_future or cand_plan):
            audit_rows.append({
                "ENCRYPTED_PAT_ID": pid,
                "NOTE_ID": note_id,
                "NOTE_DATE": note_date,
                "NOTE_TYPE": note_type,
                "SOURCE_FILE": safe_str(r.get(source_col, "")),
                "BUCKET": bucket,
                "HAS_STAGE2_STRICT": int(has_stage2),
                "CANDIDATE_PLANNING": int(cand_plan),
                "PLANNING_PATTERN": plan_pat,
                "HAS_EXCHANGE": int(has_exchange),
                "HAS_IMPLANT": int(has_implant),
                "HAS_INTRAOP": int(has_intraop),
                "IS_FUTURE_PLAN": int(is_future),
                "TEXT_SNIPPET": re.sub(r"\s+", " ", t[:520]).strip()
            })

        # update best strict evidence
        if has_stage2 == 1:
            prev = best_strict.get(pid)
            cur = (score, note_date, note_id)
            if (prev is None) or (cur > (prev[0], prev[1], prev[2])):
                best_strict[pid] = (score, note_date, note_id, note_type, bucket)

        # update best planning evidence (separate)
        if cand_plan == 1:
            # planning score: prefer CLINIC plan-stage2 > other plan; then date/id
            pscore = 0
            if plan_pat in ["CLINIC_PLAN_STAGE2_PHRASE", "CLINIC_PLAN_EXCHANGE", "CLINIC_PLAN_IMPLANT"]:
                pscore = 20
            elif plan_pat.startswith("CLINIC_"):
                pscore = 10
            else:
                pscore = 5
            prevp = best_plan.get(pid)
            curp = (pscore, note_date, note_id)
            if (prevp is None) or (curp > (prevp[0], prevp[1], prevp[2])):
                best_plan[pid] = (pscore, note_date, note_id, note_type, plan_pat)

            planning_rows.append({
                "ENCRYPTED_PAT_ID": pid,
                "NOTE_ID": note_id,
                "NOTE_DATE": note_date,
                "NOTE_TYPE": note_type,
                "SOURCE_FILE": safe_str(r.get(source_col, "")),
                "PLANNING_PATTERN": plan_pat,
                "TEXT_SNIPPET": re.sub(r"\s+", " ", t[:520]).strip()
            })

    audit = pd.DataFrame(audit_rows)
    planning = pd.DataFrame(planning_rows)

    # Bucket counts
    if not audit.empty:
        bucket_counts = audit.groupby("BUCKET", as_index=False).size().rename(columns={"size": "COUNT"})
    else:
        bucket_counts = pd.DataFrame({"BUCKET": [], "COUNT": []})

    # Build patient summary with required columns + new planning columns
    all_pats = df["ENCRYPTED_PAT_ID"].map(normalize_id)
    all_pats = all_pats[all_pats != ""].drop_duplicates()

    out_rows = []
    for pid in all_pats:
        # strict
        if pid in best_strict:
            _, s_date, s_note_id, s_note_type, s_bucket = best_strict[pid]
            has_stage2 = 1
            out_rows.append({
                "ENCRYPTED_PAT_ID": pid,
                "HAS_STAGE2": 1,
                "STAGE2_DATE": safe_str(s_date),
                "STAGE2_NOTE_ID": safe_str(s_note_id),
                "STAGE2_NOTE_TYPE": safe_str(s_note_type),
                "STAGE2_MATCH_PATTERN": safe_str(s_bucket),
                "STAGE2_HITS": 1
            })
        else:
            out_rows.append({
                "ENCRYPTED_PAT_ID": pid,
                "HAS_STAGE2": 0,
                "STAGE2_DATE": "",
                "STAGE2_NOTE_ID": "",
                "STAGE2_NOTE_TYPE": "",
                "STAGE2_MATCH_PATTERN": "",
                "STAGE2_HITS": 0
            })

    out = pd.DataFrame(out_rows)

    # Add separate planning columns (does not affect validator)
    out["CANDIDATE_PLANNING"] = 0
    out["PLANNING_DATE"] = ""
    out["PLANNING_NOTE_ID"] = ""
    out["PLANNING_NOTE_TYPE"] = ""
    out["PLANNING_MATCH_PATTERN"] = ""

    for pid, tup in best_plan.items():
        _, p_date, p_note_id, p_note_type, p_pat = tup
        m = (out["ENCRYPTED_PAT_ID"] == pid)
        out.loc[m, "CANDIDATE_PLANNING"] = 1
        out.loc[m, "PLANNING_DATE"] = safe_str(p_date)
        out.loc[m, "PLANNING_NOTE_ID"] = safe_str(p_note_id)
        out.loc[m, "PLANNING_NOTE_TYPE"] = safe_str(p_note_type)
        out.loc[m, "PLANNING_MATCH_PATTERN"] = safe_str(p_pat)

    # Write outputs
    out.to_csv(OUT_PATIENT_SUMMARY, index=False)
    audit.to_csv(OUT_AUDIT_HITS, index=False)
    bucket_counts.to_csv(OUT_AUDIT_BUCKETS, index=False)
    planning.to_csv(OUT_PLANNING_HITS, index=False)

    # Console summary
    print("Staging complete.")
    print("Patients:", int(out["ENCRYPTED_PAT_ID"].nunique()))
    print("HAS_STAGE2=1 (strict):", int((out["HAS_STAGE2"] == 1).sum()))
    print("CANDIDATE_PLANNING=1:", int((out["CANDIDATE_PLANNING"] == 1).sum()))
    print("Wrote:")
    print(" ", OUT_PATIENT_SUMMARY)
    print(" ", OUT_AUDIT_HITS)
    print(" ", OUT_AUDIT_BUCKETS)
    print(" ", OUT_PLANNING_HITS)


if __name__ == "__main__":
    main()
