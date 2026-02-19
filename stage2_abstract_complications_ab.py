# stage2_abstract_complications_ab.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   From Stage2-confirmed AB patients + Stage2-anchored rows, produce:
#     1) Row-level complication hits (category + inferred treatment + minor/major)
#     2) Patient-level S2_Comp1..S2_Comp3 fields aligned to Breast RESTORE dictionary
#
# Inputs (expected in current working dir unless paths edited):
#   - stage2_final_ab_patient_level.csv
#   - stage2_anchor_rows_with_bins.csv
#
# Outputs:
#   - stage2_ab_complications_row_hits.csv
#   - stage2_ab_complications_patient_level.csv
#   - stage2_ab_complications_summary.txt
#
# Anchoring:
#   - Requires EVENT_DT >= Stage2 date (same-day included)
#   - No upper time limit
#
# Notes:
#   - Reads CSVs using latin1(errors=replace) for robustness on WVD.
#   - Auto-detects likely date/text columns in the anchor rows file.

from __future__ import print_function

import re
import sys
import pandas as pd


# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
STAGE2_AB_PATIENTS_CSV = "stage2_final_ab_patient_level.csv"
ANCHOR_ROWS_CSV = "stage2_anchor_rows_with_bins.csv"

OUT_ROW_HITS = "stage2_ab_complications_row_hits.csv"
OUT_PATIENT_LEVEL = "stage2_ab_complications_patient_level.csv"
OUT_SUMMARY = "stage2_ab_complications_summary.txt"

CHUNKSIZE = 150000  # anchor rows can be large

# Required/expected id column name
COL_PATIENT = "patient_id"

# -------------------------
# Robust CSV reading (Python 3.6 safe)
# -------------------------
def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
    finally:
        if "chunksize" not in kwargs:
            try:
                f.close()
            except Exception:
                pass

def iter_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        for chunk in pd.read_csv(f, engine="python", **kwargs):
            yield chunk
    finally:
        try:
            f.close()
        except Exception:
            pass

def to_dt(x):
    return pd.to_datetime(x, errors="coerce")

def norm_text(x):
    if x is None:
        return ""
    s = str(x)
    try:
        s = s.replace(u"\xa0", " ")
    except Exception:
        pass
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def snippet(s, n=260):
    t = norm_text(s)
    return (t[:n] + "...") if len(t) > n else t

def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# -------------------------
# Complication category patterns (dictionary-aligned)
# -------------------------
# Note: These are intentionally explicit. You can expand later.
COMP_PATTERNS = [
    ("Hematoma", re.compile(r"\bhematoma\b", re.I)),
    ("Wound dehiscence", re.compile(r"\b(dehiscence|wound\s+dehisce|incision\s+dehisce|wound\s+separation)\b", re.I)),
    ("Wound infection", re.compile(r"\b(infection|infected|cellulitis|abscess|purulence|purulent|ssi|surgical\s+site\s+infection)\b", re.I)),
    ("Mastectomy skin flap necrosis", re.compile(r"\b(skin\s+flap\s+necrosis|mastectomy\s+flap\s+necrosis|msfn|flap\s+necrosis)\b", re.I)),
    ("Seroma", re.compile(r"\bseroma\b", re.I)),

    # Implant complications (dictionary subtypes)
    ("Capsular contracture", re.compile(r"\b(capsular\s+contracture)\b", re.I)),
    ("Implant malposition", re.compile(r"\b(implant\s+malposition|malposition)\b", re.I)),
    ("Implant rupture/leak/deflation", re.compile(r"\b(implant\s+(rupture|ruptured|leak|leakage|deflation|deflated))\b|\b(ruptured\s+implant|leaking\s+implant)\b", re.I)),
    ("Implant/expander extrusion", re.compile(r"\b(extrusion|exposed\s+(implant|expander)|implant\s+exposure|expander\s+exposure)\b", re.I)),

    # Flap complications (if present; even though AB is expander->implant, keep for safety)
    ("Acute partial flap necrosis", re.compile(r"\b(partial\s+flap\s+necrosis)\b", re.I)),
    ("Total flap loss", re.compile(r"\b(total\s+flap\s+loss|flap\s+loss)\b", re.I)),
]

# Optional: capture "Other" systemic complications only if explicitly named (very conservative)
OTHER_SYSTEMIC_RE = re.compile(r"\b(pulmonary\s+embolism|pe\b|deep\s+vein\s+thrombosis|dvt\b|pneumonia|sepsis)\b", re.I)

# -------------------------
# Treatment inference (dictionary-aligned buckets)
# -------------------------
# These are heuristic and meant for QA; they can be refined later.
RX_REOP = re.compile(
    r"\b(return(ed)?\s+to\s+or|take\s*back|takeback|re-?operation|reop|washout|operative\s+debridement|"
    r"incision\s+and\s+drainage|i\s*&\s*d|explant|explanted|implant\s+removal|remove(d)?\s+implant|"
    r"expander\s+removal|remove(d)?\s+expander|capsulectomy|capsulotomy)\b",
    re.I
)

RX_REHOSP = re.compile(
    r"\b(readmit(ted)?|re-?admit(ted)?|hospitaliz(ed|ation)|inpatient|admitted|admission|"
    r"presented\s+to\s+ed|emergency\s+department)\b",
    re.I
)

RX_NONOP = re.compile(
    r"\b(oral\s+antibiotic|iv\s+antibiotic|antibiotic(s)?|augmentin|keflex|clindamycin|vancomycin|"
    r"drainage|aspiration|tap(ped)?|percutaneous|ir\s+drain|wound\s+care|dressing\s+changes|packing|"
    r"topical|clinic\s+follow-?up)\b",
    re.I
)

RX_NOTX = re.compile(
    r"\b(no\s+treatment|no\s+intervention|observe|observation|monitor|watchful\s+waiting)\b",
    re.I
)

def infer_treatment_bucket(text):
    t = norm_text(text)
    # Priority: REOP > REHOSP > NON-OP > NO TREATMENT
    if RX_REOP.search(t):
        return "REOPERATION"
    if RX_REHOSP.search(t):
        return "REHOSPITALIZATION"
    if RX_NONOP.search(t):
        return "NON-OPERATIVE"
    if RX_NOTX.search(t):
        return "NO TREATMENT"
    # Unknown -> treat as NON-OPERATIVE? No. Keep as UNKNOWN for QA.
    return "UNKNOWN"

def major_minor_from_treatment(bucket):
    if bucket in ["REOPERATION", "REHOSPITALIZATION"]:
        return "MAJOR"
    if bucket in ["NON-OPERATIVE", "NO TREATMENT"]:
        return "MINOR"
    return "UNKNOWN"

# -------------------------
# Column auto-detection
# -------------------------
def detect_stage2_date_col(cols):
    # prefer the final column you created earlier
    candidates = [
        "stage2_date_final",
        "stage2_dt_final",
        "stage2_event_dt_best",
        "stage2_dt_best",
        "stage2_date_best",
        "stage2_date",
        "stage2_dt",
    ]
    return first_existing(cols, candidates)

def detect_event_dt_col(cols):
    candidates = ["EVENT_DT", "event_dt", "NOTE_DATE_OF_SERVICE", "note_dt", "OPERATION_DATE", "op_dt"]
    return first_existing(cols, candidates)

def detect_text_col(cols):
    candidates = ["NOTE_TEXT", "NOTE_TEXT_CLEAN", "note_text", "note_text_norm", "TEXT", "snippet"]
    return first_existing(cols, candidates)

def detect_note_type_col(cols):
    candidates = ["NOTE_TYPE", "note_type", "best_note_type"]
    return first_existing(cols, candidates)

def detect_note_id_col(cols):
    candidates = ["NOTE_ID", "note_id", "best_note_id"]
    return first_existing(cols, candidates)

def detect_file_tag_col(cols):
    candidates = ["file_tag", "FILE_TAG", "source_file", "SOURCE"]
    return first_existing(cols, candidates)

def detect_delta_col(cols):
    candidates = ["DELTA_DAYS_FROM_STAGE2", "delta_days_from_stage2", "stage2_delta_days", "stage2_delta_days_from_stage1"]
    return first_existing(cols, candidates)

# -------------------------
# Main
# -------------------------
def main():
    # 1) Load AB Stage2 patients
    ab = read_csv_safe(STAGE2_AB_PATIENTS_CSV)
    if COL_PATIENT not in ab.columns:
        raise RuntimeError("Missing '{}' in {}".format(COL_PATIENT, STAGE2_AB_PATIENTS_CSV))

    stage2_col = detect_stage2_date_col(ab.columns)
    if stage2_col is None:
        raise RuntimeError("No Stage2 date column found in {}. Expected one of stage2_date_final/stage2_event_dt_best/etc.".format(
            STAGE2_AB_PATIENTS_CSV
        ))

    ab[COL_PATIENT] = ab[COL_PATIENT].fillna("").astype(str)
    ab["STAGE2_DT"] = to_dt(ab[stage2_col])
    ab = ab[ab["STAGE2_DT"].notnull()].copy()

    ab_ids = set(ab[COL_PATIENT].tolist())
    stage2_map = dict(zip(ab[COL_PATIENT], ab["STAGE2_DT"]))

    if not ab_ids:
        raise RuntimeError("No AB patients with a Stage2 date found in {}".format(STAGE2_AB_PATIENTS_CSV))

    # 2) Peek anchor rows header
    head = read_csv_safe(ANCHOR_ROWS_CSV, nrows=10)
    if COL_PATIENT not in head.columns:
        raise RuntimeError("Missing '{}' in {}".format(COL_PATIENT, ANCHOR_ROWS_CSV))

    event_col = detect_event_dt_col(head.columns)
    text_col = detect_text_col(head.columns)

    if event_col is None:
        raise RuntimeError("Could not detect an event date column in {} (need EVENT_DT or similar).".format(ANCHOR_ROWS_CSV))
    if text_col is None:
        raise RuntimeError("Could not detect a note text column in {} (need NOTE_TEXT/NOTE_TEXT_CLEAN/snippet).".format(ANCHOR_ROWS_CSV))

    note_type_col = detect_note_type_col(head.columns)
    note_id_col = detect_note_id_col(head.columns)
    file_tag_col = detect_file_tag_col(head.columns)
    delta_col = detect_delta_col(head.columns)

    # Only load what we need
    usecols = [COL_PATIENT, event_col, text_col]
    for c in [note_type_col, note_id_col, file_tag_col, delta_col]:
        if c is not None and c not in usecols:
            usecols.append(c)

    # 3) Scan anchor rows and extract complication hits
    hits = []
    total_rows = 0
    rows_after_prefilter = 0
    rows_after_anchor = 0
    unique_pat_anyhit = set()

    for chunk in iter_csv_safe(ANCHOR_ROWS_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PATIENT].isin(ab_ids)].copy()
        if chunk.empty:
            continue
        rows_after_prefilter += len(chunk)

        chunk["EVENT_DT"] = to_dt(chunk[event_col])
        chunk["STAGE2_DT"] = chunk[COL_PATIENT].map(stage2_map)

        # anchor: same-day included (>= 0)
        chunk["DELTA_DAYS_FROM_STAGE2"] = (chunk["EVENT_DT"] - chunk["STAGE2_DT"]).dt.days
        chunk = chunk[chunk["DELTA_DAYS_FROM_STAGE2"].notnull() & (chunk["DELTA_DAYS_FROM_STAGE2"] >= 0)].copy()
        if chunk.empty:
            continue
        rows_after_anchor += len(chunk)

        # normalize text once
        chunk["_TEXT_NORM"] = chunk[text_col].fillna("").apply(norm_text)

        # iterate rows and emit per-comp hits
        for _, r in chunk.iterrows():
            pid = r.get(COL_PATIENT, "")
            txt = r.get("_TEXT_NORM", "")
            if not pid or not txt:
                continue

            found_any = False
            for comp_name, comp_re in COMP_PATTERNS:
                if comp_re.search(txt):
                    found_any = True
                    bucket = infer_treatment_bucket(txt)
                    mm = major_minor_from_treatment(bucket)

                    hits.append({
                        "patient_id": pid,
                        "EVENT_DT": r.get("EVENT_DT", None),
                        "STAGE2_DT": r.get("STAGE2_DT", None),
                        "DELTA_DAYS_FROM_STAGE2": r.get("DELTA_DAYS_FROM_STAGE2", None),
                        "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                        "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                        "file_tag": r.get(file_tag_col, "") if file_tag_col else "",
                        "complication": comp_name,
                        "treatment_bucket": bucket,
                        "comp_classification": mm,
                        "snippet": snippet(r.get(text_col, ""), 260),
                    })

            # conservative "Other" systemic capture
            if OTHER_SYSTEMIC_RE.search(txt):
                found_any = True
                bucket = infer_treatment_bucket(txt)
                mm = major_minor_from_treatment(bucket)
                hits.append({
                    "patient_id": pid,
                    "EVENT_DT": r.get("EVENT_DT", None),
                    "STAGE2_DT": r.get("STAGE2_DT", None),
                    "DELTA_DAYS_FROM_STAGE2": r.get("DELTA_DAYS_FROM_STAGE2", None),
                    "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                    "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                    "file_tag": r.get(file_tag_col, "") if file_tag_col else "",
                    "complication": "Other (systemic)",
                    "treatment_bucket": bucket,
                    "comp_classification": mm,
                    "snippet": snippet(r.get(text_col, ""), 260),
                })

            if found_any:
                unique_pat_anyhit.add(pid)

    # 4) Write row hits
    if hits:
        hits_df = pd.DataFrame(hits)
        hits_df = hits_df.sort_values(by=["patient_id", "EVENT_DT", "complication"], ascending=[True, True, True])
    else:
        hits_df = pd.DataFrame(columns=[
            "patient_id","EVENT_DT","STAGE2_DT","DELTA_DAYS_FROM_STAGE2",
            "NOTE_TYPE","NOTE_ID","file_tag","complication","treatment_bucket","comp_classification","snippet"
        ])
    hits_df.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    # 5) Build patient-level Comp1..3 (chronological)
    #    Note: dictionary wants up to 3 complications; if more, you escalate manually.
    patient_rows = []
    if not hits_df.empty:
        for pid, g in hits_df.groupby("patient_id"):
            g2 = g.sort_values(by=["EVENT_DT", "complication"], ascending=[True, True]).copy()
            # collapse duplicates at same date+complication (keep first)
            g2["dedup_key"] = g2["EVENT_DT"].astype(str) + "||" + g2["complication"].astype(str)
            g2 = g2.drop_duplicates(subset=["dedup_key"], keep="first")

            comps = g2.head(3).to_dict("records")

            row = {
                "patient_id": pid,
                "stage2_dt": stage2_map.get(pid, None),
            }

            # Fill S2_Comp1..3 fields
            for i in range(3):
                idx = i + 1
                if i < len(comps):
                    c = comps[i]
                    row["S2_Comp{}_Date".format(idx)] = c.get("EVENT_DT", None)
                    row["S2_Comp{}".format(idx)] = c.get("complication", "")
                    row["S2_Comp{}_Treatment".format(idx)] = c.get("treatment_bucket", "")
                    row["S2_Comp{}_Classification".format(idx)] = c.get("comp_classification", "")
                    row["S2_Comp{}_NoteType".format(idx)] = c.get("NOTE_TYPE", "")
                    row["S2_Comp{}_NoteID".format(idx)] = c.get("NOTE_ID", "")
                    row["S2_Comp{}_FileTag".format(idx)] = c.get("file_tag", "")
                else:
                    row["S2_Comp{}_Date".format(idx)] = None
                    row["S2_Comp{}".format(idx)] = ""
                    row["S2_Comp{}_Treatment".format(idx)] = ""
                    row["S2_Comp{}_Classification".format(idx)] = ""
                    row["S2_Comp{}_NoteType".format(idx)] = ""
                    row["S2_Comp{}_NoteID".format(idx)] = ""
                    row["S2_Comp{}_FileTag".format(idx)] = ""

            patient_rows.append(row)

    # Include AB patients even if no complications detected
    pl = pd.DataFrame({"patient_id": list(ab_ids)})
    pl["stage2_dt"] = pl["patient_id"].map(stage2_map)

    if patient_rows:
        pl_hits = pd.DataFrame(patient_rows)
        pl = pl.merge(pl_hits, on=["patient_id", "stage2_dt"], how="left")
    else:
        # create empty columns for consistency
        for idx in [1,2,3]:
            pl["S2_Comp{}_Date".format(idx)] = None
            pl["S2_Comp{}".format(idx)] = ""
            pl["S2_Comp{}_Treatment".format(idx)] = ""
            pl["S2_Comp{}_Classification".format(idx)] = ""
            pl["S2_Comp{}_NoteType".format(idx)] = ""
            pl["S2_Comp{}_NoteID".format(idx)] = ""
            pl["S2_Comp{}_FileTag".format(idx)] = ""

    pl = pl.sort_values(by=["patient_id"], ascending=[True])
    pl.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # 6) Summary
    n_ab = len(ab_ids)
    n_pat_anyhit = len(unique_pat_anyhit)
    n_rows_hits = 0 if hits_df.empty else len(hits_df)

    # counts by complication and by treatment
    comp_counts = hits_df["complication"].value_counts() if not hits_df.empty else pd.Series([])
    treat_counts = hits_df["treatment_bucket"].value_counts() if not hits_df.empty else pd.Series([])

    lines = []
    lines.append("=== Stage2 AB Complication Abstraction (anchored rows -> S2_Comp1..3) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")
    lines.append("Inputs:")
    lines.append("  - {}".format(STAGE2_AB_PATIENTS_CSV))
    lines.append("    Stage2 date column used: {}".format(stage2_col))
    lines.append("  - {}".format(ANCHOR_ROWS_CSV))
    lines.append("    Event date column used: {}".format(event_col))
    lines.append("    Text column used: {}".format(text_col))
    lines.append("")
    lines.append("AB Stage2 patients (with Stage2 date): {}".format(n_ab))
    lines.append("Anchor rows scanned (all): {}".format(total_rows))
    lines.append("Anchor rows after AB prefilter: {}".format(rows_after_prefilter))
    lines.append("Anchor rows after Stage2 anchor (EVENT_DT >= Stage2): {}".format(rows_after_anchor))
    lines.append("")
    lines.append("Row-level complication hits written: {}".format(n_rows_hits))
    lines.append("Patients with >=1 detected complication hit: {} ({:.1f}%)".format(
        n_pat_anyhit, (100.0 * n_pat_anyhit / n_ab) if n_ab else 0.0
    ))

    if not hits_df.empty:
        lines.append("")
        lines.append("Top complication categories (top 15):")
        for k, v in comp_counts.head(15).items():
            lines.append("  {:>6}  {}".format(int(v), k))

        lines.append("")
        lines.append("Treatment buckets:")
        for k, v in treat_counts.items():
            lines.append("  {:>6}  {}".format(int(v), k))

    lines.append("")
    lines.append("Outputs:")
    lines.append("  - {}".format(OUT_ROW_HITS))
    lines.append("  - {}".format(OUT_PATIENT_LEVEL))
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
