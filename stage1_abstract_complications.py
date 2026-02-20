# stage1_abstract_complications.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Stage 1 complication abstraction (0–365d after Stage1 date):
#     1) Row-level complication hits (category + inferred treatment + minor/major + snippet)
#     2) Patient-level S1_Comp1..S1_Comp3 fields
#
# Inputs:
#   - patient_recon_staging_refined.csv        (must contain patient_id + stage1_date)
#   - stage1_anchor_rows_with_bins.csv         (anchored rows with EVENT_DT + note text)
#
# Outputs:
#   - stage1_complications_row_hits.csv
#   - stage1_complications_patient_level.csv
#   - stage1_complications_summary.txt
#
# Notes:
#   - Read encoding: latin1(errors=replace)
#   - Write encoding: utf-8
#   - Filters to 0 <= delta <= 365 (Stage1 window)
#   - Suppressors reduce false positives:
#       * negation (no infection / denies / negative for)
#       * history/prior
#       * rule-out / possible / concern for
#       * "risk of complications" counseling language
#       * prophylaxis antibiotic mentions

from __future__ import print_function

import re
import sys
import pandas as pd

STAGING_CSV = "patient_recon_staging_refined.csv"
ANCHOR_ROWS_CSV = "stage1_anchor_rows_with_bins.csv"

OUT_ROW_HITS = "stage1_complications_row_hits.csv"
OUT_PATIENT_LEVEL = "stage1_complications_patient_level.csv"
OUT_SUMMARY = "stage1_complications_summary.txt"

CHUNKSIZE = 120000
COL_PATIENT = "patient_id"


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


def snippet(s, n=240):
    t = norm_text(s)
    return (t[:n] + "...") if len(t) > n else t


def norm_colname(c):
    return str(c).strip().lower().replace(" ", "_")


def pick_first_present(cols, candidates):
    cset = set([norm_colname(c) for c in cols])
    for cand in candidates:
        if norm_colname(cand) in cset:
            for c in cols:
                if norm_colname(c) == norm_colname(cand):
                    return c
    return None


def fuzzy_find_col(cols, tokens_any):
    ucols = [(c, str(c).upper()) for c in cols]
    for c, uc in ucols:
        for t in tokens_any:
            if t in uc:
                return c
    return None


# -------------------------
# Complication category patterns
# -------------------------
COMP_PATTERNS = [
    ("Hematoma", re.compile(r"\bhematoma\b", re.I)),
    ("Wound dehiscence", re.compile(r"\b(dehiscence|wound\s+dehisce|incision\s+dehisce|wound\s+separation)\b", re.I)),
    ("Wound infection", re.compile(r"\b(infection|infected|cellulitis|abscess|purulence|purulent|ssi|surgical\s+site\s+infection)\b", re.I)),
    ("Mastectomy skin flap necrosis", re.compile(r"\b(skin\s+flap\s+necrosis|mastectomy\s+flap\s+necrosis|msfn|flap\s+necrosis)\b", re.I)),
    ("Seroma", re.compile(r"\bseroma\b", re.I)),
    ("Capsular contracture", re.compile(r"\b(capsular\s+contracture)\b", re.I)),
    ("Implant malposition", re.compile(r"\b(implant\s+malposition|malposition)\b", re.I)),
    ("Implant rupture/leak/deflation", re.compile(r"\b(implant\s+(rupture|ruptured|leak|leakage|deflation|deflated))\b|\b(ruptured\s+implant|leaking\s+implant)\b", re.I)),
    ("Implant/expander extrusion", re.compile(r"\b(extrusion|exposed\s+(implant|expander)|implant\s+exposure|expander\s+exposure)\b", re.I)),
    ("Acute partial flap necrosis", re.compile(r"\b(partial\s+flap\s+necrosis)\b", re.I)),
    ("Total flap loss", re.compile(r"\b(total\s+flap\s+loss|flap\s+loss)\b", re.I)),
]

# Systemic: avoid "PE" alone (too many false positives)
OTHER_SYSTEMIC_RE = re.compile(
    r"\b(pulmonary\s+embolism|deep\s+vein\s+thrombosis|dvt\b|pneumonia|sepsis)\b",
    re.I
)

# -------------------------
# Suppressors
# -------------------------
RX_NEGATION = re.compile(
    r"\b(no|not|without|denies|negative\s+for|no\s+evidence\s+of|free\s+of)\b.{0,40}\b"
    r"(infection|cellulitis|abscess|purulence|seroma|hematoma|dehiscence|necrosis|pneumonia|sepsis|dvt|deep\s+vein\s+thrombosis|pulmonary\s+embolism)\b",
    re.I
)

RX_HISTORY_CONTEXT = re.compile(
    r"\b(history\s+of|hx\s+of|prior|past\s+history\s+of|previous|remote\s+history)\b.{0,60}\b"
    r"(infection|cellulitis|abscess|purulence|seroma|hematoma|dehiscence|necrosis|pneumonia|sepsis|dvt|deep\s+vein\s+thrombosis|pulmonary\s+embolism)\b",
    re.I
)

# FIXED: removed the invalid "?question" token
RX_RULEOUT = re.compile(
    r"\b(rule\s+out|r\/o|evaluate\s+for|work\s*up\s+for|concern\s+for|possible|question\s+of|suspicious\s+for)\b",
    re.I
)

RX_RISK_COUNSELING = re.compile(
    r"\b(risks?\s+(include|of)|discussed\s+risks?|counsel(ed|ing)\s+on\s+risks?|"
    r"risk\s+of\s+(infection|seroma|hematoma|necrosis|dehiscence))\b",
    re.I
)

RX_PROPHYLACTIC_ABX = re.compile(
    r"\b(prophylaxis|prophylactic|peri-?op|pre-?op|post-?op)\b.{0,40}\b(antibiotic|abx|ancef|cefazolin)\b",
    re.I
)

# -------------------------
# Treatment inference
# -------------------------
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
    if RX_REOP.search(t):
        return "REOPERATION"
    if RX_REHOSP.search(t):
        return "REHOSPITALIZATION"
    if RX_NONOP.search(t):
        return "NON-OPERATIVE"
    if RX_NOTX.search(t):
        return "NO TREATMENT"
    return "UNKNOWN"


def major_minor_from_treatment(bucket):
    if bucket in ["REOPERATION", "REHOSPITALIZATION"]:
        return "MAJOR"
    if bucket in ["NON-OPERATIVE", "NO TREATMENT"]:
        return "MINOR"
    return "UNKNOWN"


def should_suppress(txt_norm):
    if not txt_norm:
        return True
    if RX_RISK_COUNSELING.search(txt_norm):
        return True
    if RX_RULEOUT.search(txt_norm):
        return True
    if RX_NEGATION.search(txt_norm):
        return True
    if RX_HISTORY_CONTEXT.search(txt_norm):
        return True
    if RX_PROPHYLACTIC_ABX.search(txt_norm):
        return True
    return False


def main():
    # 1) Load staging file for Stage1 date
    st = read_csv_safe(STAGING_CSV)
    if st is None or st.empty:
        raise RuntimeError("Could not read staging file: {}".format(STAGING_CSV))

    if COL_PATIENT not in st.columns:
        raise RuntimeError("Staging file missing '{}': {}".format(COL_PATIENT, STAGING_CSV))

    stage1_col = pick_first_present(st.columns.tolist(), ["stage1_date", "stage1_dt", "stage1"])
    if stage1_col is None:
        stage1_col = fuzzy_find_col(st.columns.tolist(), ["STAGE1", "STAGE_1"])
    if stage1_col is None:
        raise RuntimeError("Could not detect Stage1 date column in staging file: {}".format(STAGING_CSV))

    st[COL_PATIENT] = st[COL_PATIENT].fillna("").astype(str)
    st["STAGE1_DT"] = to_dt(st[stage1_col])
    st = st[(st[COL_PATIENT] != "") & (st["STAGE1_DT"].notnull())].copy()

    if st.empty:
        raise RuntimeError("No patients with non-null Stage1 date after parsing.")

    stage1_map = dict(zip(st[COL_PATIENT].tolist(), st["STAGE1_DT"].tolist()))
    stage1_patients = set(stage1_map.keys())

    # 2) Detect columns in anchor rows file
    head = read_csv_safe(ANCHOR_ROWS_CSV, nrows=25)
    if head is None or head.empty:
        raise RuntimeError("Could not read anchor rows file: {}".format(ANCHOR_ROWS_CSV))

    if COL_PATIENT not in head.columns and "ENCRYPTED_PAT_ID" not in head.columns:
        raise RuntimeError("Anchor rows file missing patient_id: {}".format(ANCHOR_ROWS_CSV))

    event_col = pick_first_present(head.columns.tolist(), ["EVENT_DT", "event_dt", "note_date", "NOTE_DATE_OF_SERVICE"])
    if event_col is None:
        event_col = fuzzy_find_col(head.columns.tolist(), ["EVENT_DT", "DATE_OF_SERVICE", "NOTE_DATE", "DATE"])
    if event_col is None:
        raise RuntimeError("Could not detect event date column in anchor rows file: {}".format(ANCHOR_ROWS_CSV))

    text_col = pick_first_present(head.columns.tolist(), ["note_text", "NOTE_TEXT", "NOTE_TEXT_CLEAN", "SNIPPET", "NOTE_SNIPPET", "TEXT"])
    if text_col is None:
        text_col = fuzzy_find_col(head.columns.tolist(), ["NOTE_TEXT", "TEXT", "SNIPPET"])
    if text_col is None:
        raise RuntimeError("Could not detect note text column in anchor rows file: {}".format(ANCHOR_ROWS_CSV))

    note_type_col = pick_first_present(head.columns.tolist(), ["note_type", "NOTE_TYPE"])
    if note_type_col is None:
        note_type_col = fuzzy_find_col(head.columns.tolist(), ["NOTE_TYPE", "TYPE"])

    note_id_col = pick_first_present(head.columns.tolist(), ["note_id", "NOTE_ID"])
    if note_id_col is None:
        note_id_col = fuzzy_find_col(head.columns.tolist(), ["NOTE_ID", "ID"])

    # 3) Scan anchor rows and extract hits
    total_rows = 0
    rows_after_patient_filter = 0
    rows_after_window = 0
    unique_pat_anyhit = set()
    hits = []

    usecols = [event_col, text_col]
    if COL_PATIENT in head.columns:
        usecols = [COL_PATIENT] + usecols
        pid_is_encrypted = False
    else:
        usecols = ["ENCRYPTED_PAT_ID"] + usecols
        pid_is_encrypted = True

    for c in [note_type_col, note_id_col]:
        if c is not None and c not in usecols:
            usecols.append(c)

    for chunk in iter_csv_safe(ANCHOR_ROWS_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        if pid_is_encrypted:
            chunk = chunk.rename(columns={"ENCRYPTED_PAT_ID": COL_PATIENT})

        chunk[COL_PATIENT] = chunk[COL_PATIENT].fillna("").astype(str)
        chunk = chunk[chunk[COL_PATIENT].isin(stage1_patients)].copy()
        if chunk.empty:
            continue
        rows_after_patient_filter += len(chunk)

        chunk["EVENT_DT"] = to_dt(chunk[event_col])
        chunk["STAGE1_DT"] = chunk[COL_PATIENT].map(stage1_map)

        chunk["DELTA_DAYS_FROM_STAGE1"] = (chunk["EVENT_DT"] - chunk["STAGE1_DT"]).dt.days
        chunk = chunk[chunk["DELTA_DAYS_FROM_STAGE1"].notnull()].copy()
        chunk = chunk[(chunk["DELTA_DAYS_FROM_STAGE1"] >= 0) & (chunk["DELTA_DAYS_FROM_STAGE1"] <= 365)].copy()
        if chunk.empty:
            continue
        rows_after_window += len(chunk)

        texts = chunk[text_col].fillna("").tolist()
        pids = chunk[COL_PATIENT].tolist()
        evs = chunk["EVENT_DT"].tolist()
        deltas = chunk["DELTA_DAYS_FROM_STAGE1"].tolist()

        ntypes = chunk[note_type_col].fillna("").tolist() if note_type_col else [""] * len(chunk)
        nids = chunk[note_id_col].fillna("").tolist() if note_id_col else [""] * len(chunk)

        for i in range(len(chunk)):
            pid = pids[i]
            txt_raw = texts[i]
            txt = norm_text(txt_raw)

            if should_suppress(txt):
                continue

            found_any = False

            for comp_name, comp_re in COMP_PATTERNS:
                if comp_re.search(txt):
                    found_any = True
                    bucket = infer_treatment_bucket(txt)
                    mm = major_minor_from_treatment(bucket)
                    hits.append({
                        "patient_id": pid,
                        "EVENT_DT": evs[i],
                        "STAGE1_DT": stage1_map.get(pid, None),
                        "DELTA_DAYS_FROM_STAGE1": deltas[i],
                        "NOTE_TYPE": ntypes[i],
                        "NOTE_ID": nids[i],
                        "complication": comp_name,
                        "treatment_bucket": bucket,
                        "comp_classification": mm,
                        "snippet": snippet(txt_raw, 240),
                    })

            if OTHER_SYSTEMIC_RE.search(txt):
                found_any = True
                bucket = infer_treatment_bucket(txt)
                mm = major_minor_from_treatment(bucket)
                hits.append({
                    "patient_id": pid,
                    "EVENT_DT": evs[i],
                    "STAGE1_DT": stage1_map.get(pid, None),
                    "DELTA_DAYS_FROM_STAGE1": deltas[i],
                    "NOTE_TYPE": ntypes[i],
                    "NOTE_ID": nids[i],
                    "complication": "Other (systemic)",
                    "treatment_bucket": bucket,
                    "comp_classification": mm,
                    "snippet": snippet(txt_raw, 240),
                })

            if found_any:
                unique_pat_anyhit.add(pid)

    # 4) Write row hits
    if hits:
        hits_df = pd.DataFrame(hits)
        hits_df = hits_df.sort_values(by=["patient_id", "EVENT_DT", "complication"], ascending=[True, True, True])
    else:
        hits_df = pd.DataFrame(columns=[
            "patient_id", "EVENT_DT", "STAGE1_DT", "DELTA_DAYS_FROM_STAGE1",
            "NOTE_TYPE", "NOTE_ID", "complication", "treatment_bucket", "comp_classification", "snippet"
        ])

    hits_df.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    # 5) Patient-level Comp1..3
    pl = pd.DataFrame({"patient_id": list(stage1_patients)})
    pl["stage1_dt"] = pl["patient_id"].map(stage1_map)

    patient_rows = []
    if not hits_df.empty:
        for pid, g in hits_df.groupby("patient_id"):
            g2 = g.sort_values(by=["EVENT_DT", "complication"], ascending=[True, True]).copy()
            g2["dedup_key"] = g2["EVENT_DT"].astype(str) + "||" + g2["complication"].astype(str)
            g2 = g2.drop_duplicates(subset=["dedup_key"], keep="first")

            comps = g2.head(3).to_dict("records")

            row = {"patient_id": pid, "stage1_dt": stage1_map.get(pid, None)}
            for i in range(3):
                idx = i + 1
                if i < len(comps):
                    c = comps[i]
                    row["S1_Comp{}_Date".format(idx)] = c.get("EVENT_DT", None)
                    row["S1_Comp{}".format(idx)] = c.get("complication", "")
                    row["S1_Comp{}_Treatment".format(idx)] = c.get("treatment_bucket", "")
                    row["S1_Comp{}_Classification".format(idx)] = c.get("comp_classification", "")
                    row["S1_Comp{}_NoteType".format(idx)] = c.get("NOTE_TYPE", "")
                    row["S1_Comp{}_NoteID".format(idx)] = c.get("NOTE_ID", "")
                else:
                    row["S1_Comp{}_Date".format(idx)] = None
                    row["S1_Comp{}".format(idx)] = ""
                    row["S1_Comp{}_Treatment".format(idx)] = ""
                    row["S1_Comp{}_Classification".format(idx)] = ""
                    row["S1_Comp{}_NoteType".format(idx)] = ""
                    row["S1_Comp{}_NoteID".format(idx)] = ""
            patient_rows.append(row)

    if patient_rows:
        pl_hits = pd.DataFrame(patient_rows)
        pl = pl.merge(pl_hits, on=["patient_id", "stage1_dt"], how="left")
    else:
        for idx in [1, 2, 3]:
            pl["S1_Comp{}_Date".format(idx)] = None
            pl["S1_Comp{}".format(idx)] = ""
            pl["S1_Comp{}_Treatment".format(idx)] = ""
            pl["S1_Comp{}_Classification".format(idx)] = ""
            pl["S1_Comp{}_NoteType".format(idx)] = ""
            pl["S1_Comp{}_NoteID".format(idx)] = ""

    pl = pl.sort_values(by=["patient_id"], ascending=[True])
    pl.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # 6) Summary
    n_stage1 = int(pl["patient_id"].nunique())
    n_pat_anyhit = len(unique_pat_anyhit)
    n_rows_hits = 0 if hits_df.empty else int(len(hits_df))

    comp_counts = hits_df["complication"].value_counts() if not hits_df.empty else pd.Series(dtype=int)
    treat_counts = hits_df["treatment_bucket"].value_counts() if not hits_df.empty else pd.Series(dtype=int)

    lines = []
    lines.append("=== Stage 1 Complication Abstraction (anchored rows -> S1_Comp1..3) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")
    lines.append("Inputs:")
    lines.append("  Staging file: {}".format(STAGING_CSV))
    lines.append("  Anchor rows file: {}".format(ANCHOR_ROWS_CSV))
    lines.append("")
    lines.append("Detected columns:")
    lines.append("  patient_id: {}".format(COL_PATIENT))
    lines.append("  Stage1 date col: {}".format(stage1_col))
    lines.append("  event date col: {}".format(event_col))
    lines.append("  text col: {}".format(text_col))
    if note_type_col:
        lines.append("  NOTE_TYPE: {}".format(note_type_col))
    if note_id_col:
        lines.append("  NOTE_ID: {}".format(note_id_col))
    lines.append("")
    lines.append("Row counts:")
    lines.append("  Total rows read: {}".format(total_rows))
    lines.append("  Rows after patient filter: {}".format(rows_after_patient_filter))
    lines.append("  Rows kept in 0–365d window: {}".format(rows_after_window))
    lines.append("")
    lines.append("Row-level complication hits: {}".format(n_rows_hits))
    lines.append("Patients with >=1 hit: {} ({:.1f}%)".format(
        n_pat_anyhit, (100.0 * n_pat_anyhit / n_stage1) if n_stage1 else 0.0
    ))
    lines.append("")

    if not hits_df.empty:
        lines.append("Top complication categories (top 15):")
        for k, v in comp_counts.head(15).items():
            lines.append("  {:>6}  {}".format(int(v), k))
        lines.append("")
        lines.append("Treatment buckets:")
        for k, v in treat_counts.items():
            lines.append("  {:>6}  {}".format(int(v), k))

    lines.append("")
    lines.append("Wrote:")
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
