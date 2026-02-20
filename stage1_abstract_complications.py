# stage1_abstract_complications.py
# Python 3.6.8+
#
# Input:
#   stage1_anchor_rows_with_bins.csv   (Stage1 anchored rows, 0-365d, already binned)
#
# Output:
#   stage1_complications_row_hits.csv
#   stage1_complications_patient_level.csv
#   stage1_complications_summary.txt

from __future__ import print_function

import re
import sys
import pandas as pd

INFILE = "stage1_anchor_rows_with_bins.csv"

OUT_ROW_HITS = "stage1_complications_row_hits.csv"
OUT_PATIENT_LEVEL = "stage1_complications_patient_level.csv"
OUT_SUMMARY = "stage1_complications_summary.txt"

COL_PATIENT = "patient_id"


def read_csv_safe(path, **kwargs):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python", **kwargs)
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


def norm_colname(c):
    return str(c).strip().lower().replace(" ", "_")


def pick_first_present(cols, candidates):
    cset = set([str(c).lower() for c in cols])
    for cand in candidates:
        if cand.lower() in cset:
            for c in cols:
                if str(c).lower() == cand.lower():
                    return c
    return None


def fuzzy_find_col(cols, tokens_any):
    up = [(c, str(c).upper()) for c in cols]
    for c, uc in up:
        for tok in tokens_any:
            if tok in uc:
                return c
    return None


# -------------------------
# Complication patterns
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

# IMPORTANT FIX: remove ambiguous "PE". Keep only explicit "pulmonary embolism".
OTHER_SYSTEMIC_RE = re.compile(
    r"\b(pulmonary\s+embolism|deep\s+vein\s+thrombosis|dvt\b|pneumonia|sepsis)\b",
    re.I
)

# -------------------------
# Stage1 false-positive suppressors
# -------------------------

# Suppress instructions / counseling / risk talk (these were showing up in your snippets)
RX_RISK_CONTEXT = re.compile(
    r"\b("
    r"signs?\s+of\s+infection|"
    r"watch\s+for|return\s+precautions|"
    r"call\s+(?:us|clinic|office|provider)|"
    r"risks?\s+include|risk\s+of|possible\s+complications?|may\s+include|"
    r"discussed\s+risks?|counseled\s+on|informed\s+consent|"
    r"post-?op\s+instructions?|"
    r"seek\s+care\s+if|"
    r"redness,\s*warmth,\s*fever|"
    r"chills|foul\s+odor"
    r")\b",
    re.I
)

# Negated infection (no evidence of infection, denies infection, etc.)
RX_INF_NEG = re.compile(
    r"\b(no|not|without|denies|negative\s+for|no\s+evidence\s+of|free\s+of)\b.{0,35}\b"
    r"(infection|cellulitis|abscess|purulence|purulent|ssi|surgical\s+site\s+infection)\b",
    re.I
)

# -------------------------
# Treatment inference (same buckets as Stage2)
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
    t = text
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


def main():
    df = read_csv_safe(INFILE)
    if df is None or df.empty:
        raise RuntimeError("Could not read input: {}".format(INFILE))

    if COL_PATIENT not in df.columns:
        # allow ENCRYPTED_PAT_ID fallback
        if "ENCRYPTED_PAT_ID" in df.columns:
            df = df.rename(columns={"ENCRYPTED_PAT_ID": COL_PATIENT})
        else:
            raise RuntimeError("Missing patient_id in {}".format(INFILE))

    # detect columns
    event_col = pick_first_present(df.columns.tolist(), ["EVENT_DT", "event_dt", "note_date", "NOTE_DATE_OF_SERVICE"])
    if event_col is None:
        event_col = fuzzy_find_col(df.columns.tolist(), ["EVENT_DT", "DATE_OF_SERVICE", "NOTE_DATE", "DATE"])

    text_col = pick_first_present(df.columns.tolist(), ["note_text", "NOTE_TEXT", "NOTE_SNIPPET", "snippet", "TEXT"])
    if text_col is None:
        text_col = fuzzy_find_col(df.columns.tolist(), ["NOTE_TEXT", "TEXT", "SNIPPET"])

    stage1_col = pick_first_present(df.columns.tolist(), ["stage1_dt", "STAGE1_DT", "stage1_date", "STAGE1_DATE"])
    if stage1_col is None:
        stage1_col = fuzzy_find_col(df.columns.tolist(), ["STAGE1", "STAGE_1", "STAGE1_DT", "STAGE1_DATE"])

    note_type_col = pick_first_present(df.columns.tolist(), ["NOTE_TYPE", "note_type"])
    note_id_col = pick_first_present(df.columns.tolist(), ["NOTE_ID", "note_id"])
    delta_col = pick_first_present(df.columns.tolist(), ["DELTA_DAYS_FROM_STAGE1", "delta_days_from_stage1"])

    if event_col is None:
        raise RuntimeError("Could not find event date column (EVENT_DT-like) in {}".format(INFILE))
    if text_col is None:
        raise RuntimeError("Could not find note text column (NOTE_TEXT/NOTE_SNIPPET/etc) in {}".format(INFILE))
    if stage1_col is None:
        raise RuntimeError("Could not find stage1 date column (stage1_dt-like) in {}".format(INFILE))

    # normalize types
    df[COL_PATIENT] = df[COL_PATIENT].fillna("").astype(str)
    df["EVENT_DT_STD"] = to_dt(df[event_col])
    df["STAGE1_DT_STD"] = to_dt(df[stage1_col])

    if delta_col is None:
        df["DELTA_DAYS_FROM_STAGE1"] = (df["EVENT_DT_STD"] - df["STAGE1_DT_STD"]).dt.days
    else:
        df["DELTA_DAYS_FROM_STAGE1"] = pd.to_numeric(df[delta_col], errors="coerce")

    # enforce anchor window: 0-365
    df = df[df["DELTA_DAYS_FROM_STAGE1"].notnull()].copy()
    df = df[(df["DELTA_DAYS_FROM_STAGE1"] >= 0) & (df["DELTA_DAYS_FROM_STAGE1"] <= 365)].copy()

    # scan rows
    hits = []
    unique_pat_anyhit = set()

    for _, r in df.iterrows():
        pid = r.get(COL_PATIENT, "")
        if not pid:
            continue

        txt_raw = r.get(text_col, "")
        txt = norm_text(txt_raw)
        if not txt:
            continue

        # suppress risk/counseling/instruction text blocks
        if RX_RISK_CONTEXT.search(txt):
            continue

        found_any = False

        for comp_name, comp_re in COMP_PATTERNS:
            if not comp_re.search(txt):
                continue

            # infection negation suppressor
            if comp_name == "Wound infection" and RX_INF_NEG.search(txt):
                continue

            found_any = True
            bucket = infer_treatment_bucket(txt)
            mm = major_minor_from_treatment(bucket)

            hits.append({
                "patient_id": pid,
                "EVENT_DT": r.get("EVENT_DT_STD", None),
                "stage1_dt": r.get("STAGE1_DT_STD", None),
                "DELTA_DAYS_FROM_STAGE1": r.get("DELTA_DAYS_FROM_STAGE1", None),
                "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                "complication": comp_name,
                "treatment_bucket": bucket,
                "comp_classification": mm,
                "snippet": snippet(txt_raw, 260),
            })

        # systemic bucket (after suppressor + no "PE" keyword)
        if OTHER_SYSTEMIC_RE.search(txt):
            found_any = True
            bucket = infer_treatment_bucket(txt)
            mm = major_minor_from_treatment(bucket)

            hits.append({
                "patient_id": pid,
                "EVENT_DT": r.get("EVENT_DT_STD", None),
                "stage1_dt": r.get("STAGE1_DT_STD", None),
                "DELTA_DAYS_FROM_STAGE1": r.get("DELTA_DAYS_FROM_STAGE1", None),
                "NOTE_TYPE": r.get(note_type_col, "") if note_type_col else "",
                "NOTE_ID": r.get(note_id_col, "") if note_id_col else "",
                "complication": "Other (systemic)",
                "treatment_bucket": bucket,
                "comp_classification": mm,
                "snippet": snippet(txt_raw, 260),
            })

        if found_any:
            unique_pat_anyhit.add(pid)

    # write row hits
    if hits:
        hits_df = pd.DataFrame(hits)
        hits_df = hits_df.sort_values(by=["patient_id", "EVENT_DT", "complication"], ascending=[True, True, True])
    else:
        hits_df = pd.DataFrame(columns=[
            "patient_id","EVENT_DT","stage1_dt","DELTA_DAYS_FROM_STAGE1",
            "NOTE_TYPE","NOTE_ID","complication","treatment_bucket","comp_classification","snippet"
        ])

    hits_df.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    # patient-level Comp1..3
    patient_level = df[[COL_PATIENT]].drop_duplicates().copy()
    patient_level = patient_level.rename(columns={COL_PATIENT: "patient_id"})
    # best stage1_dt per patient (should be constant)
    stage1_map = df.groupby("patient_id")["STAGE1_DT_STD"].min().to_dict()
    patient_level["stage1_dt"] = patient_level["patient_id"].map(stage1_map)

    if not hits_df.empty:
        rows = []
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
            rows.append(row)

        pl_hits = pd.DataFrame(rows)
        patient_level = patient_level.merge(pl_hits, on=["patient_id", "stage1_dt"], how="left")
    else:
        for idx in [1, 2, 3]:
            patient_level["S1_Comp{}_Date".format(idx)] = None
            patient_level["S1_Comp{}".format(idx)] = ""
            patient_level["S1_Comp{}_Treatment".format(idx)] = ""
            patient_level["S1_Comp{}_Classification".format(idx)] = ""
            patient_level["S1_Comp{}_NoteType".format(idx)] = ""
            patient_level["S1_Comp{}_NoteID".format(idx)] = ""

    patient_level = patient_level.sort_values(by=["patient_id"], ascending=[True])
    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # summary
    n_anchor_rows = int(len(df))
    n_anchor_pats = int(df["patient_id"].nunique())
    n_hit_rows = int(len(hits_df)) if not hits_df.empty else 0
    n_hit_pats = int(len(unique_pat_anyhit))

    comp_counts = hits_df["complication"].value_counts() if not hits_df.empty else pd.Series(dtype=int)
    treat_counts = hits_df["treatment_bucket"].value_counts() if not hits_df.empty else pd.Series(dtype=int)

    lines = []
    lines.append("=== Stage1 Complication Abstraction (anchored rows -> S1_Comp1..3) ===")
    lines.append("Input: {}".format(INFILE))
    lines.append("")
    lines.append("Detected columns:")
    lines.append("  patient_id: {}".format(COL_PATIENT))
    lines.append("  stage1 date col: {}".format(stage1_col))
    lines.append("  event date col: {}".format(event_col))
    lines.append("  text col: {}".format(text_col))
    if note_type_col:
        lines.append("  NOTE_TYPE: {}".format(note_type_col))
    if note_id_col:
        lines.append("  NOTE_ID: {}".format(note_id_col))
    lines.append("")
    lines.append("Anchored rows (0-365d) scanned: {}".format(n_anchor_rows))
    lines.append("Unique patients in anchor rows: {}".format(n_anchor_pats))
    lines.append("Row-level complication hits: {}".format(n_hit_rows))
    lines.append("Patients with >=1 hit: {} ({:.1f}%)".format(
        n_hit_pats, (100.0 * n_hit_pats / n_anchor_pats) if n_anchor_pats else 0.0
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
