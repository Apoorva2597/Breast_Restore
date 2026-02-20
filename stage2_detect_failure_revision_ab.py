# stage2_detect_failure_revision_ab.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   Detect Stage 2 FAILURE and REVISION signals (AB cohort) from Stage2-anchored rows.
#
# Inputs:
#   1) stage2_final_ab_patient_level.csv        (must contain patient_id + stage2 date column)
#   2) stage2_anchor_rows_with_bins.csv         (anchored rows with EVENT_DT + text column like NOTE_SNIPPET)
#
# Outputs:
#   1) stage2_ab_failure_revision_row_hits.csv
#   2) stage2_ab_failure_revision_patient_level.csv
#   3) stage2_ab_failure_revision_summary.txt
#
# Design (conservative):
#   - Failure: explicit device removal/explant AND signals of NOT replaced / flat chest / delayed reconstruction.
#              If "removed and replaced/exchanged/implanted" is present nearby, we suppress failure.
#   - Revision: revision-type procedures (capsulectomy, fat grafting, scar revision, mastopexy, etc.)
#              occurring after Stage2 date (already anchored rows are >=0 days).
#
# Notes:
#   - Uses NOTE_SNIPPET (or similar) if full note text isn't available.
#   - Read encoding latin1(errors=replace) for WVD robustness.

from __future__ import print_function

import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
AB_STAGE2_PATIENTS_CSV = "stage2_final_ab_patient_level.csv"
ANCHOR_ROWS_CSV = "stage2_anchor_rows_with_bins.csv"

OUT_ROW_HITS = "stage2_ab_failure_revision_row_hits.csv"
OUT_PATIENT_LEVEL = "stage2_ab_failure_revision_patient_level.csv"
OUT_SUMMARY = "stage2_ab_failure_revision_summary.txt"

CHUNKSIZE = 150000

# -------------------------
# CSV reading helpers (Py3.6 safe)
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


def to_dt(series):
    return pd.to_datetime(series, errors="coerce")


# -------------------------
# Auto-detect column names
# -------------------------
def pick_first_present(cols, candidates):
    cset = set([c.lower() for c in cols])
    for cand in candidates:
        if cand.lower() in cset:
            # return actual column name with original casing
            for c in cols:
                if c.lower() == cand.lower():
                    return c
    return None


def detect_stage2_date_col(cols):
    # your pipeline recently used stage2_date_final; keep broad fallback list
    cands = [
        "stage2_date_final",
        "stage2_dt",
        "stage2_dt_best",
        "stage2_event_dt_best",
        "stage2_date_best",
        "stage2_date",
        "stage2_date_ab",
        "stage2_dt_best_ab",
    ]
    return pick_first_present(cols, cands)


def detect_text_col(cols):
    cands = [
        "NOTE_SNIPPET",
        "NOTE_TEXT",
        "NOTE_TEXT_CLEAN",
        "snippet",
        "SNIPPET",
        "note_text",
        "note_text_clean",
        "TEXT",
    ]
    return pick_first_present(cols, cands)


# -------------------------
# Regex libraries
# -------------------------

# Device keywords
RX_DEVICE = re.compile(r"\b(implant|implnt|prosthesis|tissue\s*expander|expander|\bte\b)\b", re.I)

# Removal keywords
RX_REMOVE = re.compile(r"\b(remove|removed|removal|explant|explanted|take\s+out|taken\s+out)\b", re.I)

# Replacement / exchange keywords (suppresses failure if present close-by)
RX_REPLACE = re.compile(r"\b(replace|replaced|replacement|exchange|exchanged|insert|inserted|implantation|placed|place)\b", re.I)

# Explicit “not replaced” / flat chest / delayed recon language (strong failure evidence)
RX_NOT_REPLACED = re.compile(
    r"\b(not\s+replaced|without\s+replacement|no\s+replacement|left\s+without|"
    r"flat\s+chest|went\s+flat|remain(?:ed)?\s+flat|"
    r"delayed\s+reconstruction|staged\s+reconstruction\s+later|"
    r"planned\s+reconstruction\s+later|to\s+return\s+for\s+reconstruction)\b",
    re.I
)

# “Removed due to infection/extrusion” can be failure OR major comp; we treat as failure only if not replaced
RX_STRONG_REMOVE_CONTEXT = re.compile(r"\b(extrusion|exposed\s+implant|infect(?:ion|ed)|necrosis)\b", re.I)

# Revision procedure keywords (includes items from protocol examples)
RX_REVISION = re.compile(
    r"\b("
    r"capsulotomy|capsulectomy|"
    r"fat\s+graft(?:ing)?|lipofill|lipofilling|"
    r"scar\s+revision|revision\s+of\s+scar|"
    r"mastopexy|breast\s+lift|"
    r"reduction\s+mammoplasty|breast\s+reduction|"
    r"augmentation|"
    r"contour\s+deformity|contour\s+correction|"
    r"symmetr(?:y|ization)|symmetri(?:c|z)ation|"
    r"dog\s+ear|standing\s+cutaneous\s+deformity|"
    r"areolar\s+revision|nipple\s+reconstruction|nipple\s+revision"
    r")\b",
    re.I
)

# Revision suppressor: nipple reconstruction ALONE should not count as revision per protocol,
# but your protocol says nipple reconstruction by itself is NOT revision.
# If note mentions ONLY nipple reconstruction and nothing else revision-like, suppress.
RX_NIPPLE_ONLY = re.compile(r"\b(nipple\s+reconstruction|nipple\s+revision)\b", re.I)

# Helper: detect “nipple only” scenario
def is_nipple_only_revision(text):
    t = text
    if not RX_NIPPLE_ONLY.search(t):
        return False
    # If other revision keywords exist beyond nipple terms, it's not nipple-only.
    # Remove nipple phrases and re-check revision keywords.
    t2 = re.sub(r"(nipple\s+reconstruction|nipple\s+revision)", " ", t, flags=re.I)
    return bool(RX_REVISION.search(t2)) is False


# -------------------------
# Row classification
# -------------------------
def classify_failure_revision(text):
    """
    Returns:
      failure_flag(bool), failure_rule(str|None),
      revision_flag(bool), revision_rule(str|None)
    """
    t = norm_text(text)

    # REVISION detection (conservative nipple-only suppression)
    revision = False
    rev_rule = None
    if RX_REVISION.search(t):
        if is_nipple_only_revision(t):
            revision = False
            rev_rule = "NIPPLE_ONLY_SUPPRESSED"
        else:
            revision = True
            rev_rule = "REVISION_KEYWORD"

    # FAILURE detection (very conservative)
    failure = False
    fail_rule = None

    has_device = bool(RX_DEVICE.search(t))
    has_remove = bool(RX_REMOVE.search(t))

    if has_device and has_remove:
        # If “not replaced / flat / delayed recon later” => failure strong
        if RX_NOT_REPLACED.search(t):
            failure = True
            fail_rule = "REMOVE_WITH_NOT_REPLACED_OR_FLAT"
        else:
            # If removal context (infection/extrusion/etc) and NO replacement/exchange language => possible failure
            # But still conservative: require no replace/exchange terms.
            if RX_STRONG_REMOVE_CONTEXT.search(t) and (not RX_REPLACE.search(t)):
                failure = True
                fail_rule = "REMOVE_WITH_STRONG_CONTEXT_NO_REPLACE"
            else:
                # If replacement/exchange/inserted present, treat as NOT failure
                failure = False
                fail_rule = None

    return failure, fail_rule, revision, rev_rule


# -------------------------
# Main
# -------------------------
def main():
    # Load AB Stage2 patients
    ab = read_csv_safe(AB_STAGE2_PATIENTS_CSV)
    if ab is None or ab.empty:
        raise RuntimeError("Could not read AB Stage2 patients file: {}".format(AB_STAGE2_PATIENTS_CSV))

    if "patient_id" not in ab.columns:
        raise RuntimeError("AB file missing required column: patient_id")

    s2_col = detect_stage2_date_col(ab.columns.tolist())
    if not s2_col:
        raise RuntimeError(
            "No Stage2 date column found in {}. Expected one of common names (e.g., stage2_date_final).".format(
                AB_STAGE2_PATIENTS_CSV
            )
        )

    ab["patient_id"] = ab["patient_id"].fillna("").astype(str)
    ab[s2_col] = to_dt(ab[s2_col])

    # Keep only those with non-null Stage2 date
    ab = ab[ab[s2_col].notnull()].copy()
    ab_ids = set(ab["patient_id"].tolist())

    if not ab_ids:
        raise RuntimeError("No AB patients with non-null Stage2 date found in {}".format(AB_STAGE2_PATIENTS_CSV))

    stage2_map = dict(zip(ab["patient_id"], ab[s2_col]))

    # Detect columns in anchor rows
    head = None
    for chunk in iter_csv_safe(ANCHOR_ROWS_CSV, nrows=5):
        head = chunk
        break
    if head is None:
        raise RuntimeError("Could not read: {}".format(ANCHOR_ROWS_CSV))

    if "patient_id" not in head.columns:
        raise RuntimeError("Anchor rows file missing required column: patient_id")

    event_col = pick_first_present(head.columns.tolist(), ["EVENT_DT", "event_dt", "NOTE_DATE_OF_SERVICE", "OPERATION_DATE"])
    if not event_col:
        raise RuntimeError("Could not detect EVENT_DT-like column in anchor rows file: {}".format(ANCHOR_ROWS_CSV))

    text_col = detect_text_col(head.columns.tolist())
    if not text_col:
        raise RuntimeError(
            "Could not detect a note text column in {} (need NOTE_SNIPPET/NOTE_TEXT/snippet/etc).".format(ANCHOR_ROWS_CSV)
        )

    note_type_col = pick_first_present(head.columns.tolist(), ["NOTE_TYPE", "note_type"])
    note_id_col = pick_first_present(head.columns.tolist(), ["NOTE_ID", "note_id"])

    file_tag_col = pick_first_present(head.columns.tolist(), ["file_tag", "FILE_TAG"])
    delta_col = pick_first_present(head.columns.tolist(), ["DELTA_DAYS_FROM_STAGE2", "delta_days_from_stage2"])

    # Scan anchor rows (already stage2-anchored in your pipeline, but we still filter to AB ids)
    total_rows = 0
    kept_rows = 0
    hit_rows = []

    counts = {
        "failure_rows": 0,
        "revision_rows": 0,
        "both_rows": 0
    }

    for chunk in iter_csv_safe(ANCHOR_ROWS_CSV, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
        chunk = chunk[chunk["patient_id"].isin(ab_ids)].copy()
        if chunk.empty:
            continue

        kept_rows += len(chunk)

        # Ensure event_dt parsed for safety
        chunk[event_col] = to_dt(chunk[event_col])
        # Attach Stage2 dt
        chunk["stage2_dt"] = chunk["patient_id"].map(stage2_map)

        # Keep only rows with EVENT_DT >= stage2_dt (should already be true, but enforce)
        chunk = chunk[(chunk[event_col].notnull()) & (chunk["stage2_dt"].notnull())].copy()
        chunk = chunk[chunk[event_col] >= chunk["stage2_dt"]].copy()
        if chunk.empty:
            continue

        # classify
        failures = []
        fail_rules = []
        revisions = []
        rev_rules = []

        texts = chunk[text_col].fillna("").tolist()
        for txt in texts:
            f, fr, r, rr = classify_failure_revision(txt)
            failures.append(bool(f))
            fail_rules.append(fr)
            revisions.append(bool(r))
            rev_rules.append(rr)

        chunk["S2_Failure_Flag"] = failures
        chunk["S2_Failure_Rule"] = fail_rules
        chunk["S2_Revision_Flag"] = revisions
        chunk["S2_Revision_Rule"] = rev_rules

        # keep hit rows only
        hits = chunk[(chunk["S2_Failure_Flag"]) | (chunk["S2_Revision_Flag"])].copy()
        if hits.empty:
            continue

        counts["failure_rows"] += int(hits["S2_Failure_Flag"].sum())
        counts["revision_rows"] += int(hits["S2_Revision_Flag"].sum())
        counts["both_rows"] += int(((hits["S2_Failure_Flag"]) & (hits["S2_Revision_Flag"])).sum())

        # compute delta days if needed
        hits["DELTA_DAYS_FROM_STAGE2"] = (hits[event_col] - hits["stage2_dt"]).dt.days

        # output cols
        out_cols = ["patient_id", event_col, "stage2_dt", "DELTA_DAYS_FROM_STAGE2"]
        if delta_col and delta_col not in out_cols:
            out_cols.append(delta_col)
        if note_type_col:
            out_cols.append(note_type_col)
        if note_id_col:
            out_cols.append(note_id_col)
        if file_tag_col:
            out_cols.append(file_tag_col)

        out_cols += [
            "S2_Failure_Flag", "S2_Failure_Rule",
            "S2_Revision_Flag", "S2_Revision_Rule",
            text_col
        ]

        # ensure unique and existing
        out_cols_final = []
        for c in out_cols:
            if c in hits.columns and c not in out_cols_final:
                out_cols_final.append(c)

        hit_rows.append(hits[out_cols_final])

    hits_all = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()

    # Patient-level rollup
    patient_level = ab[["patient_id"]].drop_duplicates().copy()
    patient_level["stage2_dt"] = patient_level["patient_id"].map(stage2_map)

    if hits_all.empty:
        patient_level["Stage2_Failure"] = 0
        patient_level["Stage2_Revision"] = 0
        patient_level["Stage2_Failure_FirstDate"] = pd.NaT
        patient_level["Stage2_Revision_FirstDate"] = pd.NaT
        patient_level["Stage2_Failure_FirstNoteType"] = ""
        patient_level["Stage2_Failure_FirstNoteID"] = ""
        patient_level["Stage2_Revision_FirstNoteType"] = ""
        patient_level["Stage2_Revision_FirstNoteID"] = ""
    else:
        # Ensure EVENT_DT column name standard for grouping
        hits_all["EVENT_DT_STD"] = to_dt(hits_all[event_col])

        # Earliest failure hit per patient
        fail_hits = hits_all[hits_all["S2_Failure_Flag"]].copy()
        fail_hits = fail_hits.sort_values(by=["patient_id", "EVENT_DT_STD"], ascending=[True, True])
        first_fail = fail_hits.groupby("patient_id", as_index=False).head(1).copy()

        # Earliest revision hit per patient
        rev_hits = hits_all[hits_all["S2_Revision_Flag"]].copy()
        rev_hits = rev_hits.sort_values(by=["patient_id", "EVENT_DT_STD"], ascending=[True, True])
        first_rev = rev_hits.groupby("patient_id", as_index=False).head(1).copy()

        patient_level["Stage2_Failure"] = patient_level["patient_id"].isin(set(first_fail["patient_id"])).astype(int)
        patient_level["Stage2_Revision"] = patient_level["patient_id"].isin(set(first_rev["patient_id"])).astype(int)

        # attach first dates and identifiers
        patient_level = patient_level.merge(
            first_fail[["patient_id", "EVENT_DT_STD"] + ([note_type_col] if note_type_col else []) + ([note_id_col] if note_id_col else [])].rename(
                columns={"EVENT_DT_STD": "Stage2_Failure_FirstDate",
                         note_type_col: "Stage2_Failure_FirstNoteType" if note_type_col else note_type_col,
                         note_id_col: "Stage2_Failure_FirstNoteID" if note_id_col else note_id_col}
            ),
            on="patient_id",
            how="left"
        )

        patient_level = patient_level.merge(
            first_rev[["patient_id", "EVENT_DT_STD"] + ([note_type_col] if note_type_col else []) + ([note_id_col] if note_id_col else [])].rename(
                columns={"EVENT_DT_STD": "Stage2_Revision_FirstDate",
                         note_type_col: "Stage2_Revision_FirstNoteType" if note_type_col else note_type_col,
                         note_id_col: "Stage2_Revision_FirstNoteID" if note_id_col else note_id_col}
            ),
            on="patient_id",
            how="left"
        )

        # Fill missing cols if note_type/id absent
        if not note_type_col:
            patient_level["Stage2_Failure_FirstNoteType"] = ""
            patient_level["Stage2_Revision_FirstNoteType"] = ""
        if not note_id_col:
            patient_level["Stage2_Failure_FirstNoteID"] = ""
            patient_level["Stage2_Revision_FirstNoteID"] = ""

    # Write outputs
    if hits_all.empty:
        pd.DataFrame().to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")
    else:
        hits_all.to_csv(OUT_ROW_HITS, index=False, encoding="utf-8")

    patient_level.to_csv(OUT_PATIENT_LEVEL, index=False, encoding="utf-8")

    # Summary
    n_pat = int(patient_level["patient_id"].nunique())
    n_fail = int(patient_level["Stage2_Failure"].sum())
    n_rev = int(patient_level["Stage2_Revision"].sum())

    lines = []
    lines.append("=== Stage2 Failure + Revision Detector (AB) ===")
    lines.append("Python: 3.6.8 compatible | Read encoding: latin1(errors=replace) | Write: utf-8")
    lines.append("")
    lines.append("Inputs:")
    lines.append("  AB Stage2 patients: {}".format(AB_STAGE2_PATIENTS_CSV))
    lines.append("  Anchor rows:        {}".format(ANCHOR_ROWS_CSV))
    lines.append("")
    lines.append("Detected columns:")
    lines.append("  Stage2 date col (patients): {}".format(s2_col))
    lines.append("  Event date col (anchor):    {}".format(event_col))
    lines.append("  Text col (anchor):          {}".format(text_col))
    if note_type_col:
        lines.append("  NOTE_TYPE col (anchor):     {}".format(note_type_col))
    if note_id_col:
        lines.append("  NOTE_ID col (anchor):       {}".format(note_id_col))
    if file_tag_col:
        lines.append("  file_tag col (anchor):      {}".format(file_tag_col))
    lines.append("")
    lines.append("Row counts:")
    lines.append("  Total anchor rows scanned: {}".format(total_rows))
    lines.append("  Rows in AB cohort:         {}".format(kept_rows))
    lines.append("  Failure-hit rows:          {}".format(counts["failure_rows"]))
    lines.append("  Revision-hit rows:         {}".format(counts["revision_rows"]))
    lines.append("  Rows with both flags:      {}".format(counts["both_rows"]))
    lines.append("")
    lines.append("Patient counts (AB):")
    lines.append("  AB patients (with Stage2 date): {}".format(n_pat))
    lines.append("  Stage2_Failure patients:        {} ({:.1f}%)".format(n_fail, (100.0 * n_fail / n_pat) if n_pat else 0.0))
    lines.append("  Stage2_Revision patients:       {} ({:.1f}%)".format(n_rev, (100.0 * n_rev / n_pat) if n_pat else 0.0))
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
