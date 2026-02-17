# qa_clinic_inpatient_note_types.py
# Python 3.6.8+ (pandas required)
#
# Purpose:
#   For BOTH Clinic Notes and Inpatient Notes:
#     1) NOTE_TYPE distribution
#     2) Stage2-like evidence hits per NOTE_TYPE (strict + context)
#   Outputs:
#     - qa_clinic_inpatient_note_types_summary.txt
#     - qa_clinic_note_types_counts.csv
#     - qa_inpatient_note_types_counts.csv
#     - qa_clinic_stage2_hits_by_type.csv
#     - qa_inpatient_stage2_hits_by_type.csv
#
# Notes:
#   - Reads with latin1(errors=replace) to avoid UnicodeDecodeError 0xA0
#   - Designed to run on WVD after git pull

from __future__ import print_function

import re
import sys
import pandas as pd

# -------------------------
# CONFIG (EDIT PATHS ONLY)
# -------------------------
CLINIC_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Clinic Notes.csv"
INPATIENT_NOTES_CSV = "/home/apokol/my_data_Breast/HPI-11526/HPI11256/HPI11526 Inpatient Notes.csv"

OUT_SUMMARY_TXT = "qa_clinic_inpatient_note_types_summary.txt"

OUT_CLINIC_TYPE_COUNTS = "qa_clinic_note_types_counts.csv"
OUT_INPATIENT_TYPE_COUNTS = "qa_inpatient_note_types_counts.csv"

OUT_CLINIC_HITS_BY_TYPE = "qa_clinic_stage2_hits_by_type.csv"
OUT_INPATIENT_HITS_BY_TYPE = "qa_inpatient_stage2_hits_by_type.csv"

CHUNKSIZE = 120000

COL_PAT = "ENCRYPTED_PAT_ID"
COL_NOTE_TYPE = "NOTE_TYPE"
COL_NOTE_TEXT = "NOTE_TEXT"
COL_NOTE_ID = "NOTE_ID"
COL_DOS = "NOTE_DATE_OF_SERVICE"
COL_OP_DATE = "OPERATION_DATE"

# -------------------------
# Robust chunk reader
# -------------------------
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


def norm_type(x):
    if x is None:
        return "NA"
    s = str(x).strip()
    if not s:
        return "NA"
    return s.upper()


# Strict stage2 evidence
RX_STAGE2 = re.compile(
    r"("
    r"\bexchange\b.{0,120}\b(tissue\s*expander|expander|\bte\b)\b.{0,260}\b(implant|permanent\s+implant|implnt)\b"
    r"|"
    r"\b(remove|removed|explant|explanted)\b.{0,220}\b(tissue\s*expander|expander|\bte\b)\b.{0,520}\b(place|placed|insert|inserted|insertion|implantation)\b.{0,180}\b(implant|permanent\s+implant|implnt)\b"
    r")",
    re.I
)

# Context
RX_CONTEXT = re.compile(r"\b(tissue\s*expander|expander|\bte\b)\b.*\b(implant|permanent\s+implant|implnt)\b", re.I)


def analyze_file(tag, path, out_type_counts_csv, out_hits_by_type_csv):
    # probe header
    head = None
    for chunk in iter_csv_safe(path, nrows=5):
        head = chunk
        break
    if head is None:
        raise RuntimeError("Could not read: {}".format(path))

    if COL_NOTE_TEXT not in head.columns:
        raise RuntimeError("Missing NOTE_TEXT in {}".format(path))
    if COL_NOTE_TYPE not in head.columns:
        # allow but will all be NA
        pass

    usecols = []
    for c in [COL_PAT, COL_NOTE_TYPE, COL_NOTE_TEXT, COL_NOTE_ID, COL_DOS, COL_OP_DATE]:
        if c in head.columns:
            usecols.append(c)

    note_type_counts = {}
    per_type = {}

    total_rows = 0
    total_stage2_rows = 0
    total_ctx_rows = 0

    for chunk in iter_csv_safe(path, usecols=usecols, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        types = chunk[COL_NOTE_TYPE] if COL_NOTE_TYPE in chunk.columns else pd.Series(["NA"] * len(chunk))
        types = types.fillna("NA").apply(norm_type)

        for t in types.tolist():
            note_type_counts[t] = note_type_counts.get(t, 0) + 1

        texts = chunk[COL_NOTE_TEXT].fillna("").apply(norm_text)

        pats = chunk[COL_PAT] if COL_PAT in chunk.columns else pd.Series([""] * len(chunk))
        pats = pats.fillna("").astype(str)

        stage2_hit = texts.str.contains(RX_STAGE2, regex=True)
        ctx_hit = texts.str.contains(RX_CONTEXT, regex=True)

        total_stage2_rows += int(stage2_hit.sum())
        total_ctx_rows += int(ctx_hit.sum())

        for i in range(len(chunk)):
            t = types.iat[i]
            pid = pats.iat[i]
            st2 = bool(stage2_hit.iat[i])
            ctx = bool(ctx_hit.iat[i])

            if t not in per_type:
                per_type[t] = {
                    "rows": 0,
                    "stage2_rows": 0,
                    "context_rows": 0,
                    "unique_patients_any": set(),
                    "unique_patients_stage2": set(),
                }

            d = per_type[t]
            d["rows"] += 1
            if pid:
                d["unique_patients_any"].add(pid)
            if st2:
                d["stage2_rows"] += 1
                if pid:
                    d["unique_patients_stage2"].add(pid)
            if ctx:
                d["context_rows"] += 1

    nt_df = pd.DataFrame([{"NOTE_TYPE": k, "ROWS": v} for k, v in note_type_counts.items()]) \
        .sort_values(by="ROWS", ascending=False)
    nt_df.to_csv(out_type_counts_csv, index=False, encoding="utf-8")

    rows = []
    for t, d in per_type.items():
        rows.append({
            "NOTE_TYPE": t,
            "ROWS": d["rows"],
            "STAGE2_ROWS_STRICT": d["stage2_rows"],
            "CONTEXT_ROWS_EXPANDER+IMPLANT": d["context_rows"],
            "UNIQUE_PATIENTS_ANY": len(d["unique_patients_any"]),
            "UNIQUE_PATIENTS_STAGE2_STRICT": len(d["unique_patients_stage2"]),
        })
    hits_df = pd.DataFrame(rows).sort_values(by=["UNIQUE_PATIENTS_STAGE2_STRICT", "STAGE2_ROWS_STRICT"], ascending=False)
    hits_df.to_csv(out_hits_by_type_csv, index=False, encoding="utf-8")

    summary = {
        "tag": tag,
        "path": path,
        "total_rows": total_rows,
        "total_stage2_rows": total_stage2_rows,
        "total_ctx_rows": total_ctx_rows,
        "top_types": nt_df.head(15),
        "top_hit_types": hits_df.head(15),
    }
    return summary


def main():
    clinic = analyze_file(
        "CLINIC",
        CLINIC_NOTES_CSV,
        OUT_CLINIC_TYPE_COUNTS,
        OUT_CLINIC_HITS_BY_TYPE
    )
    inpatient = analyze_file(
        "INPATIENT",
        INPATIENT_NOTES_CSV,
        OUT_INPATIENT_TYPE_COUNTS,
        OUT_INPATIENT_HITS_BY_TYPE
    )

    lines = []
    lines.append("=== QA: Clinic + Inpatient NOTE_TYPE coverage + Stage2 phrase hits ===")
    lines.append("Read encoding: latin1(errors=replace) | Python 3.6.8 compatible")
    lines.append("")

    for s in [clinic, inpatient]:
        lines.append("--- {} ---".format(s["tag"]))
        lines.append("File: {}".format(s["path"]))
        lines.append("Total rows scanned: {}".format(s["total_rows"]))
        lines.append("Strict Stage2-hit rows: {}".format(s["total_stage2_rows"]))
        lines.append("Context rows (expander+implant): {}".format(s["total_ctx_rows"]))
        lines.append("")
        lines.append("Top NOTE_TYPEs (rows) [top 15]:")
        for _, r in s["top_types"].iterrows():
            lines.append("  {:>7}  {}".format(int(r["ROWS"]), r["NOTE_TYPE"]))
        lines.append("")
        lines.append("Top NOTE_TYPEs by UNIQUE_PATIENTS_STAGE2_STRICT [top 15]:")
        for _, r in s["top_hit_types"].iterrows():
            lines.append("  {:>6} patients | {:>6} rows | {}".format(
                int(r["UNIQUE_PATIENTS_STAGE2_STRICT"]),
                int(r["STAGE2_ROWS_STRICT"]),
                r["NOTE_TYPE"]
            ))
        lines.append("")

    lines.append("Wrote:")
    lines.append("  - {}".format(OUT_SUMMARY_TXT))
    lines.append("  - {}".format(OUT_CLINIC_TYPE_COUNTS))
    lines.append("  - {}".format(OUT_INPATIENT_TYPE_COUNTS))
    lines.append("  - {}".format(OUT_CLINIC_HITS_BY_TYPE))
    lines.append("  - {}".format(OUT_INPATIENT_HITS_BY_TYPE))

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
