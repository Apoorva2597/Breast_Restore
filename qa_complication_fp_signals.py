# qa_complication_fp_signals.py
# Python 3.6.8+
#
# Purpose:
#   Quick pre-gold diagnostics for complication row_hits files:
#   - How many hits by complication category?
#   - Which note types dominate hits?
#   - How many hits look like "risk/consent/education" language (common false positives)?
#
# Inputs (edit if needed):
#   - stage1_complications_row_hits.csv
#   - stage2_ab_complications_row_hits.csv
#
# Outputs:
#   - qa_complication_fp_signals_summary.txt
#   - qa_complication_fp_signals_categories.csv
#   - qa_complication_fp_signals_note_types.csv
#   - qa_complication_fp_signals_exclusion_terms.csv

from __future__ import print_function

import re
import pandas as pd

STAGE1_ROW_HITS = "stage1_complications_row_hits.csv"
STAGE2_ROW_HITS = "stage2_ab_complications_row_hits.csv"

OUT_SUMMARY_TXT = "qa_complication_fp_signals_summary.txt"
OUT_CATS_CSV = "qa_complication_fp_signals_categories.csv"
OUT_TYPES_CSV = "qa_complication_fp_signals_note_types.csv"
OUT_TERMS_CSV = "qa_complication_fp_signals_exclusion_terms.csv"


def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python")
    finally:
        try:
            f.close()
        except Exception:
            pass


def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def pick_fuzzy(cols, tokens_any):
    # returns first col whose name contains any token
    for c in cols:
        u = str(c).upper()
        for t in tokens_any:
            if t in u:
                return c
    return None


def norm_text(x):
    if x is None or (isinstance(x, float) and pd.isnull(x)):
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Common "non-event" language that causes false positives
EXCLUSION_PATTERNS = [
    ("risk_of", re.compile(r"\brisk(s)?\s+of\b", re.I)),
    ("risks_include", re.compile(r"\brisks?\s+(include|including)\b", re.I)),
    ("discussed_risks", re.compile(r"\b(discuss(ed|ing)?|review(ed|ing)?)\s+(risk|risks|complication|complications)\b", re.I)),
    ("counsel", re.compile(r"\b(counsel(ed|ing)?|counselling|counseling)\b", re.I)),
    ("consent", re.compile(r"\b(consent|informed\s+consent)\b", re.I)),
    ("possible_potential", re.compile(r"\b(possible|potential|may|might|can\s+include)\b", re.I)),
    ("warning_signs", re.compile(r"\b(warning\s+signs?|return\s+precautions?)\b", re.I)),
    ("call_if", re.compile(r"\b(call|return)\s+(if|for)\b", re.I)),
    ("education", re.compile(r"\b(education|educat(ed|ion))\b", re.I)),
]

# Extra check for "PE" ambiguity (pulmonary embolism vs physical exam)
RX_PE_WORD = re.compile(r"\bpe\b", re.I)
RX_PHYSICAL_EXAM = re.compile(r"\bphysical\s+exam\b|\bP\.?E\.?\b", re.I)
RX_PULM_EMB = re.compile(r"\bpulmonary\s+embol(ism|us)\b", re.I)


def analyze_file(tag, path):
    df = read_csv_safe(path)
    cols = df.columns.tolist()

    pid_col = pick_col(cols, ["patient_id", "ENCRYPTED_PAT_ID"]) or pick_fuzzy(cols, ["PATIENT"])
    comp_col = pick_col(cols, ["complication"]) or pick_fuzzy(cols, ["COMPLICATION"])
    type_col = pick_col(cols, ["NOTE_TYPE", "note_type"]) or pick_fuzzy(cols, ["NOTE_TYPE", "TYPE"])
    text_col = pick_col(cols, ["snippet", "NOTE_SNIPPET", "NOTE_TEXT", "note_text", "TEXT"]) or pick_fuzzy(cols, ["SNIPPET", "NOTE_TEXT", "TEXT"])

    if pid_col is None:
        raise RuntimeError("[{}] Could not find patient id column.".format(tag))
    if comp_col is None:
        raise RuntimeError("[{}] Could not find complication column.".format(tag))
    if text_col is None:
        raise RuntimeError("[{}] Could not find text/snippet column.".format(tag))

    df[pid_col] = df[pid_col].fillna("").astype(str)
    df[comp_col] = df[comp_col].fillna("").astype(str)
    df["_TEXT_"] = df[text_col].fillna("").apply(norm_text)

    total_rows = int(len(df))
    uniq_pats = int(df[pid_col].nunique())

    # Category counts
    cat_counts = df[comp_col].value_counts(dropna=False)

    # Note type counts (if present)
    if type_col is not None:
        df[type_col] = df[type_col].fillna("").astype(str)
        type_counts = df[type_col].value_counts(dropna=False)
    else:
        type_counts = pd.Series(dtype=int)

    # Exclusion hits
    excl_any = pd.Series([False] * len(df))
    term_hits = {}
    for name, rx in EXCLUSION_PATTERNS:
        m = df["_TEXT_"].str.contains(rx)
        term_hits[name] = int(m.sum())
        excl_any = excl_any | m

    excl_any_n = int(excl_any.sum())
    excl_any_pct = (100.0 * excl_any_n / total_rows) if total_rows else 0.0

    # PE ambiguity quick check (mostly for "Other (systemic)" bucket)
    pe_any = df["_TEXT_"].str.contains(RX_PE_WORD)
    pe_physical_exam = df["_TEXT_"].str.contains(RX_PHYSICAL_EXAM)
    pe_pulm_emb = df["_TEXT_"].str.contains(RX_PULM_EMB)

    pe_any_n = int(pe_any.sum())
    pe_phys_n = int((pe_any & pe_physical_exam).sum())
    pe_true_n = int((pe_any & pe_pulm_emb).sum())

    # Build small tables for CSV outputs
    cats_out = pd.DataFrame({
        "file_tag": [tag] * len(cat_counts),
        "complication": cat_counts.index.astype(str),
        "count": cat_counts.values.astype(int),
    })

    if not type_counts.empty:
        types_out = pd.DataFrame({
            "file_tag": [tag] * len(type_counts),
            "note_type": type_counts.index.astype(str),
            "count": type_counts.values.astype(int),
        })
    else:
        types_out = pd.DataFrame(columns=["file_tag", "note_type", "count"])

    terms_out = pd.DataFrame({
        "file_tag": [tag] * len(term_hits),
        "term": list(term_hits.keys()),
        "count": [term_hits[k] for k in term_hits.keys()],
    }).sort_values(by="count", ascending=False)

    # Summary lines
    lines = []
    lines.append("=== {} ===".format(tag))
    lines.append("File: {}".format(path))
    lines.append("Detected cols: patient_id='{}'  complication='{}'  text='{}'  note_type='{}'".format(
        pid_col, comp_col, text_col, type_col if type_col else "(none)"
    ))
    lines.append("Total rows: {} | Unique patients: {}".format(total_rows, uniq_pats))
    lines.append("")
    lines.append("Top complication categories (top 15):")
    for k, v in cat_counts.head(15).items():
        lines.append("  {:>6}  {}".format(int(v), str(k)))
    lines.append("")

    if not type_counts.empty:
        lines.append("Top NOTE_TYPE values (top 15):")
        for k, v in type_counts.head(15).items():
            lines.append("  {:>6}  {}".format(int(v), str(k)))
        lines.append("")

    lines.append("Possible 'risk/consent/education' language flags:")
    lines.append("  Rows matching ANY exclusion pattern: {} / {} ({:.1f}%)".format(excl_any_n, total_rows, excl_any_pct))
    lines.append("  Top exclusion terms (by count):")
    for _, r in terms_out.head(10).iterrows():
        lines.append("    {:>6}  {}".format(int(r["count"]), r["term"]))
    lines.append("")

    lines.append("Quick PE ambiguity check (token 'pe'):")
    lines.append("  Rows with word 'pe': {}".format(pe_any_n))
    lines.append("  Of those, rows mentioning Physical Exam / P.E.: {}".format(pe_phys_n))
    lines.append("  Of those, rows explicitly saying 'pulmonary embol*': {}".format(pe_true_n))
    lines.append("")

    return "\n".join(lines), cats_out, types_out, terms_out


def main():
    all_lines = []
    cats_all = []
    types_all = []
    terms_all = []

    for tag, path in [("STAGE1", STAGE1_ROW_HITS), ("STAGE2_AB", STAGE2_ROW_HITS)]:
        try:
            txt, cats, types, terms = analyze_file(tag, path)
            all_lines.append(txt)
            cats_all.append(cats)
            if not types.empty:
                types_all.append(types)
            terms_all.append(terms)
        except Exception as e:
            all_lines.append("=== {} ===\nERROR reading/analyzing {}: {}\n".format(tag, path, str(e)))

    final_txt = "\n".join(all_lines)
    print(final_txt)

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(final_txt + "\n")

    if cats_all:
        pd.concat(cats_all, ignore_index=True).to_csv(OUT_CATS_CSV, index=False, encoding="utf-8")
    if types_all:
        pd.concat(types_all, ignore_index=True).to_csv(OUT_TYPES_CSV, index=False, encoding="utf-8")
    if terms_all:
        pd.concat(terms_all, ignore_index=True).to_csv(OUT_TERMS_CSV, index=False, encoding="utf-8")

    print("Wrote:")
    print("  - {}".format(OUT_SUMMARY_TXT))
    print("  - {}".format(OUT_CATS_CSV))
    print("  - {}".format(OUT_TYPES_CSV))
    print("  - {}".format(OUT_TERMS_CSV))


if __name__ == "__main__":
    main()
