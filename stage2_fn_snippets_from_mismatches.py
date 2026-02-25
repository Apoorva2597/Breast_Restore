#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_snippets_from_mismatches.py  (Python 3.6.8)

Goal:
- Read mismatches file
- Isolate Stage2 false negatives (gold=1, pred=0)
- Map to ENCRYPTED_PAT_ID (via mismatches if present; else via notes)
- Search raw notes CSVs for broad Stage2 keywords/phrases
- Output short snippets (15 words before/after match)
- NO MRN written to output

Inputs (edit paths below if needed):
  - ./_outputs/validation_mismatches_STAGE2_ANCHOR_FINAL_FINAL.csv
  - Notes directory: /home/apokol/my_data_Breast/HPI-11526/HPI11256
Output:
  - ./_outputs/stage2_fn_keyword_snippets_FINAL_FINAL.csv
"""

from __future__ import print_function
import os
import re
import glob
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
MISMATCH_PATH = os.path.join(".", "_outputs", "validation_mismatches_STAGE2_ANCHOR_FINAL_FINAL.csv")
NOTES_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"
OUT_PATH = os.path.join(".", "_outputs", "stage2_fn_keyword_snippets_FINAL_FINAL.csv")

WORDS_BEFORE = 15
WORDS_AFTER = 15
MAX_HITS_PER_NOTE = 5  # keep concise

# Broad Stage2 keyword/phrase patterns (case-insensitive)
# Keep these practical: exchange, permanent implant, expander removal, explant, stage 2, etc.
KEY_PATTERNS = [
    r"\bsecond stage\b",
    r"\bstage\s*2\b",
    r"\bexpander[- ]?to[- ]?implant\b",
    r"\bpermanent\b.*\bimplant(s)?\b",
    r"\bimplant\b.*\b(exchange|replace|replacement)\b",
    r"\b(exchange|replace|replacement)\b.*\bimplant(s)?\b",
    r"\bexchange\b.*\b(tissue\s+expander|expander|\bte\b)\b.*\bfor\b.*\bimplant(s)?\b",
    r"\b(tissue\s+expander|expander|\bte\b)\b.*\bexchang(e|ed)\b.*\bfor\b.*\bimplant(s)?\b",
    r"\b(remov(e|al|ed)?|explant(ed)?|take\s*out)\b.*\b(tissue\s+expander|expander|\bte\b)\b",
    r"\b(tissue\s+expander|expander|\bte\b)\b.*\b(remov(e|al|ed)?|explant(ed)?|take\s*out)\b",
    r"\b(implant\s+placement|placed\s+implant|implant\s+placed)\b",
]

# Precompile
KEY_REGEXES = [(pat, re.compile(pat, re.I)) for pat in KEY_PATTERNS]

# -----------------------------
# HELPERS
# -----------------------------
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

def normalize_id(x):
    return "" if x is None else str(x).strip()

def to01(v):
    if v is None:
        return 0
    s = str(v).strip().lower()
    if s in ["1", "y", "yes", "true", "t"]:
        return 1
    if s in ["0", "n", "no", "false", "f", ""]:
        return 0
    try:
        return 1 if float(s) != 0.0 else 0
    except:
        return 0

def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def list_csvs(root):
    # Recursively find CSVs; you can narrow later if needed
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) if os.path.isfile(p)])

def looks_like_notes_csv(df):
    cols = set(df.columns)
    must = {"NOTE_TEXT"}
    has_id = ("ENCRYPTED_PAT_ID" in cols) or ("MRN" in cols)
    has_note = ("NOTE_ID" in cols)
    return must.issubset(cols) and has_id and has_note

def build_word_spans(text):
    # Returns list of (start_char, end_char, token) for each word
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans

def snippet_around_match(text, match_start, match_end, before=15, after=15):
    if not text:
        return ""
    spans = build_word_spans(text)
    if not spans:
        return ""
    # find word index containing match_start
    idx = 0
    for i, (s, e, _) in enumerate(spans):
        if s <= match_start < e:
            idx = i
            break
        if s > match_start:
            idx = max(0, i - 1)
            break
    lo = max(0, idx - before)
    hi = min(len(spans), idx + after + 1)
    return " ".join([spans[i][2] for i in range(lo, hi)])

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and (not os.path.isdir(d)):
        os.makedirs(d)

# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.isfile(MISMATCH_PATH):
        raise IOError("Missing mismatches file: {}".format(MISMATCH_PATH))
    if not os.path.isdir(NOTES_DIR):
        raise IOError("Missing notes dir: {}".format(NOTES_DIR))

    mism = normalize_cols(read_csv_robust(MISMATCH_PATH, dtype=str, low_memory=False))

    # Find gold/pred columns
    gold_col = pick_first_existing(mism, ["GOLD_HAS_STAGE2", "GOLD_STAGE2", "GOLD_STAGE2_APP", "GOLD_STAGE2_APPLICABLE"])
    pred_col = pick_first_existing(mism, ["PRED_HAS_STAGE2", "HAS_STAGE2", "PRED_STAGE2"])

    if not gold_col or not pred_col:
        raise ValueError("Could not find GOLD/PRED stage2 columns in mismatches. Columns found: {}".format(list(mism.columns)))

    mism["_gold"] = mism[gold_col].map(to01).astype(int)
    mism["_pred"] = mism[pred_col].map(to01).astype(int)

    # Identify FN rows
    fn = mism[(mism["_gold"] == 1) & (mism["_pred"] == 0)].copy()
    if fn.empty:
        print("No false negatives found in mismatches.")
        return

    # Prefer ENCRYPTED_PAT_ID directly if present; else carry MRN to map via notes
    enc_col = pick_first_existing(fn, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
    mrn_col = pick_first_existing(fn, ["MRN", "mrn"])

    fn["ENCRYPTED_PAT_ID"] = ""
    fn["MRN"] = ""

    if enc_col:
        fn["ENCRYPTED_PAT_ID"] = fn[enc_col].map(normalize_id)
    if mrn_col:
        fn["MRN"] = fn[mrn_col].map(normalize_id)

    # Load notes CSVs and create mapping MRN->ENCRYPTED_PAT_ID if needed
    csvs = list_csvs(NOTES_DIR)
    if not csvs:
        raise IOError("No CSVs found under notes dir: {}".format(NOTES_DIR))

    # We'll scan CSVs until we find at least one notes-like file to build id_map
    id_map = None
    notes_files = []

    for p in csvs:
        try:
            df0 = normalize_cols(read_csv_robust(p, nrows=25, dtype=str, low_memory=False))
        except Exception:
            continue
        if looks_like_notes_csv(df0):
            notes_files.append(p)

    if not notes_files:
        raise IOError("Could not find any notes-like CSVs under {} (need NOTE_TEXT + NOTE_ID + MRN/ENCRYPTED_PAT_ID).".format(NOTES_DIR))

    # If FN lacks ENCRYPTED_PAT_ID, build mapping from notes
    need_map = fn["ENCRYPTED_PAT_ID"].map(lambda x: 1 if normalize_id(x) == "" else 0).sum() > 0

    if need_map:
        # build MRN->ENCRYPTED_PAT_ID map from first few notes files (stop once map is non-trivial)
        maps = []
        for p in notes_files[:10]:
            df = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
            mrn_c = pick_first_existing(df, ["MRN", "mrn"])
            enc_c = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
            if not mrn_c or not enc_c:
                continue
            tmp = df[[mrn_c, enc_c]].dropna()
            tmp.columns = ["MRN", "ENCRYPTED_PAT_ID"]
            tmp["MRN"] = tmp["MRN"].map(normalize_id)
            tmp["ENCRYPTED_PAT_ID"] = tmp["ENCRYPTED_PAT_ID"].map(normalize_id)
            tmp = tmp[(tmp["MRN"] != "") & (tmp["ENCRYPTED_PAT_ID"] != "")]
            tmp = tmp.drop_duplicates()
            if len(tmp) > 0:
                maps.append(tmp)
        if not maps:
            raise ValueError("Need MRN->ENCRYPTED_PAT_ID mapping, but could not build it from notes files.")
        id_map = pd.concat(maps, axis=0, ignore_index=True).drop_duplicates()

        # merge to fill ENCRYPTED_PAT_ID
        fn = fn.merge(id_map, on="MRN", how="left", suffixes=("", "_map"))
        fn["ENCRYPTED_PAT_ID"] = fn["ENCRYPTED_PAT_ID"].fillna("").map(normalize_id)
        fn["ENCRYPTED_PAT_ID"] = fn["ENCRYPTED_PAT_ID"].where(fn["ENCRYPTED_PAT_ID"] != "", fn["ENCRYPTED_PAT_ID_map"].fillna("").map(normalize_id))
        fn = fn.drop([c for c in fn.columns if c.endswith("_map")], axis=1, errors="ignore")

    # Keep only rows with ENCRYPTED_PAT_ID (we won't output MRN)
    fn = fn[fn["ENCRYPTED_PAT_ID"] != ""].copy()
    if fn.empty:
        raise ValueError("After mapping, no FN rows have ENCRYPTED_PAT_ID. Check mismatches columns or mapping source.")

    fn_ids = set(fn["ENCRYPTED_PAT_ID"].tolist())

    # Scan notes and extract snippets
    out_rows = []
    for p in notes_files:
        df = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
        if "NOTE_TEXT" not in df.columns:
            continue
        enc_c = pick_first_existing(df, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
        if not enc_c:
            continue
        note_id_c = pick_first_existing(df, ["NOTE_ID", "NoteID", "NOTEID"])
        note_type_c = pick_first_existing(df, ["NOTE_TYPE", "NoteType", "TYPE"])
        if not note_id_c:
            continue

        df[enc_c] = df[enc_c].map(normalize_id)
        sub = df[df[enc_c].isin(fn_ids)].copy()
        if sub.empty:
            continue

        # aggregate NOTE_TEXT per (patient, note_id) if LINE exists
        line_c = pick_first_existing(sub, ["LINE", "Line", "LINE_NUM"])
        if line_c and line_c in sub.columns:
            try:
                sub[line_c] = sub[line_c].fillna("0").map(lambda x: int(float(str(x).strip())) if str(x).strip() != "" else 0)
            except:
                sub[line_c] = 0
            sub["__text__"] = sub["NOTE_TEXT"].fillna("").map(str)
            sub = sub.sort_values([enc_c, note_id_c, line_c])
            grp = sub.groupby([enc_c, note_id_c], as_index=False).agg({
                "__text__": lambda x: " ".join([t for t in x if t]),
                (note_type_c if note_type_c else note_id_c): "first"
            })
            grp = grp.rename(columns={enc_c: "ENCRYPTED_PAT_ID", note_id_c: "NOTE_ID"})
            if note_type_c:
                grp = grp.rename(columns={note_type_c: "NOTE_TYPE"})
            else:
                grp["NOTE_TYPE"] = ""
            grp["NOTE_TEXT_FULL"] = grp["__text__"]
        else:
            grp = sub[[enc_c, note_id_c, "NOTE_TEXT"]].copy()
            grp = grp.rename(columns={enc_c: "ENCRYPTED_PAT_ID", note_id_c: "NOTE_ID"})
            grp["NOTE_TYPE"] = sub[note_type_c].values if (note_type_c and note_type_c in sub.columns) else ""
            grp["NOTE_TEXT_FULL"] = grp["NOTE_TEXT"].fillna("").map(str)

        # search snippets
        for _, r in grp.iterrows():
            pid = normalize_id(r.get("ENCRYPTED_PAT_ID"))
            note_id = normalize_id(r.get("NOTE_ID"))
            note_type = normalize_id(r.get("NOTE_TYPE"))
            text = r.get("NOTE_TEXT_FULL") or ""
            text_l = text.lower()

            hits = 0
            for pat, rx in KEY_REGEXES:
                for m in rx.finditer(text_l):
                    snip = snippet_around_match(text, m.start(), m.end(), before=WORDS_BEFORE, after=WORDS_AFTER)
                    if snip:
                        out_rows.append({
                            "ENCRYPTED_PAT_ID": pid,
                            "NOTE_ID": note_id,
                            "NOTE_TYPE": note_type,
                            "MATCH_PATTERN": pat,
                            "SNIPPET": snip,
                            "SOURCE_FILE": os.path.basename(p)
                        })
                        hits += 1
                        if hits >= MAX_HITS_PER_NOTE:
                            break
                if hits >= MAX_HITS_PER_NOTE:
                    break

    ensure_dir(OUT_PATH)
    if not out_rows:
        # still write an empty file with headers (so pipeline doesn't break)
        pd.DataFrame(columns=["ENCRYPTED_PAT_ID","NOTE_ID","NOTE_TYPE","MATCH_PATTERN","SNIPPET","SOURCE_FILE"]).to_csv(OUT_PATH, index=False)
        print("Wrote (no hits):", OUT_PATH)
        return

    out_df = pd.DataFrame(out_rows)

    # Ensure NO MRN columns ever leak
    for c in list(out_df.columns):
        if "MRN" in c.upper():
            out_df = out_df.drop([c], axis=1, errors="ignore")

    # Deduplicate obvious repeats
    out_df = out_df.drop_duplicates(subset=["ENCRYPTED_PAT_ID", "NOTE_ID", "MATCH_PATTERN", "SNIPPET"])

    out_df.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()
