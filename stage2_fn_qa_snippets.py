#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_snippets_from_mismatches.py  (Python 3.6.8)

Fixes for "no hits":
- Much more robust column detection (NOTE_TEXT variants + ENCRYPTED_PAT_ID variants)
- Normalizes IDs to remove trailing ".0" and handle numeric-like values
- Broad keyword regex (includes simple "exchange|exchanged|implant|expander|remove|revision|capsulectomy")
- If STILL no keyword hits for a FN note, emits a short PREVIEW snippet (first ~40 words)
- Output contains NO MRN (only ENCRYPTED_PAT_ID)
"""

from __future__ import print_function
import os
import re
import glob
import pandas as pd

MISMATCH_PATH = os.path.join(".", "_outputs", "validation_mismatches_STAGE2_ANCHOR_FINAL_FINAL.csv")
NOTES_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"
OUT_PATH = os.path.join(".", "_outputs", "stage2_fn_keyword_snippets_FINAL_FINAL.csv")

WORDS_BEFORE = 15
WORDS_AFTER = 15
MAX_HITS_PER_NOTE = 5
ALWAYS_PREVIEW_IF_NO_HITS = True
PREVIEW_WORDS = 40

# Broad keyword regex (intentionally broad for FN QA)
BROAD_RX = re.compile(
    r"\b("
    r"exchange|exchang(ed|e|ing)?|"
    r"expander(s)?|tissue\s+expander(s)?|\bte\b|"
    r"implant(s)?|permanent\s+implant(s)?|"
    r"remove(d|al)?|removal|explant(ed)?|take\s*out|"
    r"replace(d|ment)?|revision|capsulectomy|capsulotomy|"
    r"second\s+stage|stage\s*2|expander[- ]?to[- ]?implant"
    r")\b",
    re.I
)

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
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    # strip trailing .0 from numeric-like IDs
    if re.match(r"^\d+\.0$", s):
        s = s.split(".")[0]
    # if it's a float string like "12345.00"
    if re.match(r"^\d+\.\d+$", s):
        try:
            f = float(s)
            if abs(f - int(f)) < 1e-9:
                s = str(int(f))
        except:
            pass
    return s

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

def find_text_col(df):
    # Prefer explicit NOTE_TEXT-like columns
    prefs = ["NOTE_TEXT", "NOTE_TEXT_FULL", "NOTE_TEXT_COMBINED", "NOTE_BODY", "TEXT", "NOTE"]
    for c in prefs:
        if c in df.columns:
            return c
    # fallback: any column containing NOTE_TEXT
    for c in df.columns:
        cu = str(c).upper()
        if "NOTE_TEXT" in cu:
            return c
    # fallback: any column containing TEXT
    for c in df.columns:
        cu = str(c).upper()
        if cu.endswith("TEXT") or " TEXT" in cu or cu == "TEXT":
            return c
    return None

def list_csvs(root):
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) if os.path.isfile(p)])

def build_word_spans(text):
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans

def snippet_around_match(text, match_start, before=15, after=15):
    if not text:
        return ""
    spans = build_word_spans(text)
    if not spans:
        return ""
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

def preview_snippet(text, n_words=40):
    if not text:
        return ""
    spans = build_word_spans(text)
    if not spans:
        return ""
    hi = min(len(spans), n_words)
    return " ".join([spans[i][2] for i in range(0, hi)])

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and (not os.path.isdir(d)):
        os.makedirs(d)

def main():
    if not os.path.isfile(MISMATCH_PATH):
        raise IOError("Missing mismatches file: {}".format(MISMATCH_PATH))
    if not os.path.isdir(NOTES_DIR):
        raise IOError("Missing notes dir: {}".format(NOTES_DIR))

    mism = normalize_cols(read_csv_robust(MISMATCH_PATH, dtype=str, low_memory=False))

    gold_col = pick_first_existing(mism, ["GOLD_HAS_STAGE2", "GOLD_STAGE2", "Stage2_Applicable", "STAGE2_APPLICABLE"])
    pred_col = pick_first_existing(mism, ["PRED_HAS_STAGE2", "HAS_STAGE2", "PRED_STAGE2"])
    if not gold_col or not pred_col:
        raise ValueError("Missing GOLD/PRED cols in mismatches. Found: {}".format(list(mism.columns)))

    mism["_gold"] = mism[gold_col].map(to01).astype(int)
    mism["_pred"] = mism[pred_col].map(to01).astype(int)
    fn = mism[(mism["_gold"] == 1) & (mism["_pred"] == 0)].copy()
    if fn.empty:
        print("No false negatives found.")
        return

    enc_col = pick_first_existing(fn, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
    mrn_col = pick_first_existing(fn, ["MRN", "mrn"])

    fn["ENCRYPTED_PAT_ID"] = fn[enc_col].map(normalize_id) if enc_col else ""
    fn["MRN"] = fn[mrn_col].map(normalize_id) if mrn_col else ""

    csvs = list_csvs(NOTES_DIR)
    if not csvs:
        raise IOError("No CSVs found under notes dir: {}".format(NOTES_DIR))

    # Build MRN->ENCRYPTED map if needed
    need_map = fn["ENCRYPTED_PAT_ID"].map(lambda x: 1 if normalize_id(x) == "" else 0).sum() > 0
    id_map = None
    if need_map and (fn["MRN"].map(lambda x: 1 if normalize_id(x) != "" else 0).sum() == 0):
        raise ValueError("FN rows missing ENCRYPTED_PAT_ID and MRN; cannot map IDs.")

    if need_map:
        maps = []
        for p in csvs[:30]:
            try:
                df0 = normalize_cols(read_csv_robust(p, nrows=50, dtype=str, low_memory=False))
            except Exception:
                continue
            mrn_c = pick_first_existing(df0, ["MRN", "mrn"])
            enc_c = pick_first_existing(df0, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
            txt_c = find_text_col(df0)
            nid_c = pick_first_existing(df0, ["NOTE_ID", "NoteID", "NOTEID"])
            if (not mrn_c) or (not enc_c) or (not txt_c) or (not nid_c):
                continue
            df = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
            mrn_vals = df[mrn_c].map(normalize_id)
            enc_vals = df[enc_c].map(normalize_id)
            tmp = pd.DataFrame({"MRN": mrn_vals, "ENCRYPTED_PAT_ID": enc_vals})
            tmp = tmp[(tmp["MRN"] != "") & (tmp["ENCRYPTED_PAT_ID"] != "")].drop_duplicates()
            if len(tmp) > 0:
                maps.append(tmp)
        if not maps:
            raise ValueError("Could not build MRN->ENCRYPTED_PAT_ID map from notes CSVs.")
        id_map = pd.concat(maps, axis=0, ignore_index=True).drop_duplicates()

        fn = fn.merge(id_map, on="MRN", how="left", suffixes=("", "_map"))
        fn["ENCRYPTED_PAT_ID"] = fn["ENCRYPTED_PAT_ID"].where(fn["ENCRYPTED_PAT_ID"] != "", fn["ENCRYPTED_PAT_ID_map"].fillna("").map(normalize_id))
        fn = fn.drop([c for c in fn.columns if c.endswith("_map")], axis=1, errors="ignore")

    fn = fn[fn["ENCRYPTED_PAT_ID"] != ""].copy()
    if fn.empty:
        raise ValueError("No FN rows have ENCRYPTED_PAT_ID after mapping.")

    fn_ids = set(fn["ENCRYPTED_PAT_ID"].tolist())

    out_rows = []

    for p in csvs:
        try:
            df0 = normalize_cols(read_csv_robust(p, nrows=50, dtype=str, low_memory=False))
        except Exception:
            continue

        enc_c = pick_first_existing(df0, ["ENCRYPTED_PAT_ID", "ENCRYPTED_PATID", "ENCRYPTED_PATIENT_ID", "PatientID"])
        nid_c = pick_first_existing(df0, ["NOTE_ID", "NoteID", "NOTEID"])
        ntype_c = pick_first_existing(df0, ["NOTE_TYPE", "NoteType", "TYPE"])
        line_c = pick_first_existing(df0, ["LINE", "Line", "LINE_NUM"])
        txt_c = find_text_col(df0)

        if (not enc_c) or (not nid_c) or (not txt_c):
            continue

        df = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
        df[enc_c] = df[enc_c].map(normalize_id)
        sub = df[df[enc_c].isin(fn_ids)].copy()
        if sub.empty:
            continue

        # Aggregate per note if line exists
        if line_c and line_c in sub.columns:
            try:
                sub[line_c] = sub[line_c].fillna("0").map(lambda x: int(float(str(x).strip())) if str(x).strip() != "" else 0)
            except:
                sub[line_c] = 0
            sub["__text__"] = sub[txt_c].fillna("").map(str)
            sub = sub.sort_values([enc_c, nid_c, line_c])
            agg = {"__text__": lambda x: " ".join([t for t in x if t])}
            if ntype_c and ntype_c in sub.columns:
                agg[ntype_c] = "first"
            grp = sub.groupby([enc_c, nid_c], as_index=False).agg(agg)
            grp = grp.rename(columns={enc_c: "ENCRYPTED_PAT_ID", nid_c: "NOTE_ID"})
            grp["NOTE_TYPE"] = grp[ntype_c] if (ntype_c and ntype_c in grp.columns) else ""
            grp["NOTE_TEXT_FULL"] = grp["__text__"]
        else:
            grp = sub[[enc_c, nid_c, txt_c]].copy()
            grp = grp.rename(columns={enc_c: "ENCRYPTED_PAT_ID", nid_c: "NOTE_ID", txt_c: "NOTE_TEXT_FULL"})
            grp["NOTE_TYPE"] = sub[ntype_c].values if (ntype_c and ntype_c in sub.columns) else ""

        for _, r in grp.iterrows():
            pid = normalize_id(r.get("ENCRYPTED_PAT_ID"))
            note_id = normalize_id(r.get("NOTE_ID"))
            note_type = normalize_id(r.get("NOTE_TYPE"))
            text = r.get("NOTE_TEXT_FULL") or ""
            if not text:
                continue

            hits = 0
            for m in BROAD_RX.finditer(text):
                snip = snippet_around_match(text, m.start(), before=WORDS_BEFORE, after=WORDS_AFTER)
                if snip:
                    out_rows.append({
                        "ENCRYPTED_PAT_ID": pid,
                        "NOTE_ID": note_id,
                        "NOTE_TYPE": note_type,
                        "MATCH_TERM": m.group(0),
                        "SNIPPET": snip,
                        "SOURCE_FILE": os.path.basename(p)
                    })
                    hits += 1
                    if hits >= MAX_HITS_PER_NOTE:
                        break

            if hits == 0 and ALWAYS_PREVIEW_IF_NO_HITS:
                out_rows.append({
                    "ENCRYPTED_PAT_ID": pid,
                    "NOTE_ID": note_id,
                    "NOTE_TYPE": note_type,
                    "MATCH_TERM": "NO_KEYWORD_HIT_PREVIEW",
                    "SNIPPET": preview_snippet(text, n_words=PREVIEW_WORDS),
                    "SOURCE_FILE": os.path.basename(p)
                })

    ensure_dir(OUT_PATH)

    if not out_rows:
        pd.DataFrame(columns=["ENCRYPTED_PAT_ID","NOTE_ID","NOTE_TYPE","MATCH_TERM","SNIPPET","SOURCE_FILE"]).to_csv(OUT_PATH, index=False)
        print("Wrote (no rows):", OUT_PATH)
        return

    out_df = pd.DataFrame(out_rows)

    # Ensure NO MRN leaks
    for c in list(out_df.columns):
        if "MRN" in str(c).upper():
            out_df = out_df.drop([c], axis=1, errors="ignore")

    out_df = out_df.drop_duplicates(subset=["ENCRYPTED_PAT_ID","NOTE_ID","MATCH_TERM","SNIPPET"])
    out_df.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)
    print("Rows:", len(out_df))

if __name__ == "__main__":
    main()
