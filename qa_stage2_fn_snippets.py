#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage2_fn_snippets_from_mismatches.py

UPDATED:
- Uses MRN as primary join key (diagnostic showed MRN↔MRN overlap only)
- Maps MRN → ENCRYPTED_PAT_ID from notes
- Extracts broad Stage2 keyword snippets (15 words before/after)
- NO MRN written to output
- Output: ./_outputs/stage2_fn_keyword_snippets_FINAL_FINAL.csv
"""

from __future__ import print_function
import os
import re
import glob
import pandas as pd

MISMATCH_PATH = "./_outputs/validation_mismatches_STAGE2_ANCHOR_FINAL_FINAL.csv"
NOTES_DIR = "/home/apokol/my_data_Breast/HPI-11526/HPI11256"
OUT_PATH = "./_outputs/stage2_fn_keyword_snippets_FINAL_FINAL.csv"

WORDS_BEFORE = 15
WORDS_AFTER = 15
MAX_HITS_PER_NOTE = 5

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
    raise IOError("Failed to read CSV: {}".format(path))

def normalize_cols(df):
    df.columns = [str(c).strip().replace(u"\xa0"," ") for c in df.columns]
    return df

def normalize_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    if re.match(r"^\d+\.0$", s):
        s = s.split(".")[0]
    return s

def to01(v):
    s = str(v).strip().lower()
    if s in ["1","y","yes","true","t"]:
        return 1
    return 0

def pick_first_existing(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def list_csvs(root):
    return sorted([p for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) if os.path.isfile(p)])

def build_word_spans(text):
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans

def snippet_around(text, start):
    spans = build_word_spans(text)
    if not spans:
        return ""
    idx = 0
    for i,(s,e,_) in enumerate(spans):
        if s <= start < e:
            idx = i
            break
    lo = max(0, idx - WORDS_BEFORE)
    hi = min(len(spans), idx + WORDS_AFTER + 1)
    return " ".join([spans[i][2] for i in range(lo,hi)])

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)

def main():

    mism = normalize_cols(read_csv_robust(MISMATCH_PATH, dtype=str, low_memory=False))
    gold_col = pick_first_existing(mism, ["GOLD_HAS_STAGE2","GOLD_STAGE2"])
    pred_col = pick_first_existing(mism, ["PRED_HAS_STAGE2","HAS_STAGE2"])

    mism["_gold"] = mism[gold_col].map(to01)
    mism["_pred"] = mism[pred_col].map(to01)

    fn = mism[(mism["_gold"]==1) & (mism["_pred"]==0)].copy()
    mrn_col = pick_first_existing(fn, ["MRN","mrn"])
    if not mrn_col:
        raise ValueError("MRN column required in mismatches.")

    fn["MRN"] = fn[mrn_col].map(normalize_id)
    fn_mrns = set([x for x in fn["MRN"] if x!=""])

    note_csvs = list_csvs(NOTES_DIR)
    out_rows = []

    for p in note_csvs:

        try:
            df = normalize_cols(read_csv_robust(p, dtype=str, low_memory=False))
        except:
            continue

        mrn_c = pick_first_existing(df, ["MRN","mrn"])
        enc_c = pick_first_existing(df, ["ENCRYPTED_PAT_ID","ENCRYPTED_PATID","ENCRYPTED_PATIENT_ID","PatientID"])
        txt_c = pick_first_existing(df, ["NOTE_TEXT","NOTE_TEXT_FULL","NOTE_BODY","TEXT"])
        noteid_c = pick_first_existing(df, ["NOTE_ID","NoteID"])

        if not mrn_c or not enc_c or not txt_c or not noteid_c:
            continue

        df["MRN"] = df[mrn_c].map(normalize_id)
        sub = df[df["MRN"].isin(fn_mrns)].copy()
        if sub.empty:
            continue

        for _,r in sub.iterrows():
            text = str(r[txt_c]) if pd.notnull(r[txt_c]) else ""
            if not text:
                continue
            hits = 0
            for m in BROAD_RX.finditer(text):
                snip = snippet_around(text, m.start())
                if snip:
                    out_rows.append({
                        "ENCRYPTED_PAT_ID": normalize_id(r[enc_c]),
                        "NOTE_ID": normalize_id(r[noteid_c]),
                        "MATCH_TERM": m.group(0),
                        "SNIPPET": snip,
                        "SOURCE_FILE": os.path.basename(p)
                    })
                    hits += 1
                    if hits >= MAX_HITS_PER_NOTE:
                        break

    ensure_dir(OUT_PATH)
    if not out_rows:
        pd.DataFrame(columns=["ENCRYPTED_PAT_ID","NOTE_ID","MATCH_TERM","SNIPPET","SOURCE_FILE"]).to_csv(OUT_PATH,index=False)
        print("No hits.")
        return

    out_df = pd.DataFrame(out_rows).drop_duplicates()
    out_df.to_csv(OUT_PATH,index=False)
    print("Wrote:",OUT_PATH)
    print("Rows:",len(out_df))

if __name__=="__main__":
    main()
