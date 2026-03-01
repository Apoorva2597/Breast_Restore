#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combine_qa_deid_snippets.py
Python 3.6.8 compatible

Reads:
  ./QA_DEID_BUNDLES/<patient_id>/note_*.txt

Writes:
  ./_outputs/FP_deid_note_snippets.csv
"""

from __future__ import print_function
import os
import pandas as pd

ROOT = os.path.abspath(".")
QA_DIR = os.path.join(ROOT, "QA_DEID_BUNDLES")
OUT_PATH = os.path.join(ROOT, "_outputs", "FP_deid_note_snippets.csv")

rows = []

for pid in os.listdir(QA_DIR):
    pid_dir = os.path.join(QA_DIR, pid)

    if not os.path.isdir(pid_dir):
        continue
    if pid == "logs":
        continue

    for fname in os.listdir(pid_dir):
        if not fname.startswith("note_") or not fname.endswith(".txt"):
            continue

        fpath = os.path.join(pid_dir, fname)

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        snippet = text.strip()[:800]

        rows.append({
            "patient_id": pid,
            "note_file": fname,
            "snippet": snippet
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)

print("Wrote:", OUT_PATH)
print("Total snippets:", len(df))
