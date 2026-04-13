#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_dead_extractors.py
Python 3.6.8 compatible

PURPOSE:
    Find out WHY these variables always output zero positives:
        - PBS_Breast Reduction
        - PBS_Mastopexy
        - PBS_Other
        - Stage2_MinorComp
        - Stage2_Rehospitalization

    For each target, we:
    1) Search raw note text for the trigger patterns
    2) If found, check whether suppression rules are killing it
    3) Print a sample of hits and suppressions so we can see what is happening

USAGE:
    Edit NOTE_FILES below to point to your note CSVs on CEDAR.
    Then run:
        python diagnose_dead_extractors.py > diag_output.txt 2>&1

OUTPUT:
    diag_output.txt  -- paste this back and we will fix the rules
"""

from __future__ import print_function
import os
import re
import sys
import pandas as pd

# ============================================================
# EDIT THESE PATHS TO MATCH YOUR CEDAR SETUP
# ============================================================
BASE_DIR = "/home/apokol/Breast_Restore"

NOTE_FILES = [
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Clinic Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Operation Notes.csv"),
    os.path.join(BASE_DIR, "_staging_inputs", "HPI11526 Inpatient Notes.csv"),
]

# If staging_inputs doesn't exist try the original data dir
FALLBACK_GLOBS = [
    os.path.join(BASE_DIR, "**", "HPI11526*Notes.csv"),
    os.path.join(BASE_DIR, "**", "HPI11526*notes.csv"),
]

MAX_EXAMPLES = 8   # how many snippets to print per pattern
SNIPPET_CHARS = 300

# ============================================================
# TARGET PATTERNS
# Each entry: (variable_name, trigger_patterns, suppression_patterns)
# ============================================================

TARGETS = [
    {
        "name": "PBS_Breast Reduction",
        "triggers": [
            r"\bbreast\s+reduction\b",
            r"\breduction\s+mammaplasty\b",
            r"\breduction\s+mammoplasty\b",
        ],
        "suppressions": [
            r"\bno\s+prior\s+breast\s+surgery\b",
            r"\bno\s+history\s+of\s+breast\s+surgery\b",
            r"\bdenies\s+prior\s+breast\s+surgery\b",
            r"\bfamily\s+history\b",
            r"\breview\s+of\s+systems\b",
            r"\ballerg(y|ies)\b",
            r"\brisk\s+reduction\b",           # risk reduction != breast reduction
            r"\bcontralateral\s+risk\s+reduction\b",
            r"\bweight\s+reduction\b",
        ],
        "history_required": [
            r"\bs/p\b",
            r"\bstatus\s+post\b",
            r"\bhistory\s+of\b",
            r"\bh/o\b",
            r"\bprior\b",
            r"\bprevious\b",
            r"\bpreviously\b",
            r"\bunderwent\b",
            r"\bpast\s+surgical\s+history\b",
        ],
    },
    {
        "name": "PBS_Mastopexy",
        "triggers": [
            r"\bmastopexy\b",
            r"\bbreast\s+lift\b",
        ],
        "suppressions": [
            r"\bno\s+prior\s+breast\s+surgery\b",
            r"\bno\s+history\s+of\s+breast\s+surgery\b",
            r"\bfamily\s+history\b",
            r"\breview\s+of\s+systems\b",
        ],
        "history_required": [
            r"\bs/p\b",
            r"\bstatus\s+post\b",
            r"\bhistory\s+of\b",
            r"\bh/o\b",
            r"\bprior\b",
            r"\bprevious\b",
            r"\bpreviously\b",
            r"\bunderwent\b",
            r"\bpast\s+surgical\s+history\b",
        ],
    },
    {
        "name": "PBS_Other",
        "triggers": [
            r"\bexcisional\s+biopsy\b",
            r"\bopen\s+breast\s+biopsy\b",
            r"\bbenign\s+excision\b",
        ],
        "suppressions": [
            r"\bfamily\s+history\b",
            r"\breview\s+of\s+systems\b",
            r"\bno\s+prior\s+breast\s+surgery\b",
        ],
        "history_required": [
            r"\bs/p\b",
            r"\bstatus\s+post\b",
            r"\bhistory\s+of\b",
            r"\bprior\b",
            r"\bprevious\b",
            r"\bunderwent\b",
        ],
    },
    {
        "name": "Stage2_MinorComp",
        "triggers": [
            r"\bhematoma\b",
            r"\bseroma\b",
            r"\bwound\s+dehiscence\b",
            r"\bdehiscence\b",
            r"\bcellulitis\b",
            r"\babscess\b",
            r"\bfat\s+necrosis\b",
            r"\bcapsular\s+contracture\b",
            r"\bimplant\s+malposition\b",
            r"\bwound\s+infection\b",
            r"\bwound\s+breakdown\b",
        ],
        "suppressions": [
            r"\brisk\s+of\b",
            r"\bcomplication\s+risk\b",
            r"\bwarn(ed|ing)\b",
            r"\bcounseled\b",
            r"\bdiscussed\b",
            r"\bpossible\b",
            r"\bpotential\b",
            r"\bprevent\b",
            r"\bno\s+hematoma\b",
            r"\bno\s+seroma\b",
            r"\bno\s+dehiscence\b",
            r"\bno\s+infection\b",
            r"\bno\s+evidence\s+of\b",
            r"\bwithout\s+complication\b",
            r"\buncomplicated\b",
        ],
        "history_required": [],  # not required for complications
    },
    {
        "name": "Stage2_Rehospitalization",
        "triggers": [
            r"\breadmit(ted|sion)?\b",
            r"\bre-?admit(ted|sion)?\b",
            r"\brehospitali[sz](ed|ation)?\b",
            r"\breturn(ed)?\s+to\s+hospital\b",
            r"\breturn(ed)?\s+to\s+(the\s+)?ed\b",
            r"\bemergency\s+(department|room|visit)\b",
        ],
        "suppressions": [
            r"\bno\s+readmission\b",
            r"\bno\s+rehospitali[sz]ation\b",
            r"\bdenies\b.{0,40}\breadmit\b",
            r"\binstruction(s)?\s+to\s+return\b",
            r"\bif\s+you\s+(have|develop|experience)\b",
            r"\bprecaution(s)?\b",
            r"\bdischarge\s+instruction\b",
        ],
        "history_required": [],
    },
]

# ============================================================
# HELPERS
# ============================================================

def snippet_around(text, match_start, match_end, chars=SNIPPET_CHARS):
    half = chars // 2
    lo = max(0, match_start - half)
    hi = min(len(text), match_end + half)
    s = text[lo:hi]
    s = re.sub(r"\s+", " ", s).strip()
    if lo > 0:
        s = "..." + s
    if hi < len(text):
        s = s + "..."
    return s


def has_any(patterns, text):
    for p in patterns:
        if re.search(p, text, re.I):
            return True
    return False


def read_notes(paths):
    """
    Try to load note CSVs. Returns a DataFrame with
    columns: note_id, note_type, note_text
    """
    from glob import glob

    found = []
    for p in paths:
        if os.path.isfile(p):
            found.append(p)

    if not found:
        print("Primary paths not found, trying fallback globs...")
        for pattern in FALLBACK_GLOBS:
            matched = glob(pattern, recursive=True)
            found.extend(matched)

    if not found:
        print("ERROR: No note files found. Edit NOTE_FILES in the script.")
        sys.exit(1)

    print("Loading note files:")
    frames = []
    for p in found:
        print("  " + p)
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(
                    p, dtype=str,
                    encoding=enc, engine="python"
                )
                df.columns = [c.strip().upper() for c in df.columns]

                # find text col
                text_col = None
                for c in ["NOTE_TEXT", "TEXT", "NOTE_BODY",
                           "NOTE_CONTENT", "FULLTEXT", "FULL_TEXT"]:
                    if c in df.columns:
                        text_col = c
                        break
                if text_col is None:
                    print("    WARNING: no text column found in " + p)
                    break

                # find id / type cols
                id_col = None
                for c in ["NOTE_ID", "NOTEID", "ID"]:
                    if c in df.columns:
                        id_col = c
                        break

                type_col = None
                for c in ["NOTE_TYPE", "NOTE_TYPE_NAME",
                           "TYPE", "NOTE_CATEGORY"]:
                    if c in df.columns:
                        type_col = c
                        break

                sub = pd.DataFrame()
                sub["note_text"] = df[text_col].fillna("").astype(str)
                sub["note_id"]   = df[id_col].astype(str) if id_col else ""
                sub["note_type"] = df[type_col].astype(str) if type_col else ""
                sub["source"]    = os.path.basename(p)
                frames.append(sub)
                print("    Loaded {:,} rows (enc={})".format(len(sub), enc))
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print("    ERROR reading {}: {}".format(p, e))
                break

    if not frames:
        print("ERROR: Could not load any notes.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["note_text"].str.strip() != ""]
    print("\nTotal notes loaded: {:,}".format(len(combined)))
    return combined


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def diagnose(notes_df, target):
    name = target["name"]
    triggers = target["triggers"]
    suppressions = target["suppressions"]
    hist_req = target["history_required"]

    print("\n" + "=" * 70)
    print("VARIABLE: {}".format(name))
    print("=" * 70)

    total_notes = len(notes_df)
    trigger_hits = []       # (note_idx, trigger_pat, match_start, match_end, text)
    suppressed = []
    no_history = []
    accepted = []

    for idx, row in notes_df.iterrows():
        text = row["note_text"]
        if not text:
            continue

        for tpat in triggers:
            m = re.search(tpat, text, re.I)
            if m:
                trigger_hits.append({
                    "idx": idx,
                    "pat": tpat,
                    "start": m.start(),
                    "end": m.end(),
                    "text": text,
                    "note_type": row["note_type"],
                    "source": row["source"],
                })
                break  # one hit per note is enough

    print("\nNotes with trigger match: {} / {}".format(
        len(trigger_hits), total_notes))

    if not trigger_hits:
        print("\n>>> DIAGNOSIS: Trigger patterns NEVER match any note text.")
        print(">>> The words are simply not present in your notes.")
        print(">>> ACTION: We need to broaden the patterns.")
        return

    # For each hit, check suppression and history
    for h in trigger_hits:
        ctx = snippet_around(h["text"], h["start"], h["end"], 400)
        ctx_low = ctx.lower()

        suppressed_by = None
        for spat in suppressions:
            if re.search(spat, ctx, re.I):
                suppressed_by = spat
                break

        if suppressed_by:
            suppressed.append((h, suppressed_by, ctx))
            continue

        if hist_req:
            has_hist = has_any(hist_req, ctx)
            if not has_hist:
                no_history.append((h, ctx))
                continue

        accepted.append((h, ctx))

    print("  -> Suppressed by rules:     {}".format(len(suppressed)))
    print("  -> No history context:      {}".format(len(no_history)))
    print("  -> Would be accepted:       {}".format(len(accepted)))

    # ---- Diagnosis ----
    if len(accepted) == 0 and len(trigger_hits) > 0:
        if len(suppressed) > len(no_history):
            print("\n>>> DIAGNOSIS: Patterns fire but SUPPRESSION rules kill everything.")
            print(">>> ACTION: Suppression rules are too aggressive. See examples below.")
        else:
            print("\n>>> DIAGNOSIS: Patterns fire but HISTORY requirement kills everything.")
            print(">>> ACTION: history_required filter is too strict. See examples below.")
    elif len(accepted) > 0:
        print("\n>>> DIAGNOSIS: Extractor WOULD produce {} positives.".format(len(accepted)))
        print(">>> ACTION: Bug is likely DOWNSTREAM (aggregation or column mapping).")

    # ---- Show suppression examples ----
    if suppressed:
        print("\n--- SUPPRESSED examples (first {}) ---".format(
            min(MAX_EXAMPLES, len(suppressed))))
        for h, spat, ctx in suppressed[:MAX_EXAMPLES]:
            print("  Note type : {}".format(h["note_type"]))
            print("  Source    : {}".format(h["source"]))
            print("  Killed by : {}".format(spat))
            print("  Snippet   : {}".format(ctx[:300]))
            print()

    # ---- Show no-history examples ----
    if no_history:
        print("\n--- NO HISTORY CONTEXT examples (first {}) ---".format(
            min(MAX_EXAMPLES, len(no_history))))
        for h, ctx in no_history[:MAX_EXAMPLES]:
            print("  Note type : {}".format(h["note_type"]))
            print("  Source    : {}".format(h["source"]))
            print("  Snippet   : {}".format(ctx[:300]))
            print()

    # ---- Show accepted examples ----
    if accepted:
        print("\n--- WOULD-BE ACCEPTED examples (first {}) ---".format(
            min(MAX_EXAMPLES, len(accepted))))
        for h, ctx in accepted[:MAX_EXAMPLES]:
            print("  Note type : {}".format(h["note_type"]))
            print("  Source    : {}".format(h["source"]))
            print("  Snippet   : {}".format(ctx[:300]))
            print()


def main():
    print("=" * 70)
    print("DEAD EXTRACTOR DIAGNOSTIC")
    print("=" * 70)

    notes_df = read_notes(NOTE_FILES)

    for target in TARGETS:
        diagnose(notes_df, target)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("Paste the full output back and we will fix the rules.")
    print("=" * 70)


if __name__ == "__main__":
    main()
