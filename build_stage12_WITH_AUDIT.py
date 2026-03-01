# build_stage12_WITH_AUDIT.py
# Python 3.6.8 compatible
# STAGING ONLY — NO VALIDATION
# Outputs (unchanged):
#   _outputs/patient_stage_summary.csv
#   _outputs/stage2_fn_raw_note_snippets.csv

import os
import re
import csv
import pandas as pd

INPUT_NOTES = "_staging_inputs/HPI11526 Operation Notes.csv"
OUTPUT_SUMMARY = "_outputs/patient_stage_summary.csv"
OUTPUT_AUDIT = "_outputs/stage2_fn_raw_note_snippets.csv"

# ----------------------------
# REGEX DEFINITIONS
# ----------------------------

EXCHANGE_STRICT = re.compile(
    r"""
    (
        (underwent|performed|taken\s+to\s+the\s+OR|returned\s+to\s+the\s+operating\s+room)
        .{0,120}?
        (exchange|removal|removed|replacement)
        .{0,120}?
        (tissue\s+expander|implant)
    )
    |
    (
        (exchange|removal|removed|replacement)
        .{0,80}?
        (tissue\s+expander|implant)
        .{0,80}?
        (for|with)
        .{0,80}?
        (permanent\s+)?(silicone|saline)?\s*(implant)
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

INTRAOP_SIGNALS = re.compile(
    r"""
    (estimated\s+blood\s+loss|EBL|
     specimen(s)?\s+sent|
     drains?\s+(placed|inserted)|
     anesthesia|
     incision\s+made|
     pocket\s+created|
     implant\s+placed|
     expander\s+removed|
     capsulotomy|capsulectomy|
     \bml\b\s+(removed|placed|instilled))
    """,
    re.IGNORECASE | re.VERBOSE,
)

NEGATIVE_CONTEXT = re.compile(
    r"""
    (scheduled\s+for|
     will\s+undergo|
     planning\s+to|
     considering|
     history\s+of|
     status\s+post|
     \bs/p\b|
     discussed|
     interested\s+in|
     plan:|
     follow[-\s]?up|
     here\s+for\s+follow)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ----------------------------
# LOGIC
# ----------------------------

def is_true_exchange(note_text):
    if not EXCHANGE_STRICT.search(note_text):
        return False

    if NEGATIVE_CONTEXT.search(note_text):
        return False

    if not INTRAOP_SIGNALS.search(note_text):
        return False

    return True


def get_snippet(text, match_obj, window=200):
    start = max(match_obj.start() - window, 0)
    end = min(match_obj.end() + window, len(text))
    return text[start:end].replace("\n", " ").strip()


# ----------------------------
# MAIN
# ----------------------------

def main():

    # Fix UnicodeDecodeError (Python 3.6 safe handling)
    df = pd.read_csv(
        INPUT_NOTES,
        encoding="latin1",
        engine="python"
    )

    required_cols = set(["ENCRYPTED_PAT_ID", "NOTE_ID", "NOTE_TEXT"])
    if not required_cols.issubset(set(df.columns)):
        raise ValueError("Input must contain columns: ENCRYPTED_PAT_ID, NOTE_ID, NOTE_TEXT")

    stage2_patients = set()
    audit_rows = []

    total_events = 0

    for _, row in df.iterrows():

        pat_id = row["ENCRYPTED_PAT_ID"]
        note_id = row["NOTE_ID"]
        text = str(row["NOTE_TEXT"])

        total_events += 1

        if is_true_exchange(text):

            stage2_patients.add(pat_id)

            for match in EXCHANGE_STRICT.finditer(text):
                snippet = get_snippet(text, match)

                audit_rows.append({
                    "ENCRYPTED_PAT_ID": pat_id,
                    "NOTE_ID": note_id,
                    "MATCH_TERM": "exchange_strict",
                    "SNIPPET": snippet,
                    "SOURCE_FILE": os.path.basename(INPUT_NOTES),
                })

    # Patient-level summary
    unique_patients = df["ENCRYPTED_PAT_ID"].dropna().unique()

    summary_rows = []
    for pid in unique_patients:
        summary_rows.append({
            "ENCRYPTED_PAT_ID": pid,
            "STAGE2_ANCHOR": 1 if pid in stage2_patients else 0
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_AUDIT, index=False)

    print("Staging complete.")
    print("Patients:", len(unique_patients))
    print("Events:", total_events)


if __name__ == "__main__":
    main()
