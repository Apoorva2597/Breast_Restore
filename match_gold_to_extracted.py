# match_gold_to_extracted.py
#
# Link gold_variables.csv (per patient) to our extracted notes
# via exact match on (Age, BMI).
#
# Output: gold_extracted_links.csv

import csv

# Input files
GOLD_FILE = "gold_variables.csv"
EXTRACTED_FILE = "age_bmi_for_linking.csv"
OUTPUT_FILE = "gold_extracted_links.csv"

# --- COLUMN NAMES in GOLD FILE ---
GOLD_ID_COL = "PatientID"   # your team’s synthetic patient ID
GOLD_AGE_COL = "Age"        # integer ages
GOLD_BMI_COL = "BMI"        # numeric BMI

# --- COLUMN NAMES in EXTRACTED FILE ---
EXT_NOTE_ID_COL = "note_id"
EXT_TYPE_COL = "note_type"
EXT_DATE_COL = "note_date"
EXT_AGE_COL = "Age_DOS"
EXT_BMI_COL = "BMI"


def normalize_age(raw):
    """Convert age to stable integer string."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return None


def normalize_bmi(raw):
    """Convert BMI to 1-decimal string for exact matching."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        v = float(s)
        return "{:.1f}".format(v)
    except ValueError:
        return None


def main():
    # ----------------------------------------------
    # 1. Load GOLD dataset
    # ----------------------------------------------
    gold_index = {}   # key = (Age, BMI)
    gold_rows = []

    with open(GOLD_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age_norm = normalize_age(row.get(GOLD_AGE_COL))
            bmi_norm = normalize_bmi(row.get(GOLD_BMI_COL))

            if age_norm is None or bmi_norm is None:
                continue

            key = (age_norm, bmi_norm)
            gold_rows.append(row)
            gold_index.setdefault(key, []).append(row)

    print(
        "Loaded {} gold rows ({} usable Age & BMI).".format(
            len(gold_rows),
            sum(
                1 for r in gold_rows
                if normalize_age(r.get(GOLD_AGE_COL)) is not None
                and normalize_bmi(r.get(GOLD_BMI_COL)) is not None
            ),
        )
    )

    # ----------------------------------------------
    # 2. Load our EXTRACTED note-level file
    # ----------------------------------------------
    extracted_rows = []
    with open(EXTRACTED_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            extracted_rows.append(row)

    print("Loaded {} extracted rows.".format(len(extracted_rows)))

    # ----------------------------------------------
    # 3. Try to match on exact (Age, BMI)
    # ----------------------------------------------
    matches = []

    for row in extracted_rows:
        age_norm = normalize_age(row.get(EXT_AGE_COL))
        bmi_norm = normalize_bmi(row.get(EXT_BMI_COL))

        if age_norm is None or bmi_norm is None:
            continue

        key = (age_norm, bmi_norm)
        gold_candidates = gold_index.get(key, [])

        for g in gold_candidates:
            matches.append({
                "gold_patient_id": g.get(GOLD_ID_COL, ""),
                "gold_age": g.get(GOLD_AGE_COL, ""),
                "gold_bmi": g.get(GOLD_BMI_COL, ""),
                "note_id": row.get(EXT_NOTE_ID_COL, ""),
                "note_type": row.get(EXT_TYPE_COL, ""),
                "note_date": row.get(EXT_DATE_COL, ""),
                "extracted_age": row.get(EXT_AGE_COL, ""),
                "extracted_bmi": row.get(EXT_BMI_COL, ""),
            })

    # ----------------------------------------------
    # 4. Write matches out
    # ----------------------------------------------
    fieldnames = [
        "gold_patient_id",
        "gold_age",
        "gold_bmi",
        "note_id",
        "note_type",
        "note_date",
        "extracted_age",
        "extracted_bmi",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for m in matches:
            writer.writerow(m)

    print("Wrote {} matched rows → {}".format(len(matches), OUTPUT_FILE))


if __name__ == "__main__":
    main()
