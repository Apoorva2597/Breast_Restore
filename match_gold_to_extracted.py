# match_gold_to_extracted.py
#
# Link gold_variables.csv (per patient) to our extracted notes
# via exact match on (Age, BMI), and print debugging stats.
#
# Output: gold_extracted_links.csv

import csv

# Input files
GOLD_FILE = "gold_variables.csv"
EXTRACTED_FILE = "age_bmi_for_linking.csv"
OUTPUT_FILE = "gold_extracted_links.csv"

# --- COLUMN NAMES in GOLD FILE ---
GOLD_ID_COL = "1. PatientID"   # your team’s synthetic patient ID
GOLD_AGE_COL = "4. Age"        # integer ages in the gold sheet
GOLD_BMI_COL = "5. BMI"        # numeric BMI in the gold sheet

# --- COLUMN NAMES in EXTRACTED FILE ---
EXT_NOTE_ID_COL = "note_id"
EXT_TYPE_COL = "note_type"
EXT_DATE_COL = "note_date"
EXT_AGE_COL = "Age_DOS"
EXT_BMI_COL = "BMI"


def normalize_age(raw):
    """
    Convert age to a stable integer string.

    Examples:
      "49"   -> "49"
      "49.0" -> "49"
    """
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
    """
    Convert BMI to a 1-decimal string for exact matching.

    Examples:
      "33.8"   -> "33.8"
      "33.75"  -> "33.8"
      " 33 "   -> "33.0"
    """
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
    # -------------------------------------------------
    # 1. Load GOLD dataset and build (Age, BMI) index
    # -------------------------------------------------
    gold_index = {}   # key = (age_norm, bmi_norm) -> list of gold rows
    gold_pairs = set()
    total_gold_rows = 0
    usable_gold_rows = 0

    with open(GOLD_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_gold_rows += 1
            age_norm = normalize_age(row.get(GOLD_AGE_COL))
            bmi_norm = normalize_bmi(row.get(GOLD_BMI_COL))

            if age_norm is None or bmi_norm is None:
                continue

            usable_gold_rows += 1
            key = (age_norm, bmi_norm)
            gold_pairs.add(key)
            gold_index.setdefault(key, []).append(row)

    print("GOLD: total rows = {}, usable Age+BMI = {}".format(
        total_gold_rows, usable_gold_rows
    ))
    print("GOLD: distinct (Age,BMI) pairs = {}".format(len(gold_pairs)))

    # -------------------------------------------------
    # 2. Load EXTRACTED note-level file
    # -------------------------------------------------
    extracted_rows = []
    extracted_pairs = set()
    total_extracted_rows = 0
    usable_extracted_rows = 0

    with open(EXTRACTED_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_extracted_rows += 1
            age_norm = normalize_age(row.get(EXT_AGE_COL))
            bmi_norm = normalize_bmi(row.get(EXT_BMI_COL))

            if age_norm is not None and bmi_norm is not None:
                usable_extracted_rows += 1
                extracted_pairs.add((age_norm, bmi_norm))

            extracted_rows.append(row)

    print("EXTRACTED: total rows = {}, usable Age+BMI = {}".format(
        total_extracted_rows, usable_extracted_rows
    ))
    print("EXTRACTED: distinct (Age,BMI) pairs = {}".format(len(extracted_pairs)))

    # -------------------------------------------------
    # 3. Check overlap of (Age,BMI) keys
    # -------------------------------------------------
    overlap_pairs = gold_pairs.intersection(extracted_pairs)
    print("Overlap: distinct (Age,BMI) pairs in common = {}".format(
        len(overlap_pairs)
    ))

    # If there is no overlap at all, matching will produce 0 rows.
    if not overlap_pairs:
        print("WARNING: No shared (Age,BMI) pairs between gold and extracted.")
        print("         0 matches is expected with exact matching.")
        # We still continue and run matching anyway, just in case.

    # -------------------------------------------------
    # 4. Do the actual matching (exact Age,BMI)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 5. Write matches out
    # -------------------------------------------------
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
