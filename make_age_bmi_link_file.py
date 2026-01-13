# make_age_bmi_link_file.py
import csv

# Input: Phase 2 output with patient_id, note_id, field, value, etc.
INPUT = "all_phase2_final.csv"       # or the merged Phase 2 file
OUTPUT = "age_bmi_for_linking.csv"

# Column names in the Phase 2 file
PATIENT_ID_COL = "patient_id"
NOTE_ID_COL = "note_id"
NOTE_TYPE_COL = "note_type"
NOTE_DATE_COL = "note_date"
FIELD_COL = "field"
VALUE_COL = "value"
STATUS_COL = "status"
CONF_COL = "confidence"


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    # 1. Read all Phase 2 rows, keep only BMI / Age_DOS (and not 'denied')
    rows = []
    with open(INPUT, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row.get(FIELD_COL, "")
            if field not in ("BMI", "Age_DOS"):
                continue

            status = (row.get(STATUS_COL, "") or "").lower()
            if status == "denied":
                continue

            rows.append(row)

    # 2. For each (note_id, field), keep the highest-confidence value
    best_by_note_field = {}  # (note_id, field) -> row
    for row in rows:
        note_id = row.get(NOTE_ID_COL, "")
        field = row.get(FIELD_COL, "")
        key = (note_id, field)

        conf = _safe_float(row.get(CONF_COL))
        prev = best_by_note_field.get(key)
        if prev is None:
            best_by_note_field[key] = row
        else:
            prev_conf = _safe_float(prev.get(CONF_COL))
            if conf > prev_conf:
                best_by_note_field[key] = row

    # 3. Pivot to one row per (patient_id, note_id)
    #    Columns: patient_id, note_id, note_type, note_date, Age_DOS, BMI
    per_note = {}  # (patient_id, note_id) -> dict
    for (note_id, field), row in best_by_note_field.items():
        patient_id = row.get(PATIENT_ID_COL, "")

        key = (patient_id, note_id)
        if key not in per_note:
            per_note[key] = {
                PATIENT_ID_COL: patient_id,
                NOTE_ID_COL: note_id,
                NOTE_TYPE_COL: row.get(NOTE_TYPE_COL, ""),
                NOTE_DATE_COL: row.get(NOTE_DATE_COL, ""),
                "Age_DOS": "",
                "BMI": "",
            }

        if field == "BMI":
            per_note[key]["BMI"] = row.get(VALUE_COL, "")
        elif field == "Age_DOS":
            per_note[key]["Age_DOS"] = row.get(VALUE_COL, "")

    # 4. Write out CSV
    out_fields = [
        PATIENT_ID_COL,
        NOTE_ID_COL,
        NOTE_TYPE_COL,
        NOTE_DATE_COL,
        "Age_DOS",
        "BMI",
    ]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for key in per_note:
            writer.writerow(per_note[key])


if __name__ == "__main__":
    main()
