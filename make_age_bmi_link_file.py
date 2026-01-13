# make_age_bmi_link_file.py
import csv

INPUT = "all_phase2_final.csv"       # or concat of op/clinic/inpatient
OUTPUT = "age_bmi_for_linking.csv"

# Adjust these if your column order is different
FIELD_COL = "field"
VALUE_COL = "value"
NOTE_ID_COL = "note_id"
NOTE_TYPE_COL = "note_type"
NOTE_DATE_COL = "note_date"
STATUS_COL = "status"
CONF_COL = "confidence"


def main():
    # 1. Read everything and keep only BMI / Age_DOS
    rows = []
    with open(INPUT, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row[FIELD_COL]
            if field not in ("BMI", "Age_DOS"):
                continue
            if row.get(STATUS_COL, "").lower() == "denied":
                continue
            rows.append(row)

    # 2. For each (note_id, field), keep the highest-confidence value
    best = {}  # (note_id, field) -> row
    for row in rows:
        key = (row[NOTE_ID_COL], row[FIELD_COL])
        conf = float(row.get(CONF_COL, "0") or "0")
        prev = best.get(key)
        if prev is None or conf > float(prev.get(CONF_COL, "0") or "0"):
            best[key] = row

    # 3. Pivot to one row per note: note_id, note_type, note_date, Age_DOS, BMI
    per_note = {}
    for (note_id, field), row in best.items():
        d = per_note.setdefault(
            note_id,
            {
                NOTE_ID_COL: note_id,
                NOTE_TYPE_COL: row.get(NOTE_TYPE_COL, ""),
                NOTE_DATE_COL: row.get(NOTE_DATE_COL, ""),
                "Age_DOS": "",
                "BMI": "",
            },
        )
        if field == "BMI":
            d["BMI"] = row[VALUE_COL]
        elif field == "Age_DOS":
            d["Age_DOS"] = row[VALUE_COL]

    # 4. Write out CSV
    out_fields = [NOTE_ID_COL, NOTE_TYPE_COL, NOTE_DATE_COL, "Age_DOS", "BMI"]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for note_id, d in per_note.items():
            writer.writerow(d)


if __name__ == "__main__":
    main()
