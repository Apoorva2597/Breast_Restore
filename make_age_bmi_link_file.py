# make_age_bmi_link_file.py
import csv

INPUT = "all_phase2_final.csv"       # Phase 2 resolved file
OUTPUT = "age_bmi_for_linking.csv"

FIELD_COL = "field"
VALUE_COL = "value"
NOTE_ID_COL = "note_id"
NOTE_TYPE_COL = "note_type"
NOTE_DATE_COL = "note_date"
STATUS_COL = "status"
CONF_COL = "confidence"


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    # 1. Read Phase 2 and keep only BMI / Age_DOS that are not 'denied'
    rows = []
    with open(INPUT, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = (row.get(FIELD_COL) or "").strip()
            if field not in ("BMI", "Age_DOS"):
                continue

            status = (row.get(STATUS_COL) or "").strip().lower()
            if status == "denied":
                continue

            rows.append(row)

    # 2. For each (note_id, field), keep the highest-confidence value
    best = {}  # (note_id, field) -> row
    for row in rows:
        key = (row.get(NOTE_ID_COL), row.get(FIELD_COL))
        conf = safe_float(row.get(CONF_COL))
        prev = best.get(key)
        if prev is None or conf > safe_float(prev.get(CONF_COL)):
            best[key] = row

    # 3. Pivot to one row per note: note_id, note_type, note_date, Age_DOS, BMI
    per_note = {}  # note_id -> dict
    for (note_id, field), row in best.items():
        if not note_id:
            continue

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

        val = row.get(VALUE_COL, "")
        if field == "BMI":
            d["BMI"] = val
        elif field == "Age_DOS":
            d["Age_DOS"] = val

    # 4. Write out CSV
    out_fields = [NOTE_ID_COL, NOTE_TYPE_COL, NOTE_DATE_COL, "Age_DOS", "BMI"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for note_id in sorted(per_note.keys()):
            writer.writerow(per_note[note_id])

    print("Wrote {} rows to {}".format(len(per_note), OUTPUT))


if __name__ == "__main__":
    main()
