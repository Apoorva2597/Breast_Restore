import pandas as pd

from normalize.sectionizer import sectionize
from models import SectionedNote
from extractors import extract_all
from aggregate.rules import aggregate_patient
from config import PHASE1_FIELDS


def read_patient_index(path):
    """
    patient_note_index.csv was written by pandas (UTF-8).
    We still guard in case the environment writes/reads differently.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def main():
    df = read_patient_index("patient_note_index.csv")

    # basic hygiene
    for col in ["patient_id", "note_id", "note_type", "note_text"]:
        if col not in df.columns:
            raise RuntimeError("Missing required column in patient_note_index.csv: {}".format(col))

    # pick a deterministic pilot set: first 50 patient_ids in sorted order
    patient_ids = sorted(df["patient_id"].dropna().unique().tolist())[:50]
    print("Pilot patients:", len(patient_ids))

    df = df[df["patient_id"].isin(patient_ids)].copy()

    # parse note_date if present; do not crash if messy
    if "note_date" in df.columns:
        df["note_date_parsed"] = pd.to_datetime(df["note_date"], errors="coerce")
        df = df.sort_values(["patient_id", "note_date_parsed"])
    else:
        df = df.sort_values(["patient_id"])

    patient_rows = []
    evidence_rows = []

    # run patient-by-patient
    for i, pid in enumerate(patient_ids, start=1):
        sub = df[df["patient_id"] == pid]
        all_cands = []

        for _, row in sub.iterrows():
            note_id = str(row["note_id"])
            note_type = str(row["note_type"]) if pd.notnull(row["note_type"]) else ""
            note_date = None
            if "note_date" in row and pd.notnull(row["note_date"]):
                note_date = str(row["note_date"])

            text = row["note_text"]
            if pd.isnull(text):
                continue
            text = str(text)

            # sectionize -> SectionedNote -> extract
            sections = sectionize(text)
            sec = SectionedNote(
                note_id=note_id,
                note_type=note_type,
                sections=sections,
                note_date=note_date
            )

            cands = extract_all(sec)
            if cands:
                all_cands.extend(cands)

        # aggregate to FinalField dict (only PHASE1_FIELDS by design)
        final_map = aggregate_patient(all_cands)

        # patient-level output row (values only)
        out = {"patient_id": pid}
        for field in PHASE1_FIELDS:
            out[field] = final_map[field].value if field in final_map else None
        patient_rows.append(out)

        # evidence log (one row per selected FinalField)
        for field, ff in final_map.items():
            evidence_rows.append({
                "patient_id": pid,
                "field": ff.field,
                "value": ff.value,
                "status": ff.status,
                "evidence": ff.evidence,   # NOTE: may contain PHI; keep separate
                "section": ff.section,
                "note_type": ff.note_type,
                "note_id": ff.note_id,
                "note_date": ff.note_date,
                "rule": ff.rule
            })

        if i % 10 == 0:
            print("Processed patients:", i)

    out_df = pd.DataFrame(patient_rows)
    ev_df = pd.DataFrame(evidence_rows)

    out_df.to_csv("patient_level_phase1_p50.csv", index=False)
    ev_df.to_csv("evidence_log_phase1_p50.csv", index=False)

    print("Wrote patient_level_phase1_p50.csv (rows={})".format(out_df.shape[0]))
    print("Wrote evidence_log_phase1_p50.csv (rows={})".format(ev_df.shape[0]))


if __name__ == "__main__":
    main()
