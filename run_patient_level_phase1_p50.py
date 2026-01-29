# run_patient_level_phase1.py
import argparse
import pandas as pd

from normalize.sectionizer import sectionize
from models import SectionedNote
from extractors import extract_all
from aggregate.rules import aggregate_patient
from config import PHASE1_FIELDS


def read_patient_index(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="patient_note_index.csv")
    ap.add_argument("--n", type=int, default=50, help="Pilot N patients (ignored if --all)")
    ap.add_argument("--all", action="store_true", help="Run all patients")
    ap.add_argument("--out_prefix", default="patient_level_phase1")
    args = ap.parse_args()

    df = read_patient_index(args.index)

    for col in ["patient_id", "note_id", "note_type", "note_text"]:
        if col not in df.columns:
            raise RuntimeError("Missing required column in {}: {}".format(args.index, col))

    all_patient_ids = sorted(df["patient_id"].dropna().unique().tolist())
    if args.all:
        patient_ids = all_patient_ids
        print("Patients: ALL ({})".format(len(patient_ids)))
    else:
        patient_ids = all_patient_ids[: args.n]
        print("Patients: PILOT ({})".format(len(patient_ids)))

    df = df[df["patient_id"].isin(patient_ids)].copy()

    if "note_date" in df.columns:
        df["note_date_parsed"] = pd.to_datetime(df["note_date"], errors="coerce")
        df = df.sort_values(["patient_id", "note_date_parsed"])
    else:
        df = df.sort_values(["patient_id"])

    patient_rows = []
    evidence_rows = []

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

        final_map = aggregate_patient(all_cands)

        out = {"patient_id": pid}
        for field in PHASE1_FIELDS:
            out[field] = final_map[field].value if field in final_map else None
        patient_rows.append(out)

        for _, ff in final_map.items():
            evidence_rows.append({
                "patient_id": pid,
                "field": ff.field,
                "value": ff.value,
                "status": ff.status,
                "evidence": ff.evidence,   # may contain PHI
                "section": ff.section,
                "note_type": ff.note_type,
                "note_id": ff.note_id,
                "note_date": ff.note_date,
                "rule": ff.rule
            })

        if i % 50 == 0:
            print("Processed patients:", i)

    out_df = pd.DataFrame(patient_rows)
    ev_df = pd.DataFrame(evidence_rows)

    suffix = "all" if args.all else "p{}".format(len(patient_ids))
    out_df.to_csv("{}_{}.csv".format(args.out_prefix, suffix), index=False)
    ev_df.to_csv("evidence_log_phase1_{}.csv".format(suffix), index=False)

    print("Wrote {}_{}.csv (rows={})".format(args.out_prefix, suffix, out_df.shape[0]))
    print("Wrote evidence_log_phase1_{}.csv (rows={})".format(suffix, ev_df.shape[0]))


if __name__ == "__main__":
    main()
