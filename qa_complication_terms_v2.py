# qa_complication_terms_v2.py
# Python 3.6.8 compatible
import argparse
import re
import sys
import pandas as pd

PATIENT_INDEX_CSV = "patient_note_index.csv"
SNIPPET_CHARS = 180

# -----------------------
# Term dictionaries (same as yours)
# -----------------------
TERM_GROUPS = [
    ("Recon_Hematoma", [r"\bhematoma\b", r"\bpost[- ]op(erative)?\s+hematoma\b"]),
    ("Recon_Seroma", [r"\bseroma\b"]),
    ("Recon_Wound_Dehiscence", [
        r"\bdehiscen(ce|t)\b",
        r"\bwound\s+dehiscen(ce|t)\b",
        r"\bwound\s+separation\b",
        r"\bincision(al)?\s+separation\b",
    ]),
    ("Recon_Wound_Infection", [
        r"\bwound\s+infection\b",
        r"\bsurgical\s+site\s+infection\b",
        r"\bSSI\b",
        r"\bcellulitis\b",
        r"\babscess\b",
        r"\binfected\b",
        r"\binfection\b",
    ]),
    ("Recon_Mastectomy_Skin_Flap_Necrosis", [
        r"\bskin\s+flap\s+necrosis\b",
        r"\bmastectomy\s+skin\s+flap\s+necrosis\b",
        r"\bmsfn\b",
        r"\bflap\s+necrosis\b",
        r"\bischemi(a|c)\b",
        r"\bskin\s+slough\b",
    ]),
    ("Implant_Capsular_Contracture", [
        r"\bcapsular\s+contracture\b",
        r"\bBaker\s+(I|II|III|IV|1|2|3|4)\b",
    ]),
    ("Implant_Malposition", [
        r"\bimplant\s+malposition\b",
        r"\bmalposition\b",
        r"\bbottoming\s+out\b",
        r"\bsymmastia\b",
    ]),
    ("Implant_Rupture_Leak_Deflation", [
        r"\bimplant\s+rupture\b",
        r"\brupture\b",
        r"\bleak(age|ing)?\b",
        r"\bdeflation\b",
    ]),
    ("Implant_Extrusion", [
        r"\bextrusion\b",
        r"\bexpander\s+extrusion\b",
        r"\bimplant\s+extrusion\b",
        r"\bexposed\s+(implant|expander)\b",
    ]),
    ("Flap_Partial_Necrosis", [r"\bpartial\s+flap\s+necrosis\b", r"\bpartial\s+necrosis\b"]),
    ("Flap_Total_Loss", [
        r"\btotal\s+flap\s+loss\b",
        r"\bflap\s+loss\b",
        r"\bfailed\s+flap\b",
        r"\bnon[- ]viable\s+flap\b",
    ]),
    ("Donor_Hematoma", [r"\bdonor\s+site\b.*\bhematoma\b", r"\bhematoma\b.*\bdonor\s+site\b"]),
    ("Donor_Seroma", [r"\bdonor\s+site\b.*\bseroma\b", r"\bseroma\b.*\bdonor\s+site\b"]),
    ("Donor_Infection", [r"\bdonor\s+site\b.*\binfection\b", r"\binfection\b.*\bdonor\s+site\b"]),
    ("Donor_Dehiscence", [r"\bdonor\s+site\b.*\bdehiscen(ce|t)\b", r"\bdehiscen(ce|t)\b.*\bdonor\s+site\b"]),
    ("Donor_Necrosis", [r"\bdonor\s+site\b.*\bnecrosis\b", r"\bnecrosis\b.*\bdonor\s+site\b"]),
    ("Donor_Fat_Necrosis", [r"\bfat\s+necrosis\b"]),
    ("Abdominal_Bulge_Hernia", [
        r"\babdominal\s+wall\s+(bulge|laxity)\b",
        r"\bventral\s+hernia\b",
        r"\bincisional\s+hernia\b",
        r"\bhernia\b",
        r"\bbulge\b",
    ]),
    ("Treatment_NonOperative", [
        r"\blocal\s+wound\s+care\b",
        r"\bdressing\s+changes?\b",
        r"\bwet\s+to\s+dry\b",
        r"\bpacking\b",
        r"\bbedside\b.*\bI&D\b",
        r"\bincision\s+and\s+drainage\b",
        r"\bdrain(ed|age)\b",
        r"\baspirat(e|ion)\b",
        r"\bantibiotic(s)?\b",
        r"\bPO\s+antibiotic(s)?\b",
        r"\bIV\s+antibiotic(s)?\b",
    ]),
    ("Treatment_Reoperation", [
        r"\breturn(ed)?\s+to\s+OR\b",
        r"\bback\s+to\s+OR\b",
        r"\bre-?operation\b",
        r"\bre-?explor(ation|e)\b",
        r"\bwashout\b",
        r"\bdebridement\b",
        r"\bexplant\b",
        r"\bimplant\s+removal\b",
        r"\bexpander\s+removal\b",
        r"\bflap\s+takeback\b",
    ]),
    ("Treatment_Rehospitalization", [
        r"\bre-?admit(ted|)\b",
        r"\breadmission\b",
        r"\brehospitali[sz]ation\b",
        r"\breturned\s+to\s+hospital\b",
    ]),
    ("Failure_Removal", [
        r"\bexplant\b",
        r"\bimplant\s+removal\b",
        r"\bexpander\s+removal\b",
        r"\bflap\s+loss\b",
        r"\bremove(d)?\s+(the\s+)?(implant|expander|flap)\b",
    ]),
    ("Revision_Surgery", [
        r"\brev(ision|ise|ised)\b",
        r"\bfat\s+graft(ing)?\b",
        r"\bcap(sulectomy|sulotomy)\b",
        r"\bscar\s+revision\b",
        r"\bcontralateral\b.*\b(mastopexy|reduction|augmentation)\b",
        r"\bsymmetr(y|ization)\b",
    ]),
]


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", engine="python")


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def snippet_around(text, start, end, n=SNIPPET_CHARS):
    lo = max(0, start - n)
    hi = min(len(text), end + n)
    return norm_ws(text[lo:hi])


def compile_terms():
    compiled = []
    for name, pats in TERM_GROUPS:
        rx = re.compile("|".join(["(" + p + ")" for p in pats]), re.IGNORECASE)
        compiled.append((name, rx, pats))
    return compiled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=PATIENT_INDEX_CSV)
    ap.add_argument("--n", type=int, default=None, help="Optional pilot N patients (default: all patients)")
    ap.add_argument("--max_examples", type=int, default=0, help="Print up to N snippets per term to stdout")
    ap.add_argument("--write_evidence", action="store_true", help="Write evidence CSV (may contain PHI)")
    ap.add_argument("--max_evidence_rows", type=int, default=200000, help="Safety cap for evidence rows")
    ap.add_argument("--note_types", default="", help="Comma-separated allowlist of note_type values (exact match)")
    ap.add_argument("--out_prefix", default="qa_comp_terms")
    args = ap.parse_args()

    df = read_csv_fallback(args.index)
    required = ["patient_id", "note_id", "note_type", "note_text"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing required column in {}: {}".format(args.index, c))

    df["note_text"] = df["note_text"].fillna("").astype(str)
    df["note_type"] = df["note_type"].fillna("").astype(str)
    df["note_id"] = df["note_id"].fillna("").astype(str)

    patient_ids = sorted(df["patient_id"].dropna().unique().tolist())
    if args.n is not None:
        patient_ids = patient_ids[: args.n]
        print("Patients: PILOT ({})".format(len(patient_ids)))
    else:
        print("Patients: ALL ({})".format(len(patient_ids)))

    d0 = df[df["patient_id"].isin(patient_ids)].copy()

    allow_types = [x.strip() for x in args.note_types.split(",") if x.strip()]
    if allow_types:
        d0 = d0[d0["note_type"].isin(allow_types)].copy()
        print("Filtered to note_types ({}). Notes now: {}".format(len(allow_types), d0.shape[0]))

    print("Notes scanned:", d0.shape[0])

    compiled = compile_terms()

    summary_rows = []
    by_type_rows = []
    evidence_rows = []
    evidence_count = 0

    # Precompute note_type distribution overall (helps QA)
    overall_note_type_counts = d0["note_type"].value_counts(dropna=False)

    for term_name, rx, pats in compiled:
        # note-level hits (count notes with >=1 match)
        hits_mask = d0["note_text"].apply(lambda t: bool(rx.search(t)))
        hits_df = d0[hits_mask]

        note_hits = int(hits_df.shape[0])
        uniq_pat = int(hits_df["patient_id"].nunique()) if note_hits else 0

        # per note_type breakdown
        if note_hits:
            vc = hits_df["note_type"].value_counts(dropna=False)
            for nt, cnt in vc.items():
                by_type_rows.append({
                    "term": term_name,
                    "note_type": nt,
                    "note_hits": int(cnt),
                    "unique_patients_with_hit": int(hits_df[hits_df["note_type"] == nt]["patient_id"].nunique()),
                })

        summary_rows.append({
            "term": term_name,
            "note_hits": note_hits,
            "unique_patients_with_hit": uniq_pat,
            "patterns": " | ".join(pats),
        })

        # optional printing + evidence capture (first match per note)
        if (args.max_examples > 0 or args.write_evidence) and note_hits:
            printed = 0
            for _, row in hits_df.iterrows():
                if args.max_examples > 0 and printed < args.max_examples:
                    text = row["note_text"]
                    m = rx.search(text)
                    if m:
                        print("\n=== TERM: {} ===".format(term_name))
                        print("note_type: {} | note_id: {}".format(row["note_type"], row["note_id"]))
                        print("snippet: {}".format(snippet_around(text, m.start(), m.end())[:500]))
                        printed += 1

                if args.write_evidence and evidence_count < args.max_evidence_rows:
                    text = row["note_text"]
                    m = rx.search(text)
                    if not m:
                        continue
                    evidence_rows.append({
                        "term": term_name,
                        "patient_id": row["patient_id"],
                        "note_type": row["note_type"],
                        "note_id": row["note_id"],
                        "match_span": text[m.start():m.end()],
                        "snippet": snippet_around(text, m.start(), m.end()),
                    })
                    evidence_count += 1

        print("TERM SUMMARY: {} | note_hits={} | unique_patients={}".format(
            term_name, note_hits, uniq_pat
        ))

    sum_df = pd.DataFrame(summary_rows).sort_values(
        ["unique_patients_with_hit", "note_hits"], ascending=False
    )
    sum_path = "{}_summary.csv".format(args.out_prefix)
    sum_df.to_csv(sum_path, index=False)
    print("Wrote", sum_path, "(rows={})".format(sum_df.shape[0]))

    by_type_df = pd.DataFrame(by_type_rows)
    by_type_path = "{}_by_notetype.csv".format(args.out_prefix)
    by_type_df.to_csv(by_type_path, index=False)
    print("Wrote", by_type_path, "(rows={})".format(by_type_df.shape[0]))

    # Overall note_type distribution (baseline)
    baseline_path = "{}_notetype_baseline.csv".format(args.out_prefix)
    overall_note_type_counts.to_csv(baseline_path, header=["note_count"])
    print("Wrote", baseline_path, "(rows={})".format(overall_note_type_counts.shape[0]))

    if args.write_evidence:
        ev_df = pd.DataFrame(evidence_rows)
        ev_path = "{}_evidence.csv".format(args.out_prefix)
        ev_df.to_csv(ev_path, index=False)
        print("Wrote", ev_path, "(rows={})".format(ev_df.shape[0]))
        print("NOTE: Evidence snippets may contain PHI. Keep this file private/local.")

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
