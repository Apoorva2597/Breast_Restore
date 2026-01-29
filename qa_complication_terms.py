# qa_complication_terms.py
# Python 3.6.8 compatible
# Scans all note types for complication vocabulary coverage (pilot: first 50 patients).
#
# Outputs:
#   1) qa_complication_terms_summary.csv  (counts only; no evidence)
#   2) qa_complication_terms_evidence.csv (optional; may contain PHI)

import re
import sys
import pandas as pd

# -----------------------
# Settings
# -----------------------
PATIENT_INDEX_CSV = "patient_note_index.csv"
PILOT_N_PATIENTS = 50
MAX_EXAMPLES_PER_TERM = 5
SNIPPET_CHARS = 180

# IMPORTANT: evidence snippets may contain PHI. Keep local and do not share externally.
WRITE_EVIDENCE_CSV = True


# -----------------------
# CSV read helpers
# -----------------------
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


# -----------------------
# Term dictionaries
# -----------------------
# NOTE: We keep patterns fairly broad here (high recall).
# The point of this script is to discover vocabulary; precision comes later in extractors.

TERM_GROUPS = [
    # --- Reconstruction site complications ---
    ("Recon_Hematoma", [
        r"\bhematoma\b",
        r"\bpost[- ]op(erative)?\s+hematoma\b",
    ]),
    ("Recon_Seroma", [
        r"\bseroma\b",
    ]),
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
    ("Flap_Partial_Necrosis", [
        r"\bpartial\s+flap\s+necrosis\b",
        r"\bpartial\s+necrosis\b",
    ]),
    ("Flap_Total_Loss", [
        r"\btotal\s+flap\s+loss\b",
        r"\bflap\s+loss\b",
        r"\bfailed\s+flap\b",
        r"\bnon[- ]viable\s+flap\b",
    ]),

    # --- Donor site complications ---
    ("Donor_Hematoma", [
        r"\bdonor\s+site\b.*\bhematoma\b",
        r"\bhematoma\b.*\bdonor\s+site\b",
    ]),
    ("Donor_Seroma", [
        r"\bdonor\s+site\b.*\bseroma\b",
        r"\bseroma\b.*\bdonor\s+site\b",
    ]),
    ("Donor_Infection", [
        r"\bdonor\s+site\b.*\binfection\b",
        r"\binfection\b.*\bdonor\s+site\b",
    ]),
    ("Donor_Dehiscence", [
        r"\bdonor\s+site\b.*\bdehiscen(ce|t)\b",
        r"\bdehiscen(ce|t)\b.*\bdonor\s+site\b",
    ]),
    ("Donor_Necrosis", [
        r"\bdonor\s+site\b.*\bnecrosis\b",
        r"\bnecrosis\b.*\bdonor\s+site\b",
    ]),
    ("Donor_Fat_Necrosis", [
        r"\bfat\s+necrosis\b",
    ]),
    ("Abdominal_Bulge_Hernia", [
        r"\babdominal\s+wall\s+(bulge|laxity)\b",
        r"\bventral\s+hernia\b",
        r"\bincisional\s+hernia\b",
        r"\bhernia\b",
        r"\bbulge\b",
    ]),

    # --- Treatment / severity signals ---
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

    # --- Failure / revision signals ---
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


def compile_terms():
    out = []
    for name, pats in TERM_GROUPS:
        rx = re.compile("|".join(["(" + p + ")" for p in pats]), re.IGNORECASE)
        out.append((name, rx, pats))
    return out


def main():
    df = read_csv_fallback(PATIENT_INDEX_CSV)

    required = ["patient_id", "note_id", "note_type", "note_text"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError("Missing required column in {}: {}".format(PATIENT_INDEX_CSV, c))

    # Deterministic pilot: first N patients in sorted order
    patient_ids = sorted(df["patient_id"].dropna().unique().tolist())[:PILOT_N_PATIENTS]
    print("Pilot patients: {}".format(len(patient_ids)))

    d0 = df[df["patient_id"].isin(patient_ids)].copy()
    print("Pilot notes: {}".format(d0.shape[0]))

    compiled = compile_terms()

    # Summary counts
    summary_rows = []
    evidence_rows = []

    # Pre-normalize note_text to string
    d0["note_text"] = d0["note_text"].fillna("").astype(str)
    d0["note_type"] = d0["note_type"].fillna("").astype(str)
    d0["note_id"] = d0["note_id"].fillna("").astype(str)

    # Scan term-by-term to keep memory stable and make output clearer
    for term_name, rx, pats in compiled:
        total_hits = 0
        hit_patients = set()
        examples = 0

        # iterate notes
        for idx, row in d0.iterrows():
            text = row["note_text"]
            if not text:
                continue

            m = rx.search(text)
            if not m:
                continue

            total_hits += 1
            hit_patients.add(row["patient_id"])

            if examples < MAX_EXAMPLES_PER_TERM:
                snip = snippet_around(text, m.start(), m.end(), n=SNIPPET_CHARS)
                print("\n=== TERM: {} ===".format(term_name))
                print("note_type: {} | note_id: {}".format(row["note_type"], row["note_id"]))
                print("snippet: {}".format(snip[:500]))
                examples += 1

            if WRITE_EVIDENCE_CSV:
                # Collect ONE row per note hit (first match per note)
                evidence_rows.append({
                    "term": term_name,
                    "patient_id": row["patient_id"],
                    "note_type": row["note_type"],
                    "note_id": row["note_id"],
                    "match_span": text[m.start():m.end()],
                    "snippet": snippet_around(text, m.start(), m.end(), n=SNIPPET_CHARS),
                })

        summary_rows.append({
            "term": term_name,
            "note_hits": int(total_hits),
            "unique_patients_with_hit": int(len(hit_patients)),
            "patterns": " | ".join(pats),
        })

        print("\nTERM SUMMARY: {} | note_hits={} | unique_patients={}".format(
            term_name, total_hits, len(hit_patients)
        ))

    # Write outputs
    sum_df = pd.DataFrame(summary_rows).sort_values(["unique_patients_with_hit", "note_hits"], ascending=False)
    sum_df.to_csv("qa_complication_terms_summary.csv", index=False)

    print("\nWrote qa_complication_terms_summary.csv (rows={})".format(sum_df.shape[0]))

    if WRITE_EVIDENCE_CSV:
        ev_df = pd.DataFrame(evidence_rows)
        ev_df.to_csv("qa_complication_terms_evidence.csv", index=False)
        print("Wrote qa_complication_terms_evidence.csv (rows={})".format(ev_df.shape[0]))
        print("NOTE: Evidence snippets may contain PHI. Keep this file private/local.")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
