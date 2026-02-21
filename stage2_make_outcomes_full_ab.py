# stage2_make_outcomes_full_ab.py
# Merge Stage2 outcomes + failure/revision into one AB outcomes file

from __future__ import print_function
import sys
import pandas as pd

IN_OUTCOMES = "stage2_ab_outcomes_patient_level.csv"
IN_FAILREV  = "stage2_ab_failure_revision_patient_level.csv"
OUT_FILE    = "stage2_outcomes_full_ab.csv"

def read_csv_safe(path):
    f = open(path, "r", encoding="latin1", errors="replace")
    try:
        return pd.read_csv(f, engine="python")
    finally:
        try:
            f.close()
        except Exception:
            pass

def pick_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def to_int01(x):
    # normalize to 0/1 int
    s = pd.to_numeric(x, errors="coerce")
    s = s.fillna(0)
    s = (s != 0).astype(int)
    return s

def main():
    a = read_csv_safe(IN_OUTCOMES)
    b = read_csv_safe(IN_FAILREV)

    if "patient_id" not in a.columns:
        raise RuntimeError("Missing patient_id in {}".format(IN_OUTCOMES))
    if "patient_id" not in b.columns:
        raise RuntimeError("Missing patient_id in {}".format(IN_FAILREV))

    a["patient_id"] = a["patient_id"].fillna("").astype(str)
    b["patient_id"] = b["patient_id"].fillna("").astype(str)

    # Stage2 flags in outcomes file
    col_minor = pick_col(a.columns, ["Stage2_MinorComp", "S2_MinorComp", "stage2_minorcomp"])
    col_major = pick_col(a.columns, ["Stage2_MajorComp", "S2_MajorComp", "stage2_majorcomp"])
    col_reop  = pick_col(a.columns, ["Stage2_Reoperation", "S2_Reoperation", "stage2_reoperation"])
    col_rehsp = pick_col(a.columns, ["Stage2_Rehospitalization", "S2_Rehospitalization", "stage2_rehospitalization"])

    if not col_minor or not col_major or not col_reop or not col_rehsp:
        raise RuntimeError(
            "Could not find required Stage2 columns in {}.\nFound: minor={}, major={}, reop={}, rehosp={}".format(
                IN_OUTCOMES, col_minor, col_major, col_reop, col_rehsp
            )
        )

    # Failure/Revision cols (from failrev file)
    col_fail = pick_col(b.columns, ["Stage2_Failure", "S2_Failure", "stage2_failure"])
    col_rev  = pick_col(b.columns, ["Stage2_Revision", "S2_Revision", "stage2_revision"])

    if not col_fail or not col_rev:
        raise RuntimeError(
            "Could not find Stage2_Failure/Stage2_Revision in {}.\nFound: failure={}, revision={}".format(
                IN_FAILREV, col_fail, col_rev
            )
        )

    # Keep only what we need (avoid accidental collisions)
    a2 = a[["patient_id", col_minor, col_major, col_reop, col_rehsp]].copy()
    b2 = b[["patient_id", col_fail, col_rev]].copy()

    # Rename to standard names
    a2 = a2.rename(columns={
        col_minor: "Stage2_MinorComp",
        col_major: "Stage2_MajorComp",
        col_reop:  "Stage2_Reoperation",
        col_rehsp: "Stage2_Rehospitalization"
    })
    b2 = b2.rename(columns={
        col_fail: "Stage2_Failure",
        col_rev:  "Stage2_Revision"
    })

    # Merge (AB cohort should come from a2; left join is safest)
    m = a2.merge(b2, on="patient_id", how="left")

    # Normalize flags to 0/1
    for c in ["Stage2_MinorComp","Stage2_MajorComp","Stage2_Reoperation","Stage2_Rehospitalization","Stage2_Failure","Stage2_Revision"]:
        if c in m.columns:
            m[c] = to_int01(m[c])
        else:
            m[c] = 0

    # Write
    m.to_csv(OUT_FILE, index=False, encoding="utf-8")

    # Print summary
    n = int(m["patient_id"].nunique())
    def pct(k):
        return (100.0 * k / n) if n else 0.0

    n_minor = int(m["Stage2_MinorComp"].sum())
    n_major = int(m["Stage2_MajorComp"].sum())
    n_reop  = int(m["Stage2_Reoperation"].sum())
    n_rehsp = int(m["Stage2_Rehospitalization"].sum())
    n_fail  = int(m["Stage2_Failure"].sum())
    n_rev   = int(m["Stage2_Revision"].sum())

    print("Patients:", n)
    print("Stage2_MinorComp:", n_minor, "({:.1f}%)".format(pct(n_minor)))
    print("Stage2_MajorComp:", n_major, "({:.1f}%)".format(pct(n_major)))
    print("Stage2_Reoperation:", n_reop, "({:.1f}%)".format(pct(n_reop)))
    print("Stage2_Rehospitalization:", n_rehsp, "({:.1f}%)".format(pct(n_rehsp)))
    print("Stage2_Failure:", n_fail, "({:.1f}%)".format(pct(n_fail)))
    print("Stage2_Revision:", n_rev, "({:.1f}%)".format(pct(n_rev)))
    print("Wrote:", OUT_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", str(e))
        sys.exit(1)
