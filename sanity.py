# pred_spine_sanity_check.py
# Terminal-only sanity check for prediction spine
# Python 3.6.8 compatible

from __future__ import print_function
import pandas as pd

FILE = "pred_spine_stage1_stage2.csv"

df = pd.read_csv(FILE, encoding="latin1")

df["patient_id"] = df["patient_id"].astype(str)

def pct(x, n):
    return round(100.0 * x / n, 1) if n else 0.0

n = df["patient_id"].nunique()

print("\n=== Prediction Spine Sanity Check ===")
print("Total patients:", n)

# ---- Stage 1 ----
s1_major = int(df["Stage1_MajorComp_pred"].sum())
s1_minor = int(df["Stage1_MinorComp_pred"].sum())
s1_reop  = int(df["Stage1_Reoperation_pred"].sum())
s1_rehsp = int(df["Stage1_Rehospitalization_pred"].sum())

any_s1 = ((df["Stage1_MajorComp_pred"] == 1) |
          (df["Stage1_MinorComp_pred"] == 1)).sum()

print("\n--- Stage 1 ---")
print("Any Stage1 Complication:", int(any_s1), "({:.1f}%)".format(pct(any_s1, n)))
print("Major:", s1_major, "({:.1f}%)".format(pct(s1_major, n)))
print("Minor:", s1_minor, "({:.1f}%)".format(pct(s1_minor, n)))
print("Reoperation:", s1_reop, "({:.1f}%)".format(pct(s1_reop, n)))
print("Rehospitalization:", s1_rehsp, "({:.1f}%)".format(pct(s1_rehsp, n)))

# ---- Stage 2 ----
s2_major = int(df["Stage2_MajorComp"].sum())
s2_minor = int(df["Stage2_MinorComp"].sum())
s2_reop  = int(df["Stage2_Reoperation"].sum())
s2_rehsp = int(df["Stage2_Rehospitalization"].sum())
s2_fail  = int(df["Stage2_Failure"].sum())
s2_rev   = int(df["Stage2_Revision"].sum())

any_s2 = ((df["Stage2_MajorComp"] == 1) |
          (df["Stage2_MinorComp"] == 1) |
          (df["Stage2_Failure"] == 1) |
          (df["Stage2_Revision"] == 1)).sum()

print("\n--- Stage 2 (All 848 patients) ---")
print("Any Stage2 Signal:", int(any_s2), "({:.1f}%)".format(pct(any_s2, n)))
print("Major:", s2_major, "({:.1f}%)".format(pct(s2_major, n)))
print("Minor:", s2_minor, "({:.1f}%)".format(pct(s2_minor, n)))
print("Reoperation:", s2_reop, "({:.1f}%)".format(pct(s2_reop, n)))
print("Rehospitalization:", s2_rehsp, "({:.1f}%)".format(pct(s2_rehsp, n)))
print("Failure:", s2_fail, "({:.1f}%)".format(pct(s2_fail, n)))
print("Revision:", s2_rev, "({:.1f}%)".format(pct(s2_rev, n)))

# ---- Logical Consistency Checks ----
print("\n--- Logical Checks ---")

# Major should include reop or rehosp sometimes
maj_without_trigger = df[
    (df["Stage1_MajorComp_pred"] == 1) &
    (df["Stage1_Reoperation_pred"] == 0) &
    (df["Stage1_Rehospitalization_pred"] == 0)
].shape[0]

print("Stage1 Major without Reop/Rehosp:", maj_without_trigger)

# Minor + Major both 1 should not happen
minor_and_major = df[
    (df["Stage1_MajorComp_pred"] == 1) &
    (df["Stage1_MinorComp_pred"] == 1)
].shape[0]

print("Stage1 Minor AND Major both 1 (should be 0):", minor_and_major)

print("\nSanity check complete.\n")
