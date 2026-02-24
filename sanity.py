#!/usr/bin/env python3
import os
from collections import defaultdict

QA_DIR = "/home/apokol/Breast_Restore/QA_DEID_BUNDLES"

targets = [
    "encounters_timeline.csv",
    "timeline.csv",
    "ALL_NOTES_COMBINED.txt",
    "stage2_anchor_summary.txt",
    "stage2_anchor_summary_v2.txt",
]

have = defaultdict(int)

for pid in os.listdir(QA_DIR):
    pdir = os.path.join(QA_DIR, pid)
    if not os.path.isdir(pdir) or pid == "logs":
        continue
    for t in targets:
        if os.path.exists(os.path.join(pdir, t)):
            have[t] += 1

print("Counts (patients with file):")
for t in targets:
    print("{:<28} {}".format(t, have[t]))
