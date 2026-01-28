import re
import pandas as pd

FILE = "evidence_log_phase1_p50.csv"

print("Loading evidence file...")
df = pd.read_csv(FILE, encoding="utf-8", engine="python")

# Combine all evidence text
texts = df["evidence"].dropna().astype(str).str.lower()

print("Total evidence rows:", len(texts))

# ---------------------------------------
# TERMS WE ALREADY COVER
# ---------------------------------------
base_patterns = [
    "radiation", "radiotherapy", "xrt", "pmrt",
    "chemo", "chemotherapy"
]

# ---------------------------------------
# POSSIBLE ADDITIONAL TERMS TO DISCOVER
# (regimens, drug names, abbreviations)
# ---------------------------------------
candidate_patterns = [
    "taxol", "paclitaxel", "docetaxel",
    "adriamycin", "doxorubicin",
    "cyclophosphamide", "carboplatin", "cisplatin",
    "ac ", "tc ", "tch", "ddac",
    "trastuzumab", "herceptin",
    "pertuzumab", "perjeta",
    "capecitabine", "xeloda",
    "femara", "arimidex", "tamoxifen",  # to confirm endocrine appears
]

def search_patterns(patterns, label):
    print("\n=== Checking:", label, "===")
    for pat in patterns:
        rx = re.compile(r"\b" + re.escape(pat.strip()) + r"\b", re.IGNORECASE)
        matches = texts[texts.str.contains(rx)]
        if len(matches) > 0:
            print("\nTerm:", pat, "| hits:", len(matches))
            print(matches.head(3).to_string(index=False))

# 1️⃣ What we already detect
search_patterns(base_patterns, "BASE TREATMENT TERMS")

# 2️⃣ What might exist but we don't yet capture
search_patterns(candidate_patterns, "POTENTIAL NEW TERMS")

print("\nDone.")
