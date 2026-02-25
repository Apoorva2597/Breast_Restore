#!/usr/bin/env python3
from __future__ import print_function
import os, shutil, sys

CANDIDATES = [
    "HPI11526 Operation Notes.csv",
    "HPI11526_Operation_Notes.csv",
]

SRC_DIRS = [
    os.path.expanduser("~/my_data_Breast/HPI-11526/HPI11256"),
    os.path.expanduser("~/my_data_Breast/HPI-11526"),
]

DST_DIR = os.path.join(os.getcwd(), "_staging_inputs")

def main():
    if not os.path.isdir(DST_DIR):
        os.makedirs(DST_DIR)

    found = None
    for d in SRC_DIRS:
        for f in CANDIDATES:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                found = p
                break
        if found:
            break

    if not found:
        print("ERROR: Could not find Operation Notes CSV in:")
        for d in SRC_DIRS:
            print(" -", d)
        print("Tried filenames:", CANDIDATES)
        sys.exit(1)

    dst = os.path.join(DST_DIR, os.path.basename(found))
    shutil.copy2(found, dst)

    print("OK: staged file")
    print("  from:", found)
    print("  to:  ", dst)

if __name__ == "__main__":
    main()
