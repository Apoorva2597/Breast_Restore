#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_pbs_laterality.py
Python 3.6.8 compatible

Fixes the laterality rejection bug in update_pbs_only.py.

The bug: PBS_Breast Reduction, PBS_Mastopexy, PBS_Augmentation, PBS_Other
are being rejected with "reject_unknown_recon_laterality" because the
patient's reconstruction laterality is unknown. But these fields are
PAST COSMETIC HISTORY -- laterality of current reconstruction is irrelevant.

This patch replaces only the broken decision block, leaving everything
else in update_pbs_only.py untouched.

USAGE (run from ~/Breast_Restore):
    python patch_pbs_laterality.py

It will write update_pbs_only.py with the fix applied.
A backup is saved as update_pbs_only.py.bak first.
"""

from __future__ import print_function
import os
import shutil

TARGET = "update_pbs_only.py"
BACKUP = "update_pbs_only.py.bak"

# ---------------------------------------------------------------
# The exact block we are replacing (must match the file exactly)
# ---------------------------------------------------------------
OLD_BLOCK = '''            elif day_diff < 0:
                if lat_decision == "accept":
                    if history_ok:
                        accept = True
                        reason = "accept_pre_recon_historical"
                    else:
                        if field == "PBS_Lumpectomy":
                            accept = True
                            reason = "accept_pre_recon_lumpectomy"
                        else:
                            accept = False
                            reason = "reject_pre_recon_no_history"
                elif lat_decision == "reject_contralateral":
                    accept = False
                    reason = "reject_contralateral"
                elif lat_decision == "unknown_unilateral":
                    if field == "PBS_Lumpectomy":
                        inferred = infer_laterality_from_field_context(field, combined_context)
                        if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                            accept = True
                            reason = "accept_inferred_laterality"
                        else:
                            accept = False
                            reason = "reject_unknown_laterality_unilateral"
                    else:
                        accept = False
                        reason = "reject_unknown_laterality_unilateral"
                else:
                    accept = False
                    reason = "reject_unknown_recon_laterality"

            else:
                if field == "PBS_Lumpectomy":
                    # restore the broader, better-performing post-recon lumpectomy behavior
                    if not history_ok:
                        accept = False
                        reason = "reject_post_recon_not_historical"
                    else:
                        if lat_decision == "accept":
                            accept = True
                          reason = "accept_post_recon_historical"
                        elif lat_decision == "reject_contralateral":
                            accept = False
                            reason = "reject_contralateral"
                        elif lat_decision == "unknown_unilateral":
                            inferred = infer_laterality_from_field_context(field, combined_context)
                            if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                                accept = True
                                reason = "accept_post_recon_inferred_laterality"
                            else:
                                accept = False
                                reason = "reject_unknown_laterality_unilateral"
                        else:
                            if history_ok:
                                accept = True
                                reason = "accept_post_recon_history_no_recon_lat"
                            else:
                                accept = False
                                reason = "reject_unknown_recon_laterality"
                else:
                    if not history_ok:
                        accept = False
                        reason = "reject_post_recon_not_historical"
                    else:
                        if lat_decision == "accept":
                            accept = True
                            reason = "accept_post_recon_historical"
                        elif lat_decision == "reject_contralateral":
                            accept = False
                            reason = "reject_contralateral"
                        elif lat_decision == "unknown_unilateral":
                            accept = False
                            reason = "reject_unknown_laterality_unilateral"
                        else:
                            accept = False
                            reason = "reject_unknown_recon_laterality"'''

# ---------------------------------------------------------------
# The replacement block
# ---------------------------------------------------------------
NEW_BLOCK = '''            elif field != "PBS_Lumpectomy":
                # PBS_Breast Reduction, PBS_Mastopexy, PBS_Augmentation,
                # PBS_Other: these are PAST COSMETIC/SURGICAL HISTORY.
                # Laterality of current reconstruction is irrelevant.
                # Only require history context.
                if not history_ok:
                    accept = False
                    reason = "reject_no_history_context"
                else:
                    accept = True
                    reason = "accept_non_lumpectomy_history"

            elif day_diff < 0:
                # PBS_Lumpectomy pre-recon
                if lat_decision == "accept":
                    if history_ok:
                        accept = True
                        reason = "accept_pre_recon_historical"
                    else:
                        accept = True
                        reason = "accept_pre_recon_lumpectomy"
                elif lat_decision == "reject_contralateral":
                    accept = False
                    reason = "reject_contralateral"
                elif lat_decision == "unknown_unilateral":
                    inferred = infer_laterality_from_field_context(field, combined_context)
                    if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                        accept = True
                        reason = "accept_inferred_laterality"
                    else:
                        accept = False
                        reason = "reject_unknown_laterality_unilateral"
                else:
                    # recon_lat unknown - accept if history present
                    if history_ok:
                        accept = True
                        reason = "accept_pre_recon_unknown_lat_history"
                    else:
                        accept = False
                        reason = "reject_unknown_recon_laterality"

            else:
                # PBS_Lumpectomy post-recon
                if not history_ok:
                    accept = False
                    reason = "reject_post_recon_not_historical"
                else:
                    if lat_decision == "accept":
                        accept = True
                        reason = "accept_post_recon_historical"
                    elif lat_decision == "reject_contralateral":
                        accept = False
                        reason = "reject_contralateral"
                    elif lat_decision == "unknown_unilateral":
                        inferred = infer_laterality_from_field_context(field, combined_context)
                        if inferred and laterality_relation(recon_lat, inferred, combined_context) == "accept":
                            accept = True
                            reason = "accept_post_recon_inferred_laterality"
                        else:
                            accept = False
                            reason = "reject_unknown_laterality_unilateral"
                    else:
                        # recon_lat unknown - accept if history present
                        accept = True
                        reason = "accept_post_recon_history_no_recon_lat"'''


def main():
    if not os.path.isfile(TARGET):
        print("ERROR: {} not found. Run from ~/Breast_Restore".format(TARGET))
        return

    with open(TARGET, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize line endings
    content_unix = content.replace("\r\n", "\n").replace("\r", "\n")
    old_unix = OLD_BLOCK.replace("\r\n", "\n").replace("\r", "\n")

    if old_unix not in content_unix:
        print("ERROR: Could not find the target block in {}.".format(TARGET))
        print("")
        print("This means the file on CEDAR differs from what was in project knowledge.")
        print("Please run on CEDAR:")
        print("  grep -n 'reject_unknown_recon_laterality' update_pbs_only.py")
        print("and share the line numbers so we can adjust the patch.")
        return

    count = content_unix.count(old_unix)
    if count > 1:
        print("WARNING: Found {} occurrences of target block. Replacing first only.".format(count))

    new_content = content_unix.replace(old_unix, NEW_BLOCK, 1)

    # Backup
    shutil.copy2(TARGET, BACKUP)
    print("Backup saved: {}".format(BACKUP))

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("Patched: {}".format(TARGET))
    print("")
    print("Verify the patch applied correctly:")
    print("  grep -n 'accept_non_lumpectomy_history' update_pbs_only.py")
    print("")
    print("If that prints a line number, the patch worked.")
    print("Then run:")
    print("  python update_pbs_only.py")


if __name__ == "__main__":
    main()
