    # Get reconstruction / anchor date from mismatch file if present,
    # otherwise pull it from master.
    recon_col = None
    for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE"]:
        if c in target.columns:
            recon_col = c
            break

    if recon_col is None:
        master_recon_col = None
        for c in ["Recon_Date", "RECONSTRUCTION_DATE", "recon_date", "ANCHOR_DATE"]:
            if c in master.columns:
                master_recon_col = c
                break

        if master_recon_col is None:
            raise RuntimeError(
                "Could not find reconstruction date column in mismatch file or master file."
            )

        target = target.merge(
            master[[MERGE_KEY, master_recon_col]].drop_duplicates(),
            on=MERGE_KEY,
            how="left"
        )
        recon_col = master_recon_col

    print("Using reconstruction date column: {0}".format(recon_col))
