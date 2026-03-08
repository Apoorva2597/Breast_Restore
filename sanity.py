def choose_best_bmi_recon_rows(struct_df):
    out_best = {}

    if len(struct_df) == 0:
        return out_best

    source_priority = {
        "clinic": 1,
        "operation": 2,
        "inpatient": 3
    }

    preferred_cpts = set([
        "19357",
        "19340",
        "19342",
        "19361",
        "19364",
        "19367",
        "S2068"
    ])

    primary_exclude_cpts = set([
        "19325",
        "19330"
    ])

    fallback_allowed_cpts = set([
        "19350",
        "19380"
    ])

    eligible_sources = struct_df[struct_df["STRUCT_SOURCE"].isin(["clinic", "operation", "inpatient"])].copy()
    if len(eligible_sources) == 0:
        return out_best

    has_preferred_cpt = {}

    for mrn, g in eligible_sources.groupby(MERGE_KEY):
        found = False
        for val in g["CPT_CODE_STRUCT"].fillna("").astype(str).tolist():
            cpt = clean_cell(val).upper()
            if cpt in preferred_cpts:
                found = True
                break
        has_preferred_cpt[mrn] = found

    for _, row in eligible_sources.iterrows():
        mrn = clean_cell(row.get(MERGE_KEY, ""))
        if not mrn:
            continue

        source = clean_cell(row.get("STRUCT_SOURCE", "")).lower()
        if source not in source_priority:
            continue

        admit_date = parse_date_safe(row.get("ADMIT_DATE_STRUCT", ""))
        recon_date = parse_date_safe(row.get("RECONSTRUCTION_DATE_STRUCT", ""))

        cpt_code = clean_cell(row.get("CPT_CODE_STRUCT", "")).upper()
        procedure = clean_cell(row.get("PROCEDURE_STRUCT", "")).lower()

        if admit_date is None or recon_date is None:
            continue

        if cpt_code in primary_exclude_cpts:
            continue

        if has_preferred_cpt.get(mrn, False) and cpt_code in fallback_allowed_cpts:
            continue

        is_anchor = False

        if cpt_code in preferred_cpts:
            is_anchor = True

        if (not has_preferred_cpt.get(mrn, False)) and (cpt_code in fallback_allowed_cpts):
            is_anchor = True

        if not is_anchor:
            if (
                ("tissue expander" in procedure) or
                ("breast recon" in procedure) or
                ("implant on same day of mastectomy" in procedure) or
                ("insert or replcmnt breast implnt on sep day from mastectomy" in procedure) or
                ("latissimus" in procedure) or
                ("diep" in procedure) or
                ("tram" in procedure) or
                ("flap" in procedure)
            ):
                is_anchor = True

        if not is_anchor:
            continue

        score = (
            source_priority[source],
            recon_date,
            admit_date
        )

        current_best = out_best.get(mrn)

        if current_best is None or score < current_best["score"]:
            out_best[mrn] = {
                "recon_date": recon_date.strftime("%Y-%m-%d"),
                "source": source,
                "cpt_code": cpt_code,
                "procedure": clean_cell(row.get("PROCEDURE_STRUCT", "")),
                "score": score
            }

    return out_best
