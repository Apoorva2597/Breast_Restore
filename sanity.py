print("\nSanity check: rows that match DROP_EXCEL_ROWS (before dropping):")
print(df[df["__excel_row__"].isin(DROP_EXCEL_ROWS)][["__excel_row__", "PatientID", "MRN"]].head(20))
print("Count present:", int(df["__excel_row__"].isin(DROP_EXCEL_ROWS).sum()))
