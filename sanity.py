sed -i 's/df\.to_csv(out_path, index=False)/df.to_csv(out_path, index=False)\n    merged.to_csv("_outputs\/validation_merged_patient_level.csv", index=False)/' validate_abstraction.py
