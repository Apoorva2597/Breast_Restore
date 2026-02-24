
import os, glob, pandas as pd

BASE="/home/apokol/Breast_Restore/QA_DEID_BUNDLES"
pids=[p for p in os.listdir(BASE) if os.path.isdir(os.path.join(BASE,p)) and p!="logs"]
rows=[]
for pid in sorted(pids):
    d=os.path.join(BASE,pid)
    rec={"patient_id":pid}
    # files
    rec["has_timeline"]=os.path.exists(os.path.join(d,"timeline.csv"))
    rec["has_encounters"]=os.path.exists(os.path.join(d,"encounters_timeline.csv"))
    rec["has_anchor_summary"]=os.path.exists(os.path.join(d,"stage2_anchor_summary.txt"))
    # note types
    if rec["has_timeline"]:
        df=pd.read_csv(os.path.join(d,"timeline.csv"), dtype=str)
        if "NOTE_TYPE" in df.columns:
            vc=df["NOTE_TYPE"].fillna("").value_counts()
            rec["n_notes"]=len(df)
            rec["top_note_type"]=vc.index[0] if len(vc) else ""
        else:
            rec["n_notes"]=len(df)
    # anchor info (light parse)
    if rec["has_anchor_summary"]:
        txt=open(os.path.join(d,"stage2_anchor_summary.txt"),"r",errors="replace").read()
        def grab(prefix):
            for line in txt.splitlines():
                if line.strip().startswith(prefix):
                    return line.split(":",1)[1].strip() if ":" in line else line.strip()
            return ""
        rec["anchor_date"]=grab("ANCHOR_DATE")
        rec["anchor_source"]=grab("ANCHOR_SOURCE")
    rows.append(rec)

out="/home/apokol/Breast_Restore/qa_bundle_summary.csv"
pd.DataFrame(rows).to_csv(out,index=False)
print("WROTE:", out, "rows:", len(rows))
