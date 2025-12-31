from __future__ import annotations
import argparse
import json
from pathlib import Path

from phase1_pipeline.ingest.docx_reader import read_docx
from phase1_pipeline.normalize.sectionizer import sectionize

def main():
    ap = argparse.ArgumentParser() 
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.docx"))
    if not files:
        raise SystemExit(f"No .docx files found in {in_dir}")

    for f in files:
        note = read_docx(str(f), note_id=f.stem)
        secs = sectionize(note.text)

        # Human-readable view
        txt_path = out_dir / f"{f.stem}.sections.txt"
        with open(txt_path, "w", encoding="utf-8") as w:
            for name, body in secs.items():
                w.write("=" * 90 + "\n")
                w.write(f"[{name}]\n")
                w.write("=" * 90 + "\n")
                w.write(body.strip() + "\n\n")

        # Machine-readable view
        json_path = out_dir / f"{f.stem}.sections.json"
        with open(json_path, "w", encoding="utf-8") as w:
            json.dump({"note_id": f.stem, "section_order": list(secs.keys()), "sections": secs},
                      w, ensure_ascii=False, indent=2)

    print(f"Wrote section debug outputs for {len(files)} notes to {out_dir}")

if __name__ == "__main__":
    main()
