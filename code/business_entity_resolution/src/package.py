"""Build <team_name>_submission.zip in the organisers' layout.

    <team_name>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/  (src/, README.md, requirements.txt)
    └── Documentation_template.md

The output files are validated first; nothing is packaged if validation fails.

Usage:  python -m src.package --team "<team name>"
"""

import argparse
import sys
import zipfile

from src import config
from src.submit import OUTPUT_DIR, validate

ROOT = config.WORK_DIR.parent
CODE_DIR = ROOT / "code" / "business_entity_resolution"
DOC = ROOT / "Documentation_template.md"


def main():
    parser = argparse.ArgumentParser(description="Build the final submission zip")
    parser.add_argument("--team", required=True, help="team name, used in the zip file name")
    args = parser.parse_args()

    if not validate(OUTPUT_DIR):
        print("Validation failed: not packaging.")
        return 1
    if "[Your Team Name]" in DOC.read_text(encoding="utf-8"):
        print("WARNING: Documentation_template.md still has the team-name placeholder.")

    target = ROOT / f"{args.team}_submission.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(OUTPUT_DIR / name, f"output/{name}")
        for f in sorted(CODE_DIR.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f"code/business_entity_resolution/{f.relative_to(CODE_DIR).as_posix()}")
        z.write(DOC, "Documentation_template.md")
        names = z.namelist()
    print(f"{target} ({target.stat().st_size / 1e6:.0f} MB, {len(names)} files)")
    for n in names:
        print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
