#!/usr/bin/env python3
"""
link_yes_no.py

Cross-link every pair of files in two directories that were produced by
`convert_questions.py` – one directory whose questions were answered 'NO',
the other 'YES'.

For each matching question whose x/y values are reversed between a NO file
and the corresponding YES file, add:

    • yes_question_id   (to the NO question object)
    • no_question_id    (to the YES question object)

Usage
-----
    python link_yes_no.py gt_NO_1 gt_YES_1
    python link_yes_no.py gt_NO_1 gt_YES_1 -s _linked   # save as *.linked.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


def load_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def cross_link(no_path: Path, yes_path: Path) -> None:
    no_data = load_json(no_path)
    yes_data = load_json(yes_path)

    # Build lookup: (x, y) ➜ question dict
    yes_lookup: Dict[Tuple[str, str], Dict] = {
        (q["x_value"], q["y_value"]): q for q in yes_data["questions"]
    }

    for q_no in no_data["questions"]:
        reversed_pair = (q_no["y_value"], q_no["x_value"])
        q_yes = yes_lookup.get(reversed_pair)
        if q_yes:
            q_no["yes_question_id"] = q_yes["question_id"]
            q_yes["no_question_id"] = q_no["question_id"]

    save_json(no_data, no_path)
    save_json(yes_data, yes_path)
    print(f"✓ linked {no_path.name} ↔ {yes_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-link matching NO/YES question files."
    )
    parser.add_argument("no_dir", help="Directory with *_NO_* JSON files")
    parser.add_argument("yes_dir", help="Directory with *_YES_* JSON files")
    parser.add_argument(
        "-s",
        "--suffix",
        help="Write output to new files with this suffix instead of overwriting "
        "(e.g. '_linked' → foo.json → foo_linked.json)",
        default="",
    )
    args = parser.parse_args()

    no_dir = Path(args.no_dir)
    yes_dir = Path(args.yes_dir)
    if not (no_dir.is_dir() and yes_dir.is_dir()):
        parser.error("Both arguments must be directories that exist.")

    no_files = sorted(no_dir.glob("*.json"))
    yes_files = sorted(yes_dir.glob("*.json"))

    if len(no_files) != len(yes_files):
        print(
            f"⚠ Warning: directory sizes differ "
            f"({len(no_files)} NO files vs {len(yes_files)} YES files). "
            f"Only matching indices will be processed.",
            file=sys.stderr,
        )

    for no_f, yes_f in zip(no_files, yes_files):
        # Optionally write to copies instead of overwriting
        no_target = no_f if not args.suffix else no_f.with_stem(no_f.stem + args.suffix)
        yes_target = (
            yes_f if not args.suffix else yes_f.with_stem(yes_f.stem + args.suffix)
        )
        # If saving as copies, duplicate originals first
        if args.suffix:
            no_target.write_bytes(no_f.read_bytes())
            yes_target.write_bytes(yes_f.read_bytes())
        cross_link(no_target, yes_target)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
