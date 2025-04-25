#!/usr/bin/env python3
"""
convert_questions.py

Convert a YAML file of ‘params / question_by_qid / …’ data into JSON,
adding a monotonically-increasing “question_id” field to each question.

Usage
-----
    python convert_questions.py input.yaml            # writes input.json
    python convert_questions.py input.yaml -o out.json
    python convert_questions.py ./directory_of_yaml   # converts every *.yaml inside
"""

import argparse
import json
import sys
from pathlib import Path

import yaml  # pip install pyyaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def transform(raw: dict) -> dict:
    """
    Return a new dict that keeps the original top-level 'params', but replaces
    'question_by_qid' with a flat 'questions' list, each element carrying:

        - question_id  (1, 2, 3… in file order)
        - qid          (original hash key)
        - all fields inside that question node
    """
    q_list = []
    for i, (qid, q_block) in enumerate(raw.get("question_by_qid", {}).items(), start=1):
        q_obj = {"question_id": i, "qid": qid, **q_block}
        q_list.append(q_obj)

    return {"params": raw.get("params", {}), "questions": q_list}


def write_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def convert_one(in_path: Path, out_path: Path) -> None:
    data = load_yaml(in_path)
    result = transform(data)
    write_json(result, out_path)
    print(f"✓ {in_path.name}  →  {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YAML question files to JSON.")
    parser.add_argument(
        "input",
        help="Either a YAML file or a directory containing YAML files",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (for a single file). "
        "If omitted, uses the input name with .json extension.",
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        parser.error(f"{src} does not exist")

    if src.is_dir():
        # batch-convert every *.yaml in the directory
        for yaml_path in sorted(src.glob("*.yaml")):
            convert_one(yaml_path, yaml_path.with_suffix(".json"))
    else:
        out_path = Path(args.output) if args.output else src.with_suffix(".json")
        convert_one(src, out_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
