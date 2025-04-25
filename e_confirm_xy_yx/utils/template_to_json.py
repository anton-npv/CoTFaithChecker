#!/usr/bin/env python3
"""
convert_instructions.py
-----------------------

Convert any YAML file that contains instruction templates (like *instr-v0*,
*instr-nyt*, *instr-wm* …) into pretty-printed JSON.

Usage
-----
    # single file → same name but .json
    python convert_instructions.py templates.yaml

    # single file with explicit output name
    python convert_instructions.py templates.yaml -o templates.json

    # convert every *.yaml in a directory, keeping the same base-name
    python convert_instructions.py ./dir_with_yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml  # pip install pyyaml


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        # ensure_ascii=False keeps the text readable; indent=2 pretty-prints
        json.dump(obj, f, ensure_ascii=False, indent=2)


def convert_one(src: Path, dst: Path):
    data = load_yaml(src)
    write_json(data, dst)
    print(f"✓ {src.name}  →  {dst.name}")


def main():
    p = argparse.ArgumentParser(description="Convert instruction-template YAML to JSON.")
    p.add_argument("input", help="YAML file or directory containing YAML files")
    p.add_argument("-o", "--output", help="Output filename (single-file mode only)")
    args = p.parse_args()

    source = Path(args.input)
    if not source.exists():
        p.error(f"{source} does not exist")

    if source.is_dir():
        for yml in sorted(source.glob("*.yaml")):
            convert_one(yml, yml.with_suffix(".json"))
    else:
        out = Path(args.output) if args.output else source.with_suffix(".json")
        convert_one(source, out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
