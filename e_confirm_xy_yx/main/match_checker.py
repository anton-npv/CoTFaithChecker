import json
from pathlib import Path
from typing import Dict, List
from e_confirm_xy_yx.main.logging_utils import init_logger

__all__ = ["check_matches"]

logger = init_logger(Path.cwd() / "logs", "match_checker")


def _index_by_question_id(records: List[Dict]) -> Dict[int, Dict]:
    return {r["question_id"]: r for r in records}


def check_matches(
    no_file: Path,
    yes_file: Path,
    output_path: Path,
):
    """
    Compare the verified answers from the NO-file to the YES-file using cross-links.
    """
    """with no_file.open() as f:
        no_records = json.load(f)
    with yes_file.open() as f:
        yes_records = json.load(f)"""
    # if the caller passed a placeholder or non-existent YES file,
    # locate the real one by property prefix (hashes often differ).
    if not yes_file.exists():
        prop_prefix = no_file.stem.split("_gt_")[0]        # e.g. "wm-book-length"
        candidates   = list(
            yes_file.parent.glob(f"{prop_prefix}_gt_YES_*_verified.json")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No YES verified file found for property '{prop_prefix}'"
            )
        yes_file = candidates[0]

    with yes_file.open() as f:
        yes_records = json.load(f)

    yes_by_id = _index_by_question_id(yes_records)

    final_records = []
    for no_rec in no_records:
        yid = no_rec["yes_question_id"]
        yes_rec = yes_by_id[yid]
        matched = no_rec["verified_answer"] == yes_rec["verified_answer"]
        final_records.append(
            {
                "no_file_q_id": no_rec["question_id"],
                "yes_question_id": yid,
                "no_answer": no_rec["verified_answer"],
                "yes_answer": yes_rec["verified_answer"],
                "matched": matched,
            }
        )

    with output_path.open("w") as f:
        json.dump(final_records, f, indent=2)
    logger.info(f"Saved match report → {output_path}")
