import json
from pathlib import Path
from typing import Dict, List
from cot_faithfulness.logging_utils import init_logger

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
    with no_file.open() as f:
        no_records = json.load(f)
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
