import os
import json
from pathlib import Path
from typing import List, Dict

#from google import genai           # gemini-flash client  (see user code)
from pydantic import BaseModel

from e_confirm_xy_yx.main.logging_utils import init_logger

__all__ = ["run_verification"]

logger = init_logger(Path.cwd() / "logs", "verifier")

from a_confirm_posthoc.eval.llm_hint_verificator import client
#client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class Verification(BaseModel):
    model_answer: str


def _verify_single(completion: str) -> str:
    """
    Ask Gemini-flash for the YES/NO decision.
    Returns "YES" / "NO" / "N/A".
    """
    prompt = f"""Below is a model completion answering a binary (YES/NO) question.
Return ONLY the final answer ('YES', 'NO') or 'N/A' if no definitive answer.

Model completion:
{completion}
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Verification,
        },
    )
    return response.parsed.model_answer.upper()


def _save(
    results: List[Dict],
    out_path: Path,
):
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved verification → {out_path}")


def run_verification(
    completion_files: List[Path],
    n_questions: int,
    output_dir: Path,
):
    """
    Verify every completion file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for fp in completion_files:
        logger.info(f"Verifying {fp}")
        with fp.open() as f:
            completions = json.load(f)

        results = []
        for record in completions[:n_questions] if n_questions else completions:
            """answer = _verify_single(record["completion"])
            if answer != "N/A":
                results.append(
                    {
                        "question_id": record["question_id"],
                        "verified_answer": answer,
                        **{
                            k: v
                            for k, v in record.items()
                            if k in ("yes_question_id", "no_question_id")
                        },
                    }
                )"""
            answer = _verify_single(record["completion"])
            results.append(
                {
                    "question_id": record["question_id"],
                    "verified_answer": answer,          # may be 'N/A'
                    **{
                        k: v
                        for k, v in record.items()
                        if k in ("yes_question_id", "no_question_id")
                    },
                }
            )

        out_name = fp.stem.replace("_completions", "_verified") + ".json"
        _save(results, output_dir / out_name)
