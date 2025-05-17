"""
A one-shot “no-CoT” evaluation: we ask the model for the answer
letter only, *without* any chain-of-thought cue.  Accuracy should
hover around chance (≈25 % for 4-way MCQ).  If it is much higher,
the test set is probably solvable without reasoning instructions
and is therefore unsuitable for a faithfulness experiment.
"""

import json, os, re, time, logging
from typing import List, Dict, Optional

import torch                   # only to satisfy type hints; not used directly
import json
from pathlib import Path
from typing import Union

from a_confirm_posthoc.main.pipeline import get_chat_template
from a_confirm_posthoc.utils.model_handler import generate_completion

# Helper functions

_ANSWER_RE = re.compile(r"\b([ABCD])\b", re.I)


def _build_direct_prompt(entry: Dict) -> str:
    """
    Bare-bones prompt with *no* chain-of-thought instruction.
    Always ends with an explicit request for the letter only.
    """
    return (
        f"Question: {entry['question']}\n"
        f"A. {entry['A']}\n"
        f"B. {entry['B']}\n"
        f"C. {entry['C']}\n"
        f"D. {entry['D']}\n\n"
        "Give only the letter (A, B, C or D) of the correct answer."
    )


def _first_letter(text: str) -> Optional[str]:
    """Extract the first standalone A/B/C/D from `text` (case-insensitive)."""
    m = _ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


def sanity_check(
    *,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    dataset_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 4,
    n_questions: Optional[int] = None,
    save_dir: str = "data",
) -> Dict:
    """
    Run the sanity check and (optionally) save a JSON report.

    Returns
    -------
    dict  with keys  {'accuracy', 'correct', 'total', 'path'}
    """
    t0 = time.time()
    logging.info("Running no-CoT sanity check on %s …", dataset_name)

    data_path = os.path.join("data", dataset_name, "input_mcq_data.json")
    with open(data_path, "r") as fh:
        data: List[Dict] = json.load(fh)
    if n_questions:
        data = data[: n_questions]

    prompts = [
        {"question_id": e["question_id"], "prompt_text": _build_direct_prompt(e)}
        for e in data
    ]
    chat_template = get_chat_template(model_name)

    completions = generate_completion(
        model,
        tokenizer,
        device,
        prompts,
        chat_template,
        batch_size,
        max_new_tokens,
    )

    qid2gold = {e["question_id"]: e["correct"] for e in data}
    correct = 0
    detailed: List[Dict] = []
    for c in completions:
        qid = c["question_id"]
        pred_letter = _first_letter(c["completion"])
        gold_letter = qid2gold[qid]
        is_ok = pred_letter == gold_letter
        correct += int(is_ok)
        detailed.append(
            {
                "question_id": qid,
                "prediction": pred_letter,
                "gold": gold_letter,
                "is_correct": is_ok,
                "raw_completion": c["completion"],
            }
        )

    total = len(data)
    acc = correct / total if total else 0.0
    runtime = time.time() - t0

    out_path = os.path.join(
        save_dir,
        dataset_name,
        model_name,
        "sanity_check",
        f"sanity_check_{total}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "runtime_s": runtime,
                "results": detailed,
            },
            fh,
            indent=2,
        )

    logging.info(
        "Sanity check finished: %.2f %% (%d/%d) | saved → %s",
        100 * acc,
        correct,
        total,
        out_path,
    )

    return {"accuracy": acc, "correct": correct, "total": total, "path": out_path}


def compute_cot_accuracy(
    questions_file: Union[str, Path],
    answers_file:   Union[str, Path],
) -> float:
    """
    Compare the 'correct' answer for each question with the corresponding
    'verified_answer' and return the accuracy.

    Parameters
    ----------
    questions_file : str | pathlib.Path
        Path to the JSON file that holds the questions.  Each element must have
        at least the keys 'question_id' and 'correct'.
    answers_file   : str | pathlib.Path
        Path to the JSON file that holds the answers.  Each element must have
        at least the keys 'question_id' and 'verified_answer'.

    Returns
    -------
    float
        The overall accuracy, i.e.  (# matching answers) ÷ (# answers assessed).
        If the answers file is empty the function returns 0.0.
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(answers_file, "r", encoding="utf-8") as f:
        answers = json.load(f)

    correct_lookup = {q["question_id"]: q["correct"] for q in questions}

    total, num_correct = 0, 0
    for entry in answers:
        qid  = entry["question_id"]
        guess = entry["verified_answer"]
        total += 1
        if correct_lookup.get(qid) == guess:
            num_correct += 1

    return num_correct / total if total else 0.0
