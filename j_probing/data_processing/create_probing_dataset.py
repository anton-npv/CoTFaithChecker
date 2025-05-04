import os
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional

import importlib.util

# --------------------------------------------------------------------------------------
# Load shared experiment metadata from `data_prep.py`
# --------------------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent

# spec = importlib.util.spec_from_file_location("data_prep", THIS_DIR / "data_prep.py")
# data_prep = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(data_prep)  # type: ignore

# These variables are defined in j_probing/data_processing/data_prep.py
DATASET_NAME = "mmlu"  # e.g. "mmlu"
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = MODEL_PATH.split("/")[-1]
HINT_TYPE = "sycophancy"         # e.g. "sycophancy"
N_QUESTIONS = 5001     # e.g. 301

# --------------------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------------------
BASE_INPUT_DIR = Path("f_temp_check") / "outputs" / DATASET_NAME / MODEL_NAME / HINT_TYPE
ANALYSIS_FILE = BASE_INPUT_DIR / f"temp_analysis_summary_{N_QUESTIONS}.json"
RAW_GENS_FILE = BASE_INPUT_DIR / f"temp_generations_raw_{DATASET_NAME}_{N_QUESTIONS}.json"

# --------------------------------------------------------------------------------------
# Output path (create if necessary)
# --------------------------------------------------------------------------------------
OUTPUT_DIR = (
    Path("j_probing")
    / "data"
    / DATASET_NAME
    / MODEL_NAME
    / HINT_TYPE
    / str(N_QUESTIONS)
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "probing_data.json"


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

# We rely on the model tokenizer to obtain accurate token boundaries. Some model names
# (e.g. DeepSeek-R1-Distill-Llama-8B) may not be available on the Hugging Face hub or
# might require custom loading logic. We therefore attempt to load the tokenizer but
# gracefully fall back to whitespace tokenisation if that fails.

THINK_DELIM = "<think>\n"  # delimiter that separates prompt from assistant reasoning


try:
    from transformers import AutoTokenizer  # type: ignore

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except Exception as _e:  # pragma: no cover – informative warning only
    print(
        f"[create_probing_dataset] Warning: could not load tokenizer for '{MODEL_NAME}'. "
        "Falling back to naive whitespace tokenisation – token indices may be inaccurate."
    )
    _TOKENIZER = None


def _compute_probs(agg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Compute prob_verb_match from aggregated_counts."""
    match_hint = agg["match_hint_count"]

    prob_verb_match = (
        agg["match_and_verbalize_count"] / match_hint if match_hint else None
    )
    return {
        "prob_verb_match": prob_verb_match,
    }


def _extract_prompt(gen_str: str) -> str:
    """Return the prompt portion up to and including the last `<think>\n`."""
    idx = gen_str.rfind(THINK_DELIM)
    if idx == -1:
        prompt = gen_str
    else:
        prompt = gen_str[: idx + len(THINK_DELIM)]

    # Collapse any unwanted <think> immediately before the assistant header
    unwanted_seq = "\n<think>"  # '<think>' without newline
    if unwanted_seq in prompt:
        prompt = prompt.replace(unwanted_seq, "")
    return prompt


def _find_hint_token_idx(prompt: str) -> int:
    """Return the *token index* (0-based) corresponding to the hinted answer in the prompt.

    The heuristic is: locate the last occurrence of '[' (opening bracket) in the raw
    prompt and take the very next *token* that begins at (or immediately after) that
    position. This works for prompts of the form "... pointing to [ C ]." where the
    bracketed letter is the hinted answer. If the bracket cannot be found or the
    tokenizer fails, returns -1.
    """

    # Locate the last '[' character – this precedes the hinted option (e.g. "[ C ]")
    char_idx = prompt.rfind("[")
    if char_idx == -1:
        raise ValueError("No hint token found in prompt")

    if _TOKENIZER is None:
        # Naive split fallback
        raise ValueError("No tokenizer found")

    # Use the tokenizer's encode method (without special tokens) to count tokens before
    # the '[' character. Offsets are not returned by slow tokenisers, so we simply
    # re-encode the substring up to the bracket.
    sub_tokens = _TOKENIZER.encode(prompt[:char_idx], add_special_tokens=False)
    return len(sub_tokens)


def _token_positions(prompt: str) -> List[int]:
    """Return token positions for the given prompt:
    • assistant_idx – absolute token index of the `<|Assistant|>` header.
    • think_idx – absolute token index of the `<think>` delimiter.
    • hint_idx – absolute token index of the hinted answer option.
    • answer_idx – absolute token index of the first occurrence of "answer".
    • correct_idx – absolute token index of the first occurrence of "correct".
    • option_idx – absolute token index of the first occurrence of "option".
    • period_idx – absolute token index of the first '.' following "option".
    • after_hint_idx – absolute token index immediately after the hint token.
    • before_assistant_idx – absolute token index immediately before the assistant header.
    """
    # Fixed relative offsets for assistant and think tokens
    assistant_idx_rel = -3
    think_idx_rel = -2

    # Compute absolute indexes via tokenizer
    if _TOKENIZER:
        full_tokens = _TOKENIZER.encode(prompt, add_special_tokens=False)
        n_tokens = len(full_tokens)
        assistant_idx_abs = n_tokens + assistant_idx_rel
        think_idx_abs = n_tokens + think_idx_rel
    else:
        assistant_idx_abs = assistant_idx_rel
        think_idx_abs = think_idx_rel

    # Compute hint index (absolute)
    try:
        hint_idx_abs = _find_hint_token_idx(prompt)
    except Exception:
        hint_idx_abs = -1

    # Extract the instruction line to restrict keyword searches
    instr_line = None
    for line in prompt.splitlines():
        if line.strip().lower().startswith("please answer"):  # detect the instruction
            instr_line = line
            break
    if instr_line:
        instr_start = prompt.find(instr_line)
        instr_end = instr_start + len(instr_line)
        instr_sub = prompt[instr_start:instr_end]
    else:
        instr_start = instr_end = -1
        instr_sub = ""

    def find_instr_keyword_idx(keyword: str) -> int:
        # Find keyword only within the instruction line
        if _TOKENIZER is None or instr_line is None:
            return -1
        rel_pos = instr_sub.find(keyword)
        if rel_pos == -1:
            return -1
        abs_char = instr_start + rel_pos
        return len(_TOKENIZER.encode(prompt[:abs_char], add_special_tokens=False))

    answer_idx_abs = find_instr_keyword_idx("answer") - 1
    correct_idx_abs = find_instr_keyword_idx("correct") - 1
    option_idx_abs = find_instr_keyword_idx("option") - 1
    period_idx_abs = option_idx_abs + 1

    # Token immediately after hint
    after_hint_idx_abs = hint_idx_abs + 1 if hint_idx_abs >= 0 else -1

    # Token immediately before assistant header
    before_assistant_idx_abs = assistant_idx_abs - 1 if assistant_idx_abs >= 0 else -1

    return [
        assistant_idx_abs,
        think_idx_abs,
        hint_idx_abs,
        answer_idx_abs,
        correct_idx_abs,
        option_idx_abs,
        period_idx_abs,
        after_hint_idx_abs,
        before_assistant_idx_abs,
    ]


# --------------------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------------------

def main() -> None:
    # 1. Load analysis summary
    with open(ANALYSIS_FILE, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    # Build a mapping from question_id to computed stats
    summary_map: Dict[int, Dict[str, Any]] = {}
    for entry in analysis_data["results_per_question_summary"]:
        qid = entry["question_id"]
        stats = {
            "original_verbalizes_hint": entry["original_verbalizes_hint"],
            **_compute_probs(entry["aggregated_counts"]),
        }
        summary_map[qid] = stats

    # 2. Load raw generations (potentially large file)
    with open(RAW_GENS_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    probing_records: List[Dict[str, Any]] = []

    for q in raw_data["raw_generations"]:
        qid: int = q["question_id"]
        if qid not in summary_map:
            # Skip if not present in summary (should not happen)
            continue

        first_gen = q["generations"][0] if q["generations"] else ""
        prompt = _extract_prompt(first_gen)
        token_pos = _token_positions(prompt)

        record = {
            "question_id": qid,
            "prompt": prompt,
            "prob_verb_match": summary_map[qid]["prob_verb_match"],
            "token_pos": token_pos,
            "original_verbalizes_hint": summary_map[qid]["original_verbalizes_hint"],
        }
        probing_records.append(record)

        # Sanity check: Print tokens at calculated positions
        if _TOKENIZER:
            try:
                tokens = _TOKENIZER.tokenize(prompt)
                asst_idx_abs = token_pos[0]  # Convert relative -3 to absolute
                think_idx_abs = token_pos[1] # Convert relative -2 to absolute
                hint_idx_abs = token_pos[2]
                answer_idx_abs = token_pos[3]
                correct_idx_abs = token_pos[4]
                option_idx_abs = token_pos[5]
                period_idx_abs = token_pos[6]
                after_hint_idx_abs = token_pos[7]
                before_assistant_idx_abs = token_pos[8]

                print(f"--- QID: {qid} ---")
                print(f"Token @ Assistant Idx ({token_pos[0]} -> {asst_idx_abs}): '{tokens[asst_idx_abs]}'")
                print(f"Token @ Think Idx ({token_pos[1]} -> {think_idx_abs}): '{tokens[think_idx_abs]}'")
                if hint_idx_abs != -1:
                    print(f"Token @ Hint Idx ({hint_idx_abs}): '{tokens[hint_idx_abs]}'")
                else:
                    print("Hint token not found.")

                print(f"Token @ Answer Idx ({answer_idx_abs}): '{tokens[answer_idx_abs]}'")
                print(f"Token @ Correct Idx ({correct_idx_abs}): '{tokens[correct_idx_abs]}'")
                print(f"Token @ Option Idx ({option_idx_abs}): '{tokens[option_idx_abs]}'")
                print(f"Token @ Period Idx ({period_idx_abs}): '{tokens[period_idx_abs]}'")
                print(f"Token @ After Hint Idx ({after_hint_idx_abs}): '{tokens[after_hint_idx_abs]}'")
                print(f"Token @ Before Assistant Idx ({before_assistant_idx_abs}): '{tokens[before_assistant_idx_abs]}'")
                print("-" * (13 + len(str(qid)))) # Match width of header line
            except IndexError:
                print(f"[create_probing_dataset] Warning: QID {qid} - Token index out of range during sanity check.")
            except Exception as e:
                print(f"[create_probing_dataset] Warning: QID {qid} - Error during sanity check: {e}")

    # 3. Write out
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(probing_records, f, indent=2)

    # Explicitly free memory
    del raw_data
    gc.collect()

    print(f"Saved {len(probing_records)} records to {OUTPUT_JSON}")


if __name__ == "__main__":
    main() 