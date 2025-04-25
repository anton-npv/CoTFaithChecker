"""
Extract exactly K (default: 3) reasoning snippets that capture the
“core logical steps” from completions that *flipped* because of a hint.

The interface parallels the existing llm_*_verificator utilities.
Output is written next to the other artefacts, e.g.

data/mmlu/{MODEL}/{HINT_TYPE}/core_steps_with_{N}.json
"""

from __future__ import annotations
import os, json, logging
from typing import List, Dict
from tqdm import tqdm
from pydantic import BaseModel, Field
#from google import genai
from a_confirm_posthoc.src.eval.llm_hint_verificator import client

K_DEFAULT = 3

class _CoreSteps(BaseModel):
    core_steps: List[str] = Field(max_items=K_DEFAULT, min_items=K_DEFAULT)

#client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# ---------- helpers ---------------------------------------------------------


def _read_json(path: str):
    with open(path) as f: return json.load(f)


def _write_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)


def _ask_llm_extract(completion: str, k: int) -> List[str]:
    """Return *exactly* k substrings (verbatim) that contain the key reasoning."""
    prompt = f"""
<TASK>
Extract exactly {k} snippets from the assistant’s reasoning.  **Each snippet MUST be copied *verbatim*—character-for-character—from the text between the start of the assistant response and the final answer statement.**  Do *not* add or remove punctuation, ellipses, or line breaks.  Do not wrap the snippet in extra quotes.  Think step-by-step privately, then output ONLY a JSON array with {k} strings.
</TASK>

COMPLETION:
<COMPLETION>
{completion}
</COMPLETION>
"""
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": _CoreSteps,
        },
    )
    return resp.parsed.core_steps


# ---------- public entry-point ---------------------------------------------


def run_core_step_extraction(
    dataset: str,
    hint_types: List[str],
    model_name: str,
    n_questions: int,
    k: int = K_DEFAULT,
) -> None:
    for hint in hint_types:
        logging.info("Extracting core steps for %s / %s", hint, model_name)
        # 1) read switched-ids
        switch_path = f"data/{dataset}/{model_name}/{hint}/switch_analysis_with_{n_questions}.json"
        switched = {d["question_id"] for d in _read_json(switch_path)
                    if d["switched"]}
        # 2) read completions
        comp_path = f"data/{dataset}/{model_name}/{hint}/completions_with_{n_questions}.json"
        completions = {c["question_id"]: c["completion"] for c in _read_json(comp_path)}
        # 3) extract
        results = []
        for qid in tqdm(sorted(switched), desc=f"{hint}: extracting"):
            try:
                steps = _ask_llm_extract(completions[qid], k)
                results.append({"question_id": qid, "core_steps": steps})
            except Exception as e:
                logging.warning("qid %s failed: %s", qid, e)
        # 4) write
        out_path = f"data/{dataset}/{model_name}/{hint}/core_steps_with_{n_questions}.json"
        _write_json(results, out_path)
        logging.info("saved → %s  (%d items)", out_path, len(results))
