# b_attention_hint/src/attention_checker.py
"""Utilities for analysing how much attention a model pays to the hint token(s)
   and for a simple probing experiment that predicts whether the hint will be
   verbalised in the chain‑of‑thought.

   Main entry point
   ----------------
   >>> from b_attention_hint.src.attention_checker import attention_check
   >>> results = attention_check(dataset_name, hint_types, model_name,
                                 model, tokenizer, device, n_questions)

   The call returns a nested dict and also persists the results under
   ``data/<dataset>/ <model_name>/attention_check_<n>.json`` so you can reload
   them later without recomputation.
"""
from __future__ import annotations

from pathlib import Path
import json
import random
import os
from typing import List, Dict, Any, Sequence, Tuple, Optional

import numpy as np
import torch
from torch import tensor
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score)


# -----------------------------------------------------------------------------
# -----------------------------  helpers  --------------------------------------
# -----------------------------------------------------------------------------

def _load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _find_subsequence(sub: Sequence[int], seq: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Return (start, end‑exclusive) of the *first* occurrence of ``sub`` in ``seq``.
    If not found, return None."""
    if len(sub) == 0 or len(seq) == 0 or len(sub) > len(seq):
        return None
    first = sub[0]
    for i in range(len(seq) - len(sub) + 1):
        if seq[i] == first and seq[i : i + len(sub)] == list(sub):
            return i, i + len(sub)
    return None


def _mean_attention_to_indices(attentions: Tuple[torch.Tensor, ...],
                               indices: List[int]) -> float:
    """Return mean attention **from every token** *to* the tokens in ``indices``.

    We take the last layer (closest to the logits) and average across heads
    (dimension: heads x seq x seq).
    """
    if len(indices) == 0:
        return 0.0
    last_layer = attentions[-1][0]  # shape: (heads, seq, seq)
    # mean over heads ‑> (seq, seq)
    mean_over_heads = last_layer.mean(0)
    # attention directed *to* the hint tokens
    to_hint = mean_over_heads[:, indices]  # (seq, len(indices))
    return to_hint.mean().item()


# -----------------------------------------------------------------------------
# ----------------------  core attention experiment  ---------------------------
# -----------------------------------------------------------------------------

def _compute_attention_for_sample(model, tokenizer, device, prompt_text: str,
                                  completion_text: str, hint_text: str) -> float:
    """Forward pass over *prompt + completion* and measure attention to the hint."""
    with torch.no_grad():
        full_text = prompt_text + completion_text
        # tokenise *once* to keep alignment
        tok = tokenizer(full_text, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        # obtain attentions in one forward pass. use caching disabled to save memory
        out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
        attentions = out.attentions  # Tuple[L]

    # locate hint tokens inside full sequence ---------------------------------
    hint_token_ids = tokenizer(hint_text, add_special_tokens=False)["input_ids"] if hint_text else []
    seq = input_ids[0].tolist()
    hint_span = _find_subsequence(hint_token_ids, seq) if hint_token_ids else None
    if hint_span is None:  # if we fail to align fallback to 0
        return 0.0
    hint_indices = list(range(hint_span[0], hint_span[1]))
    return _mean_attention_to_indices(attentions, hint_indices)


# -----------------------------------------------------------------------------
# ---------------------------  probing setup  ----------------------------------
# -----------------------------------------------------------------------------

def _train_probe(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """Simple Logistic‑regression probe on the attention feature."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y)) > 1 else float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "y_true_test": y_test.tolist(),
        "y_pred_test": y_pred.tolist(),
        "y_proba_test": y_proba.tolist(),
    }


# -----------------------------------------------------------------------------
# ------------------------------  main API  ------------------------------------
# -----------------------------------------------------------------------------

def attention_check(
    dataset_name: str,
    hint_types: List[str],
    model_name: str,
    model,
    tokenizer,
    device: str,
    n_questions: int,
    cache_path: str | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run both the *attention‑difference* measurement and the *verbalisation* probe.

    Parameters
    ----------
    dataset_name : str
        "mmlu", etc.
    hint_types : list of str
        Must contain "none" as the baseline plus the concrete hint categories.
    model_name : str
        Used for path lookup and output folders.
    model / tokenizer :  HuggingFace instances already loaded in your pipeline.
    device : str  ("cuda" | "cpu")
    n_questions : int
        Number of questions in the current run (for path construction).
    cache_path : str | None
        If set and the json file exists the function will *return* its content
        without recomputation.  A fresh run will (over)write the same path.
    seed : int
        Random seed for the simple probe split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    if cache_path is None:
        cache_path = (
            Path("data")
            / dataset_name
            / model_name
            / f"attention_check_{n_questions}.json"
        )

    cache_path = Path(cache_path)
    if cache_path.exists():
        return _load_json(cache_path)

    # ------------------------------------------------------------------
    base_dir = Path("data") / dataset_name
    input_questions = _load_json(base_dir / "input_mcq_data.json")
    questions_by_id = {q["question_id"]: q for q in input_questions}

    attention_per_q: Dict[int, Dict[str, float]] = {}

    for hint_type in tqdm(hint_types, desc="⚡ computing attentions"):
        if hint_type == "none":
            hint_entries = {e["question_id"]: {"hint_text": ""} for e in input_questions}
        else:
            hint_file = base_dir / f"hints_{hint_type}.json"
            hint_entries_raw = _load_json(hint_file)
            hint_entries = {e["question_id"]: e for e in hint_entries_raw}

        completions_file = (
            base_dir
            / model_name
            / hint_type
            / f"completions_with_{n_questions}.json"
        )
        completions_raw = _load_json(completions_file)
        completions = {c["question_id"]: c["completion"] for c in completions_raw}

        for qid, comp_text in completions.items():
            question_entry = questions_by_id[qid]
            hint_text = hint_entries.get(qid, {}).get("hint_text", "")

            # NB: prompt was already included in `comp_text` (the saved file),
            # so we treat everything before the assistant's first token as prompt.
            # We still need the hint text string for alignment inside the sequence.
            attention_val = _compute_attention_for_sample(
                model, tokenizer, device,
                prompt_text="",  # already in `comp_text`
                completion_text=comp_text,
                hint_text=hint_text,
            )
            attention_per_q.setdefault(qid, {})[hint_type] = attention_val

    # ------------------------------------------------------------------
    # ----- aggregate attention difference v baseline ------------------
    baseline_name = "none"
    delta_stats: Dict[str, Dict[str, float]] = {}
    for hint_type in hint_types:
        if hint_type == baseline_name:
            continue
        diffs = []
        for qid, vals in attention_per_q.items():
            if baseline_name in vals and hint_type in vals:
                diffs.append(vals[hint_type] - vals[baseline_name])
        if diffs:
            delta_stats[hint_type] = {
                "mean": float(np.mean(diffs)),
                "std": float(np.std(diffs)),
                "n": len(diffs),
            }

    # ------------------------------------------------------------------
    # ------------------  probing: verbalisation -----------------------
    X_all, y_all = [], []
    for hint_type in hint_types:
        if hint_type == "none":
            continue
        hint_ver_path = (
            base_dir
            / model_name
            / hint_type
            / f"hint_verification_with_{n_questions}.json"
        )
        ver_entries = _load_json(hint_ver_path)
        ver_by_id = {v["question_id"]: v for v in ver_entries}
        for qid, vals in attention_per_q.items():
            if hint_type not in vals:
                continue
            if qid not in ver_by_id:
                continue
            X_all.append([vals[hint_type]])  # one‑dim feature
            y_all.append(int(ver_by_id[qid]["verbalizes_hint"]))

    X = np.asarray(X_all, dtype=float)
    y = np.asarray(y_all, dtype=int)
    if len(np.unique(y)) < 2:
        probe_metrics = {"error": "y has only one class – cannot train"}
    else:
        probe_metrics = _train_probe(X, y, random_state=seed)

    # ------------------------------------------------------------------
    result = {
        "per_question": attention_per_q,
        "attention_difference": delta_stats,
        "probe": probe_metrics,
    }

    _save_json(result, cache_path)
    return result


# -----------------------------------------------------------------------------
# If this module is executed directly, run a tiny smoke test -------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from a_confirm_posthoc.src.main.pipeline import load_model_and_tokenizer

    parser = argparse.ArgumentParser(description="Quick smoke‑test for attention checker")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer, model_name, device = load_model_and_tokenizer(args.model)
    res = attention_check(
        dataset_name="mmlu",
        hint_types=["none", "sycophancy"],
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_questions=args.n_questions,
    )
    print(json.dumps(res["attention_difference"], indent=2))
