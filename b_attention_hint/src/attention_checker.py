# b_attention_hint/src/attention_checker.py
import os, json, logging, re, math, random, pathlib, gc, warnings
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------
# ----------  low-level helpers -----------------------------
# -----------------------------------------------------------
def _char_span_of_substring(full: str, sub: str) -> Tuple[int, int]:
    """Return (start, end) char indices (inclusive, exclusive). Raises ValueError if not found."""
    start = full.find(sub)
    if start == -1:
        raise ValueError("Could not find hint_text in prompt_text.")
    return start, start + len(sub)

def _hint_token_indices(tokenizer, prompt_text: str, hint_text: str):
    """
    Return the indices (list[int]) of tokens whose offsets overlap the hint span.
    Uses the fast tokenizer's offset mapping.
    """
    start, end = _char_span_of_substring(prompt_text, hint_text)
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    hint_token_ids = []
    for idx, (s, e) in enumerate(enc["offset_mapping"]):
        if e is None:  # some tokenizers return (0,0) for special tokens
            continue
        if (s < end) and (e > start):  # overlap
            hint_token_ids.append(idx)
    return hint_token_ids, enc["input_ids"]

def _mean_attention_to_indices(attn_tensor: torch.Tensor,
                               target_indices: List[int]) -> torch.Tensor:
    """
    attn_tensor: shape (num_layers, num_heads, seq_len, seq_len)
    returns tensor shape (seq_len,) giving layer+head averaged attention
            from each query position TO the target_indices (hint tokens)
    """
    if len(target_indices) == 0:
        raise ValueError("No target indices for hint?")

    # average over layers and heads first
    # -> shape (seq_len, seq_len)
    a = attn_tensor.mean(dim=0).mean(dim=0)
    # sum over key positions that belong to the hint
    # -> shape (seq_len,)
    scores = a[:, target_indices].sum(dim=-1)
    return scores  # length == seq_len

# -----------------------------------------------------------
# ----------  main analysis utilities -----------------------
# -----------------------------------------------------------
def _compute_attention_curve(model, tokenizer, prompt_text, completion_text,
                             hint_text, device="cuda"):
    """
    Returns a list[float] of length len(completion_tokens)
    giving attention-to-hint for each generated token.
    """
    # find hint token indices _within prompt_
    try:
        hint_token_indices, prompt_ids = _hint_token_indices(tokenizer, prompt_text, hint_text)
    except ValueError:
        logger.warning("Hint text not found; skipping example.")
        return None

    # full input (prompt + completion); keep EOS
    full_text = prompt_text + completion_text
    enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc_full["input_ids"].to(device)
    prompt_len = len(prompt_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True)

    # At inference the model returns attentions per layer; convert to tensor
    # (layers, batch, heads, seq, seq) -> squeeze batch
    attn = torch.stack(out.attentions).squeeze(1).to("cpu")  # (L, H, S, S)

    scores = _mean_attention_to_indices(attn, hint_token_indices)  # (S,)
    # we only need the part that corresponds to generated tokens
    gen_scores = scores[prompt_len:].tolist()
    return gen_scores


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------
# ----------  public API  -----------------------------------
# -----------------------------------------------------------
def attention_check(model,
                    tokenizer,
                    model_name: str,
                    device: torch.device,
                    dataset_name: str,
                    hint_type: str,
                    baseline_hint_type: str = "none",
                    n_questions: Optional[int] = None,
                    probe_first_k: int = 10,
                    seed: int = 42,
                    verbose: bool = True):
    """
    High-level convenience function. Returns a dict with:
      - mean_curve_hint / baseline / diff
      - auc_hint / auc_baseline / auc_diff
      - probe_acc / probe_roc_auc
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def _path(ht):
        base = f"data/{dataset_name}/{model_name}/{ht}/completions_with_{n_questions}.json" \
               if n_questions is not None else \
               f"data/{dataset_name}/{model_name}/{ht}/completions.json"
        if not os.path.exists(base):
            raise FileNotFoundError(base)
        return base

    hint_path     = _path(hint_type)
    baseline_path = _path(baseline_hint_type)

    hint_examples     = _load_json(hint_path)
    baseline_examples = _load_json(baseline_path)

    if n_questions is not None:
        hint_examples     = hint_examples[:n_questions]
        baseline_examples = baseline_examples[:n_questions]

    # --- 1. compute curves ---------------------------------------------------
    curves_hint, curves_base = [], []

    for ex in hint_examples:
        curve = _compute_attention_curve(model, tokenizer,
                                         ex["prompt_text"],
                                         ex["generated_text"],
                                         ex.get("hint_text", ""),
                                         device)
        if curve:  # skip None
            curves_hint.append(curve)

    for ex in baseline_examples:
        curve = _compute_attention_curve(model, tokenizer,
                                         ex["prompt_text"],
                                         ex["generated_text"],
                                         ex.get("hint_text", ""),  # empty
                                         device)
        if curve:
            curves_base.append(curve)

    # pad to equal length with NaN then np.nanmean
    max_len = max(max(map(len, curves_hint)), max(map(len, curves_base)))
    def _pad(l):
        return np.array([np.pad(x, (0, max_len-len(x)), constant_values=np.nan)
                         for x in l])

    mean_curve_hint = np.nanmean(_pad(curves_hint), axis=0)
    mean_curve_base = np.nanmean(_pad(curves_base), axis=0)
    diff_curve      = mean_curve_hint - mean_curve_base

    auc_hint = np.nansum(mean_curve_hint)
    auc_base = np.nansum(mean_curve_base)
    auc_diff = auc_hint - auc_base

    # --- 2. probing ----------------------------------------------------------
    # Load labels: whether hint was verbalised
    hv_path = f"data/{dataset_name}/{model_name}/{hint_type}/hint_verification_with_{n_questions}.json" \
              if n_questions is not None else \
              f"data/{dataset_name}/{model_name}/{hint_type}/hint_verification.json"
    hv      = _load_json(hv_path)
    hv_dict = {d["question_id"]: d["verbalised"] for d in hv}

    X, y = [], []
    for ex, curve in zip(hint_examples, curves_hint):
        qid = ex["question_id"]
        if qid not in hv_dict:
            continue
        feature = np.array(curve[:probe_first_k] + [0.0]*(probe_first_k-len(curve)))
        X.append(feature)
        y.append(int(hv_dict[qid]))

    X, y = np.array(X), np.array(y)
    if len(set(y)) < 2:
        warnings.warn("Not enough positive/negative examples for probe.")
        probe_acc = probe_auc = float("nan")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:,1]

        probe_acc = accuracy_score(y_test, y_pred)
        probe_auc = roc_auc_score(y_test, y_proba)

    # --- 3. return results ---------------------------------------------------
    res = dict(
        num_examples_hint=len(curves_hint),
        num_examples_baseline=len(curves_base),
        mean_curve_hint   = mean_curve_hint.tolist(),
        mean_curve_base   = mean_curve_base.tolist(),
        diff_curve        = diff_curve.tolist(),
        auc_hint=float(auc_hint),
        auc_baseline=float(auc_base),
        auc_diff=float(auc_diff),
        probe_first_k=probe_first_k,
        probe_acc=float(probe_acc),
        probe_auc=float(probe_auc),
    )

    if verbose:
        logger.info("---------- Attention to hint summary ----------")
        logger.info(f"AUC (hint)     : {auc_hint:.4f}")
        logger.info(f"AUC (baseline) : {auc_base:.4f}")
        logger.info(f"AUC diff       : {auc_diff:.4f}")
        logger.info(f"Probe acc       {probe_acc:.3f} | ROC-AUC {probe_auc:.3f}")
        logger.info("----------------------------------------------")

    # cleanup
    torch.cuda.empty_cache(); gc.collect()
    return res
