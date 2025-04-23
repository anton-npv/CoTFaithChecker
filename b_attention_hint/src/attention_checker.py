# b_attention_hint/src/attention_checker.py
import os, json, logging, random, warnings, gc, re
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# ------------------------- helpers ----------------------------------------- #
# --------------------------------------------------------------------------- #
_ASSISTANT_RE = re.compile(r"(?:^|\n)\s*assistant\s*:?", re.I)

def _char_span_of_substring(full: str, sub: str) -> Tuple[int, int]:
    start = full.find(sub)
    if start == -1:
        raise ValueError("Could not find hint_text in prompt_text.")
    return start, start + len(sub)

def _hint_token_indices(tokenizer, prompt_text: str, hint_text: str):
    start, end = _char_span_of_substring(prompt_text, hint_text)
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    idxs = [i for i, (s, e) in enumerate(enc["offset_mapping"])
            if e is not None and (s < end) and (e > start)]
    return idxs, enc["input_ids"]

def _mean_attn_to(attn: torch.Tensor, key_indices: List[int]) -> torch.Tensor:
    if not key_indices:
        raise ValueError("No hint token indices.")
    a = attn.mean(dim=0).mean(dim=0)         # (seq, seq)
    return a[:, key_indices].sum(dim=-1)      # (seq,)

def _split_user_assistant(full: str) -> Tuple[str, str]:
    """
    Split a single string that contains both sides of the chat into
    (prompt_text, assistant_text).  Falls back to half-split if pattern
    not found.
    """
    m = _ASSISTANT_RE.search(full)
    if not m:
        # crude fallback – treat whole as prompt, empty completion
        logger.warning("Could not locate 'Assistant:' marker – skipping example.")
        return None, None
    cut = m.start()
    return full[:cut], full[cut + len(m.group(0)) :]

def _load_json(path: str):
    with open(path, "r") as fh:
        return json.load(fh)

# --------------------------------------------------------------------------- #
# ------------------------- core -------------------------------------------- #
# --------------------------------------------------------------------------- #
def _compute_attention_curve(model,
                             tokenizer,
                             prompt_text: str,
                             completion_text: str,
                             hint_text: str,
                             device="cuda"):
    try:
        hint_token_idx, prompt_ids = _hint_token_indices(tokenizer,
                                                         prompt_text,
                                                         hint_text)
    except ValueError:
        # hint not present in prompt – nothing to measure
        return None

    full_text = prompt_text + completion_text
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    prompt_len = len(prompt_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True)

    attn = torch.stack(out.attentions).squeeze(1).cpu()   # (L, H, S, S)
    scores = _mean_attn_to(attn, hint_token_idx)          # (seq,)
    return scores[prompt_len:].tolist()                   # only generation part

# --------------------------------------------------------------------------- #
# ------------------------- public API -------------------------------------- #
# --------------------------------------------------------------------------- #
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
                    verbose: bool = True) -> Dict:

    rng = random.Random(seed)
    np.random.seed(seed); torch.manual_seed(seed); rng.seed(seed)

    def _path(ht: str) -> str:
        if n_questions is not None:
            p = f"data/{dataset_name}/{model_name}/{ht}/completions_with_{n_questions}.json"
        else:
            p = f"data/{dataset_name}/{model_name}/{ht}/completions.json"
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    hint_examples     = _load_json(_path(hint_type))
    baseline_examples = _load_json(_path(baseline_hint_type))

    if n_questions is not None:
        hint_examples     = hint_examples[:n_questions]
        baseline_examples = baseline_examples[:n_questions]

    # --- load hint texts ----------------------------------------------------
    hints_file = f"data/{dataset_name}/hints_{hint_type}.json"
    hints_dict = {h["question_id"]: h["hint_text"] for h in _load_json(hints_file)} \
                 if os.path.exists(hints_file) else {}

    def _extract(ex, with_hint: bool):
        """Return prompt_text, completion_text, hint_text (may be empty)."""
        prompt = ex.get("prompt_text")
        completion = ex.get("generated_text")
        if prompt is None or completion is None:
            # legacy single-string format
            all_text = ex.get("completion") or ""
            prompt, completion = _split_user_assistant(all_text)
            if prompt is None:
                return None, None, None
        hint = ex.get("hint_text") or (hints_dict.get(ex["question_id"], "") if with_hint else "")
        return prompt, completion, hint

    # --- compute curves -----------------------------------------------------
    curves_hint, curves_base = [], []

    for ex in hint_examples:
        p, c, h = _extract(ex, with_hint=True)
        if p is None:
            continue
        curve = _compute_attention_curve(model, tokenizer, p, c, h, device)
        if curve:
            curves_hint.append(curve)

    for ex in baseline_examples:
        p, c, _ = _extract(ex, with_hint=False)
        if p is None:
            continue
        curve = _compute_attention_curve(model, tokenizer, p, c, "", device)
        if curve:
            curves_base.append(curve)

    if not curves_hint or not curves_base:
        raise RuntimeError("No valid examples after parsing; cannot compute attention curves.")

    # pad to same length
    max_len = max(max(map(len, curves_hint)), max(map(len, curves_base)))
    pad = lambda lst: np.array([np.pad(x, (0, max_len - len(x)), np.nan) for x in lst])

    mean_hint = np.nanmean(pad(curves_hint), axis=0)
    mean_base = np.nanmean(pad(curves_base), axis=0)
    diff      = mean_hint - mean_base

    auc_hint = float(np.nansum(mean_hint))
    auc_base = float(np.nansum(mean_base))
    auc_diff = auc_hint - auc_base

    # ---------- probing -----------------------------------------------------
    hv_path = f"data/{dataset_name}/{model_name}/{hint_type}/hint_verification_with_{n_questions}.json" \
              if n_questions is not None else \
              f"data/{dataset_name}/{model_name}/{hint_type}/hint_verification.json"
    hv = _load_json(hv_path) if os.path.exists(hv_path) else []
    hv_dict = {d["question_id"]: d["verbalizes_hint"] if "verbalizes_hint" in d else d["verbalised"]
               for d in hv}

    X, y = [], []
    for ex, curve in zip(hint_examples, curves_hint):
        qid = ex["question_id"]
        if qid not in hv_dict:
            continue
        feat = np.array(curve[:probe_first_k] + [0] * max(0, probe_first_k - len(curve)))
        X.append(feat); y.append(int(hv_dict[qid]))

    if len(set(y)) < 2:
        warnings.warn("Probe skipped – only one class present.")
        probe_acc = probe_auc = float("nan")
    else:
        X = np.array(X); y = np.array(y)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                              random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        probe_acc = accuracy_score(yte, clf.predict(Xte))
        probe_auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    if verbose:
        logger.info("----------- attention-to-hint summary ------------")
        logger.info("AUC hint      %.4f", auc_hint)
        logger.info("AUC baseline  %.4f", auc_base)
        logger.info("Δ AUC         %.4f", auc_diff)
        logger.info("Probe  acc %.3f | ROC-AUC %.3f", probe_acc, probe_auc)
        logger.info("--------------------------------------------------")

    torch.cuda.empty_cache(); gc.collect()

    return dict(
        num_examples_hint=len(curves_hint),
        num_examples_baseline=len(curves_base),
        mean_curve_hint=mean_hint.tolist(),
        mean_curve_base=mean_base.tolist(),
        diff_curve=diff.tolist(),
        auc_hint=auc_hint,
        auc_baseline=auc_base,
        auc_diff=auc_diff,
        probe_first_k=probe_first_k,
        probe_acc=probe_acc,
        probe_auc=probe_auc,
    )
