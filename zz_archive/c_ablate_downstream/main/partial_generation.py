"""
Generate MC-question answers *incrementally* and record the model’s
confidence at several %-of-generation checkpoints.
"""
from typing import List, Dict
import torch, json, os, math, logging, tqdm
from ..utils.confidence_tools import ANSWER_TOKEN_IDS, option_probs
from ..utils.prompt_constructor import construct_prompt

# --- top of file -----------------------------------------------------------
from typing import List, Dict
import os, math, json, tqdm, torch, logging
from ..utils.confidence_tools import ANSWER_TOKEN_IDS
from ..utils.prompt_constructor import construct_prompt


# helper: compute P(A-D | provisional prompt)
# helper ────────────────────────────────────────────────────────────────────
def _probe_probs(model, tokenizer, past, device, answer_ids):
    """
    Advance the model state with the probe prompt *token-by-token* so the
    KV cache stays consistent, then return a renormalised distribution over
    A/B/C/D for the first token *after* the "[".
    """
    probe_ids = tokenizer(
        "\n\nProvisional answer (A–D only): [",
        add_special_tokens=False
    )["input_ids"]

    with torch.no_grad():
        for tid in probe_ids:                            # ► one token each
            inp = torch.tensor([[tid]], device=device)
            out = model(input_ids=inp,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True)
            past = out.past_key_values                  # update cache

        logits = out.logits[0, -1]                      # after the "["

    # restrict to A–D and renormalise
    ids   = [answer_ids[ltr] for ltr in answer_ids.keys()]
    vals  = torch.softmax(logits[ids], dim=-1)
    return {ltr: vals[i].item() for i, ltr in enumerate(answer_ids.keys())}



# --- public api -----------------------------------------------------------

def run_partial_inference(model, tokenizer, device,
                          questions: List[Dict],
                          hint_type: str,
                          out_dir: str,
                          stop_percents=(0.0, .10, .20, .30),
                          max_new_tokens: int = 128,
                          temperature: float = 0.0):

    os.makedirs(out_dir, exist_ok=True)
    answer_ids = ANSWER_TOKEN_IDS(tokenizer)
    checkpoints = {round(p * max_new_tokens) for p in stop_percents}
    results = []

    for q in tqdm.tqdm(questions, desc=f"partial-gen {hint_type}"):
        prompt = construct_prompt(q)
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        past = None
        generated: List[int] = []

        for step in range(max_new_tokens + 1):
            # -- forward ---------------------------------------------------
            with torch.no_grad():
                if step == 0:
                    out = model(**enc, use_cache=True, return_dict=True)
                else:
                    nid = torch.tensor([[generated[-1]]], device=device)
                    out = model(input_ids=nid, past_key_values=past,
                                use_cache=True, return_dict=True)

            past   = out.past_key_values
            logits = out.logits[0, -1]

            # -- probe at checkpoints -------------------------------------
            if step in checkpoints:
                probs = _probe_probs(model, tokenizer, past, device, answer_ids)
                results.append({
                    "question_id":      q["question_id"],
                    "hint_type":        hint_type,
                    "tokens_generated": step,
                    "percent":          round(step / max_new_tokens, 3),
                    "probabilities":    probs
                })
                if step == max(checkpoints):
                    break

            # -- normal generation ----------------------------------------
            next_id = logits.argmax()
            generated.append(next_id.item())

        torch.cuda.empty_cache()      # keeps 80 GB A100 happy

    json.dump(results,
              open(os.path.join(out_dir, "confidence_snapshots.json"), "w"),
              indent=2)
    logging.info("confidence snapshots written to %s", out_dir)
