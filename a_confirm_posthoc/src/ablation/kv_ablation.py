"""
Run *token-level KV-cache ablations*:

 - first reasoning step
 - second reasoning step
 - third reasoning step
 - all three steps
 - entire CoT (everything after the user prompt)

Produces a JSON file for every ablation kind with the final answer and
a boolean 'changed' flag (relative to the verified answer).

Ablations can be disabled globally via kwargs.
"""

from __future__ import annotations
import os, json, logging, copy
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# ---------------------------------------------------------------------------

def _read_json(p):            return json.load(open(p))
def _write_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=2)

# ----------------------- token span helpers --------------------------------

def _char_span_to_token_span(text: str, span: str, tokenizer) -> range:
    """Return the *token indices* that cover the first occurrence of `span`."""
    start = text.find(span)
     # exact match first
    start = text.find(span)
    if start == -1:
         # tolerate collapsed whitespace / stray backticks
        import re
        normalise = lambda s: re.sub(r"\s+", " ", s.strip("` "))
        start = text.find(normalise(span))
    if start == -1:
        raise ValueError("substring not found")
    end = start + len(span)
    tok_ids = tokenizer(text, add_special_tokens=False).input_ids
    # reconstruct mapping char→token naive but robust
    offsets = tokenizer(text,
                        add_special_tokens=False,
                        return_offsets_mapping=True).offset_mapping
    tk_start = next(i for i,(s,_) in enumerate(offsets) if s>=start or start in range(s,_))
    tk_end   = max(i for i,(_,e) in enumerate(offsets) if e<=end)
    return range(tk_start, tk_end+1)

# -------------------------- KV hook ----------------------------------------

# ---------------------------------------------------------------------------
# KV-cache ablator – zero-out key & value tensors for selected token spans
# ---------------------------------------------------------------------------

class Ablator:
    def __init__(self, spans_to_ablate):
        self.spans = spans_to_ablate
        self.hook  = None

    def register(self, model):
        def _zero_kv(module, inputs, output):
            pkv = getattr(output, "past_key_values", None)
            if pkv is None:
                return
            for layer_past in pkv:
                k, v = layer_past[:2]           # robust: just take first 2 tensors
                #for span in self.spans:
                #    k[..., span, :] = 0
                #    v[..., span, :] = 0

                seq_len = k.size(-2)
                for span in self.spans:
                    # clip to valid range
                    s = max(0, span.start)
                    e = min(seq_len, span.stop)
                    if s >= e:            # span lies beyond sequence
                        continue          # -> nothing to zero here
                    k[..., s:e, :] = 0
                    v[..., s:e, :] = 0
        self.hook = model.register_forward_hook(_zero_kv)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None


# ------------------------- main runner -------------------------------------

EXPERIMENTS = {
    "first":  ("first reasoning step",  ["first"]),
    "second": ("second reasoning step", ["second"]),
    "third":  ("third reasoning step",  ["third"]),
    "all":    ("steps 1+2+3",           ["first","second","third"]),
    "cot":    ("full chain-of-thought", ["cot"]),
}

def run_kv_ablation_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    *,
    dataset: str,
    hint_types: List[str],
    model_name: str,
    n_questions: int,
    run_first=True, run_second=True, run_third=True,
    run_all=True, run_cot=True,
):
    # map bool flags → list of exp-ids we really run
    to_run = {k for k,flag in
              zip(("first","second","third","all","cot"),
                  (run_first,run_second,run_third,run_all,run_cot)) if flag}
    # load verified answers once
    verified_none = _read_json(
        f"data/{dataset}/{model_name}/none/verification_with_{n_questions}.json")
    gold = {d["question_id"]: d["verified_answer"] for d in verified_none}

    for hint in hint_types:
        # completions (only switched so they already have core_steps json)
        steps = _read_json(
            f"data/{dataset}/{model_name}/{hint}/core_steps_with_{n_questions}.json")
        step_map = {d["question_id"]: d["core_steps"] for d in steps}

        #results_dir = f"data/{dataset}/{model_name}/{hint}/kv_ablation/"
        #os.makedirs(results_dir, exist_ok=True)
        results_dir = f"data/{dataset}/{model_name}/{hint}/kv_ablation/"
        os.makedirs(results_dir, exist_ok=True)
        all_records = []          # ← collect everything for this hint-type

        skipped = 0

        for qid, substrings in tqdm(step_map.items(), desc=f"{hint}: ablation"):
            # build ablation token spans
            #completion_path = f"data/{dataset}/{model_name}/{hint}/completions_with_{n_questions}.json"
            #completion_text = next(c["completion"] for c in _read_json(completion_path)
            #                       if c["question_id"]==qid)

            # token spans for each step
            #span_first  = _char_span_to_token_span(completion_text, substrings[0], tokenizer)
            #span_second = _char_span_to_token_span(completion_text, substrings[1], tokenizer)
            #span_third  = _char_span_to_token_span(completion_text, substrings[2], tokenizer)

            #completion_path = f".../completions_with_{n_questions}.json"
            completion_path = f"data/{dataset}/{model_name}/{hint}/completions_with_{n_questions}.json"
            completion_text = next(c["completion"] for c in _read_json(completion_path)
                                if c["question_id"]==qid)

            try:
                span_first  = _char_span_to_token_span(completion_text, substrings[0], tokenizer)
                span_second = _char_span_to_token_span(completion_text, substrings[1], tokenizer)
                span_third  = _char_span_to_token_span(completion_text, substrings[2], tokenizer)
            except ValueError:
                logging.warning("qid %s skipped – core-step substring not found.", qid)
                logging.info("%d of %d questions skipped for %s", skipped, len(step_map), hint)
                skipped += 1
                continue      # ► skip this example and move on

            # everything after the user→assistant boundary
            cot_start = completion_text.index("assistant")  # using template
            span_cot  = _char_span_to_token_span(completion_text,
                                                 completion_text[cot_start:], tokenizer)

            span_map = {
                "first":  [span_first],
                "second": [span_second],
                "third":  [span_third],
                "all":    [span_first, span_second, span_third],
                "cot":    [span_cot],
            }

            prompt_only = completion_text.split("assistant")[0]  # user part

            for exp in to_run:
                spans = span_map[exp]
                #abl = Ablator(spans, num_layers=len(model.config.hidden_size))
                #abl = Ablator(spans, num_layers=model.config.num_hidden_layers)
                abl = Ablator(spans)
                abl.register(model)
                #out = model.generate(
                #    **tokenizer(prompt_only, return_tensors="pt").to(device),
                #    max_new_tokens=0,  # we only need logits at EOS
                context_text = completion_text.split("**Answer")[0] 
                out = model.generate(
                    **tokenizer(context_text, return_tensors="pt").to(device),
                    max_new_tokens=1,  #stub token :/
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True)
                abl.remove()

                # pick the most probable choice letter
                #logits = out.scores[-1][0]  # vocab-size
                logits = out.scores[0][0]   # only one score tensor was returned
                letter_tokens = tokenizer.convert_tokens_to_ids(["A","B","C","D"])
                probs = torch.softmax(logits[letter_tokens], dim=0)
                answer = ["A","B","C","D"][probs.argmax().item()]

                changed = answer != gold[qid]
                """_write_json(
                    {
                        "question_id": qid,
                        "experiment": exp,
                        "new_answer": answer,
                        "changed": changed,
                    },
                    f"{results_dir}/{qid}_{exp}.json"
                )"""
                all_records.append({
                    "question_id": qid,
                    "experiment": exp,
                    "new_answer": answer,
                    "changed": changed,
                })

        # after **all** QIDs processed, dump one file for this hint-type
        out_path = os.path.join(results_dir,
                                f"kv_ablation_results_with_{n_questions}.json")
        _write_json(all_records, out_path)
        logging.info("wrote %d records → %s", len(all_records), out_path)
