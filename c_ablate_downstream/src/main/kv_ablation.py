"""
Regenerate answers while *ablating* the KV rows corresponding to
tokens that realise a particular reasoning step.
"""
import json, os, torch, tqdm, logging
from typing import Dict, List, Tuple
from ..utils.prompt_constructor import construct_prompt
from ..utils.confidence_tools import ANSWER_TOKEN_IDS

def _zero_kv_rows(past, rows):
    return tuple(tuple(
        (k.index_copy(2, rows, torch.zeros_like(k[:, :, rows])),
         v.index_copy(2, rows, torch.zeros_like(v[:, :, rows])))
        for k,v in layer) for layer in past)

def run_kv_ablation(model, tokenizer, device,
                    questions: List[Dict],
                    steps_file: str,
                    out_dir: str,
                    max_new_tokens: int = 128):
    steps = {e["question_id"]: e["steps"] for e in json.load(open(steps_file))}
    os.makedirs(out_dir, exist_ok=True)
    answer_ids = ANSWER_TOKEN_IDS(tokenizer)
    results = []

    for q in tqdm.tqdm(questions, desc="KV-ablate"):
        qid = q["question_id"]
        prompt = construct_prompt(q)
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.input_ids.shape[1]

        # ---- locate token spans of each of the 3 steps -------------
        # (heuristic: take first exact substring match)
        step_spans: List[Tuple[int,int]] = []
        decoded = tokenizer.decode(input_ids.input_ids[0])
        for step in steps[qid]:
            idx = decoded.find(step)
            if idx == -1:
                step_spans.append((0,0))          # fallback: no ablation
                continue
            t_start = len(tokenizer.decode(tokenizer(decoded[:idx]).input_ids))
            t_end   = len(tokenizer.decode(tokenizer(decoded[:idx+len(step)]).input_ids))
            step_spans.append((t_start, t_end))

        # ---- run three ablation variants ---------------------------
        for i, (s,e) in enumerate(step_spans, start=1):
            with torch.no_grad():
                out = model(input_ids, use_cache=True, return_dict=True)
                past = list(list(p) for p in out.past_key_values)
                if e > s:
                    rows = torch.arange(s, e, device=device)
                    past = _zero_kv_rows(past, rows)

                generated_ids = []
                for _ in range(max_new_tokens):
                    logits = out.logits[0, -1]
                    next_id = logits.argmax()
                    generated_ids.append(next_id.item())
                    out = model(torch.tensor([[next_id]], device=device),
                                use_cache=True,
                                past_key_values=tuple(tuple(p) for p in past),
                                return_dict=True)
                    past = list(list(p) for p in out.past_key_values)

                answer = tokenizer.decode(generated_ids).strip()
                results.append({"question_id": qid,
                                "ablate_step": i,
                                "raw_answer": answer})

    json.dump(results,
              open(os.path.join(out_dir,"kv_ablation_results.json"),"w"), indent=2)
    logging.info("KV-ablation results written to %s", out_dir)
