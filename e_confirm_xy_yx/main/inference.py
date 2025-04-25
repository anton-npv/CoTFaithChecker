import json
from pathlib import Path
from typing import Dict, List, Optional
import re
from collections import defaultdict
from e_confirm_xy_yx.main.data_loader import detect_cluster

import torch
from e_confirm_xy_yx.main.logging_utils import init_logger
from e_confirm_xy_yx.main.data_loader import load_dataset
from e_confirm_xy_yx.main.prompt_builder import PromptBuilder

__all__ = ["run_inference"]

logger = init_logger(Path.cwd() / "logs", "inference")


def _save_hidden_or_attention(tensor_list: List[torch.Tensor], out_file: Path):
    """
    Utility for serialising tensors.
    """
    torch.save(tensor_list, out_file)
    logger.info(f"Saved tensor list → {out_file}")


def _generate(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
) -> Dict:
    """
    Core batched generation, keeping hidden/attention if requested.
    No try/except: any failure stops the run and will be visible in the notebook & log.
    """
    #encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = encodings.input_ids.shape[1]   # length of the fixed prompt

    """with torch.no_grad():
        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            output_hidden_states=save_hidden,
            output_attentions=save_attention,
            return_dict_in_generate=True,
        )"""
    # ---------------- generation kwargs ----------------
    gen_kwargs = {
        "output_hidden_states": save_hidden,
        "output_attentions":   save_attention,
        "return_dict_in_generate": True,
        # if the caller passed None fall back to 128 tokens
        "max_new_tokens": 1024 if max_new_tokens is None else max_new_tokens,
    }

    with torch.no_grad():
        outputs = model.generate(**encodings, **gen_kwargs)

    #completions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    # keep only the NEW tokens (drop the prompt echo)
    completions = tokenizer.batch_decode(
        outputs.sequences[:, prompt_len:],
        skip_special_tokens=True,
    )

    extra = {}
    if save_hidden:
        hidden = [
            [outputs.hidden_states[i][layer].cpu() for layer in hidden_layers]
            for i in range(len(prompts))
        ]
        extra["hidden"] = hidden
    if save_attention:
        atts = [
            [outputs.attentions[i][layer].cpu() for layer in attn_layers]
            for i in range(len(prompts))
        ]
        extra["attention"] = atts

    return {"completions": completions, **extra}


def _process_single_file(
    json_path: Path,
    prompt_builder: PromptBuilder,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
    output_dir: Path,
) -> Path:
    """
    Process one dataset JSON; write a *completion* JSON next to it and return its Path.
    """
    logger.info(f"Processing {json_path.name}")
    data = load_dataset(json_path)
    questions: List[Dict] = data["questions"]

    prompts = prompt_builder.batch([q["q_str"] for q in questions])

    # Batch-wise generation
    completions: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        gen_out = _generate(
            model,
            tokenizer,
            batch_prompts,
            device,
            max_new_tokens,
            save_hidden,
            hidden_layers,
            save_attention,
            attn_layers,
        )
        completions.extend(gen_out["completions"])

        # optionals
        if save_hidden:
            out_hidden = output_dir / f"{json_path.stem}_hidden.pt"
            _save_hidden_or_attention(gen_out["hidden"], out_hidden)
        if save_attention:
            out_attn = output_dir / f"{json_path.stem}_attn.pt"
            _save_hidden_or_attention(gen_out["attention"], out_attn)

    output_records = []
    for q, comp in zip(questions, completions):
        entry = {
            "question_id": q["question_id"],
            "completion": comp,
        }
        # keep cross-link IDs
        if "yes_question_id" in q:
            entry["yes_question_id"] = q["yes_question_id"]
        if "no_question_id" in q:
            entry["no_question_id"] = q["no_question_id"]
        output_records.append(entry)

    fname = f"{json_path.stem}_{model_name}_completions.json"
    out_path = output_dir / fname
    with out_path.open("w") as f:
        json.dump(output_records, f, indent=2)
    logger.info(f"Saved completions → {out_path}")
    return out_path


def run_inference(
    dataset_files: List[Path],
    prompt_builder: PromptBuilder,
    model,
    tokenizer,
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_new_tokens: Optional[int],
    save_hidden: bool,
    hidden_layers: List[int],
    save_attention: bool,
    attn_layers: List[int],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: defaultdict[tuple, list] = defaultdict(list)
    # matches “…_gt_YES_…”  or “…_lt_NO_…”
    pattern = re.compile(r"_(gt|lt)_(YES|NO)_")
    for fp in dataset_files:
        out_path =_process_single_file(
            fp,
            prompt_builder,
            model,
            tokenizer,
            model_name,
            device,
            batch_size,
            max_new_tokens,
            save_hidden,
            hidden_layers,
            save_attention,
            attn_layers,
            output_dir,
        )
        
            # ───── aggregate per cluster & answer type ─────
        match = pattern.search(fp.stem)
        if match:
            comparison, expected = match.group(1), match.group(2)
        else:
            comparison, expected = "unknown", "unknown"

        cluster = detect_cluster(fp.stem)
        with out_path.open() as f:
            buckets[(cluster, comparison, expected)].extend(json.load(f))

    # ───── save aggregated cluster files ─────
    cluster_dir = output_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    for (cluster, cmp, exp), records in buckets.items():
        fname = f"{cluster}_{cmp}_{exp}_{model_name}_completions.json"
        with (cluster_dir / fname).open("w") as f:
            json.dump(records, f, indent=2)
        logger.info(
            f"Aggregated {len(records)} records → {cluster_dir / fname}"
        )

