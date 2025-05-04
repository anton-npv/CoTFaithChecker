#!/usr/bin/env python3
import os
import sys
import json
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable, Union 

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int 

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1] 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Running script from CWD: {os.getcwd()}")
print(f"Updated Python Path: {sys.path}")

from j_probing.utils.training_utils import (
    load_target_data, 
    load_question_ids,
    get_data_splits, 
    setup_determinism,
    get_cleaned_qids 
)

# --- Configuration --- 
@dataclass
class ContinuousSteeringConfig:
    # Model & Tokenizer
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # Input Data Config
    base_dir: str = "j_probing"
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B"
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"
    # Steering Vector Info
    steering_vector_base_dir: str = "k_steering/steering_vectors"
    source_layer: int = 18
    vector_clean_data: bool = True
    # Output Paths
    output_base_dir: str = "k_steering"
    output_filename_suffix: str = "_continuous_steered_gens.json"
    # Steering Parameters
    target_layers: List[int] = field(default_factory=lambda: list(range(32)))
    alpha_values: List[float] = field(default_factory=lambda: [0.2, -0.2])
    hook_point: str = "resid_post"
    # Generation Parameters
    num_generations_per_prompt: int = 5
    batch_size: int = 20
    temperature: float = 0.7
    max_new_tokens: int = 2000
    stop_at_eos: bool = True
    # Test Data Split Parameters
    val_frac: float = 0.05
    test_frac: float = 0.1
    split_seed: int = 42
    # Data Cleaning Parameters (to match vector calculation)
    clean_data: bool = True
    verbalized_low_threshold: float = 0.2
    nonverbalized_high_threshold: float = 0.8
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- Continuous Steering Hook Class ---
class ContinuousSteeringHook:
    def __init__(self, steering_vector, alpha, target_layers):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.target_layers = set(target_layers)

    def __call__(self, activation: torch.Tensor, hook: HookPoint):
        if hook.layer() not in self.target_layers:
            return activation 
        activation[:, -1, :] = activation[:, -1, :] + self.alpha * self.steering_vector.to(activation.dtype)
        return activation

# --- Custom Generation Function with Hooks ---

# Disable gradient tracking for the whole generation routine to speed up inference & save memory.
@torch.no_grad()
def generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    toks: Int[torch.Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 100,
    fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
    temperature: float = 0.0,
    top_k: int = -1,
    stop_at_eos: bool = True
) -> List[str]:
    """Generates text token-by-token applying hooks at each step."""
    model.eval()
    batch_size, seq_len = toks.shape
    device = toks.device
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None: pad_token_id = eos_token_id
    
    all_toks = torch.cat(
        (toks, torch.full((batch_size, max_tokens_generated), pad_token_id, dtype=torch.long, device=device)),
        dim=1
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Add a print statement before the loop
    print(f"  Starting generation loop for batch size {batch_size}, seq_len {seq_len}...")
    
    for i in range(max_tokens_generated):
        current_seq_len = seq_len + i
        
        # Add print statement inside the loop
        if i % 50 == 0: # Print every 50 tokens to avoid spamming
             print(f"    Generating token {i+1}/{max_tokens_generated}...")
            
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :current_seq_len])[:, -1, :]
        if temperature == 0.0:
            next_tokens = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_logits, top_indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, -float('inf'))
                mask.scatter_(1, top_indices, top_logits)
                logits = mask
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        next_tokens = next_tokens.to(device)
        just_finished = (next_tokens == eos_token_id)
        finished = finished | just_finished
        token_to_add = next_tokens if not stop_at_eos else torch.where(just_finished, eos_token_id, next_tokens)
        all_toks[:, current_seq_len] = torch.where(finished, pad_token_id, token_to_add)
        if stop_at_eos and finished.all(): break
            
    results = []
    for b in range(batch_size):
        eos_pos = -1
        if stop_at_eos:
            eos_indices = (all_toks[b, seq_len:] == eos_token_id).nonzero()
            if len(eos_indices) > 0: eos_pos = eos_indices[0].item()
        gen_toks = all_toks[b, seq_len : seq_len + eos_pos] if eos_pos != -1 else all_toks[b, seq_len:]
        results.append(tokenizer.decode(gen_toks, skip_special_tokens=True))
    return results

# --- Main Generation Function ---
def generate_continuously_steered_completions(cfg: ContinuousSteeringConfig):
    """Loads model, vector, data, and generates completions with steering applied continuously."""
    print(f"Configuration: {cfg}")
    setup_determinism(42)

    # --- Resolve Paths ---
    base_probing_dir = project_root / cfg.base_dir
    data_dir = base_probing_dir / "data" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    acts_dir = base_probing_dir / "acts" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    probing_data_full_path = data_dir / "probing_data.json"
    meta_full_path = acts_dir / "meta.json"
    
    clean_suffix = "_cleaned" if cfg.vector_clean_data else ""
    vector_filename = f"{cfg.hint_type}_verbalization_L{cfg.source_layer}{clean_suffix}.pt"
    steering_vector_base_path = project_root / cfg.steering_vector_base_dir / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    steering_vector_full_path = steering_vector_base_path / vector_filename
    
    output_full_dir = project_root / cfg.output_base_dir / "outputs" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    output_full_dir.mkdir(parents=True, exist_ok=True)
    output_filename = vector_filename.replace('.pt', '') + cfg.output_filename_suffix
    output_full_path = output_full_dir / output_filename

    print(f"Input Probing Data: {probing_data_full_path}")
    print(f"Input Meta Path: {meta_full_path}")
    print(f"Loading steering vector from: {steering_vector_full_path}")
    print(f"Output Generations Path: {output_full_path}")

    if not steering_vector_full_path.exists():
        raise FileNotFoundError(f"Steering vector not found: {steering_vector_full_path}")
    steering_vector = torch.load(steering_vector_full_path, map_location=cfg.device).to(cfg.dtype)
    print(f"Steering vector loaded. Shape: {steering_vector.shape}, dtype: {steering_vector.dtype}")

    # --- Load Model & Tokenizer ---
    print(f"Loading model: {cfg.model_path}")
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=True, 
        dtype=cfg.dtype,
        device=cfg.device,
        default_padding_side='left'
    )
    model.eval()
    torch.set_grad_enabled(False)
    tokenizer = model.tokenizer 
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" 
    
    # --- Load and Prepare Test Data ---
    print("Loading data to identify and filter test set...")
    target_map = load_target_data(probing_data_full_path)
    all_qids_ordered, n_layers, d_model = load_question_ids(meta_full_path)
    if steering_vector.shape[-1] != d_model: raise ValueError(f"Vector d_model != model d_model")

    _, _, initial_test_qids_int = get_data_splits(all_qids_ordered, target_map, cfg.val_frac, cfg.test_frac, cfg.split_seed)
    print(f"Identified {len(initial_test_qids_int)} initial test QIDs.")

    with open(probing_data_full_path, "r", encoding="utf-8") as f: probing_data_full = json.load(f)
    
    _, _, kept_qids_set = get_cleaned_qids(
        probing_data_full=probing_data_full, target_map=target_map,
        clean_data=cfg.clean_data,
        verbalized_low_threshold=cfg.verbalized_low_threshold,
        nonverbalized_high_threshold=cfg.nonverbalized_high_threshold
    )
    print(f"Total QIDs meeting cleaning criteria: {len(kept_qids_set)}")

    final_test_qids_set = {qid for qid in initial_test_qids_int if qid in kept_qids_set}
    print(f"Filtered test set size: {len(final_test_qids_set)}")

    # Prepare final prompt list using only the filtered test QIDs
    # No need to split based on original_verbalizes_hint anymore
    test_prompts_list = []
    test_qids_list = []
    qid_to_original_verbalizes = {}
    for record in probing_data_full:
        qid = record["question_id"]
        if qid in final_test_qids_set:
            test_prompts_list.append(record["prompt"])
            test_qids_list.append(qid)
            # Still store original verbalization status for potential later analysis
            qid_to_original_verbalizes[qid] = record.get("original_verbalizes_hint")

    print(f"Prepared {len(test_prompts_list)} final prompts for steered generation.")
    if not test_prompts_list: print("No test prompts remaining after filtering. Exiting."); return

    # --- Generation --- 
    # Results dictionary keyed by QID, then by alpha
    results_dict = {qid: {"original_verbalizes_hint": qid_to_original_verbalizes.get(qid), "generations": {}} 
                    for qid in test_qids_list} 
    target_layers_set = set(cfg.target_layers if cfg.target_layers else range(n_layers))
    hook_point_name_filter = lambda name: name.endswith(cfg.hook_point)
    print(f"Applying hook to layers: {sorted(list(target_layers_set))} at point {cfg.hook_point}")

    # Iterate through alpha values directly
    for alpha in tqdm(cfg.alpha_values, desc="Alpha values"):
        print(f"\n--- Generating for alpha = {alpha:.2f} ---")
        steering_hook_instance = ContinuousSteeringHook(steering_vector, alpha, target_layers_set)
        fwd_hooks = [(hook_point_name_filter, steering_hook_instance)]
        
        # Use the full filtered test prompt list
        prompt_list = test_prompts_list
        qid_list = test_qids_list
        
        for i in tqdm(range(0, len(prompt_list), cfg.batch_size), desc=f"  Batches (alpha={alpha:.2f})", leave=False):
            batch_prompts = prompt_list[i : i + cfg.batch_size]
            batch_qids = qid_list[i : i + cfg.batch_size]

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(cfg.device)
            input_toks = inputs.input_ids

            batch_generations = []
            for n in range(cfg.num_generations_per_prompt):
                generated_texts = generate_with_hooks(
                    model=model, tokenizer=tokenizer, toks=input_toks,
                    max_tokens_generated=cfg.max_new_tokens, fwd_hooks=fwd_hooks,
                    temperature=cfg.temperature, stop_at_eos=cfg.stop_at_eos
                )
                if n == 0: batch_generations = [[] for _ in range(len(batch_prompts))]
                for j, text in enumerate(generated_texts):
                    full_text = batch_prompts[j] + text
                    batch_generations[j].append(full_text)
            
            # Store results using the direct alpha value as the key
            for j, qid in enumerate(batch_qids):
                alpha_key = f"{alpha:.2f}"
                # Ensure the alpha key exists before extending
                if alpha_key not in results_dict[qid]["generations"]:
                    results_dict[qid]["generations"][alpha_key] = []
                results_dict[qid]["generations"][alpha_key].extend(batch_generations[j])
            
            gc.collect(); torch.cuda.empty_cache()

    # --- Save Final Results ---
    print(f"Saving final results to {output_full_path}")
    save_json(results_dict, output_full_path)
    print("Continuously steered generation complete.")

# --- Helper to save JSON ---
def save_json(data: Any, file_path: Path):
    print(f"Saving JSON to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e: print(f"[Error] Saving JSON failed: {e}")

if __name__ == "__main__":
    config = ContinuousSteeringConfig()
    generate_continuously_steered_completions(config) 