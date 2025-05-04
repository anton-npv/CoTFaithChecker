#!/usr/bin/env python3
import os
import sys
import json
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable, Union # Added Union

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int # Added jaxtyping

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parents[1] # Go up one level from k_steering/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Running script from CWD: {os.getcwd()}")
print(f"Updated Python Path: {sys.path}")

# Import necessary functions from utils
from j_probing.utils.training_utils import (
    load_target_data, # Still needed to identify test set
    load_question_ids,
    get_data_splits, # To determine test set
    setup_determinism,
    get_cleaned_qids # Import the cleaning function
)

# --- Configuration (Similar to SteeringConfig, adjusted output suffix) ---

@dataclass
class ContinuousSteeringConfig:
    # Model & Tokenizer
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # Input Data Config (for test set identification)
    base_dir: str = "j_probing" 
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B" 
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"
    # Steering Vector Path 
    steering_vector_dir: str = "k_steering/steering_vectors" # Just the base directory
    steering_vector_filename: str = "hint_verbalization_vector_L5-28_cleaned.pt" 
    # Output Paths
    output_base_dir: str = "k_steering" # Base directory for steering outputs
    output_filename_suffix: str = "_L5-28_cleaned_continuous_steered_gens.json" # Changed suffix
    # Steering Parameters
    target_layers: List[int] = field(default_factory=lambda: list(range(32))) # Apply to ALL layers
    alpha_magnitudes: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    hook_point: str = "resid_post" # Where to hook into
    # Generation Parameters
    num_generations_per_prompt: int = 5
    batch_size: int = 4 
    temperature: float = 0.7
    max_new_tokens: int = 512
    stop_at_eos: bool = True # Stop generation when EOS is produced
    # Test Data Split Parameters
    val_frac: float = 0.10
    test_frac: float = 0.10
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
        # activation shape: [batch_size, seq_pos, d_model]
        if hook.layer() not in self.target_layers:
            return activation 

        # Add the scaled steering vector to the activation of the LAST token
        activation[:, -1, :] = activation[:, -1, :] + self.alpha * self.steering_vector.to(activation.dtype)
        
        return activation

# --- Custom Generation Function with Hooks (from reasoning_demo) --- 

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
    """ 
    Generates text token-by-token applying hooks at each step.
    """
    model.eval()
    batch_size, seq_len = toks.shape
    device = toks.device
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    
    all_toks = torch.cat(
        (toks, torch.full((batch_size, max_tokens_generated), pad_token_id, dtype=torch.long, device=device)),
        dim=1
    )
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(max_tokens_generated):
        current_seq_len = seq_len + i
        
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
            
        if stop_at_eos and finished.all():
            break
            
    results = []
    for b in range(batch_size):
        eos_pos = -1
        if stop_at_eos:
            eos_indices = (all_toks[b, seq_len:] == eos_token_id).nonzero()
            if len(eos_indices) > 0:
                eos_pos = eos_indices[0].item()
        gen_toks = all_toks[b, seq_len : seq_len + eos_pos] if eos_pos != -1 else all_toks[b, seq_len:]
        results.append(tokenizer.decode(gen_toks, skip_special_tokens=True))
        
    return results

# --- Main Generation Function --- 

def generate_continuously_steered_completions(cfg: ContinuousSteeringConfig):
    """Loads model, vector, data, and generates completions with steering applied continuously."""
    print(f"Configuration: {cfg}")
    setup_determinism(42)

    # --- Resolve Paths Dynamically --- 
    base_probing_dir = project_root / cfg.base_dir
    data_dir = base_probing_dir / "data" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    acts_dir = base_probing_dir / "acts" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    # Construct steering vector path using the specific model/dataset info (ds_name first, include hint_type)
    steering_vector_base_path = project_root / cfg.steering_vector_dir / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    steering_vector_full_path = steering_vector_base_path / cfg.steering_vector_filename
    probing_data_full_path = data_dir / "probing_data.json"
    meta_full_path = acts_dir / "meta.json"
    
    # Construct output path using output_base_dir and specific model/dataset info (ds_name first, include hint_type)
    output_full_dir = project_root / cfg.output_base_dir / "outputs" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    output_full_dir.mkdir(parents=True, exist_ok=True)
    output_filename = cfg.steering_vector_filename.replace('.pt', '') + cfg.output_filename_suffix
    output_full_path = output_full_dir / output_filename

    print(f"Input Probing Data: {probing_data_full_path}")
    print(f"Input Meta Path: {meta_full_path}")
    print(f"Attempting to load steering vector from: {steering_vector_full_path}")



    if not steering_vector_full_path.exists():
        raise FileNotFoundError(f"Steering vector not found: {steering_vector_full_path}")
    steering_vector = torch.load(steering_vector_full_path, map_location=cfg.device).to(cfg.dtype)
    print(f"Steering vector loaded. Shape: {steering_vector.shape}, dtype: {steering_vector.dtype}")

    # --- Load Model & Tokenizer --- 
    print(f"Loading model: {cfg.model_path}")
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.1-8B-Instruct",
        local_files_only=True,  # Set to True if using local models
        dtype=cfg.dtype,
        device=cfg.device,
        default_padding_side='left'
    )
    model.eval()
    torch.set_grad_enabled(False)
    tokenizer = model.tokenizer 
    if tokenizer.pad_token_id is None:
         tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" 
    
    # --- Load and Prepare Test Data (Only need prompts) ---
    print("Loading data to identify test set...")
    target_map = load_target_data(probing_data_full_path)
    all_qids_ordered, _, d_model = load_question_ids(meta_full_path)
    if steering_vector.shape[-1] != d_model:
         raise ValueError(f"Steering vector d_model != model d_model")

    # Get the initial test split QIDs
    _, _, initial_test_qids_int = get_data_splits(
        all_qids_ordered, target_map, cfg.val_frac, cfg.test_frac, cfg.split_seed
    )
    print(f"Identified {len(initial_test_qids_int)} initial test QIDs.")

    # Load full probing data for cleaning filter
    with open(probing_data_full_path, "r", encoding="utf-8") as f:
        probing_data_full = json.load(f)

    # Get the set of QIDs that pass the cleaning criteria
    _, _, kept_qids_set = get_cleaned_qids(
        probing_data_full=probing_data_full,
        target_map=target_map,
        clean_data=cfg.clean_data, # Use same cleaning config as vector calculation
        verbalized_low_threshold=cfg.verbalized_low_threshold,
        nonverbalized_high_threshold=cfg.nonverbalized_high_threshold
    )
    print(f"Total QIDs meeting cleaning criteria: {len(kept_qids_set)}")

    # Filter the initial test set QIDs
    final_test_qids_set = {qid for qid in initial_test_qids_int if qid in kept_qids_set}
    print(f"Filtered test set size: {len(final_test_qids_set)}")

    # Split test data based on original_verbalizes_hint
    test_prompts_false = []
    test_qids_false = []
    test_prompts_true = []
    test_qids_true = []
    qid_to_original_verbalizes = {}

    for record in probing_data_full:
        qid = record["question_id"]
        if qid in final_test_qids_set:
            original_verbalizes = record.get("original_verbalizes_hint")
            if original_verbalizes is None: continue
            qid_to_original_verbalizes[qid] = original_verbalizes
            prompt = record["prompt"]
            if original_verbalizes:
                test_prompts_true.append(prompt)
                test_qids_true.append(qid)
            else:
                test_prompts_false.append(prompt)
                test_qids_false.append(qid)

    print(f"Prepared {len(test_prompts_false)} prompts (orig_verb=False) and {len(test_prompts_true)} (orig_verb=True)")
    if not test_prompts_false and not test_prompts_true: print("No test prompts. Exiting."); return

    # --- Generation --- 
    results_dict = {qid: {"original_verbalizes_hint": qid_to_original_verbalizes[qid], "generations": {}} 
                    for qid in final_test_qids_set if qid in qid_to_original_verbalizes}
    target_layers_set = set(cfg.target_layers if cfg.target_layers else range(n_layers))
    hook_point_name_filter = lambda name: name.endswith(cfg.hook_point)
    print(f"Applying hook to layers: {sorted(list(target_layers_set))} at point {cfg.hook_point}")

    for alpha_magnitude in tqdm(cfg.alpha_magnitudes, desc="Alpha magnitudes"):
        print(f"\n--- Processing alpha magnitude = {alpha_magnitude:.2f} ---")
        
        # Determine prompts and signed alpha for this magnitude
        prompts_to_process_with_alpha = {}
        if alpha_magnitude == 0.0:
            prompts_to_process_with_alpha[0.0] = (test_prompts_false + test_prompts_true, test_qids_false + test_qids_true)
        else:
            if test_prompts_false: prompts_to_process_with_alpha[+alpha_magnitude] = (test_prompts_false, test_qids_false)
            if test_prompts_true: prompts_to_process_with_alpha[-alpha_magnitude] = (test_prompts_true, test_qids_true)

        for alpha, (prompt_list, qid_list) in prompts_to_process_with_alpha.items():
            if not prompt_list: continue
            print(f"  Running with alpha = {alpha:.2f} for {len(prompt_list)} prompts")
            steering_hook_instance = ContinuousSteeringHook(steering_vector, alpha, target_layers_set)
            fwd_hooks = [(hook_point_name_filter, steering_hook_instance)]
            
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
                
                for j, qid in enumerate(batch_qids):
                    # Use the actual signed alpha as the key
                    alpha_key = f"{alpha:.2f}"
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
    """Saves data to a JSON file."""
    print(f"Saving JSON to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[Error] An unexpected error occurred saving to {file_path}: {e}")

# --- Main Execution --- 
if __name__ == "__main__":
    config = ContinuousSteeringConfig()
    generate_continuously_steered_completions(config) 