#!/usr/bin/env python3
import os
import sys
import json
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

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
class FixedPosSteeringConfig:
    # Model & Tokenizer
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # Input Data Config
    base_dir: str = "j_probing"
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B"
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"
    # Steering Vector Info (Source Layer defines which vector to load)
    steering_vector_base_dir: str = "k_steering/steering_vectors" 
    source_layer: int = 18 # Layer vector was derived from
    vector_clean_data: bool = True # Whether the vector used cleaned data
    # Output Paths 
    output_base_dir: str = "k_steering"
    output_filename_suffix: str = "_fixed_pos_steered_gens.json"
    # Steering Parameters
    # target_layers: List[int] = field(default_factory=lambda: list(range(32))) # Apply to ALL layers by default
    target_layers: List[int] = field(default_factory=lambda: [18]) # Apply to ALL layers by default
    alpha_magnitudes: List[float] = field(default_factory=lambda: [0.05])
    token_position_target: str = "hint" # Position to apply the vector at
    hook_point: str = "resid_post" # Where to hook into ('resid_pre', 'resid_mid', 'resid_post')
    # Generation Parameters
    num_generations_per_prompt: int = 2
    batch_size: int = 20
    temperature: float = 0.7
    max_new_tokens: int = 512
    # Test Data Split Parameters
    val_frac: float = 0.10
    test_frac: float = 0.05
    split_seed: int = 42
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- Steering Hook Class (Fixed Position) ---
class FixedPosSteeringHook:
    def __init__(self, steering_vector, alpha, target_layers):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.target_layers = set(target_layers) # Use set for faster lookup
        self.reset_state()

    def reset_state(self):
        self.target_token_indices = None
        self.current_batch_size = 0

    def set_batch_state(self, target_token_indices: List[int]):
        if not target_token_indices:
            raise ValueError("target_token_indices cannot be empty")
        self.target_token_indices = torch.tensor(target_token_indices, device=self.steering_vector.device)
        self.current_batch_size = len(target_token_indices)

    def __call__(self, activation: torch.Tensor, hook: HookPoint):
        if hook.layer() not in self.target_layers:
            return activation
        if self.target_token_indices is None:
            print("[Warning] Hook called without target_token_indices set."); return activation
        if activation.shape[0] != self.current_batch_size:
            print(f"[Warning] Activation batch size mismatch ({activation.shape[0]} vs {self.current_batch_size}). Skipping."); return activation

        for i in range(self.current_batch_size):
            target_idx = self.target_token_indices[i].item()
            if target_idx < activation.shape[1]:
                activation[i, target_idx, :] += self.alpha * self.steering_vector.to(activation.dtype)
        return activation

# --- Main Generation Function ---
def generate_fixed_pos_steered_completions(cfg: FixedPosSteeringConfig):
    """Generates completions applying steering vector once at a fixed position."""
    print(f"Configuration: {cfg}")
    setup_determinism(42)

    # --- Resolve Paths ---
    base_probing_dir = project_root / cfg.base_dir
    data_dir = base_probing_dir / "data" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    probing_data_full_path = data_dir / "probing_data.json"
    meta_full_path = (base_probing_dir / "acts" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str / "meta.json")
    
    clean_suffix = "_cleaned" if cfg.vector_clean_data else ""
    vector_filename = f"{cfg.hint_type}_verbalization_L{cfg.source_layer}{clean_suffix}.pt"
    steering_vector_full_path = project_root / cfg.steering_vector_base_dir / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str / vector_filename
    
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
        local_files_only=True, # Assuming local for this workaround
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
        clean_data=cfg.vector_clean_data, # Use flag indicating if vector used cleaning
        verbalized_low_threshold=0.2, nonverbalized_high_threshold=0.8 # Keep thresholds consistent for now
    )
    print(f"Total QIDs meeting cleaning criteria: {len(kept_qids_set)}")

    final_test_qids_set = {qid for qid in initial_test_qids_int if qid in kept_qids_set}
    print(f"Filtered test set size: {len(final_test_qids_set)}")

    # Split test data based on original_verbalizes_hint
    test_prompts_data_false = []
    test_prompts_data_true = []
    qid_to_original_verbalizes = {}

    pos_map = {"assistant": 0, "think": 1, "hint": 2}
    if cfg.token_position_target not in pos_map:
        raise ValueError(f"Unknown target: {cfg.token_position_target}")
    target_list_idx = pos_map[cfg.token_position_target]

    for record in probing_data_full:
        qid = record["question_id"]
        if qid in final_test_qids_set:
            original_verbalizes = record.get("original_verbalizes_hint")
            if original_verbalizes is None: continue # Skip if missing original status
            qid_to_original_verbalizes[qid] = original_verbalizes

            prompt_text = record["prompt"]
            token_positions = record["token_pos"]
            if not isinstance(token_positions, list) or len(token_positions) != 3: continue
            target_token_idx = token_positions[target_list_idx]
            if cfg.token_position_target in ["assistant", "think"] and target_token_idx < 0:
                prompt_tokens_no_bos = model.to_tokens(prompt_text, prepend_bos=False)[0]
                target_token_idx = len(prompt_tokens_no_bos) + target_token_idx + 1
            elif target_token_idx >= 0: target_token_idx += 1
            else: continue

            prompt_info = {"question_id": qid, "prompt": prompt_text, "target_token_idx": target_token_idx}
            if original_verbalizes:
                test_prompts_data_true.append(prompt_info)
            else:
                test_prompts_data_false.append(prompt_info)

    print(f"Prepared {len(test_prompts_data_false)} prompts (orig_verb=False) and {len(test_prompts_data_true)} (orig_verb=True)")
    if not test_prompts_data_false and not test_prompts_data_true: print("No test prompts. Exiting."); return

    # --- Generation --- 
    results_dict = {qid: {"original_verbalizes_hint": qid_to_original_verbalizes[qid], "generations": {}} 
                    for qid in final_test_qids_set if qid in qid_to_original_verbalizes}
    target_layers_set = set(cfg.target_layers if cfg.target_layers else range(n_layers))
    hook_point = utils.get_act_name(cfg.hook_point, -1)
    hook_point_name_filter = lambda name: name.endswith(hook_point)
    print(f"Applying hook to layers: {sorted(list(target_layers_set))} at point {hook_point}")

    for alpha_magnitude in tqdm(cfg.alpha_magnitudes, desc="Alpha magnitudes"):
        print(f"\n--- Processing alpha magnitude = {alpha_magnitude:.2f} ---")
        
        # Determine signed alphas for this magnitude
        alphas_to_run = {}
        if alpha_magnitude == 0.0:
             alphas_to_run[0.0] = (test_prompts_data_false + test_prompts_data_true) # Baseline for all
        else:
             alphas_to_run[+alpha_magnitude] = test_prompts_data_false
             alphas_to_run[-alpha_magnitude] = test_prompts_data_true

        for alpha, prompt_data_list in alphas_to_run.items():
            if not prompt_data_list: continue # Skip if group is empty
            print(f"  Running with alpha = {alpha:.2f} for {len(prompt_data_list)} prompts")
            steering_hook_instance = FixedPosSteeringHook(steering_vector, alpha, target_layers_set)
            
            for i in tqdm(range(0, len(prompt_data_list), cfg.batch_size), desc=f"  Batches (alpha={alpha:.2f})", leave=False):
                batch_data = prompt_data_list[i : i + cfg.batch_size]
                batch_prompts = [item["prompt"] for item in batch_data]
                batch_target_indices = [item["target_token_idx"] for item in batch_data]
                batch_qids = [item["question_id"] for item in batch_data]

                steering_hook_instance.set_batch_state(batch_target_indices)
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(cfg.device)

                batch_generations = []
                for n in range(cfg.num_generations_per_prompt):
                    with model.hooks(fwd_hooks=[(hook_point_name_filter, steering_hook_instance)]):
                        outputs = model.generate(
                            inputs.input_ids,
                            max_new_tokens=cfg.max_new_tokens,
                            temperature=cfg.temperature,
                            do_sample=True,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    if n == 0: batch_generations = [[] for _ in range(len(batch_prompts))]
                    for j, text in enumerate(decoded_outputs):
                         batch_generations[j].append(text)
                
                # Store results in the main dictionary
                for j, qid in enumerate(batch_qids):
                     alpha_key = f"pos_{alpha_magnitude}" if alpha > 0 else (f"neg_{alpha_magnitude}" if alpha < 0 else "baseline")
                     if alpha_key not in results_dict[qid]["generations"]:
                          results_dict[qid]["generations"][alpha_key] = []
                     results_dict[qid]["generations"][alpha_key].extend(batch_generations[j]) # Store N generations
                
                gc.collect(); torch.cuda.empty_cache()

    # --- Save Final Results ---
    print(f"Saving final results to {output_full_path}")
    save_json(results_dict, output_full_path)
    print("Fixed-position steered generation complete.")

# --- Helper to save JSON ---
def save_json(data: Any, file_path: Path):
    print(f"Saving JSON to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e: print(f"[Error] Saving JSON failed: {e}")

if __name__ == "__main__":
    config = FixedPosSteeringConfig()
    # config.alpha_values = [0.0, 1.0] # Example override
    # config.num_generations_per_prompt = 1
    # config.target_layers = [10, 15, 20]
    generate_fixed_pos_steered_completions(config) 