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

# Add project root to sys.path to allow absolute imports
project_root = Path(__file__).resolve().parents[1] # Go up one level from k_steering/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Running script from CWD: {os.getcwd()}")
print(f"Updated Python Path: {sys.path}")

# Import necessary functions from utils
from j_probing.utils.training_utils import (
    load_target_data,
    load_question_ids,
    get_data_splits, # To determine test set
    setup_determinism,
    get_cleaned_qids # Import the new function
)

# --- Configuration ---

@dataclass
class SteeringConfig:
    # Model & Tokenizer
    model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # Input Paths (relative to project root)
    base_dir: str = "j_probing"
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B"
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"
    # Steering Vector Path (relative to project root)
    steering_vector_dir: str = "k_steering/steering_vectors" # Just the base directory
    steering_vector_filename: str = "hint_verbalization_vector_L5-28_cleaned.pt" # Assumes cleaned data
    # Output Paths (relative to project root)
    output_base_dir: str = "k_steering" # Base directory for steering outputs
    output_filename_suffix: str = "_L5-28_cleaned_steered_gens.json"
    # Steering Parameters
    layers_to_steer: List[int] = field(default_factory=lambda: list(range(5, 29))) # Layers 5-28 inclusive
    alpha_values: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 1.0]) # Steering strengths
    token_position_target: str = "hint" # Can be "hint", "assistant", "think" - used to get index from probing_data
    # Generation Parameters
    num_generations_per_prompt: int = 5
    batch_size: int = 4 # Adjust based on GPU memory
    temperature: float = 0.7
    max_new_tokens: int = 512
    # Test Data Split Parameters (must match probing run)
    val_frac: float = 0.10
    test_frac: float = 0.10
    split_seed: int = 42 # Seed used for the original train/val/test split
    # Data Cleaning Parameters (to match vector calculation)
    clean_data: bool = True
    verbalized_low_threshold: float = 0.2 
    nonverbalized_high_threshold: float = 0.8 
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- Steering Hook Class --- 

class SteeringHook:
    def __init__(self, steering_vector, alpha, target_layers):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.target_layers = target_layers
        self.reset_state()

    def reset_state(self):
        # State needs to be reset for each new batch
        self.target_token_indices = None # Shape: [batch_size], contains index for each prompt in batch
        self.current_batch_size = 0

    def set_batch_state(self, target_token_indices: List[int]):
        if not target_token_indices:
             raise ValueError("target_token_indices cannot be empty")
        self.target_token_indices = torch.tensor(target_token_indices, device=self.steering_vector.device)
        self.current_batch_size = len(target_token_indices)
        # print(f"Hook state set for batch size {self.current_batch_size} with indices: {self.target_token_indices}")

    def __call__(self, activation: torch.Tensor, hook: HookPoint):
        # activation shape: [batch_size, seq_pos, d_model]
        
        if hook.layer() not in self.target_layers:
             # print(f"Skipping layer {hook.layer()}")
             return activation # Pass through if not a target layer

        if self.target_token_indices is None:
             print("[Warning] Hook called without target_token_indices set. Returning original activation.")
             return activation
             
        # Check batch size match - crucial
        if activation.shape[0] != self.current_batch_size:
            print(f"[Warning] Activation batch size ({activation.shape[0]}) != hook state batch size ({self.current_batch_size}). Skipping hook.")
            return activation

        # print(f"Applying hook at layer {hook.layer()} for indices {self.target_token_indices}")

        # Iterate through batch items because target index can differ per item
        for i in range(self.current_batch_size):
            target_idx = self.target_token_indices[i].item() # Get the specific index for this batch item
            # Ensure index is valid for the current activation sequence length
            if target_idx < activation.shape[1]:
                 # Add the scaled steering vector
                 activation[i, target_idx, :] += self.alpha * self.steering_vector.to(activation.dtype)
            # else:
                 # print(f"[Debug] Target index {target_idx} out of bounds for seq len {activation.shape[1]} in batch item {i}")
                 # This might happen during generation if target token is near the end

        return activation

# --- Main Generation Function --- 

def generate_steered_completions(cfg: SteeringConfig):
    """Loads model, vector, data, and generates completions with steering."""
    print(f"Configuration: {cfg}")
    setup_determinism(42) # Seed for generation consistency

    # --- Resolve Paths Dynamically (using updated config) --- 
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
    # Construct output filename based on steering vector filename and suffix
    output_filename = cfg.steering_vector_filename.replace('.pt', '') + cfg.output_filename_suffix
    output_full_path = output_full_dir / output_filename

    print(f"Loading steering vector from: {steering_vector_full_path}")
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
    # No grad needed for generation
    torch.set_grad_enabled(False)
    tokenizer = model.tokenizer # Use model's tokenizer
    if tokenizer.pad_token_id is None:
         tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # --- Load and Prepare Test Data --- 
    print("Loading probing data to identify test set and target token indices...")
    target_map = load_target_data(probing_data_full_path)
    all_qids_ordered, n_layers, d_model = load_question_ids(meta_full_path)
    
    if steering_vector.shape[-1] != d_model:
         raise ValueError(f"Steering vector d_model ({steering_vector.shape[-1]}) doesn't match model d_model ({d_model})")

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
    final_test_qids = [qid for qid in initial_test_qids_int if qid in kept_qids_set]
    print(f"Filtered test set size: {len(final_test_qids)}")

    # Prepare final prompt list and target indices using only the filtered test QIDs
    test_prompts_data = []
    for record in probing_data_full:
        qid = record["question_id"]
        if qid in final_test_qids:
            prompt_text = record["prompt"]
            token_positions = record["token_pos"] # This is a LIST
            # Get the index based on the target position name
            if not isinstance(token_positions, list) or len(token_positions) != 3:
                print(f"[Warning] QID {qid} has unexpected token_pos format: {token_positions}. Skipping.")
                continue
                
            # Map target name to list index
            pos_map = {"assistant": 0, "think": 1, "hint": 2}
            if cfg.token_position_target not in pos_map:
                 print(f"[Warning] QID {qid} unknown token_position_target: '{cfg.token_position_target}'. Skipping.")
                 continue
                 
            target_token_idx = token_positions[pos_map[cfg.token_position_target]]

            # Adjust relative indices if necessary (assuming BOS=True was used)
            if cfg.token_position_target in ["assistant", "think"] and target_token_idx < 0:
                 prompt_tokens_no_bos = model.to_tokens(prompt_text, prepend_bos=False)[0]
                 target_token_idx = len(prompt_tokens_no_bos) + target_token_idx + 1
            elif target_token_idx >= 0: # Absolute index (like hint), just add 1 for BOS
                 target_token_idx += 1
            else:
                 print(f"[Warning] Invalid target index {target_token_idx} for QID {qid}. Skipping.")
                 continue

            test_prompts_data.append({
                "question_id": qid,
                "prompt": prompt_text,
                "target_token_idx": target_token_idx
            })

    print(f"Prepared {len(test_prompts_data)} final prompts for steered generation.")
    if not test_prompts_data:
         print("No test prompts remaining after filtering. Exiting."); return

    # --- Generation --- 
    results = {}
    target_layers_set = set(cfg.layers_to_steer)
    hook_point_name_filter = lambda name: name.endswith("resid_post") # Hook after MLP/Attn

    for alpha in tqdm(cfg.alpha_values, desc="Alpha values"):
        print(f"\n--- Generating for alpha = {alpha:.2f} ---")
        steering_hook_instance = SteeringHook(steering_vector, alpha, target_layers_set)
        alpha_results = [] # Store results for this alpha

        for i in tqdm(range(0, len(test_prompts_data), cfg.batch_size), desc="Batches", leave=False):
            batch_data = test_prompts_data[i : i + cfg.batch_size]
            batch_prompts = [item["prompt"] for item in batch_data]
            batch_target_indices = [item["target_token_idx"] for item in batch_data]
            batch_qids = [item["question_id"] for item in batch_data]

            # Set the hook state for the current batch
            steering_hook_instance.set_batch_state(batch_target_indices)

            # Tokenize batch (with BOS)
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(cfg.device)

            # Generate N completions for the batch with hooks active
            batch_generations = []
            for n in range(cfg.num_generations_per_prompt):
                # Add hooks using context manager
                with model.hooks(fwd_hooks=[(hook_point_name_filter, steering_hook_instance)]):
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id
                    )
                # Decode *full* sequence and store
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                if n == 0: # Initialize list for each prompt on first generation
                     for _ in range(len(batch_prompts)):
                          batch_generations.append([])
                for j, text in enumerate(decoded_outputs):
                     batch_generations[j].append(text)
            
            # Store results for the batch
            for j, qid in enumerate(batch_qids):
                alpha_results.append({
                    "question_id": qid,
                    "alpha": alpha,
                    "generations": batch_generations[j]
                })

            # Clear cache if needed
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[f"alpha_{alpha:.2f}"] = alpha_results

        # Save intermediate results after each alpha (optional)
        # print(f"Saving intermediate results for alpha {alpha:.2f}...")
        # save_json(results, output_full_path.with_suffix(f".alpha_{alpha:.2f}.json"))

    # --- Save Final Results --- 
    print(f"Saving final results to {output_full_path}")
    save_json(results, output_full_path)
    print("Steered generation complete.")


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
    config = SteeringConfig()
    # Example override:
    # config.alpha_values = [0.0, 1.0, 2.0]
    # config.num_generations_per_prompt = 2
    # config.batch_size = 2
    generate_steered_completions(config) 