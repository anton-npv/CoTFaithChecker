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
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list # Add gather_object and broadcast_object_list

# nohup accelerate launch generate_continuous_parallel.py


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
# --- Helper to save JSON ---
def save_json(data: Any, file_path: Path):
    print(f"Saving JSON to: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e: print(f"[Error] Saving JSON failed: {e}")

# --- Configuration --- 
@dataclass
class ContinuousSteeringConfig:
    # Model & Tokenizer
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
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
    alpha_values: List[float] = field(default_factory=lambda: [0.2])
    hook_point: str = "resid_post"
    # Generation Parameters
    num_generations_per_prompt: int = 2
    batch_size: int = 30
    temperature: float = 0.7
    max_new_tokens: int = 512
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
        # Move steering vector to the same device/dtype as the activation
        steer = self.steering_vector.to(dtype=activation.dtype, device=activation.device)
        activation[:, -1, :] = activation[:, -1, :] + self.alpha * steer
        return activation

# --- Custom Generation Function with Hooks ---
def generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    toks: torch.Tensor,  # Shape: [batch_size, seq_len]
    max_tokens_generated: int = 100,
    fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
    temperature: float = 0.0,
    top_k: int = -1,
    stop_at_eos: bool = True,
    accelerator: Accelerator = None
) -> List[str]:
    """Generate continuations using HF-style generate with KV-cache while preserving hooks."""

    model.eval()
    batch_size, seq_len = toks.shape
    device = accelerator.device

    eos_token_id = tokenizer.eos_token_id

    # Temperature / sampling setup
    do_sample = temperature > 0.0 or top_k > 0

    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            # Input should already be on the right device, but check and print for debugging
            input_cuda = toks
            # Print per-rank device info for debugging
            if accelerator is None or True:
                print(f"    [Rank {os.environ.get('LOCAL_RANK', '0')}] tokens device = {input_cuda.device}, model device = {device}")
            # Add deeper debugging of embedding matrix device
            embedding_matrix_device = model.embed.W_E.device
            print(f"    [Rank {os.environ.get('LOCAL_RANK', '0')}] embedding matrix device = {embedding_matrix_device}")
            # Force tokens to embedding matrix device specifically
            input_cuda = input_cuda.to(embedding_matrix_device)
            print(f"    [Rank {os.environ.get('LOCAL_RANK', '0')}] input moved to {input_cuda.device}")
            # Call generate with explicit parameters (no pad_token_id)
            gen_ids = model.generate(
                input_cuda,
                max_new_tokens=max_tokens_generated,
                stop_at_eos=stop_at_eos,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
                top_k=top_k if top_k > 0 else None,
                temperature=temperature if do_sample else None,
                verbose=False,
            )

    # gen_ids shape: [B, prompt+new]
    results: List[str] = []
    for b in range(batch_size):
        full_seq = gen_ids[b].tolist()
        gen_part = full_seq[seq_len:]  # tokens after the original prompt
        # Truncate at EOS if requested and not already stopped
        if stop_at_eos and eos_token_id in gen_part:
            eos_index = gen_part.index(eos_token_id)
            gen_part = gen_part[:eos_index]
        results.append(tokenizer.decode(gen_part, skip_special_tokens=True))

    return results

# --- Main Generation Function ---
def generate_continuously_steered_completions(cfg: ContinuousSteeringConfig):
    """Loads model, vector, data, and generates completions with steering applied continuously using Accelerate."""
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"Configuration: {cfg}")
        print(f"Running on {accelerator.num_processes} processes.")
    
    setup_determinism(42) # Ensure determinism across processes if needed

    # --- Resolve Paths (Main process primarily handles this for clarity) ---
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
    # Ensure output directory exists (only main process needs to create it)
    if accelerator.is_main_process:
        output_full_dir.mkdir(parents=True, exist_ok=True)
    output_filename = vector_filename.replace('.pt', '') + cfg.output_filename_suffix
    output_full_path = output_full_dir / output_filename

    if accelerator.is_main_process:
        print(f"Input Probing Data: {probing_data_full_path}")
        print(f"Input Meta Path: {meta_full_path}")
        print(f"Loading steering vector from: {steering_vector_full_path}")
        print(f"Output Generations Path: {output_full_path}")

    if not steering_vector_full_path.exists():
        # Ensure error is raised consistently across processes if file is missing
        if accelerator.is_main_process:
            raise FileNotFoundError(f"Steering vector not found: {steering_vector_full_path}")
        else: # Wait for main process to potentially error out or proceed
            accelerator.wait_for_everyone()
            # Re-check if needed, though main process should handle the exit
            if not steering_vector_full_path.exists():
                 raise FileNotFoundError(f"Steering vector not found (process {accelerator.process_index}): {steering_vector_full_path}")

    # Load steering vector onto the correct device assigned by Accelerator
    steering_vector = torch.load(steering_vector_full_path, map_location="cpu").to(accelerator.device, dtype=cfg.dtype)
    if accelerator.is_main_process:
        print(f"Steering vector loaded. Shape: {steering_vector.shape}, dtype: {steering_vector.dtype}")

    # --- Load Model & Tokenizer ---
    if accelerator.is_main_process:
        print(f"Loading model on CPU first, then manually moving to {accelerator.device}")

    # Load model on CPU first to avoid immediate GPU placement
    model = HookedTransformer.from_pretrained_no_processing(
        cfg.model_path,
        local_files_only=True,
        dtype=cfg.dtype,
        default_padding_side="left",
        device="cpu"  # Start on CPU
    )

    # Tokenizer setup before model preparation
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # DO NOT use accelerator.prepare() for the model - it wraps with DDP which complicates access
    # Instead, manually move the entire model to the right device
    model.to(accelerator.device)

    # Configure TransformerLens to use a single device (disable pipeline parallelism)
    model.cfg.device = str(accelerator.device)
    model.cfg.n_devices = 1

    # Debug print to verify environment variables
    local_rank_env = os.environ.get("LOCAL_RANK", "Not set")
    rank_env = os.environ.get("RANK", "Not set")
    world_size_env = os.environ.get("WORLD_SIZE", "Not set")
    print(f"Process environment: LOCAL_RANK={local_rank_env}, RANK={rank_env}, WORLD_SIZE={world_size_env}, device={accelerator.device}")

    # Verify both model parameters and embedding matrix are on the right device
    param_device = next(model.parameters()).device
    embed_device = model.embed.W_E.device
    print(f"[Rank {accelerator.process_index}] model param device = {param_device}, embedding device = {embed_device}")

    # --- Load and Prepare Test Data (Main process loads, then distributes) ---
    if accelerator.is_main_process:
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
        test_prompts_list_full = []
        test_qids_list_full = []
        qid_to_original_verbalizes = {}
        for record in probing_data_full:
            qid = record["question_id"]
            if qid in final_test_qids_set:
                test_prompts_list_full.append(record["prompt"])
                test_qids_list_full.append(qid)
                # Still store original verbalization status for potential later analysis
                qid_to_original_verbalizes[qid] = record.get("original_verbalizes_hint")

        print(f"Prepared {len(test_prompts_list_full)} final prompts for steered generation.")
        if not test_prompts_list_full: print("No test prompts remaining after filtering. Exiting."); return
        
        # Package data for distribution
        data_to_distribute = {
            "prompts": test_prompts_list_full,
            "qids": test_qids_list_full,
            "qid_verbalization": qid_to_original_verbalizes
        }
    else:
        data_to_distribute = None
        
    # Broadcast prompt/QID data from rank 0, then slice by rank
    if accelerator.is_main_process:
        container = [data_to_distribute]
    else:
        container = [None]

    broadcast_object_list(container)  # After call, container[0] is available on all ranks
    shared_data = container[0]

    # Slice by rank to reduce per-GPU load
    all_prompts = shared_data["prompts"]
    all_qids = shared_data["qids"]
    qid_to_original_verbalizes = shared_data["qid_verbalization"]

    rank = accelerator.process_index
    world = accelerator.num_processes
    test_prompts_list = all_prompts[rank::world]
    test_qids_list = all_qids[rank::world]
    
    if accelerator.is_main_process:
         print(f"Data distributed. Process 0 has {len(test_prompts_list)} prompts.")
    else:
         print(f"Data received. Process {accelerator.process_index} has {len(test_prompts_list)} prompts.")


    # --- Generation --- 
    # Results dictionary keyed by QID, then by alpha (each process collects its own results)
    local_results_dict = {qid: {"original_verbalizes_hint": qid_to_original_verbalizes.get(qid), "generations": {}} 
                          for qid in test_qids_list} 
    
    target_layers_set = set(cfg.target_layers if cfg.target_layers else range(model.cfg.n_layers)) # Use model config for n_layers
    hook_point_name_filter = lambda name: name.endswith(cfg.hook_point)
    if accelerator.is_main_process:
        print(f"Applying hook to layers: {sorted(list(target_layers_set))} at point {cfg.hook_point}")

    # Iterate through alpha values directly
    for alpha in tqdm(cfg.alpha_values, desc="Alpha values", disable=not accelerator.is_main_process):
        if accelerator.is_main_process:
            print(f"\n--- Generating for alpha = {alpha:.2f} ---")
            
        # Ensure steering vector is on the correct device for the hook instance
        steering_hook_instance = ContinuousSteeringHook(steering_vector.to(accelerator.device), alpha, target_layers_set)
        fwd_hooks = [(hook_point_name_filter, steering_hook_instance)]
        
        # Use the process-local prompt list
        prompt_list = test_prompts_list
        qid_list = test_qids_list
        
        # Calculate batch size per device based on total batch size and number of processes
        # Note: cfg.batch_size is now treated as PER DEVICE batch size
        effective_batch_size = cfg.batch_size 
        
        # tqdm progress bar adjusted for distributed processing
        num_local_prompts = len(prompt_list)
        progress_bar = tqdm(range(0, num_local_prompts, effective_batch_size), 
                            desc=f"  Batches (alpha={alpha:.2f}, process={accelerator.process_index})", 
                            leave=False, 
                            disable=not accelerator.is_local_main_process) # Show progress per local main process

        for i in progress_bar:
            batch_prompts = prompt_list[i : i + effective_batch_size]
            batch_qids = qid_list[i : i + effective_batch_size]

            if not batch_prompts: continue # Skip empty batches if data isn't perfectly divisible

            # Tokenize and move inputs to the same device as the model parameters
            model_device = next(model.parameters()).device
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            input_toks = inputs["input_ids"]

            batch_generations = []
            for n in range(cfg.num_generations_per_prompt):
                 # Use accelerator.unwrap_model if needed, but generate_with_hooks uses the prepared model directly
                generated_texts = generate_with_hooks(
                    model=model, # Use the accelerator-prepared model
                    tokenizer=tokenizer, 
                    toks=input_toks,
                    max_tokens_generated=cfg.max_new_tokens, 
                    fwd_hooks=fwd_hooks,
                    temperature=cfg.temperature, 
                    stop_at_eos=cfg.stop_at_eos,
                    accelerator=accelerator # Pass accelerator
                )
                if n == 0: batch_generations = [[] for _ in range(len(batch_prompts))]
                for j, text in enumerate(generated_texts):
                    full_text = batch_prompts[j] + text
                    batch_generations[j].append(full_text)
            
            # Store results using the direct alpha value as the key in the local dictionary
            for j, qid in enumerate(batch_qids):
                alpha_key = f"{alpha:.2f}"
                # Ensure the alpha key exists before extending
                if alpha_key not in local_results_dict[qid]["generations"]:
                    local_results_dict[qid]["generations"][alpha_key] = []
                local_results_dict[qid]["generations"][alpha_key].extend(batch_generations[j])
            
            # Clear cache if necessary (might be less critical with Accelerate's memory management)
            # gc.collect(); torch.cuda.empty_cache() # Consider if needed based on memory usage

        accelerator.wait_for_everyone() # Ensure all processes finish alpha before moving to next

    # --- Gather Results from All Processes ---
    if accelerator.num_processes > 1:
        if accelerator.is_main_process: print("Gathering results from all processes...")
        gathered_results_list = gather_object(local_results_dict) # Gather list of dictionaries
        
        # Combine results on the main process
        if accelerator.is_main_process:
            print("Combining gathered results...")
            final_results_dict = {}
            # Need the full original qid_to_original_verbalizes map here
            full_qid_verbalization_map = data_to_distribute["qid_verbalization"] 
            all_gathered_qids = set()
            
            for process_results in gathered_results_list:
                for qid, data in process_results.items():
                    all_gathered_qids.add(qid)
                    if qid not in final_results_dict:
                         # Initialize with original verbalization info from the full map
                        final_results_dict[qid] = {"original_verbalizes_hint": full_qid_verbalization_map.get(qid), "generations": {}}
                    
                    for alpha_key, gens in data["generations"].items():
                        if alpha_key not in final_results_dict[qid]["generations"]:
                            final_results_dict[qid]["generations"][alpha_key] = []
                        final_results_dict[qid]["generations"][alpha_key].extend(gens) 
            
            print(f"Combined results for {len(final_results_dict)} QIDs.")
            # Sanity check: ensure all expected QIDs were processed
            expected_qids = set(data_to_distribute["qids"])
            if all_gathered_qids != expected_qids:
                 print(f"[Warning] Mismatch in gathered QIDs. Expected {len(expected_qids)}, got {len(all_gathered_qids)}")
                 print(f"Missing QIDs: {expected_qids - all_gathered_qids}")
                 print(f"Extra QIDs: {all_gathered_qids - expected_qids}")
                 
    else: # Single process execution
        final_results_dict = local_results_dict

    # --- Save Final Results (Only on Main Process) ---
    if accelerator.is_main_process:
        print(f"Saving final results to {output_full_path}")
        # Use imported save_json helper
        save_json(final_results_dict, output_full_path) 
        print("Continuously steered generation complete.")
    
    accelerator.wait_for_everyone() # Final sync before script ends

if __name__ == "__main__":
    # Configuration loading could potentially use argparse or hydra in the future
    config = ContinuousSteeringConfig() 
    generate_continuously_steered_completions(config) 