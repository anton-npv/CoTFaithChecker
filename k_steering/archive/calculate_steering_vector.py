#!/usr/bin/env python3
import os
import sys
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

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
    load_activations_for_split,
    get_cleaned_qids,
    # We might need ModelConfig, DataConfig if we formalize config
)

# --- Configuration --- 

@dataclass
class SteeringVectorConfig:
    # Base Directories/Names
    base_dir: str = "j_probing" # Base directory for probing data/acts
    output_base_dir: str = "k_steering" # Base directory for steering outputs
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B"
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"

    # Output filename (can be overridden)
    output_filename: Optional[str] = None # If None, construct dynamically

    # Vector Calculation Parameters
    start_layer: int = 5
    end_layer: int = 28 # Inclusive
    token_position_index: int = 2 # Hint token

    # Data Cleaning Parameters
    clean_data: bool = True
    verbalized_low_threshold: float = 0.2 # Remove originally verbalized if prob <= this
    nonverbalized_high_threshold: float = 0.8 # Remove originally non-verbalized if prob >= this

def calculate_steering_vector(cfg: SteeringVectorConfig):
    """Calculates the steering vector based on activation differences."""
    print(f"Configuration: {cfg}")

    # --- Construct Paths Dynamically --- 
    base_probing_dir = project_root / cfg.base_dir
    data_dir = base_probing_dir / "data" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    acts_dir = base_probing_dir / "acts" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    
    probing_data_full_path = data_dir / "probing_data.json"
    activations_full_dir = acts_dir
    meta_full_path = acts_dir / "meta.json"
    
    # Update output directory structure (ds_name/model_name/hint_type/n_questions_str)
    output_full_dir = project_root / cfg.output_base_dir / "steering_vectors" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    output_full_dir.mkdir(parents=True, exist_ok=True)

    # Construct default output filename if not provided
    if cfg.output_filename is None:
        clean_suffix = "_cleaned" if cfg.clean_data else ""
        cfg.output_filename = f"hint_verbalization_vector_L{cfg.start_layer}-{cfg.end_layer}{clean_suffix}.pt"
    
    output_full_path = output_full_dir / cfg.output_filename
    
    print(f"Input Probing Data: {probing_data_full_path}")
    print(f"Input Activations Dir: {activations_full_dir}")
    print(f"Input Meta Path: {meta_full_path}")
    print(f"Output Vector Path: {output_full_path}")

    # --- Load Data --- 
    print("Loading data...")
    target_map = load_target_data(probing_data_full_path)
    all_qids_ordered, n_layers, d_model = load_question_ids(meta_full_path)
    print(f"Loaded targets for {len(target_map)} QIDs.")
    print(f"Model info: {n_layers} layers, d_model={d_model}")

    # Validate layer range
    if not (0 <= cfg.start_layer < n_layers and 0 <= cfg.end_layer < n_layers and cfg.start_layer <= cfg.end_layer):
        raise ValueError(f"Invalid layer range [{cfg.start_layer}, {cfg.end_layer}] for model with {n_layers} layers.")

    # --- Data Cleaning (using utility function) --- 
    # Load full probing data required by the cleaning function
    with open(probing_data_full_path, "r", encoding="utf-8") as f:
        probing_data_full = json.load(f)

    # Call the utility function to get cleaned QID lists
    verbalized_qids, non_verbalized_qids, _ = get_cleaned_qids(
        probing_data_full=probing_data_full,
        target_map=target_map,
        clean_data=cfg.clean_data,
        verbalized_low_threshold=cfg.verbalized_low_threshold,
        nonverbalized_high_threshold=cfg.nonverbalized_high_threshold
    )
    # The print statements about cleaning results are now inside get_cleaned_qids

    if not verbalized_qids or not non_verbalized_qids:
        raise ValueError("One or both groups (verbalized, non-verbalized) are empty after filtering. Cannot calculate vector.")

    # --- Calculate Difference Vectors Per Layer --- 
    all_diff_vectors = []
    layers_to_process = list(range(cfg.start_layer, cfg.end_layer + 1))

    print(f"Calculating difference vectors for layers {cfg.start_layer} to {cfg.end_layer}...")
    for layer in tqdm(layers_to_process, desc="Processing Layers"):
        activation_path = activations_full_dir / f"layer_{layer:02d}.bin"
        if not activation_path.exists():
            print(f"[Warning] Activation file not found for layer {layer}. Skipping.")
            continue
        
        try:
            # Load activations for the hint token for both groups
            verbalized_acts = load_activations_for_split(
                activation_path, cfg.token_position_index, all_qids_ordered, verbalized_qids, d_model
            )
            non_verbalized_acts = load_activations_for_split(
                activation_path, cfg.token_position_index, all_qids_ordered, non_verbalized_qids, d_model
            )

            # Handle potential empty arrays if load_activations failed internally somehow
            if verbalized_acts.size == 0 or non_verbalized_acts.size == 0:
                 print(f"[Warning] Empty activations loaded for layer {layer}. Skipping.")
                 continue

            # Calculate mean vectors (converting to float32 for precision)
            mean_verbalized = np.mean(verbalized_acts.astype(np.float32), axis=0)
            mean_non_verbalized = np.mean(non_verbalized_acts.astype(np.float32), axis=0)
            
            # Calculate difference vector for this layer
            diff_vector = mean_verbalized - mean_non_verbalized
            all_diff_vectors.append(torch.from_numpy(diff_vector)) # Store as torch tensor

        except Exception as e:
            print(f"[Error] Failed processing layer {layer}: {e}. Skipping.")

    if not all_diff_vectors:
        raise RuntimeError("Failed to calculate difference vectors for any layer in the specified range.")

    # --- Average Difference Vectors --- 
    print("Averaging difference vectors across layers...")
    # Stack tensors and calculate the mean along the 0-th dimension (layers)
    steering_vector_tensor = torch.mean(torch.stack(all_diff_vectors), dim=0)

    print(f"Final steering vector shape: {steering_vector_tensor.shape}")
    print(f"Final steering vector dtype: {steering_vector_tensor.dtype}")

    # --- Save the Steering Vector --- 
    print(f"Saving steering vector to: {output_full_path}")
    torch.save(steering_vector_tensor, output_full_path)

    print("Steering vector calculation complete.")


if __name__ == "__main__":
    config = SteeringVectorConfig()
    # Example: Override config defaults if needed via argparse or direct modification
    # config.clean_data = False 
    # config.output_filename = "my_vector.pt"
    calculate_steering_vector(config) 