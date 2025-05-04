#!/usr/bin/env python3
import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
print(f"Running script from CWD: {os.getcwd()}")
print(f"Updated Python Path: {sys.path}")

from j_probing.utils.training_utils import (
    load_target_data,
    load_question_ids,
    load_activations_for_split,
    get_cleaned_qids,
)

# --- Configuration ---
@dataclass
class CalcVectorConfig:
    base_dir: str = "j_probing"
    output_base_dir: str = "k_steering"
    ds_name: str = "mmlu"
    model_name_for_path: str = "DeepSeek-R1-Distill-Llama-8B"
    hint_type: str = "sycophancy"
    n_questions_str: str = "5001"
    output_filename: Optional[str] = None
    source_layer: int = 18 # Layer to calculate the vector from
    token_position_index: int = 2 # Hint token
    clean_data: bool = True
    verbalized_low_threshold: float = 0.2
    nonverbalized_high_threshold: float = 0.8
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def calculate_single_layer_steering_vector(cfg: CalcVectorConfig):
    """Calculates the steering vector from a single layer's activations."""
    print(f"Configuration: {cfg}")

    # --- Construct Paths ---
    base_probing_dir = project_root / cfg.base_dir
    data_dir = base_probing_dir / "data" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    acts_dir = base_probing_dir / "acts" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    probing_data_full_path = data_dir / "probing_data.json"
    activations_full_dir = acts_dir
    meta_full_path = acts_dir / "meta.json"
    output_full_dir = project_root / cfg.output_base_dir / "steering_vectors" / cfg.ds_name / cfg.model_name_for_path / cfg.hint_type / cfg.n_questions_str
    output_full_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output_filename is None:
        clean_suffix = "_cleaned" if cfg.clean_data else ""
        cfg.output_filename = f"{cfg.hint_type}_verbalization_L{cfg.source_layer}{clean_suffix}.pt"
    output_full_path = output_full_dir / cfg.output_filename
    
    print(f"Input Probing Data: {probing_data_full_path}")
    print(f"Input Activations Dir: {activations_full_dir}")
    print(f"Input Meta Path: {meta_full_path}")
    print(f"Output Vector Path: {output_full_path}")

    # --- Load Data ---
    print("Loading metadata and target data...")
    target_map = load_target_data(probing_data_full_path)
    all_qids_ordered, n_layers, d_model = load_question_ids(meta_full_path)
    print(f"Model info: {n_layers} layers, d_model={d_model}")

    if not (0 <= cfg.source_layer < n_layers):
        raise ValueError(f"Invalid source_layer {cfg.source_layer} for model with {n_layers} layers.")

    # --- Get Cleaned QIDs --- 
    with open(probing_data_full_path, "r", encoding="utf-8") as f:
        probing_data_full = json.load(f)
    
    verbalized_qids, non_verbalized_qids, _ = get_cleaned_qids(
        probing_data_full=probing_data_full,
        target_map=target_map,
        clean_data=cfg.clean_data,
        verbalized_low_threshold=cfg.verbalized_low_threshold,
        nonverbalized_high_threshold=cfg.nonverbalized_high_threshold
    )
    
    if not verbalized_qids or not non_verbalized_qids:
        raise ValueError("One or both groups are empty after filtering.")

    # --- Calculate Difference Vector for the Source Layer ---
    print(f"Calculating difference vector for layer {cfg.source_layer}...")
    activation_path = activations_full_dir / f"layer_{cfg.source_layer:02d}.bin"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found for source layer {cfg.source_layer}: {activation_path}")
    
    steering_vector_tensor = None
    try:
        verbalized_acts = load_activations_for_split(
            activation_path, cfg.token_position_index, all_qids_ordered, verbalized_qids, d_model
        )
        non_verbalized_acts = load_activations_for_split(
            activation_path, cfg.token_position_index, all_qids_ordered, non_verbalized_qids, d_model
        )

        if verbalized_acts.size == 0 or non_verbalized_acts.size == 0:
             raise ValueError(f"Empty activations loaded for layer {cfg.source_layer}.")

        mean_verbalized = np.mean(verbalized_acts.astype(np.float32), axis=0)
        mean_non_verbalized = np.mean(non_verbalized_acts.astype(np.float32), axis=0)
        diff_vector = mean_verbalized - mean_non_verbalized
        steering_vector_tensor = torch.from_numpy(diff_vector).to(cfg.dtype) # Cast to desired dtype

    except Exception as e:
        raise RuntimeError(f"Failed processing layer {cfg.source_layer}: {e}")

    if steering_vector_tensor is None:
        raise RuntimeError("Failed to calculate steering vector.")

    print(f"Final steering vector shape: {steering_vector_tensor.shape}")
    print(f"Final steering vector dtype: {steering_vector_tensor.dtype}")

    # --- Save the Steering Vector --- 
    print(f"Saving steering vector to: {output_full_path}")
    torch.save(steering_vector_tensor, output_full_path)
    print("Steering vector calculation complete.")

if __name__ == "__main__":
    config = CalcVectorConfig()
    # config.source_layer = 18 # Keep default or override
    # config.clean_data = False
    calculate_single_layer_steering_vector(config) 