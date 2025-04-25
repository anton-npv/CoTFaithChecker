# %% [markdown]
# Log Probability Tracking Analysis

# %%
%cd ..


# %% 
# Imports and Setup
import os
import json
import logging
import sys



# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

# %% 
# Configuration

# --- Core Parameters for a Single Run --- 
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATASET = "mmlu"

# --- Parameters for the Logprobs Experiment --- 
HINT_TYPES_TO_ANALYZE = ["induced_urgency", "sycophancy"] # List of hints to compare against baseline
INTERVENTION_TYPES = ["dots", "dots_eot"] # Types of intervention prompts
PERCENTAGE_STEPS = list(range(10, 101, 10)) # Analyze at 10%, 20%, ..., 100%

# --- Run Control & File Parameters --- 
N_QUESTIONS = 500 # Number of questions used to generate source files (e.g., completions_with_500.json)
DEMO_MODE_N = 5 # Set to None to run on all relevant questions, or integer N for first N
DATA_DIR = "./data"

# %% 
# Load Model and Tokenizer

from src.main.pipeline import load_model_and_tokenizer

logging.info(f"Loading model and tokenizer: {MODEL_PATH}")
try:
    model, tokenizer, model_name_from_load, device = load_model_and_tokenizer(MODEL_PATH)
    logging.info(f"Model loaded successfully on device: {device}")
    # Derive model_name for directory paths (consistent with other scripts)
    model_name = MODEL_PATH.split('/')[-1]
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
    # Exit or handle error appropriately - maybe raise SystemExit?
    raise SystemExit("Model/Tokenizer loading failed.")





# %% 
# Run Logprobs Analysis Pipeline

from src.experiments.logprobs.pipeline import run_logprobs_analysis_for_hint_types

if model and tokenizer:
    # Define base output directory for this model/dataset
    output_dir_base = os.path.join(DATA_DIR, DATASET, model_name)

    run_logprobs_analysis_for_hint_types(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=model_name, # Use derived name for path consistency
        dataset=DATASET,
        data_dir=DATA_DIR,
        hint_types_to_analyze=HINT_TYPES_TO_ANALYZE,
        intervention_types=INTERVENTION_TYPES,
        percentage_steps=PERCENTAGE_STEPS,
        n_questions=N_QUESTIONS,
        demo_mode_n=DEMO_MODE_N,
        output_dir_base=output_dir_base
    )
else:
    logging.error("Pipeline execution skipped due to model/tokenizer loading failure.")


# %%
