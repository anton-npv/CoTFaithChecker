"""High-level pipeline function for the logprobs experiment."""

import os
import json
import logging
from tqdm import tqdm

from .io import load_hint_verification_data, load_mcq_data, load_completion, load_switch_analysis_data, load_verification_data
from .logit_extraction import get_option_token_ids, find_reasoning_end, extract_logprobs_sequence, find_reasoning_start
from .utils import get_intervention_prompt

def run_logprobs_analysis_for_hint_types(
    model,
    tokenizer,
    device: str,
    model_name: str,
    dataset: str,
    data_dir: str,
    hint_types_to_analyze: list[str], # e.g., ["induced_urgency", "sycophancy"]
    intervention_types: list[str],
    percentage_steps: list[int],
    n_questions: int,
    demo_mode_n: int | None = None,
    # output_dir_base: str | None = None # Base directory for saving results - REMOVED
):
    """Runs the logit extraction analysis for a model/dataset across specified hint types.

    Saves results separately for baseline and each analyzed hint type.
    """
    logging.info(f"--- Starting Logprobs Analysis --- ")
    logging.info(f"Model: {model_name}, Dataset: {dataset}")
    logging.info(f"Analyzing Hint Types: {hint_types_to_analyze}")
    logging.info(f"Intervention Types: {intervention_types}")

    # --- 1. Load MCQ Data --- 
    mcq_data = load_mcq_data(data_dir=data_dir, dataset=dataset)
    if not mcq_data:
        logging.error(f"Failed to load MCQ data for {dataset}. Aborting.")
        return
    logging.info(f"Loaded MCQ data for {len(mcq_data)} questions.")

    # --- 2. Determine Relevant QIDs & Standard Options --- 
    all_relevant_qids = set()
    all_hint_verification_data = {}
    all_switch_analysis_data = {} # Store switch analysis data per hint type
    for ht in hint_types_to_analyze:
        logging.info(f"Loading hint verification data for hint type: {ht}")
        verification_data = load_hint_verification_data(
            data_dir=data_dir, dataset=dataset, model_name=model_name, 
            hint_type=ht, n_questions=n_questions
        )
        # Load switch analysis data for this hint type
        logging.info(f"Loading switch analysis data for hint type: {ht}")
        switch_data = load_switch_analysis_data(
            data_dir=data_dir, dataset=dataset, model_name=model_name,
            hint_type=ht, n_questions=n_questions
        )
        all_hint_verification_data[ht] = verification_data
        all_switch_analysis_data[ht] = switch_data # Store it
        qids_for_hint = set(verification_data.keys())
        logging.info(f"  Found {len(qids_for_hint)} questions that switched to hint '{ht}'.")
        all_relevant_qids.update(qids_for_hint)
    
    logging.info(f"Total relevant QIDs (switched to any analyzed hint): {len(all_relevant_qids)}")

    if not all_relevant_qids:
        logging.warning("No relevant questions found based on hint verification files. Nothing to process.")
        return

    # Determine standard options and get token IDs once
    # Assuming options are consistent, check first relevant QID
    first_qid = next(iter(all_relevant_qids))
    question_data = mcq_data[first_qid]
    # Extract option labels (e.g., 'A', 'B', 'C', 'D') by finding single uppercase letter keys
    options = sorted([key for key in question_data.keys() if len(key) == 1 and key.isupper()])
    if not options:
         raise ValueError(f"Could not determine standard options from keys of QID {first_qid}")
    
    logging.info(f"Determined standard options: {options}")
    standard_option_token_ids = get_option_token_ids(options, tokenizer)
    if len(standard_option_token_ids) != len(options):
         raise ValueError(f"Failed to map all standard option tokens: {options}. Found: {standard_option_token_ids}")
    logging.info(f"Standard option token IDs: {standard_option_token_ids}")

    # Apply demo mode limit if needed
    qids_to_process_final = sorted(list(all_relevant_qids))
    if demo_mode_n is not None and demo_mode_n > 0:
        qids_to_process_final = qids_to_process_final[:demo_mode_n]
        logging.warning(f"Running in DEMO mode. Processing only {len(qids_to_process_final)} relevant questions.")

    # --- 3. Process Baseline ("none") --- 
    baseline_results = {}
    logging.info(f"Processing baseline ('none') completions for {len(qids_to_process_final)} relevant questions...")
    
    # Load baseline verification data specifically for baseline processing
    logging.info("Loading baseline (none) verification data...")
    baseline_verification_map = load_verification_data(
        data_dir=data_dir, dataset=dataset, model_name=model_name, hint_type="none", n_questions=n_questions
    )
    logging.info(f"Loaded baseline verification data for {len(baseline_verification_map)} questions.")

    for qid in tqdm(qids_to_process_final, desc="Processing Baseline"):
        # Get verified answer from the baseline map
        baseline_verif_info = baseline_verification_map.get(qid, {})
        verified_answer = baseline_verif_info.get("verified_answer")

        baseline_results[qid] = {
            "verified_answer": verified_answer
        }
        completion = load_completion(data_dir, dataset, model_name, "none", n_questions, qid)
        if not completion:
            logging.warning(f"Baseline completion not found for QID {qid}. Skipping baseline processing for this QID.")
            continue # Skip this qid for baseline
        
        try:
            reasoning_start = find_reasoning_start(completion, model_name)
            reasoning_end = find_reasoning_end(completion, model_name)
            

            for intervention_type in intervention_types:
                prompt = get_intervention_prompt(intervention_type)
                if not prompt:
                    raise ValueError(f"Could not generate prompt for type {intervention_type}.")
                
                sequence = extract_logprobs_sequence(
                    completion_text=completion,
                    reasoning_start_index=reasoning_start,
                    reasoning_end_index=reasoning_end,
                    option_token_ids=standard_option_token_ids,
                    intervention_prompt=prompt,
                    percentage_steps=percentage_steps,
                    model=model, tokenizer=tokenizer, device=device
                )
                reasoning_tokens = sequence[-1]["token_index"] if sequence else 0
                baseline_results[qid][intervention_type] = {
                    "reasoning_tokens": reasoning_tokens,
                    "logprobs_sequence": sequence
                }
        except Exception as e:
            logging.error(f"Error processing baseline for QID {qid}: {e}", exc_info=True)
            # Mark as failed or remove partial results?
            # Keep verified_answer if already added
            baseline_results[qid]["error"] = str(e) 

    # --- 4. Save Baseline Results --- 
    # Define output directory directly
    logprobs_output_dir = os.path.join("b_logprobs_analysis", "outputs", dataset, model_name)
    os.makedirs(logprobs_output_dir, exist_ok=True)
    
    baseline_output_filename = os.path.join(logprobs_output_dir, "baseline_logprobs.json")
    logging.info(f"Saving baseline logprobs results to {baseline_output_filename}")
    try:
        output_data = {
            "experiment_details": {
                "dataset": dataset,
                "model": model_name,
                "hint_type": "none",
                "intervention_types": intervention_types,
                "percentage_steps": percentage_steps
            },
            "results": baseline_results
        }
        with open(baseline_output_filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        logging.info("Baseline results saved.")
    except Exception as e:
        logging.error(f"Failed to save baseline results: {e}")

    # --- 5. Loop Through Hint Types --- 
    for hint_type in hint_types_to_analyze:
        logging.info(f"--- Processing Hint Type: {hint_type} ---")
        hinted_results = {}
        hint_verification_data = all_hint_verification_data.get(hint_type, {})
        switch_analysis_data = all_switch_analysis_data.get(hint_type, {}) # Get switch data

        # Load verification data for the *current* hint type
        logging.info(f"Loading verification data for hint type: {hint_type}...")
        current_verification_map = load_verification_data(
            data_dir=data_dir, dataset=dataset, model_name=model_name, hint_type=hint_type, n_questions=n_questions
        )
        logging.info(f"Loaded verification data for {len(current_verification_map)} questions for hint {hint_type}.")
        
        # Process only the QIDs relevant to *this* hint type and also in the final processing list
        qids_for_this_hint = [qid for qid in qids_to_process_final if qid in hint_verification_data]
        
        if not qids_for_this_hint:
            logging.warning(f"No questions to process for hint type '{hint_type}' after filtering/demo mode. Skipping.")
            continue

        logging.info(f"Processing {len(qids_for_this_hint)} questions for hint type: {hint_type}")
        for qid in tqdm(qids_for_this_hint, desc=f"Processing {hint_type}"):
            # Retrieve verification and switch info safely
            verification_info = hint_verification_data.get(qid, {})
            switch_info = switch_analysis_data.get(qid, {})

            # Determine status and other fields (from hint_verification_with_N.json)
            verbalizes = verification_info.get("verbalizes_hint", False)
            status = "verbalized" if verbalizes else "non_verbalized"
            depends_on_hint = verification_info.get("depends_on_hint")
            quartiles = verification_info.get("quartiles")
            hint_option = switch_info.get("hint_option") # From switch_analysis
            is_correct_option = switch_info.get("is_correct_option") # From switch_analysis

            # Get verified answer from the hint-specific verification map (verification_with_N.json)
            current_verif_info = current_verification_map.get(qid, {})
            verified_answer = current_verif_info.get("verified_answer")

            # Initialize hinted_results with all metadata fields
            hinted_results[qid] = {
                "status": status,
                "depends_on_hint": depends_on_hint,
                "quartiles": quartiles,
                "hint_option": hint_option,
                "is_correct_option": is_correct_option,
                "verified_answer": verified_answer # Add the verified answer here
            }

            completion = load_completion(data_dir, dataset, model_name, hint_type, n_questions, qid)
            if not completion:
                logging.warning(f"Hinted completion '{hint_type}' not found for QID {qid}. Skipping processing for this QID/hint.")
                hinted_results[qid]["error"] = "Completion file not found or empty"
                continue
            
            try:
                reasoning_start = find_reasoning_start(completion, model_name)
                reasoning_end = find_reasoning_end(completion, model_name)

                if reasoning_end <= reasoning_start:
                    logging.warning(f"Invalid reasoning boundaries for hint '{hint_type}' QID {qid} (Start: {reasoning_start}, End: {reasoning_end}). Skipping.")
                    hinted_results[qid]["error"] = "Invalid reasoning boundaries"
                    continue

                for intervention_type in intervention_types:
                    prompt = get_intervention_prompt(intervention_type)
                    if not prompt:
                         raise ValueError(f"Could not generate prompt for type {intervention_type}.")

                    sequence = extract_logprobs_sequence(
                        completion_text=completion,
                        reasoning_start_index=reasoning_start,
                        reasoning_end_index=reasoning_end,
                        option_token_ids=standard_option_token_ids,
                        intervention_prompt=prompt,
                        percentage_steps=percentage_steps,
                        model=model, tokenizer=tokenizer, device=device
                    )
                    reasoning_tokens = sequence[-1]["token_index"] if sequence else 0
                    # Add the logprobs results under the intervention type key
                    hinted_results[qid][intervention_type] = {
                        "reasoning_tokens": reasoning_tokens,
                        "logprobs_sequence": sequence
                    }
            except Exception as e:
                 logging.error(f"Error processing hint '{hint_type}' for QID {qid}: {e}", exc_info=True)
                 # Add error message, preserving existing metadata
                 hinted_results[qid]["error"] = str(e)

        # --- Save Results for This Hint Type --- 
        hinted_output_filename = os.path.join(logprobs_output_dir, f"{hint_type}_logprobs.json")
        logging.info(f"Saving '{hint_type}' logprobs results to {hinted_output_filename}")
        try:
            output_data = {
                "experiment_details": {
                    "dataset": dataset,
                    "model": model_name,
                    "analyzed_hint_type": hint_type,
                    "intervention_types": intervention_types,
                    "percentage_steps": percentage_steps
                },
                "results": hinted_results
            }
            with open(hinted_output_filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"'{hint_type}' results saved.")
        except Exception as e:
            logging.error(f"Failed to save '{hint_type}' results: {e}")

    logging.info("--- Logprobs Analysis Finished --- ") 