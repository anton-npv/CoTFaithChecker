import os
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

def load_logprobs_data(filepath: str) -> dict | None:
    """Loads logprobs JSON data from a file."""
    if not os.path.exists(filepath):
        logging.error(f"Data file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None

def plot_trajectories(
    steps: list[int],
    target_traj: np.ndarray,
    other_traj: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    target_label: str,
    other_label: str,
    save_path: str
):
    """Creates and saves a plot with two trajectories."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, target_traj, marker='o', linestyle='-', label=target_label)
    plt.plot(steps, other_traj, marker='x', linestyle='--', label=other_label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(steps)
    plt.ylim(0, 1) # Probabilities range from 0 to 1
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory
    logging.info(f"Plot saved to {save_path}")

def process_condition_data(
    logprobs_data: dict,
    condition_type: str # 'baseline' or 'hinted'
) -> dict[str, dict[str, np.ndarray]]:
    """Processes loaded logprobs data to calculate average trajectories."""
    if not logprobs_data or 'results' not in logprobs_data:
        logging.warning("Invalid or empty logprobs data provided.")
        return {}

    # {intervention_type: {'target': [traj1, traj2,...], 'other': [traj1, traj2,...]}}
    trajectories = defaultdict(lambda: defaultdict(list))
    percentage_steps = logprobs_data.get("experiment_details", {}).get("percentage_steps", [])
    if not percentage_steps:
        logging.error("Could not determine percentage steps from experiment details.")
        return {}

    num_steps = len(percentage_steps)

    for qid, q_data in logprobs_data['results'].items():
        if 'error' in q_data:
            logging.warning(f"Skipping QID {qid} due to previous processing error: {q_data['error']}")
            continue

        target_option_key = 'verified_answer' if condition_type == 'baseline' else 'hint_option'
        target_option = q_data.get(target_option_key)

        if target_option is None:
            logging.warning(f"Skipping QID {qid} for condition '{condition_type}': Missing '{target_option_key}'.")
            continue

        for int_type, int_data in q_data.items():
            # Skip metadata keys and intervention types with errors
            if int_type in ['status', 'depends_on_hint', 'quartiles', 'hint_option', 'is_correct_option', 'verified_answer', 'error']:
                 continue
            if 'logprobs_sequence' not in int_data or not int_data['logprobs_sequence']:
                logging.warning(f"Skipping QID {qid}, Intervention '{int_type}': Missing or empty 'logprobs_sequence'.")
                continue

            # Ensure sequence length matches expected steps
            if len(int_data['logprobs_sequence']) != num_steps:
                 logging.warning(f"Skipping QID {qid}, Intervention '{int_type}': Sequence length mismatch (expected {num_steps}, got {len(int_data['logprobs_sequence'])}). Data might be incomplete.")
                 continue


            q_target_traj = np.full(num_steps, np.nan)
            q_other_traj = np.full(num_steps, np.nan)

            valid_sequence = True
            for i, step_data in enumerate(int_data['logprobs_sequence']):
                logprobs = step_data.get('logprobs')
                if logprobs is None:
                    logging.warning(f"Missing logprobs for QID {qid}, Intervention '{int_type}', step {i}. Skipping question for this intervention.")
                    valid_sequence = False
                    break # Skip this intervention type for this question if any step is bad

                options = list(logprobs.keys())
                logprob_values = np.array([logprobs[opt] for opt in options])
                num_options = len(options)
                
                # --- Linear Normalization --- 
                shifted_logits = logprob_values
                min_logit = np.min(logprob_values)
                if min_logit <= 0: # Shift if any logit is zero or negative
                    shift_value = abs(min_logit) + 1e-9 # Add epsilon for stability
                    shifted_logits = logprob_values + shift_value
                    # Optional: Log if shifting happened, might be noisy
                    # logging.debug(f"Shifted logits for QID {qid}, Int '{int_type}', Step {i} by {shift_value}")

                logit_sum = np.sum(shifted_logits)
                
                norm_prop_map = {}
                if abs(logit_sum) < 1e-9: # Handle potential zero sum after shift (highly unlikely but safe)
                     logging.warning(f"Logit sum near zero for QID {qid}, Int '{int_type}', Step {i}. Assigning uniform proportions.")
                     uniform_prop = 1.0 / num_options
                     norm_prop_map = {opt: uniform_prop for opt in options}
                else:
                     norm_prop_map = {opt: val / logit_sum for opt, val in zip(options, shifted_logits)}
                # --- End Linear Normalization ---

                if target_option not in norm_prop_map:
                    logging.warning(f"Target option '{target_option}' not found in normalized proportions for QID {qid}, Intervention '{int_type}', step {i}. Options: {options}. Skipping question for this intervention.")
                    valid_sequence = False
                    break

                q_target_traj[i] = norm_prop_map[target_option]

                # Find the proportion of the highest *other* option
                other_props = {opt: p for opt, p in norm_prop_map.items() if opt != target_option}
                if not other_props:
                    q_other_traj[i] = np.nan # No other options to compare against
                else:
                    max_other_prop = max(other_props.values())
                    q_other_traj[i] = max_other_prop

            if valid_sequence:
                 trajectories[int_type]['target'].append(q_target_traj)
                 trajectories[int_type]['other'].append(q_other_traj)


    # Average trajectories across questions
    averaged_trajectories = {}
    for int_type, data in trajectories.items():
        if data['target'] and data['other']: # Ensure we have data to average
            avg_target = np.nanmean(np.array(data['target']), axis=0)
            avg_other = np.nanmean(np.array(data['other']), axis=0)
            # Check if averaging resulted in NaNs (e.g., if all entries for a step were NaN)
            if np.isnan(avg_target).any() or np.isnan(avg_other).any():
                 logging.warning(f"NaN values encountered after averaging trajectories for intervention '{int_type}'. Some steps might have lacked valid data across all questions.")
            averaged_trajectories[int_type] = {
                'target': avg_target,
                'other': avg_other
            }
        else:
            logging.warning(f"No valid trajectories found to average for intervention type '{int_type}'.")


    return averaged_trajectories, percentage_steps


# --- Main Function ---

def create_trajectory_visualizations(
    dataset_name: str,
    model_name: str,
    hint_types_to_analyze: list[str],
    intervention_types: list[str]
):
    """Loads logprobs results and generates trajectory plots."""
    logging.info(f"Starting visualization for Dataset: {dataset_name}, Model: {model_name}")
    base_results_dir = os.path.join("b_logprobs_analysis", "outputs", dataset_name, model_name)

    # --- Process Baseline ---
    logging.info("--- Processing Baseline --- ")
    baseline_file = os.path.join(base_results_dir, "baseline_logprobs.json")
    baseline_data = load_logprobs_data(baseline_file)
    if baseline_data:
        avg_baseline_trajectories, steps = process_condition_data(baseline_data, 'baseline')
        if avg_baseline_trajectories:
            baseline_fig_dir = os.path.join(base_results_dir, "baseline", "figures")
            for int_type, avg_data in avg_baseline_trajectories.items():
                if int_type not in intervention_types: # Only plot requested intervention types
                    continue
                plot_title = f"Baseline: Avg. Probability Trajectory ({int_type})\n{model_name} on {dataset_name}"
                save_path = os.path.join(baseline_fig_dir, f"{int_type}_trajectory.png")
                plot_trajectories(
                    steps=steps,
                    target_traj=avg_data['target'],
                    other_traj=avg_data['other'],
                    title=plot_title,
                    xlabel="Reasoning Step (%)",
                    ylabel="Average Normalized Logit Proportion",
                    target_label="Verified Answer",
                    other_label="Max Other Option",
                    save_path=save_path
                )
        else:
            logging.warning("No averaged trajectories could be calculated for baseline.")
    else:
        logging.warning("Could not load baseline data. Skipping baseline plots.")


    # --- Process Hint Types ---
    for hint_type in hint_types_to_analyze:
        logging.info(f"--- Processing Hint Type: {hint_type} --- ")
        hint_file = os.path.join(base_results_dir, f"{hint_type}_logprobs.json")
        hint_data = load_logprobs_data(hint_file)
        if hint_data:
            avg_hint_trajectories, steps = process_condition_data(hint_data, 'hinted')
            if avg_hint_trajectories:
                hint_fig_dir = os.path.join(base_results_dir, hint_type, "figures")
                for int_type, avg_data in avg_hint_trajectories.items():
                    if int_type not in intervention_types: # Only plot requested intervention types
                         continue
                    plot_title = f"Hint Type: {hint_type} - Avg. Probability Trajectory ({int_type})\n{model_name} on {dataset_name}"
                    save_path = os.path.join(hint_fig_dir, f"{int_type}_trajectory.png")
                    plot_trajectories(
                        steps=steps,
                        target_traj=avg_data['target'],
                        other_traj=avg_data['other'],
                        title=plot_title,
                        xlabel="Reasoning Step (%)",
                        ylabel="Average Normalized Logit Proportion",
                        target_label="Hint Option",
                        other_label="Max Other Option",
                        save_path=save_path
                    )
            else:
                 logging.warning(f"No averaged trajectories could be calculated for hint type '{hint_type}'.")
        else:
             logging.warning(f"Could not load data for hint type '{hint_type}'. Skipping plots for this hint type.")

    logging.info("Visualization generation finished.")



create_trajectory_visualizations(
        dataset_name="mmlu",
        model_name="DeepSeek-R1-Distill-Qwen-1.5B",
        hint_types_to_analyze=["sycophancy"],
        intervention_types=["dots", "dots_eot"]
    )

# # --- Script Execution ---

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate log probability trajectory visualizations.")
#     parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., mmlu)")
#     parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., DeepSeek-R1-Distill-Llama-8B)")
#     parser.add_argument("--hint_types", nargs='+', required=True, help="List of hint types to analyze (e.g., induced_urgency sycophancy)")
#     parser.add_argument("--intervention_types", nargs='+', required=True, help="List of intervention types used (e.g., dots dots_eot)")

#     args = parser.parse_args()

#     create_trajectory_visualizations(
#         dataset_name=args.dataset_name,
#         model_name=args.model_name,
#         hint_types_to_analyze=args.hint_types,
#         intervention_types=args.intervention_types
#     )