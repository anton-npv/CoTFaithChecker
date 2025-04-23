import os
import json
import random
from datasets import load_dataset

def format_mmlu_data(sample_size=5000, output_dir="data/mmlu", output_filename="input_mcq_data.json"):
    """
    Loads the MMLU dataset, samples a specified number of examples,
    formats them into the desired JSON structure, and saves the result.

    Args:
        sample_size (int): The number of random samples to draw from the dataset.
        output_dir (str): The directory to save the output JSON file.
        output_filename (str): The name of the output JSON file.
    """
    print("Loading MMLU dataset (all subjects)...")
    # Load the 'test' split which contains the questions and answers
    # Using 'all' configuration loads data from all subjects
    try:
        mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
        print(f"Dataset loaded successfully. Total examples: {len(mmlu_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure sample size is not larger than the dataset
    actual_sample_size = min(sample_size, len(mmlu_dataset))
    if actual_sample_size < sample_size:
        print(f"Warning: Requested sample size ({sample_size}) is larger than dataset size ({len(mmlu_dataset)}). Using {actual_sample_size} samples instead.")

    print(f"Sampling {actual_sample_size} random examples...")
    # Get random indices
    random_indices = random.sample(range(len(mmlu_dataset)), actual_sample_size)
    # Select the samples using the indices
    sampled_data = mmlu_dataset.select(random_indices)

    print("Formatting data...")
    formatted_data = []
    for i, example in enumerate(sampled_data):
        # The 'answer' field is the index of the correct choice
        correct_answer_index = example['answer']
        choices = example['choices']
        
        # Generate letter labels ('A', 'B', 'C', ...) for choices
        choice_labels = [chr(ord('A') + j) for j in range(len(choices))]
        
        # Ensure the correct answer index is valid
        if 0 <= correct_answer_index < len(choice_labels):
            correct_label = choice_labels[correct_answer_index]
        else:
            # Handle cases where the answer index might be out of bounds (shouldn't happen in standard MMLU)
            print(f"Warning: Invalid answer index {correct_answer_index} for question {i}. Skipping.")
            continue # Or handle appropriately

        formatted_entry = {
            "question_id": i,
            "question": example['question'],
            "correct": correct_label,
        }
        
        # Add choices with their corresponding letter labels
        for label, choice_text in zip(choice_labels, choices):
            formatted_entry[label] = choice_text
            
        formatted_data.append(formatted_entry)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving formatted data to {output_path}...")
    try:
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to file: {e}")

if __name__ == "__main__":
    format_mmlu_data()
