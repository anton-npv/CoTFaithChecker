import json
from datasets import load_dataset

def format_dataset_entry(entry, index):
    """Converts a dataset entry to the desired JSON format,
    handling integer and float formatting for options."""
    formatted_entry = {
        "question_id": index,  # Simple sequential ID starting from 1
        "question": entry["Question"],
        "correct": entry["Answer"] # Already a string (A, B, C, or D)
    }
    for option_key in ["A", "B", "C", "D"]:
        value = entry[option_key]
        if isinstance(value, float) and value.is_integer():
            # If it's a float but represents a whole number, convert to int then str
            formatted_entry[option_key] = str(int(value))
        else:
            # Otherwise, just convert to str (handles actual floats and potentially other types)
            formatted_entry[option_key] = str(value)
    return formatted_entry

# Load the dataset from Hugging Face Hub
print("Loading dataset satoshidg/GSM-MC-Stage...")
# Specify cache_dir if needed, e.g., cache_dir="./hf_cache"
dataset = load_dataset("satoshidg/GSM-MC-Stage")
print("Dataset loaded.")

# Assuming you want to process the 'train' split based on the image
train_split = dataset['train']

# Format the dataset
print(f"Formatting {len(train_split)} entries from the 'train' split...")
formatted_data = [format_dataset_entry(entry, i) for i, entry in enumerate(train_split)]
print("Formatting complete.")

# Define the output file path
output_file = "data/gsm_mc_stage_formatted.json"

# Save the formatted data to a JSON file
print(f"Saving formatted data to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False) # indent=2 for readability

print(f"Successfully saved {len(formatted_data)} entries to {output_file}")

# Optional: Print the first few entries to verify
print("\nFirst 3 formatted entries:")
for i in range(min(3, len(formatted_data))):
    print(json.dumps(formatted_data[i], indent=2))
