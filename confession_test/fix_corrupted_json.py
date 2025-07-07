"""
Fix corrupted JSON completions file that has duplicate completion fields
"""

import json
import os
import argparse
import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fix_corrupted_completions(file_path: str) -> List[Dict]:
    """
    Fix corrupted JSON file that has multiple completion fields per entry
    """
    logging.info(f"Attempting to fix corrupted JSON file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into entries based on question_id patterns
    entries = []
    
    # Find all question_id patterns
    question_pattern = r'"question_id":\s*(\d+)'
    matches = list(re.finditer(question_pattern, content))
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        
        # Find the end position (next question_id or end of file)
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        # Extract this entry's content
        entry_content = content[start_pos:end_pos]
        
        # Extract question_id
        question_id = int(match.group(1))
        
        # Find the first completion field (usually the valid one)
        # Use a more robust pattern that handles nested quotes
        completion_start = entry_content.find('"completion":')
        if completion_start != -1:
            # Find the start of the completion value
            value_start = entry_content.find('"', completion_start + len('"completion":'))
            if value_start != -1:
                # Find the end of the completion value by counting quotes and escapes
                value_end = value_start + 1
                while value_end < len(entry_content):
                    if entry_content[value_end] == '"' and entry_content[value_end-1] != '\\':
                        break
                    value_end += 1
                
                if value_end < len(entry_content):
                    # Extract the completion without quotes
                    completion = entry_content[value_start + 1:value_end]
                    
                    entries.append({
                        "question_id": question_id,
                        "completion": completion
                    })
            
            if len(entries) % 100 == 0:
                logging.info(f"Processed {len(entries)} entries...")
    
    logging.info(f"Successfully extracted {len(entries)} valid entries")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Fix corrupted completions JSON file")
    parser.add_argument("--dataset", default="mmlu_new", help="Dataset name")
    parser.add_argument("--model", default="DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--hint_type", default="sycophancy", help="Hint type")
    parser.add_argument("--n_questions", type=int, default=8960, help="Number of questions")
    
    args = parser.parse_args()
    
    # Original corrupted file path
    original_path = os.path.join(
        "data", args.dataset, args.model, args.hint_type, f"completions_with_{args.n_questions}.json"
    )
    
    if not os.path.exists(original_path):
        logging.error(f"Original file not found: {original_path}")
        return
    
    # Fix the corrupted JSON
    try:
        fixed_entries = fix_corrupted_completions(original_path)
        
        # Save the fixed version
        fixed_path = os.path.join(
            "data", args.dataset, args.model, args.hint_type, f"completions_with_{args.n_questions}_fixed.json"
        )
        
        with open(fixed_path, 'w') as f:
            json.dump(fixed_entries, f, indent=2)
        
        logging.info(f"Fixed JSON saved to: {fixed_path}")
        logging.info(f"Original entries expected: {args.n_questions}")
        logging.info(f"Fixed entries extracted: {len(fixed_entries)}")
        
    except Exception as e:
        logging.error(f"Failed to fix JSON: {e}")


if __name__ == "__main__":
    main() 