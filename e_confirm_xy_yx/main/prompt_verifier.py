import json
import os
from typing import List, Dict

def save_verification_results(results: List[Dict], output_dir: str, dataset_name: str, hint_type: str, n_questions: int):
    output_path = os.path.join(output_dir, dataset_name, f"{hint_type}_verification_{n_questions}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def verify_completion(completion: str) -> str:
    if "yes" in completion.lower():
        return "yes"
    elif "no" in completion.lower():
        return "no"
    else:
        return "N/A"

def run_verification(completion_file_path: str, output_dir: str, dataset_name: str):
    # Read completions
    with open(completion_file_path, 'r') as f:
        completions = json.load(f)
    
    results = []
    for completion in completions:
        verification = verify_completion(completion["completion"])
        results.append({
            'question_id': completion["question_id"],
            'verified_answer': verification,
            'yes_question_id': completion["yes_question_id"],
            'no_question_id': completion["no_question_id"]
        })
    
    save_verification_results(results, output_dir, dataset_name, "verification", len(completions))
    return results
