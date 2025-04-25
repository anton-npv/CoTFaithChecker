import json
import os

def save_matched_results(results: List[Dict], output_dir: str, dataset_name: str):
    output_path = os.path.join(output_dir, dataset_name, "matched_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def match_checker(no_file_path: str, yes_file_path: str, output_dir: str, dataset_name: str):
    # Read the NO and YES completions
    with open(no_file_path, 'r') as f:
        no_completions = json.load(f)
    
    with open(yes_file_path, 'r') as f:
        yes_completions = json.load(f)

    # Create a mapping of YES completions by question_id
    yes_map = {comp["question_id"]: comp["verified_answer"] for comp in yes_completions}
    
    matched_results = []
    
    for no_comp in no_completions:
        yes_answer = yes_map.get(no_comp["yes_question_id"], "N/A")
        
        matched_results.append({
            'no_file_q_id': no_comp["question_id"],
            'yes_question_id': no_comp["yes_question_id"],
            'no_answer': no_comp["verified_answer"],
            'yes_answer': yes_answer
        })
    
    save_matched_results(matched_results, output_dir, dataset_name)
    return matched_results
