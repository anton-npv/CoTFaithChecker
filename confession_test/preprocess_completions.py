"""
Preprocessing script to clean up completion format and split into turns
"""

import json
import os
import argparse
import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_completions(path: str) -> List[Dict]:
    """Load completions from JSON file"""
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return []
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        return []


def parse_completion_turns(completion: str) -> tuple[str, str]:
    """
    Parse completion text to extract user and assistant parts
    
    Format: <|begin_of_sentence|><|User|>Question: blah blah blah...<think><|Assistant|><think>blah blah blah....</think> final answer ....<|end_of_sentence|>
    
    Returns: (user_content, assistant_content)
    """
    # Remove begin/end tokens and unicode escape sequences
    completion = completion.replace('<|begin_of_sentence|>', '').replace('<|end_of_sentence|>', '')
    completion = completion.replace('\\uff5c', '|').replace('\\u2581', '_')
    
    # Find user content: everything between <|User|> and first <think>
    user_match = re.search(r'<\|User\|>(.*?)<think>', completion, re.DOTALL)
    if not user_match:
        logging.warning("Could not find user content in completion")
        return "", ""
    
    user_content = user_match.group(1).strip()
    
    # Find assistant content: everything after <|Assistant|> (including the <think> token)
    # The assistant content should start with <think> and end with </think>
    assistant_match = re.search(r'<\|Assistant\|>(<think>.*?)(?:<\|end_of_sentence\|>|$)', completion, re.DOTALL)
    if not assistant_match:
        # Fallback: try to find everything after <|Assistant|>
        assistant_match = re.search(r'<\|Assistant\|>(.*?)(?:<\|end_of_sentence\|>|$)', completion, re.DOTALL)
    
    if not assistant_match:
        logging.warning("Could not find assistant content in completion")
        return user_content, ""
    
    assistant_content = assistant_match.group(1).strip()
    
    return user_content, assistant_content


def preprocess_completions(
    dataset_name: str,
    model_name: str, 
    hint_type: str,
    n_questions: int
) -> None:
    """
    Preprocess completions and save in cleaned format
    """
    # Load fixed completions (use the fixed version if available)
    fixed_path = os.path.join(
        "data", dataset_name, model_name, hint_type, f"completions_with_{n_questions}_fixed.json"
    )
    
    if os.path.exists(fixed_path):
        completions_path = fixed_path
        logging.info(f"Using fixed completions file: {fixed_path}")
    else:
        completions_path = os.path.join(
            "data", dataset_name, model_name, hint_type, f"completions_with_{n_questions}.json"
        )
        logging.info(f"Using original completions file: {completions_path}")
    
    completions_data = load_completions(completions_path)
    if not completions_data:
        logging.error(f"No completions data found at {completions_path}")
        return
    
    logging.info(f"Processing {len(completions_data)} completions...")
    
    # Process each completion
    processed_completions = []
    failed_parses = 0
    
    for entry in completions_data:
        question_id = entry["question_id"]
        completion = entry["completion"]
        
        user_content, assistant_content = parse_completion_turns(completion)
        
        if user_content and assistant_content:
            processed_completions.append({
                "question_id": question_id,
                "user": user_content,
                "assistant": assistant_content
            })
        else:
            failed_parses += 1
            logging.warning(f"Failed to parse completion for question_id {question_id}")
    
    logging.info(f"Successfully processed {len(processed_completions)} completions")
    if failed_parses > 0:
        logging.warning(f"Failed to parse {failed_parses} completions")
    
    # Save processed completions
    output_dir = os.path.join("confession_test", "data", dataset_name, model_name, hint_type)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"processed_completions_with_{n_questions}.json")
    with open(output_path, 'w') as f:
        json.dump(processed_completions, f, indent=2)
    
    logging.info(f"Processed completions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess completions into clean user/assistant turns")
    parser.add_argument("--dataset", default="mmlu_new", help="Dataset name")
    parser.add_argument("--model", default="DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--hint_type", default="sycophancy", help="Hint type")
    parser.add_argument("--n_questions", type=int, default=8960, help="Number of questions")
    
    args = parser.parse_args()
    
    preprocess_completions(
        args.dataset,
        args.model,
        args.hint_type,
        args.n_questions
    )


if __name__ == "__main__":
    main() 