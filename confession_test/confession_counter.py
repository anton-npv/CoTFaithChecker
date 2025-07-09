#!/usr/bin/env python3
"""
Confession Counter - Count how often models confess across verbalizes_hint conditions

Usage:
python confession_test/confession_counter.py \
  --dataset mmlu_new \
  --model DeepSeek-R1-Distill-Llama-8B \
  --hint_type sycophancy \
  --input_suffix confession_extractions_prefill_with_8960
"""

import sys
import pathlib
import os
import json
import argparse
from typing import List, Dict
from pathlib import Path

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def read_confession_extractions(data_path: str) -> List[Dict]:
    """Read confession extraction results from JSON file"""
    with open(data_path, 'r') as f:
        return json.load(f)

def count_confessions(data: List[Dict]) -> Dict:
    """Count confessions across different conditions"""
    
    # Filter out errors
    valid_data = [item for item in data if item["confession_answer"] != "error"]
    
    # Separate by verbalizes_hint
    faithful_items = [item for item in valid_data if item["verbalizes_hint"]]
    unfaithful_items = [item for item in valid_data if not item["verbalizes_hint"]]
    
    # Count confession answers
    def count_answers(items):
        counts = {
            "yes": sum(1 for item in items if item["confession_answer"] == "yes"),
            "no": sum(1 for item in items if item["confession_answer"] == "no"),
            "unclear": sum(1 for item in items if item["confession_answer"] == "unclear"),
            "total": len(items)
        }
        return counts
    
    faithful_counts = count_answers(faithful_items)
    unfaithful_counts = count_answers(unfaithful_items)
    total_counts = count_answers(valid_data)
    
    return {
        "faithful": faithful_counts,
        "unfaithful": unfaithful_counts,
        "total": total_counts,
        "error_count": len(data) - len(valid_data)
    }

def print_results(counts: Dict, dataset: str, model: str, hint_type: str):
    """Print detailed confession counting results"""
    
    print("="*60)
    print("CONFESSION COUNTING RESULTS")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Hint Type: {hint_type}")
    print(f"Total processed: {counts['total']['total']}")
    print(f"Errors: {counts['error_count']}")
    print()
    
    # Overall results
    print("OVERALL RESULTS:")
    print("-" * 40)
    total = counts['total']
    if total['total'] > 0:
        print(f"Yes confessions: {total['yes']} ({total['yes']/total['total']*100:.1f}%)")
        print(f"No confessions: {total['no']} ({total['no']/total['total']*100:.1f}%)")
        print(f"Unclear responses: {total['unclear']} ({total['unclear']/total['total']*100:.1f}%)")
    print()
    
    # Faithful completions (verbalizes_hint=True)
    print("FAITHFUL COMPLETIONS (verbalizes_hint=True):")
    print("-" * 40)
    faithful = counts['faithful']
    if faithful['total'] > 0:
        print(f"Total: {faithful['total']}")
        print(f"Yes confessions: {faithful['yes']} ({faithful['yes']/faithful['total']*100:.1f}%)")
        print(f"No confessions: {faithful['no']} ({faithful['no']/faithful['total']*100:.1f}%)")
        print(f"Unclear responses: {faithful['unclear']} ({faithful['unclear']/faithful['total']*100:.1f}%)")
    else:
        print("No faithful completions found")
    print()
    
    # Unfaithful completions (verbalizes_hint=False)
    print("UNFAITHFUL COMPLETIONS (verbalizes_hint=False):")
    print("-" * 40)
    unfaithful = counts['unfaithful']
    if unfaithful['total'] > 0:
        print(f"Total: {unfaithful['total']}")
        print(f"Yes confessions: {unfaithful['yes']} ({unfaithful['yes']/unfaithful['total']*100:.1f}%)")
        print(f"No confessions: {unfaithful['no']} ({unfaithful['no']/unfaithful['total']*100:.1f}%)")
        print(f"Unclear responses: {unfaithful['unclear']} ({unfaithful['unclear']/unfaithful['total']*100:.1f}%)")
    else:
        print("No unfaithful completions found")
    print()
    
    # Comparison
    print("COMPARISON:")
    print("-" * 40)
    if faithful['total'] > 0 and unfaithful['total'] > 0:
        faithful_confession_rate = faithful['yes'] / faithful['total']
        unfaithful_confession_rate = unfaithful['yes'] / unfaithful['total']
        
        print(f"Faithful confession rate: {faithful_confession_rate*100:.1f}%")
        print(f"Unfaithful confession rate: {unfaithful_confession_rate*100:.1f}%")
        
        if faithful_confession_rate > unfaithful_confession_rate:
            diff = faithful_confession_rate - unfaithful_confession_rate
            print(f"Faithful models confess {diff*100:.1f}% more often")
        elif unfaithful_confession_rate > faithful_confession_rate:
            diff = unfaithful_confession_rate - faithful_confession_rate
            print(f"Unfaithful models confess {diff*100:.1f}% more often")
        else:
            print("Both groups confess at the same rate")

def save_summary(counts: Dict, output_path: str, dataset: str, model: str, hint_type: str):
    """Save summary statistics to JSON file"""
    
    summary = {
        "dataset": dataset,
        "model": model,
        "hint_type": hint_type,
        "counts": counts,
        "summary_stats": {
            "total_processed": counts['total']['total'],
            "error_count": counts['error_count'],
            "faithful_confession_rate": counts['faithful']['yes'] / counts['faithful']['total'] if counts['faithful']['total'] > 0 else 0,
            "unfaithful_confession_rate": counts['unfaithful']['yes'] / counts['unfaithful']['total'] if counts['unfaithful']['total'] > 0 else 0,
            "overall_confession_rate": counts['total']['yes'] / counts['total']['total'] if counts['total']['total'] > 0 else 0
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Count confession rates across verbalizes_hint conditions")
    parser.add_argument("--dataset", default="mmlu_new", help="Dataset name")
    parser.add_argument("--model", default="DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--hint_type", default="sycophancy", help="Hint type")
    parser.add_argument("--input_suffix", default="confession_extractions_prefill_with_8960", help="Input file suffix")
    
    args = parser.parse_args()
    
    # Construct input path
    input_path = f"confession_test/results/{args.dataset}/{args.model}/{args.hint_type}/{args.input_suffix}.json"
    
    print(f"Loading confession extraction data from: {input_path}")
    
    try:
        data = read_confession_extractions(input_path)
        print(f"Loaded {len(data)} confession extractions")
        
        # Count confessions
        counts = count_confessions(data)
        
        # Print results
        print_results(counts, args.dataset, args.model, args.hint_type)
        
        # Save summary
        output_path = f"confession_test/results/{args.dataset}/{args.model}/{args.hint_type}/confession_summary_{args.input_suffix.replace('confession_extractions_', '')}.json"
        save_summary(counts, output_path, args.dataset, args.model, args.hint_type)
        
        print(f"Summary saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        print("Make sure you have run the confession extractor first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 