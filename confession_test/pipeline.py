import json, os, time, logging
from typing import List, Dict, Optional
import random

import torch
from a_confirm_posthoc.parallelization.model_handler import generate_completion
from accelerate.utils import gather_object

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(path: str) -> List[Dict]:
    """Load JSON data from file"""
    if not os.path.exists(path):
        logging.error(f"Data file not found: {path}")
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {path}: {e}")
        return []


def load_processed_completions(dataset_name: str, model_name: str, hint_type: str, n_questions: int) -> List[Dict]:
    """Load processed completions that have been split into user/assistant turns"""
    path = os.path.join("confession_test", "data", dataset_name, model_name, hint_type, f"processed_completions_with_{n_questions}.json")
    
    if not os.path.exists(path):
        logging.error(f"Processed completions not found: {path}")
        logging.error("Please run: python confession_test/preprocess_completions.py first")
        return []
    
    return load_data(path)


def create_confession_conversation(user_content: str, assistant_content: str, confession_question: str) -> List[Dict]:
    """
    Create a multi-turn conversation for confession test
    """
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": confession_question}
    ]


def save_results(results: List[Dict], dataset: str, hint: str, model: str, n_q: int) -> None:
    """Save confession test results"""
    out_dir = os.path.join("confession_test", "results", dataset, model, hint)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"confessions_with_{n_q}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Confession results saved to {out_path}")


def save_results_demo(results: List[Dict], dataset: str, hint: str, model: str, demo_size: int) -> None:
    """Save demo confession test results with special filename"""
    out_dir = os.path.join("confession_test", "results", dataset, model, hint)
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"confessions_demo_{demo_size}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Demo confession results saved to {out_path}")


def generate_confession_completions(
    accelerator,
    model, tokenizer, model_name, device,
    dataset_name: str,
    hint_type: str,
    confession_question: str,
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
    n_questions: Optional[int] = None,
    demo_mode: bool = False,
    demo_sample_size: int = 10,
) -> None:
    """
    Generate confession completions for existing model completions
    """
    start = time.time()
    if accelerator.is_main_process:
        logging.info(f"Running confession test on {accelerator.num_processes} GPU(s)")
    
    # Load hint verification data to get question IDs
    hint_verification_path = os.path.join(
        "data", dataset_name, model_name, hint_type, f"hint_verification_with_{n_questions or 0}.json"
    )
    hint_verification_data = load_data(hint_verification_path)
    
    if not hint_verification_data:
        logging.error(f"No hint verification data found at {hint_verification_path}")
        return
    
    # Demo mode: randomly sample a subset for quick testing
    if demo_mode:
        original_count = len(hint_verification_data)
        hint_verification_data = random.sample(hint_verification_data, min(demo_sample_size, len(hint_verification_data)))
        logging.info(f"Demo mode: Using {len(hint_verification_data)} random questions out of {original_count} total")
    
    # Load processed completions
    processed_completions = load_processed_completions(dataset_name, model_name, hint_type, n_questions or 0)
    
    if not processed_completions:
        logging.error("No processed completions found")
        return
    
    # Create a mapping of question_id to processed completion
    completions_dict = {comp["question_id"]: comp for comp in processed_completions}
    
    # Filter to only include questions from hint verification
    target_question_ids = {entry["question_id"] for entry in hint_verification_data}
    
    # Create confession conversations
    confession_data = []
    for entry in hint_verification_data:
        qid = entry["question_id"]
        if qid in completions_dict:
            completion_data = completions_dict[qid]
            user_content = completion_data["user"]
            assistant_content = completion_data["assistant"]
            
            if user_content and assistant_content:
                confession_data.append({
                    "question_id": qid,
                    "verbalizes_hint": entry["verbalizes_hint"],
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                    "confession_question": confession_question
                })
    
    if not confession_data:
        logging.error("No valid confession data created")
        return
    
    logging.info(f"Created {len(confession_data)} confession conversations")
    
    # Distribute data across GPUs
    rank, world = accelerator.process_index, accelerator.num_processes
    confession_data = confession_data[rank::world]
    
    # Build multi-turn conversation prompts
    prompts = []
    for entry in confession_data:
        conversation = create_confession_conversation(
            entry["user_content"], 
            entry["assistant_content"], 
            entry["confession_question"]
        )
        prompts.append({
            "question_id": entry["question_id"],
            "conversation": conversation,
            "verbalizes_hint": entry["verbalizes_hint"]
        })
    
    logging.info(f"Processing {len(prompts)} confession prompts on rank {rank}")
    
    # Generate confessions using modified model handler
    results = generate_confession_completion(
        model, tokenizer, device, prompts,
        batch_size, max_new_tokens
    )
    
    # Gather results from all GPUs
    accelerator.wait_for_everyone()
    gathered = gather_object(results)
    
    if accelerator.is_main_process:
        merged = [d for lst in gathered for d in (lst if isinstance(lst, list) else [lst])]
        
        # Use different filename for demo mode
        if demo_mode:
            save_results_demo(merged, dataset_name, hint_type, model_name, demo_sample_size)
        else:
            save_results(merged, dataset_name, hint_type, model_name, n_questions or 0)
        
        logging.info(f"Total confession test time: {time.time() - start:.2f} s")


def generate_confession_completion(
    model, tokenizer, device,
    prompts: List[Dict],
    batch_size: int = 8, 
    max_new_tokens: Optional[int] = 512
) -> List[Dict]:
    """
    Generate completions for confession conversations
    Modified version of the original generate_completion function
    """
    results, gen_max = [], max_new_tokens or 1024
    logging.info(f"Using max_new_tokens: {gen_max}")

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        conversations = [p["conversation"] for p in batch]
        qids = [p["question_id"] for p in batch]
        verbalizes_hints = [p["verbalizes_hint"] for p in batch]

        logging.info(f"Processing confession batch {i//batch_size+1}/"
                     f"{(len(prompts)+batch_size-1)//batch_size} "
                     f"(size {len(conversations)}, QIDs {min(qids)}-{max(qids)})")

        # DEBUG: Print the first conversation to see what it looks like
        if i == 0:
            print("DEBUG: First conversation structure:")
            for j, turn in enumerate(conversations[0]):
                print(f"Turn {j+1} ({turn['role']}): {turn['content'][:200]}...")
                if len(turn['content']) > 200:
                    print(f"... (truncated, total length: {len(turn['content'])})")
            
            # Print the actual confession question
            print(f"\nDEBUG: Confession question being asked: '{conversations[0][2]['content']}'")
            
            # Check if the confession question is about using hints
            if "hint" in conversations[0][2]['content'].lower():
                print("DEBUG: ✓ Confession question contains 'hint' - this is correct")
            else:
                print("DEBUG: ✗ Confession question does NOT contain 'hint' - this is wrong!")
        
        # Format conversations using chat template with no truncation
        formatted_prompts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        
        # DEBUG: Print the first formatted prompt
        if i == 0:
            print("DEBUG: First formatted prompt:")
            print(formatted_prompts[0][:1000] + "..." if len(formatted_prompts[0]) > 1000 else formatted_prompts[0])
            
            # Look for the confession question in the formatted prompt
            if "hint" in formatted_prompts[0].lower():
                print("DEBUG: ✓ Formatted prompt contains 'hint' - confession question is preserved")
            else:
                print("DEBUG: ✗ Formatted prompt does NOT contain 'hint' - confession question is missing!")
        
        # Tokenize with no truncation to preserve full conversation
        enc = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=False,  # Keep this False to preserve full conversation
            return_tensors="pt"
        )

        gen_model = model.module if hasattr(model, "module") else model
        gen_device = next(gen_model.parameters()).device

        input_ids = enc["input_ids"].to(gen_device)
        attention_mask = enc["attention_mask"].to(gen_device)

        print(f"DEBUG: Input sequence length: {input_ids.shape[1]}")

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the new tokens (the model's response to the confession question)
        new_token_outputs = outputs[:, input_ids.shape[1]:]
        confession_responses = tokenizer.batch_decode(new_token_outputs, skip_special_tokens=True)
        
        # DEBUG: Print the first confession response
        if i == 0:
            print("DEBUG: First confession response:")
            print(confession_responses[0][:500] + "..." if len(confession_responses[0]) > 500 else confession_responses[0])
        
        # Clean up the responses - remove any remaining special tokens or artifacts
        confession_responses = [response.strip() for response in confession_responses]
        
        for qid, confession_response, verbalizes_hint in zip(qids, confession_responses, verbalizes_hints):
            results.append({
                "question_id": qid,
                "confession_response": confession_response,
                "verbalizes_hint": verbalizes_hint
            })

    return results 