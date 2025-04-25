import os
import json
import logging
from tqdm import tqdm

def setup_logging(log_dir: str, log_file: str):
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info("Logging initialized.")

def run_inference(model, tokenizer, device, dataset_name, data_types, batch_size, max_new_tokens, output_dir, log_file):
    # Setup logging
    setup_logging(output_dir, log_file)
    
    results = []
    
    for data_type in data_types:
        data = fetch_data('data/chainscope/questions_json', *data_type)
        
        for question_data in tqdm(data, desc=f"Processing {data_type} data"):
            question = question_data["questions"][0]  # Get the first question
            prompt = construct_prompt("data/chainscope/templates/instructions.json", question, "instr-v0")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_length=max_new_tokens)
            
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Save the result
            result = {
                'question_id': question['question_id'],
                'completion': answer,
                'yes_question_id': question['yes_question_id'] if 'yes_question_id' in question else None,
                'no_question_id': question['no_question_id'] if 'no_question_id' in question else None
            }
            results.append(result)

        # Save results to file
        result_path = os.path.join(output_dir, f"{dataset_name}_completions_{data_type}.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results
