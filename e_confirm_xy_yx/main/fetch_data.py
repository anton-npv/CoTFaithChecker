import os
import json
import logging
from typing import List

def setup_logging(log_dir: str, log_file: str):
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info("Logging initialized.")

def fetch_data(data_dir: str, comparison_type: str, expected_answer: str) -> List[dict]:
    folder_name = f"{comparison_type}_{expected_answer}_1"
    data_folder_path = os.path.join(data_dir, folder_name)
    
    # Fetch all JSON files
    all_data_files = [f for f in os.listdir(data_folder_path) if f.endswith('.json')]
    
    all_data = []
    
    for data_file in all_data_files:
        file_path = os.path.join(data_folder_path, data_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    return all_data
