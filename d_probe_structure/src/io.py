"""Functions for loading and accessing experiment data."""

import json
import pandas as pd
import os
import logging 

def load_hint_verification_data(data_dir: str, dataset: str, model_name: str, hint_type: str, n_questions: int) -> dict[int, bool]:
    """Loads hint verification data and returns a map from question ID to verbalization status.

    Args:
        data_dir: The root data directory (e.g., './data').
        dataset: The name of the dataset (e.g., 'mmlu').
        model_name: The name of the model used for completions.
        hint_type: The specific hint type analyzed (e.g., 'induced_urgency').
        n_questions: The number of questions the original experiment ran on (e.g., 500).

    Returns:
        A dictionary mapping question_id (int) to verbalizes_hint (bool).
        Returns an empty dict if the file doesn't exist.
    """
    file_path = os.path.join(
        data_dir,
        dataset,
        model_name,
        hint_type,
        f"hint_verification_with_{n_questions}.json"
    )

    verbalization_status = {}
    if not os.path.exists(file_path):
        logging.warning(f"Hint verification file not found: {file_path}")
        return verbalization_status

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if 'question_id' in item and 'verbalizes_hint' in item:
                verbalization_status[item['question_id']] = item['verbalizes_hint']
            else:
                logging.warning(f"Skipping malformed entry in {file_path}: {item}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

    return verbalization_status 

def load_mcq_data(data_dir: str, dataset: str) -> dict[int, dict]:
    """Loads the input MCQ data containing questions and options.

    Args:
        data_dir: The root data directory (e.g., './data').
        dataset: The name of the dataset (e.g., 'mmlu').

    Returns:
        A dictionary mapping question_id (int) to its data (including options).
        Returns an empty dict if the file doesn't exist.
    """
    # NOTE: This loads the entire file. May need optimization for huge datasets.
    file_path = os.path.join(data_dir, dataset, "input_mcq_data.json")

    mcq_data = {}
    if not os.path.exists(file_path):
        logging.warning(f"MCQ data file not found: {file_path}")
        return mcq_data

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Assuming the input JSON is a list of dictionaries, each with a 'question_id'
        if isinstance(data, list):
            for item in data:
                if 'question_id' in item:
                    mcq_data[item['question_id']] = item
                else:
                    logging.warning(f"Skipping entry without 'question_id' in {file_path}")
        else:
            logging.error(f"Unexpected format in {file_path}. Expected a list of dicts.")

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")

    return mcq_data 

def load_completion(data_dir: str, dataset: str, model_name: str, hint_type: str, n_questions: int, question_id: int) -> str | None:
    """Loads the specific completion text for a given question ID and hint type.

    Args:
        data_dir: The root data directory.
        dataset: The dataset name.
        model_name: The model name.
        hint_type: The hint type ('none', 'induced_urgency', etc.).
        n_questions: The number of questions run.
        question_id: The ID of the question to load completion for.

    Returns:
        The completion text (str) or None if not found or error occurs.
    """
    # NOTE: This loads the entire file. May need optimization for huge datasets.
    file_path = os.path.join(
        data_dir,
        dataset,
        model_name,
        hint_type,
        f"completions_with_{n_questions}.json"
    )

    if not os.path.exists(file_path):
        logging.warning(f"Completions file not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            completions = json.load(f)

        # Check if completions is a list of dicts or dict of dicts
        if isinstance(completions, list):
             # Find the completion by iterating through the list
            for comp in completions:
                if comp.get('question_id') == question_id:
                    return comp.get('completion')
            logging.warning(f"Completion for question_id {question_id} not found in list format in {file_path}")
            return None
        elif isinstance(completions, dict):
             # Check if question_id (as string key) exists
             q_id_str = str(question_id)
             if q_id_str in completions:
                 # Assuming the value is the completion string directly or a dict containing it
                 if isinstance(completions[q_id_str], str):
                     return completions[q_id_str]
                 elif isinstance(completions[q_id_str], dict):
                     return completions[q_id_str].get('completion')
                 else:
                     logging.warning(f"Unexpected format for question_id {question_id} value in dict format in {file_path}")
                     return None
             else:
                 logging.warning(f"Completion for question_id {question_id} not found in dict format in {file_path}")
                 return None
        else:
            logging.error(f"Unexpected format in {file_path}. Expected list or dict.")
            return None

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None 