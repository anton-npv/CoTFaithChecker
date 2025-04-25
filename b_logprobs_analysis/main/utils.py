"""Utility functions for the logprobs experiment."""

import logging

INTERVENTION_PROMPTS = {
    "dots": "... The final answer is [",
    "dots_eot": "...\n</think>\n\nThe final answer is [",
    # Add more types here if needed
}

def get_intervention_prompt(intervention_type: str) -> str:
    """Constructs the intervention prompt string based on the type.

    Args:
        intervention_type: The key for the intervention type (e.g., 'dots', 'eos_then_dots').

    Returns:
        The intervention prompt string, or an empty string if type is invalid.
    """
    if intervention_type in INTERVENTION_PROMPTS:
        return INTERVENTION_PROMPTS[intervention_type]
    else:
        logging.warning(f"Unknown intervention type: {intervention_type}. Returning empty string.")
        return "" 