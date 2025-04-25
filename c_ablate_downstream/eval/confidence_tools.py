"""
Utilities for extracting answer–token ids and computing option
probabilities from a logits tensor.
"""
from typing import Dict, List
import torch

__all__ = ["ANSWER_TOKEN_IDS", "option_probs"]

# single ASCII-letter token ids – works for all Llama/Qwen style BPEs
def _letter_id(tok, letter: str) -> int:
    # prefer bare letter, fall back to leading space + letter
    tid = tok(letter)["input_ids"][-1]
    if tok.decode([tid]).strip() == letter:
        return tid
    return tok(" " + letter)["input_ids"][-1]

def ANSWER_TOKEN_IDS(tokenizer) -> Dict[str, int]:
    return {ltr: _letter_id(tokenizer, ltr) for ltr in ["A", "B", "C", "D"]}

def option_probs(next_token_logits: torch.Tensor,
                 answer_token_ids: Dict[str, int]) -> Dict[str, float]:
    """Return softmax probabilities for A/B/C/D."""
    probs = torch.softmax(next_token_logits, dim=-1)
    return {ltr: probs[tid].item() for ltr, tid in answer_token_ids.items()}
