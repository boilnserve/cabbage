import re
import math
import json
from typing import List, Dict, Optional
from loguru import logger

TEMP = 5
SCORES = ['1', '2', '3', '4', '5']

def find_score_token_index(target_word: str, tokens: List) -> Optional[int]:
    """Finds the index of the score token in a token sequence."""
    reconstructed = ""
    token_map = []

    for i, token in enumerate(tokens):
        token_map.append((i, len(reconstructed)))
        reconstructed += token.token

    match = re.search(target_word, reconstructed)
    if match:
        end = match.end()
        for idx, (token_index, start) in enumerate(token_map):
            if start >= end and tokens[token_index].token.isdigit():
                return token_index

    logger.warning(f"'{target_word}' not found in token stream.")
    return None


def compute_geval_score(top_logprobs, valid_scores: List[str] = SCORES, temp: float = TEMP) -> float:
    """Computes weighted score from token logprobs."""
    probs = {
        tok.token: math.exp(tok.logprob / temp)
        for tok in top_logprobs if tok.token in valid_scores
    }

    total_prob = sum(probs.values())
    normalized = {k: v / total_prob for k, v in probs.items()}

    return sum(int(k) * v for k, v in normalized.items())


def extract_geval_scores(completion) -> Dict[str, float]:
    """Extract G-Eval scores from model completion."""
    raw_scores = json.loads(completion.choices[0].message.content)['scores']
    score_names = list(raw_scores.keys())

    token_indices = [
        find_score_token_index(name, completion.choices[0].logprobs.content)
        for name in score_names
    ]

    if any(idx is None for idx in token_indices):
        raise ValueError("Failed to locate all score tokens in completion.")

    return {
        name: compute_geval_score(completion.choices[0].logprobs.content[idx].top_logprobs)
        for name, idx in zip(score_names, token_indices)
    }
