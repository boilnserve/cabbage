from typing import List, Optional, Dict, Any, Tuple
import random
import re
from llm_eval.utils.configuration import ModelInputConfig

def format_options(options_list: List[str]) -> str:
    """Format a list of options as lettered choices (A., B., ...). Args: options_list: List of option strings. Returns: Formatted string with each option on a new line."""
    return "\n".join([f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options_list)])

def extract_options_and_answer_letter(doc: dict, config: ModelInputConfig) -> Tuple[Optional[List[str]], Optional[str]]:
    """Extracts the options and the correct answer letter from a document based on the config. Args: doc: The question document. config: ModelInputConfig specifying question type and difficulty. Returns: Tuple of (options list, correct answer letter) or (None, None) if not applicable."""
    if config.question_type != 'multiple_choice':
        return None, None

    all_options = doc.get('options')
    if not all_options:
        return None, None

    difficulty = config.difficulty
    options_for_difficulty = (
        all_options.get(difficulty) or 
        all_options.get('default') or 
        random.choice([v for v in all_options.values() if v]) if any(all_options.values()) else None
    )

    if not options_for_difficulty:
        raise ValueError(f"No options found for difficulty '{difficulty}' and no valid fallback available.")

    correct_answer = doc.get('answer')
    if correct_answer not in options_for_difficulty:
        return None, None

    try:
        correct_index = options_for_difficulty.index(correct_answer)
    except ValueError:
        return None, None
    answer_letter = chr(ord('A') + correct_index)
    return options_for_difficulty, answer_letter

def get_answer_letter(doc: Dict, config: ModelInputConfig) -> Optional[str]:
    """Returns the correct answer letter for a multiple-choice question. Args: doc: The question document. config: ModelInputConfig. Returns: The answer letter (A, B, ...) or None."""
    options, answer_letter = extract_options_and_answer_letter(doc, config)
    return answer_letter

def format_prompt(doc: dict, config: ModelInputConfig) -> str:
    """Formats a prompt for the model, including question, options, and pre/post prompts. Args: doc: The question document. config: ModelInputConfig. Returns: The formatted prompt string."""
    prompt_parts = [config.pre_prompt, f"Question: {doc['question']}"]
    options_list, answer_letter = extract_options_and_answer_letter(doc, config)
    
    if options_list:
        prompt_parts.append(f"\nOptions:\n{format_options(options_list)}")

    prompt_parts.append(config.post_prompt)
    return "\n".join(part for part in prompt_parts if part.strip())

def extract_evaluation_step_titles(text: str) -> List[str]:
    """Extract step titles from evaluation content. Args: text: The evaluation content string. Returns: List of step titles as strings."""
    match = re.search(r"Evaluation Steps:\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return []

    steps_text = match.group(1)
    return [f"{num}. {title}" for num, title in re.findall(r"(\d+)\.\s*(.*?):", steps_text)]
