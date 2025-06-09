import os
import re
import time
import math
import json
import yaml
import openai
from typing import List, Dict, Optional, Callable, Tuple, Awaitable, Union
from loguru import logger
from evaluation_classes import EvaluationProcess, EvaluationProcessProcedural

# Constants
NUM_SECONDS_TO_SLEEP = 10
RETRIES = 5
TEMP = 5
SCORES = ['1', '2', '3', '4', '5']

# Get the absolute path to the 'docs' directory
docs_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'evaluation_prompts.yaml')


# Load prompts
with open(docs_path, 'r') as file:
    PROMPT_DICT: Dict[str, str] = yaml.safe_load(file)


# --- Utility Functions ---

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


def extract_evaluation_step_titles(text: str) -> List[str]:
    """Extract step titles from evaluation content."""
    match = re.search(r"Evaluation Steps:\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return []

    steps_text = match.group(1)
    return [f"{num}. {title}" for num, title in re.findall(r"(\d+)\.\s*(.*?):", steps_text)]


# --- Main Evaluation Class ---

class GEvaluator:
    def __init__(self, provider: Dict[str, str], evaluation_type: str) -> None:
        self.model_name = provider['model']
        self.client = openai.AsyncClient(
            base_url=provider['base_url'],
            api_key=os.getenv(provider['api_key'])
        )
        self.system_prompt = PROMPT_DICT.get("system_prompt", "")

        self.query_template: str
        if evaluation_type == 'knowledge':
            self.query_template = PROMPT_DICT.get("query_prompt_agronomic", "")
            self.evaluation_process = EvaluationProcess
        elif evaluation_type == 'procedural':
            self.query_template = PROMPT_DICT.get("query_prompt_procedural", "")
            self.evaluation_process = EvaluationProcessProcedural
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

        self._response_handler: Callable[[str], Awaitable[Tuple]]
        for key in ("gpt", "deepseek", "gemini"):
            if key in self.model_name:
                self._response_handler = getattr(self, f"_handle_{key}")
                break
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def prepare_prompt(self, doc: Dict[str, str], model_answer: str) -> Optional[str]:
        if model_answer in ["Error", "=Copy the answer="]:
            model_answer = doc.get("answer", "") if model_answer == "=Copy the answer=" else ""

        if not model_answer:
            return None

        try:
            return self.query_template.format(
                QUESTION='Question text: ' + doc.get("question", ""),
                EXPERT_ANSWER=doc.get("answer", ""),
                MODEL_ANSWER=model_answer
            )
        except Exception as e:
            logger.error(f"Failed to format prompt: {e}")
            return None

    async def _retry_with_backoff(self, func: Callable[..., Awaitable], *args, **kwargs) -> Tuple:
        for attempt in range(RETRIES):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < RETRIES - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP * (attempt + 1))
                else:
                    logger.error(f"All retries failed. Last error: {e}")
        return "",""
    async def _handle_deepseek(self, content: str) -> Tuple:
        async def call():
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{content}\nJSON SCHEMA:\n{self.evaluation_process.model_json_schema()}"}
                ],
                response_format={'type': 'json_object'},
                max_tokens=2048,
                temperature=0.4,
                logprobs=True,
                top_logprobs=10
            )
            choice = completion.choices[0]
            if not choice.message.content:
                raise ValueError("Missing message content.")
            response_obj = self.evaluation_process.model_validate(json.loads(choice.message.content))
            return response_obj, extract_geval_scores(completion)

        return await self._retry_with_backoff(call)

    async def _handle_gpt(self, content: str) -> Tuple:
        async def call():
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                response_format=self.evaluation_process,
                max_tokens=2048,
                temperature=0.4,
                logprobs=True,
                top_logprobs=10
            )
            parsed = completion.choices[0].message.parsed
            return parsed, extract_geval_scores(completion)

        return await self._retry_with_backoff(call)

    async def _handle_gemini(self, content: str) -> Tuple:
        async def call():
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}
                ],
                response_format=self.evaluation_process,
                max_tokens=2048,
                temperature=0.4,
                timeout=10
            )
            choice = completion.choices[0]
            if not choice.message.content:
                raise ValueError("Message content missing.")
            parsed = choice.message.parsed
            scores = {k: int(v) for k, v in json.loads(choice.message.content)['scores'].items()}
            return parsed, scores

        return await self._retry_with_backoff(call)

    async def get_response(self, content: str) -> Tuple:
        return await self._response_handler(content)

    async def evaluate(self, doc: Dict[str, str], model_output: str) -> Dict:
        if not model_output:
            logger.error("Empty model output received.")
            return {"scores": "invalid_format"}

        prompt = self.prepare_prompt(doc, model_output)
        if not prompt:
            logger.error("Failed to prepare prompt.")
            return {"scores": "invalid_format"}

        try:
            parsed_response, scores = await self.get_response(prompt)
            if not hasattr(parsed_response, "final_review"):
                raise AttributeError("Missing 'final_review' in response.")

            step_titles = extract_evaluation_step_titles(prompt)
            step_analysis = "\n\n".join(
                f"{title}\n {step.step_analysis}"
                for title, step in zip(step_titles, parsed_response.evaluation_steps)
            )

            return {
                "query": prompt,
                "steps": step_analysis,
                "review": parsed_response.final_review,
                "scores": scores,
                "complete_response": parsed_response.model_dump(mode="json")
            }

        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            return {"scores": "invalid_format"}
