import os
import time
import json
import openai
import tqdm.asyncio
import asyncio
from typing import List, Dict, Optional, Callable, Tuple, Awaitable, Any
from loguru import logger
from llm_eval.structured_output.evaluation_types import EvaluationProcess, EvaluationProcessProcedural
from llm_eval.utils.geval import extract_geval_scores
from llm_eval.utils.formatting import extract_evaluation_step_titles
from llm_eval.utils.configuration import EvaluatorConfig

# Constants
NUM_SECONDS_TO_SLEEP = 10
RETRIES = 5

class LLMJudge:
    """Judge class for evaluating model outputs using LLMs with different evaluation types and response handlers."""
    def __init__(self, evaluator: EvaluatorConfig, evaluation_type: Optional[str], prompts_dict: Dict) -> None:
        """Initialize the LLMJudge with evaluator config, evaluation type, and prompts. Args: evaluator: EvaluatorConfig object. evaluation_type: Type of evaluation ('knowledge' or 'procedural'). prompts_dict: Dictionary of prompt templates."""
        self.model_name = evaluator.model
        # self.client = openai.AsyncClient(
        #     base_url=evaluator.base_url,
        #     api_key=os.getenv(evaluator.api_key)
        # )
        self.evaluator = evaluator
        self.system_prompt = prompts_dict.get("system_prompt", "")

        self.query_template: str
        if evaluation_type == 'knowledge':
            self.query_template = prompts_dict.get("query_prompt_knowledge", "")
            self.evaluation_process = EvaluationProcess
        elif evaluation_type == 'procedural':
            self.query_template = prompts_dict.get("query_prompt_procedural", "")
            self.evaluation_process = EvaluationProcessProcedural
        else:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")

        self._response_handler: Callable[[str, Any], Awaitable[Tuple]]
        for key in ("gpt", "deepseek", "gemini"):
            if key in self.model_name:
                self._response_handler = getattr(self, f"_handle_{key}")
                break
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def prepare_prompt(self, doc: Dict[str, str], model_answer: str) -> Optional[str]:
        """Prepare the evaluation prompt for the LLM. Args: doc: The question document. model_answer: The model's answer. Returns: The formatted prompt string or None if invalid."""
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
        """Retry an async function with exponential backoff. Args: func: The async function to call. *args, **kwargs: Arguments to pass to the function. Returns: The result of the function or ("", "") if all retries fail."""
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
    
    async def _handle_deepseek(self, content: str, client) -> Tuple:
        """Handle evaluation using the DeepSeek model. Args: content: The prompt content. client: The async client. Returns: Tuple of (parsed response, scores)."""
        async def call():
            completion = await client.chat.completions.create(
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

    async def _handle_gpt(self, content: str, client) -> Tuple:
        """Handle evaluation using the GPT model. Args: content: The prompt content. client: The async client. Returns: Tuple of (parsed response, scores)."""
        async def call():
            completion = await client.beta.chat.completions.parse(
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

    async def _handle_gemini(self, content: str, client) -> Tuple:
        """Handle evaluation using the Gemini model. Args: content: The prompt content. client: The async client. Returns: Tuple of (parsed response, scores)."""
        async def call():
            completion = await client.beta.chat.completions.parse(
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

    async def get_response(self, content: str, client) -> Tuple:
        """Get the model's response using the appropriate handler. Args: content: The prompt content. client: The async client. Returns: Tuple of (parsed response, scores)."""
        return await self._response_handler(content, client)

    async def evaluate(self, doc: Dict[str, str], model_output: str, client) -> Dict:
        """Evaluate a single model output using the LLM. Args: doc: The question document. model_output: The model's answer. client: The async client. Returns: Dictionary with evaluation results and scores."""
        if not model_output:
            logger.error("Empty model output received.")
            return {"scores": "invalid_format"}

        prompt = self.prepare_prompt(doc, model_output)
        if not prompt:
            logger.error("Failed to prepare prompt.")
            return {"scores": "invalid_format"}

        try:
            parsed_response, scores = await self.get_response(prompt, client)
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
    
    def evaluate_until(self, examples: List[Dict]) -> List[Dict]:
        """Evaluate a list of examples until completion. Args: examples: List of example dictionaries. Returns: List of evaluation results."""
        async def _run_all():
            async with openai.AsyncClient(base_url=self.evaluator.base_url,api_key=os.getenv(self.evaluator.api_key_env_var)) as client:
                tasks = [self.evaluate(ex['original_doc'], ex['inference_result'].get('model_answer'), client) for ex in examples]
                return await tqdm.asyncio.tqdm_asyncio.gather(*tasks, desc=f"[{self.model_name}] Generating")
        return asyncio.run(_run_all())
