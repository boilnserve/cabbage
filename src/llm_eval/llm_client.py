import os
import time
import asyncio
import openai
import tqdm.asyncio

from typing import Any, List, Dict
from loguru import logger as log

from llm_eval.request_generation import ExperimentInput
from llm_eval.utils.image_utils import encode_image
from llm_eval.structured_output.response_types import RESPONSE_MODEL_MAP

# ---------- Constants ----------
NUM_SECONDS_TO_SLEEP = 5
RETRIES = 3
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Client Class ----------
class LLMClient:
    def __init__( self, model_name: str, base_url: str = "", api_key: str = "None", timeout: int = 120) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = "You are a helpful assistant."
        self.client = openai.AsyncClient(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    # ---------- Payload Assembly ----------
    def _build_payload(self, prompt: str, images: List[str]) -> Dict:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in images]
        content.append({"type": "text", "text": prompt})

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": content}
            ]
        }

    @staticmethod
    def _update_generation_params(gen_kwargs: dict) -> None:
        gen_kwargs.setdefault("max_new_tokens", 1024)
        gen_kwargs["max_new_tokens"] = min(gen_kwargs["max_new_tokens"], 4096)
        gen_kwargs.setdefault("temperature", 0.0)
    
    # ---------- Generation ----------
    async def _generate_single(self, request: ExperimentInput, client: openai.AsyncClient) -> Dict:
        
        encoded_images = [encode_image(img) for img in request.visuals]
        payload = self._build_payload(request.input, encoded_images)
        self._update_generation_params(request.gen_kwargs)

        response_model = RESPONSE_MODEL_MAP.get(request.question_type)
        if not response_model:
            raise ValueError(f"Unsupported answer_field: {request.question_type}")

        payload.update({
            "max_tokens": request.gen_kwargs["max_new_tokens"],
            "temperature": request.gen_kwargs["temperature"],
        })
        last_error=None
        for attempt in range(RETRIES):
            try:
                response = await client.beta.chat.completions.parse(
                    **payload, response_format=response_model
                )
                parsed_response = response.choices[0].message.parsed
                
                if parsed_response is not None:
                    return {'reasoning': getattr(parsed_response, 'reasoning', None), 'model_answer': getattr(parsed_response, 'answer', None)}
                else:
                    return {'error': 'Parsed response is None'}
            except Exception as e:
                log.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < RETRIES - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP * (attempt + 1))
                else:
                    log.error(f"All retries failed: {e}")
                    last_error=e
        # Fail-safe fallback (should never be reached)
        return {'error' : f'{last_error}'}

    def generate_until(self, requests: List[ExperimentInput]) -> List[Dict]:
        async def _run_all():
            async with openai.AsyncClient(base_url=self.base_url,api_key=os.getenv(self.api_key)) as client:
                tasks = [self._generate_single(request, client) for request in requests]
                return await tqdm.asyncio.tqdm_asyncio.gather(*tasks, desc=f"[{self.model_name}] Generating")
        return asyncio.run(_run_all())
