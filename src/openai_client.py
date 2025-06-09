import base64
import os
import time
import asyncio
import itertools
from copy import deepcopy
from io import BytesIO
from typing import Any, List, Literal, Optional, Tuple, Type, Union, Dict
from generate_requests import ExperimentRequest

import openai
import tqdm.asyncio
from PIL import Image
from pydantic import BaseModel, Field, create_model
from loguru import logger as log

#----------Response Classes------------

from pydantic import BaseModel
from typing import Literal


class ReasoningAnswer(BaseModel):
    reasoning: str
    answer: str


class ReasoningAnswerLetter(BaseModel):
    reasoning: str
    answer: Literal["A", "B", "C", "D", "E"]


class ReasoningLongAnswer(BaseModel):
    reasoning: str
    answer: str

ParsedResponse = Union[ReasoningAnswer, ReasoningAnswerLetter, ReasoningLongAnswer]

RESPONSE_MODEL_MAP = {
        "open_ended_short": ReasoningAnswer,
        "multiple_choice": ReasoningAnswerLetter,
        "open_ended_long": ReasoningLongAnswer,
    }

# ---------- Constants ----------
NUM_SECONDS_TO_SLEEP = 5
RETRIES = 3
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------- Client Class ----------
class OpenAIClient:
    def __init__(
        self,
        model_name: str,
        base_url: str = "",
        api_key: str = "None",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = "You are a helpful assistant."
        self.client = openai.AsyncClient(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    # ---------- Image Processing ----------
    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ---------- Payload Assembly ----------
    def _build_payload(self, prompt: str, images: List[str]) -> dict:
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
    async def _generate_single(
        self,
        request: ExperimentRequest,
        client: Any
    ) -> Dict:
        encoded_images = [self._encode_image(img) for img in request.visuals]
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
                return {'reasoning' : parsed_response.reasoning, 'model_answer' : parsed_response.answer}
            except Exception as e:
                log.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < RETRIES - 1:
                    time.sleep(NUM_SECONDS_TO_SLEEP * (attempt + 1))
                else:
                    log.error(f"All retries failed: {e}")
                    last_error=e
        # Fail-safe fallback (should never be reached)
        return {'error' : f'{last_error}'}

    def generate_until(self, requests: List[ExperimentRequest]) -> List[Dict]:
        async def _run_all():
            tasks = [self._generate_single(request, self.client) for request in requests]
            return await tqdm.asyncio.tqdm_asyncio.gather(*tasks, desc=f"[{self.model_name}] Generating")
        return asyncio.run(_run_all())
