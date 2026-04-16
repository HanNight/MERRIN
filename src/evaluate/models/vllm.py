"""Local vLLM/sglang models via OpenAI-compatible Chat Completions API."""

import time
from openai import AsyncOpenAI

from .base import BaseModel, ModelResponse
from ..config import QUERY_TEMPLATE


class VLLMModel(BaseModel):
    def __init__(self, model_id: str, base_url: str = "http://localhost:8000/v1",
                 max_tokens: int = 32768, temperature: float = 0.6,
                 top_p: float = 0.95, top_k: int = 20, min_p: float = 0.0):
        super().__init__(model_id)
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

    async def answer(
        self, question: str, condition: str = "no_search",
        prompt_template: str = "default",
    ) -> ModelResponse:
        prompt = QUERY_TEMPLATE.format(question=question)

        t0 = time.monotonic()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body={
                "top_k": self.top_k,
                "min_p": self.min_p,
            },
        )
        latency = time.monotonic() - t0

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        return ModelResponse(
            raw_response=content,
            metadata={
                "latency_s": round(latency, 2),
                "model_id": self.model_id,
                "condition": condition,
                "finish_reason": response.choices[0].finish_reason,
                "thinking_length": len(reasoning),
                "full_response": response.model_dump(mode="json"),
            },
        )
