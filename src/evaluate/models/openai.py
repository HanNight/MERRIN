"""OpenAI model client via Responses API. Supports both OpenAI and Azure OpenAI."""

import time
from openai import AsyncOpenAI

from .base import BaseModel, ModelResponse
from ..config import QUERY_TEMPLATE

# Valid reasoning effort levels for OpenAI models
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(model_id)
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = AsyncOpenAI(**kwargs)
        self.reasoning_effort = reasoning_effort

    async def answer(
        self, question: str, condition: str = "no_search",
        prompt_template: str = "default",
    ) -> ModelResponse:
        prompt = QUERY_TEMPLATE.format(question=question)
        kwargs = {
            "model": self.model_id,
            "input": prompt,
        }
        if condition in ("with_search", "with_url_context"):
            kwargs["tools"] = [{"type": "web_search", "search_context_size": "medium"}]

        # Reasoning config: effort + summary
        # Note: Azure o-series models don't support reasoning summaries
        reasoning_config = {}
        if self.reasoning_effort:
            reasoning_config["effort"] = self.reasoning_effort
        is_o_series = self.model_id.startswith("o")
        if not is_o_series:
            reasoning_config["summary"] = "auto"
        if reasoning_config:
            kwargs["reasoning"] = reasoning_config

        t0 = time.monotonic()
        response = await self.client.responses.create(**kwargs)
        latency = time.monotonic() - t0

        return ModelResponse(
            raw_response=response.output_text or "",
            metadata={
                "latency_s": round(latency, 2),
                "model_id": self.model_id,
                "condition": condition,
                "reasoning_effort": self.reasoning_effort,
                "full_response": response.model_dump(mode="json"),
            },
        )
