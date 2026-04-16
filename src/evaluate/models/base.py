"""Abstract base class for model clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    raw_response: str
    extracted_answer: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseModel(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    async def answer(
        self, question: str, condition: str = "no_search",
        prompt_template: str = "default",
    ) -> ModelResponse:
        """Answer a question under a given condition.

        Conditions: "no_search", "with_search", "with_url_context", "with_video_tool"
        Prompt templates: "default" (Option C), "with_tools" (Option A with citations)
        """
        ...
