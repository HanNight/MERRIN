from .openai import OpenAIModel
from .gemini import GeminiModel
from .vllm import VLLMModel

PROVIDERS = {
    "openai": OpenAIModel,
    "gemini": GeminiModel,
    "vllm": VLLMModel,
}
