"""Model registry and evaluation conditions."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    provider: str
    api_model_id: str
    supports_search: bool = True
    base_url: str | None = None  # For Azure or custom endpoints
    api_key_env: str | None = None  # Env var name for API key


# Azure OpenAI endpoint
AZURE_OPENAI_ENDPOINT = "https://YOUR_AZURE_ENDPOINT/openai/v1/"

MODELS: dict[str, ModelConfig] = {
    # OpenAI
    "gpt-4o": ModelConfig("openai", "gpt-4o", True),
    "gpt-5": ModelConfig("openai", "gpt-5", True),
    "o3": ModelConfig("openai", "o3", True),
    "o4-mini": ModelConfig("openai", "o4-mini", True),
    # Azure OpenAI
    "gpt-5.4-mini": ModelConfig(
        "openai", "gpt-5.4-mini", True,
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
    "gpt-5.4-nano": ModelConfig(
        "openai", "gpt-5.4-nano", True,
        base_url=AZURE_OPENAI_ENDPOINT,
        api_key_env="AZURE_OPENAI_API_KEY",
    ),
    # Google
    "gemini-2.5-pro": ModelConfig("gemini", "gemini-2.5-pro", True),
    "gemini-2.5-flash": ModelConfig("gemini", "gemini-2.5-flash", True),
    "gemini-3-flash-preview": ModelConfig("gemini", "gemini-3-flash-preview", True),
    "gemini-3-pro-preview": ModelConfig("gemini", "gemini-3-pro-preview", True),
    "gemini-3.1-pro-preview": ModelConfig("gemini", "gemini-3.1-pro-preview", True),
    "gemini-3.1-flash-lite-preview": ModelConfig("gemini", "gemini-3.1-flash-lite-preview", True),
    # Local vLLM (OpenAI-compatible Chat Completions API)
    "qwen3.5-4b": ModelConfig(
        "vllm", "Qwen/Qwen3.5-4B", False,
        base_url="http://localhost:8000/v1",
    ),
    "qwen3-4b-thinking": ModelConfig(
        "vllm", "Qwen/Qwen3-4B-Thinking-2507", False,
        base_url="http://localhost:8000/v1",
    ),
    "qwen3-4b-thinking-tc": ModelConfig(
        "vllm", "Qwen/Qwen3-4B-Thinking-2507", False,
        base_url="http://localhost:8000/v1",
    ),
    "qwen3-30b-a3b-thinking": ModelConfig(
        "vllm", "Qwen/Qwen3-30B-A3B-Thinking-2507", False,
        base_url="http://localhost:8100/v1",
    ),
    "qwen3-30b-a3b-thinking-tc": ModelConfig(
        "vllm", "Qwen/Qwen3-30B-A3B-Thinking-2507", False,
        base_url="http://localhost:8100/v1",
    ),
    "qwen3-235b-a22b-thinking": ModelConfig(
        "vllm", "Qwen/Qwen3-235B-A22B-Thinking-2507", False,
        base_url="http://localhost:8200/v1",
    ),
    "qwen3-235b-a22b-thinking-tc": ModelConfig(
        "vllm", "Qwen/Qwen3-235B-A22B-Thinking-2507", False,
        base_url="http://localhost:8200/v1",
    ),
}

CONDITIONS = ["no_search", "with_search", "with_url_context", "with_video_tool"]

# Unified prompt template for all conditions
QUERY_TEMPLATE = """\
{question}

Your response should be in the following format:
Reasoning: {{your step-by-step reasoning for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}""".strip()

GRADER_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.""".strip()
