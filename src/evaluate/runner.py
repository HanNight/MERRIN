"""Async evaluation runner."""

import asyncio
import json
import re
import time
from pathlib import Path

from .config import MODELS
from .models import PROVIDERS
from .models.base import ModelResponse


def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def extract_answer(text: str) -> str | None:
    """Extract 'Exact Answer: ...' from model response (BrowseComp/HLE format)."""
    match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for [ANSWER] brackets
    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        return match.group(1).strip()
    return None


async def evaluate_question(
    model_client, question: dict, condition: str, semaphore: asyncio.Semaphore,
    prompt_template: str = "default",
    max_retries: int = 5,
) -> dict:
    """Evaluate a single question with a model. Retries on rate limit errors."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp: ModelResponse = await model_client.answer(
                    question["question"], condition=condition,
                    prompt_template=prompt_template,
                )
                extracted = extract_answer(resp.raw_response)
                return {
                    "question_id": question["id"],
                    "question": question["question"],
                    "gold_answer": question["answer"],
                    "model_response": resp.raw_response,
                    "extracted_answer": extracted,
                    "metadata": resp.metadata,
                    "error": None,
                }
            except Exception as e:
                err_str = str(e)
                # Retry on rate limit (429)
                if "429" in err_str and attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"    Rate limited on Q{question['id']}, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                # Retry on "too many tool calls" (Gemini Interactions API)
                if "too many tool calls" in err_str.lower() and attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"    Too many tool calls on Q{question['id']}, retrying ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(wait)
                    continue
                return {
                    "question_id": question["id"],
                    "question": question["question"],
                    "gold_answer": question["answer"],
                    "model_response": "",
                    "extracted_answer": None,
                    "metadata": {},
                    "error": err_str,
                }


def load_completed(output_path: Path) -> set[int]:
    """Load already-completed question IDs for resumability."""
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["question_id"])
    return completed


async def run_model_condition(
    model_name: str,
    condition: str,
    questions: list[dict],
    output_dir: Path,
    concurrency: int = 5,
    thinking_level: str | None = None,
    prompt_template: str = "default",
    use_interactions_api: bool = False,
    run_id: int | None = None,
):
    """Run one (model, condition) pair across all questions."""
    cfg = MODELS[model_name]

    # Skip search/url_context conditions for models that don't support them
    if condition != "no_search" and not cfg.supports_search:
        print(f"  Skipping {model_name} / {condition} (not supported)")
        return
    # with_url_context and with_video_tool are Gemini-only
    if condition in ("with_url_context", "with_video_tool") and cfg.provider != "gemini":
        print(f"  Skipping {model_name} / {condition} (Gemini-only feature)")
        return

    suffix = ""
    if thinking_level:
        suffix += f"_think-{thinking_level}"
    if prompt_template != "default":
        suffix += f"_prompt-{prompt_template}"
    if use_interactions_api:
        suffix += "_interactions"
    if run_id is not None:
        suffix += f"_run-{run_id}"
    output_path = output_dir / f"{model_name}_{condition}{suffix}.jsonl"
    completed = load_completed(output_path)
    remaining = [q for q in questions if q["id"] not in completed]

    if not remaining:
        print(f"  {model_name} / {condition}: all done ({len(completed)} completed)")
        return

    think_label = f" (thinking={thinking_level})" if thinking_level else ""
    print(
        f"  {model_name} / {condition}{think_label}: {len(remaining)} remaining "
        f"({len(completed)} already done)"
    )

    client_cls = PROVIDERS[cfg.provider]
    # Pass provider-specific kwargs
    if cfg.provider == "gemini" and (thinking_level or use_interactions_api):
        from .models.gemini import GeminiModel
        client = GeminiModel(
            cfg.api_model_id,
            thinking_level=thinking_level,
            use_interactions_api=use_interactions_api,
        )
    elif cfg.provider == "openai":
        import os
        from .models.openai import OpenAIModel
        oai_kwargs = {"model_id": cfg.api_model_id}
        if cfg.base_url:
            oai_kwargs["base_url"] = cfg.base_url
        if cfg.api_key_env:
            oai_kwargs["api_key"] = os.environ.get(cfg.api_key_env, "")
        if thinking_level:
            oai_kwargs["reasoning_effort"] = thinking_level
        client = OpenAIModel(**oai_kwargs)
    elif cfg.provider == "vllm":
        from .models.vllm import VLLMModel
        client = VLLMModel(cfg.api_model_id, base_url=cfg.base_url or "http://localhost:8000/v1")
    else:
        client = client_cls(cfg.api_model_id)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        evaluate_question(client, q, condition, sem, prompt_template=prompt_template)
        for q in remaining
    ]
    results = await asyncio.gather(*tasks)

    # Append results
    with open(output_path, "a") as f:
        for r in results:
            r["model"] = model_name
            r["condition"] = condition
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if r["error"])
    print(f"    Done: {len(results)} questions, {errors} errors")


async def run_evaluation(
    questions_path: str,
    output_dir: str,
    models: list[str],
    conditions: list[str],
    concurrency: int = 5,
    thinking_level: str | None = None,
    prompt_template: str = "default",
    use_interactions_api: bool = False,
    run_id: int | None = None,
):
    """Run full evaluation across models and conditions."""
    questions = load_questions(questions_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(questions)} questions")
    print(f"Models: {models}")
    print(f"Conditions: {conditions}")
    if thinking_level:
        print(f"Thinking level: {thinking_level}")
    if prompt_template != "default":
        print(f"Prompt template: {prompt_template}")
    if use_interactions_api:
        print("API: Interactions API")
    if run_id is not None:
        print(f"Run ID: {run_id}")
    print()

    for model_name in models:
        print(f"Model: {model_name}")
        for condition in conditions:
            await run_model_condition(
                model_name, condition, questions, out, concurrency,
                thinking_level=thinking_level,
                prompt_template=prompt_template,
                use_interactions_api=use_interactions_api,
                run_id=run_id,
            )
        print()
