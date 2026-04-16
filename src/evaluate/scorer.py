"""Scoring: exact match + LLM-as-judge (BrowseComp / HLE style)."""

import asyncio
import json
import re
import string
from abc import ABC, abstractmethod

from .config import GRADER_TEMPLATE


# --- Grader backends ---

class BaseGrader(ABC):
    @abstractmethod
    async def grade(self, prompt: str) -> str:
        """Send grading prompt, return raw text response."""
        ...


class OpenAIGrader(BaseGrader):
    def __init__(self, model: str = "gpt-4o"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model

    async def grade(self, prompt: str) -> str:
        resp = await self.client.responses.create(model=self.model, input=prompt)
        return resp.output_text


class GeminiGrader(BaseGrader):
    def __init__(self, model: str = "gemini-2.5-flash"):
        from google import genai
        self.client = genai.Client()
        self.model = model

    async def grade(self, prompt: str) -> str:
        resp = await self.client.aio.models.generate_content(
            model=self.model, contents=prompt,
        )
        return resp.text or ""


GRADERS = {
    "openai": OpenAIGrader,
    "gemini": GeminiGrader,
}


def get_grader(judge_model: str) -> BaseGrader:
    """Create a grader from a string like 'openai/gpt-4o' or 'gemini/gemini-2.5-flash'.

    Shortcuts: 'gpt-4o' -> OpenAI, 'gemini-*' -> Gemini.
    """
    if "/" in judge_model:
        provider, model_id = judge_model.split("/", 1)
        return GRADERS[provider](model_id)
    if judge_model.startswith("gemini"):
        return GeminiGrader(judge_model)
    return OpenAIGrader(judge_model)


# --- Normalization & exact match ---

def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t


_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def exact_match(predicted: str, gold: str) -> bool:
    p, g = normalize(predicted), normalize(gold)
    if p == g:
        return True
    pp = _NUM_WORDS.get(p, p)
    gg = _NUM_WORDS.get(g, g)
    return pp == gg


def extract_exact_answer(text: str) -> str | None:
    """Extract 'Exact Answer: ...' from model response."""
    match = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# --- LLM judge ---

def parse_grading(grading_text: str) -> dict:
    """Parse structured fields from a grader response."""
    correct_match = re.search(r"correct:\s*(yes|no)", grading_text, re.IGNORECASE)
    is_correct = correct_match.group(1).lower() == "yes" if correct_match else False

    answer_match = re.search(
        r"extracted_final_answer:\s*(.+?)(?:\n|$)", grading_text, re.IGNORECASE
    )
    extracted = answer_match.group(1).strip() if answer_match else None

    reasoning_match = re.search(
        r"reasoning:\s*(.+?)(?:\ncorrect:|\nconfidence:|\Z)",
        grading_text, re.IGNORECASE | re.DOTALL,
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    confidence_match = re.search(r"confidence:\s*(\d+)", grading_text)
    confidence = int(confidence_match.group(1)) if confidence_match else None

    return {
        "correct": is_correct,
        "extracted_answer": extracted,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw_grading": grading_text,
    }


async def llm_judge(
    question: str,
    gold: str,
    response: str,
    grader: BaseGrader,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Grade a response using the BrowseComp/HLE grader template."""
    async with semaphore:
        prompt = GRADER_TEMPLATE.format(
            question=question, correct_answer=gold, response=response,
        )
        try:
            grading_text = await grader.grade(prompt)
            return parse_grading(grading_text)
        except Exception as e:
            return {
                "correct": False, "extracted_answer": None,
                "reasoning": f"Error: {e}", "confidence": None,
                "raw_grading": "",
            }


# --- Score file ---

async def score_file(
    results_path: str,
    use_judge: bool = True,
    judge_model: str = "gpt-4o",
) -> dict:
    """Score a results JSONL file."""
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        return {"error": "No results found"}

    # Extract answers and compute exact match
    for r in results:
        extracted = extract_exact_answer(r.get("model_response", ""))
        r["extracted_answer"] = extracted or r.get("extracted_answer")
        pred = r.get("extracted_answer") or ""
        gold = r.get("gold_answer") or ""
        r["exact_match"] = exact_match(pred, gold) if pred else False

    # LLM judge scoring
    if use_judge:
        grader = get_grader(judge_model)
        sem = asyncio.Semaphore(5)
        tasks = []
        for r in results:
            if not r.get("error"):
                tasks.append(
                    llm_judge(
                        r["question"], r["gold_answer"],
                        r["model_response"], grader, sem,
                    )
                )
            else:
                async def _default():
                    return {
                        "correct": False, "extracted_answer": None,
                        "reasoning": "No response", "confidence": None,
                        "raw_grading": "",
                    }
                tasks.append(_default())

        judgments = await asyncio.gather(*tasks)
        for r, j in zip(results, judgments):
            r["judge_model"] = judge_model
            r["judge_correct"] = j.get("correct", False)
            r["judge_extracted"] = j.get("extracted_answer")
            r["judge_reasoning"] = j.get("reasoning", "")
            r["judge_confidence"] = j.get("confidence")

    # Compute metrics
    metrics = _compute_metrics(results, use_judge)

    # Save scored results
    scored_path = results_path.replace(".jsonl", "_scored.jsonl")
    with open(scored_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Scored results saved to {scored_path}")
    return metrics


def _compute_metrics(results: list[dict], use_judge: bool) -> dict:
    total = len(results)
    em_correct = sum(1 for r in results if r.get("exact_match"))
    errors = sum(1 for r in results if r.get("error"))

    metrics = {
        "total": total,
        "exact_match_accuracy": round(em_correct / total, 4) if total else 0,
        "error_count": errors,
    }

    if use_judge:
        judge_correct = sum(1 for r in results if r.get("judge_correct"))
        metrics["judge_accuracy"] = round(judge_correct / total, 4) if total else 0

    return metrics
