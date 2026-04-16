"""Search agent using smolagents framework.

Provides a unified agent interface that works with different LLM backends
(OpenAI, Gemini, Azure, open-source) using the same tools.
"""

import json
import time
import asyncio
from pathlib import Path

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    LiteLLMModel,
    AzureOpenAIModel,
    GoogleSearchTool,
    tool,
)

from .config import MODELS, QUERY_TEMPLATE
from .runner import extract_answer


# ── Custom Tools ──────────────────────────────────────────────────────────────

# Default Gemini model for tool backends (configurable via set_tool_model)
_TOOL_MODEL = "gemini-3-flash-preview"


def set_tool_model(model_id: str):
    """Set the Gemini model used by visit_webpage and watch_video tools."""
    global _TOOL_MODEL
    _TOOL_MODEL = model_id


_VISIT_WEBPAGE_PROMPT = """\
Extract all content from the following webpage that is relevant to the user's goal. \
Examine all content on the page including text, tables, lists, images, charts, and diagrams. \
Never miss any important information.

Webpage URL: {url}
Web Search Query: {query}

Response in following format:
1. **Evidence**: Summarize all content (text, tables, images, charts, etc.) relevant to the query into concise points.
2. **Relevant Links** ([text](url)): Any links on the page that may contain additional useful information.
3. **Summary**: A concise synthesis of the findings, prioritizing clarity and relevance to the query.

If no relevant information exists, just output "No relevant information."
""".strip()


@tool
def visit_webpage(url: str, query: str) -> str:
    """Visit a webpage and extract information relevant to a specific web search query. This tool reads the full content of a webpage including text, images, charts, tables, and other visual elements, then extracts information relevant to the given query.

    Args:
        url: The URL of the webpage to visit.
        query: What specific information to look for on the page.
    """
    import os
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        prompt = _VISIT_WEBPAGE_PROMPT.format(url=url, query=query)
        response = client.models.generate_content(
            model=_TOOL_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(url_context=types.UrlContext())],
            ),
        )
        text = response.text or ""
        return text if text else "Could not extract content from the page."
    except Exception as e:
        return f"Error visiting {url}: {e}"


@tool
def search_images(query: str) -> str:
    """Search for images on the web using Google Image Search via Serper API.
    Returns image URLs, titles, and source pages.

    Args:
        query: The search query for finding images.
    """
    import os
    import requests

    try:
        resp = requests.post(
            "https://google.serper.dev/images",
            headers={
                "X-API-KEY": os.environ.get("SERPER_API_KEY", ""),
                "Content-Type": "application/json",
            },
            json={"q": query, "num": 8},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        output = []
        for r in data.get("images", [])[:8]:
            title = r.get("title", "No title")
            image_url = r.get("imageUrl", "")
            source = r.get("link", "")
            output.append(f"- {title}\n  Image: {image_url}\n  Source: {source}")
        return "\n".join(output) if output else "No images found."
    except Exception as e:
        return f"Error searching images: {e}"


_VIDEO_SEARCH_NUM = 3


def set_video_search_num(n: int):
    """Set the default number of videos returned by search_video."""
    global _VIDEO_SEARCH_NUM
    _VIDEO_SEARCH_NUM = n


# ── Custom web_search tool (subclass of smolagents GoogleSearchTool) ──────────

class CustomGoogleSearchTool(GoogleSearchTool):
    """GoogleSearchTool variant with a configurable number of results.

    smolagents's default sends no `num` parameter to Serper (10 results).
    This subclass lets us pass `num_results` via the constructor and include
    it in the Serper request body.
    """

    name = "web_search_custom"
    description = (
        "Performs a google web search for your query then returns a string of the "
        "top search results. Configurable number of results (default 10)."
    )

    def __init__(self, provider: str = "serper", num_results: int = 10):
        super().__init__(provider=provider)
        self.num_results = num_results

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import math
        import requests

        organic_results: list = []
        # Serper caps `num` at 10 per request — paginate via `page` to get more.
        pages_needed = max(1, math.ceil(self.num_results / 10))

        if self.provider == "serpapi":
            for page_num in range(1, pages_needed + 1):
                params = {
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                    "google_domain": "google.com",
                    "num": 10,
                    "start": (page_num - 1) * 10,
                }
                if filter_year is not None:
                    params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
                response = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
                if response.status_code != 200:
                    raise ValueError(response.json())
                page_data = response.json()
                page_organic = page_data.get(self.organic_key, [])
                if not page_organic:
                    break
                organic_results.extend(page_organic)
                if len(organic_results) >= self.num_results:
                    break
        else:
            for page_num in range(1, pages_needed + 1):
                payload = {"q": query, "page": page_num}
                if filter_year is not None:
                    payload["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )
                if response.status_code != 200:
                    raise ValueError(response.json())
                page_data = response.json()
                page_organic = page_data.get(self.organic_key, [])
                if not page_organic:
                    break
                organic_results.extend(page_organic)
                if len(organic_results) >= self.num_results:
                    break

        if not organic_results:
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")

        web_snippets = []
        for idx, page in enumerate(organic_results[: self.num_results]):
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
            web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


_WEB_SEARCH_NUM = 10


def set_web_search_num(n: int):
    """Set the default number of results returned by web_search."""
    global _WEB_SEARCH_NUM
    _WEB_SEARCH_NUM = n


@tool
def search_video(query: str) -> str:
    """Performs a google video search for your query then returns a string of top search results with video titles, URLs, and other information. Use this when you need to find videos to address your query.

    Args:
        query: The search query for finding videos.
    """
    import os
    import requests

    headers = {
        "X-API-KEY": os.environ.get("SERPER_API_KEY", ""),
        "Content-Type": "application/json",
    }
    try:
        output = []
        # Serper caps results per page at ~10; paginate via `page` to get more.
        # YouTube filter means we may discard many — keep paging until target met
        # or pages exhausted (cap at 5 pages to avoid runaway cost).
        max_pages = 5
        for page_num in range(1, max_pages + 1):
            resp = requests.post(
                "https://google.serper.dev/videos",
                headers=headers,
                json={"q": query, "page": page_num},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            videos = data.get("videos", [])
            if not videos:
                break
            for r in videos:
                link = r.get("link", "")
                if "youtube.com" not in link and "youtu.be" not in link:
                    continue
                title = r.get("title", "No title")
                duration = r.get("duration", "")
                channel = r.get("channel", "")
                date = r.get("date", "")
                snippet = r.get("snippet", "")
                entry = f"- {title}\n  URL: {link}\n  Channel: {channel} | Duration: {duration} | Date: {date}"
                if snippet:
                    entry += f"\n  Description: {snippet}"
                output.append(entry)
                if len(output) >= _VIDEO_SEARCH_NUM:
                    break
            if len(output) >= _VIDEO_SEARCH_NUM:
                break
        return "\n".join(output) if output else "No videos found."
    except Exception as e:
        return f"Error searching videos: {e}"


_WATCH_VIDEO_PROMPT = """\
Watch this video carefully and answer the following query based on the video.

Question: {query}

Response in following format:
1. **Evidence**: Summarize all content (visual and audio) relevant to the query into concise points.
2. **Summary**: A concise synthesis of the findings that address the query, prioritizing clarity and relevance to the query.
""".strip()


@tool
def watch_video(video_url: str, query: str) -> str:
    """Watch a video and extract information relevant to the query based on the video content (both visual and audio).

    Args:
        video_url: The video URL (e.g., https://www.youtube.com/watch?v=...).
        query: The query need to be answered based on the video.
    """
    import os
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        prompt = _WATCH_VIDEO_PROMPT.format(query=query)
        response = client.models.generate_content(
            model=_TOOL_MODEL,
            contents=types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=video_url)),
                    types.Part(text=prompt),
                ]
            ),
        )
        return response.text or "This video cannot answer the query."
    except Exception as e:
        return f"Error processing video: {e}"


# ── Model Factory ─────────────────────────────────────────────────────────────

def create_model(model_name: str, **kwargs):
    """Create a smolagents model from our config."""
    cfg = MODELS.get(model_name)
    if not cfg:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    if cfg.provider == "gemini":
        return LiteLLMModel(
            model_id=f"gemini/{cfg.api_model_id}",
            **kwargs,
        )
    elif cfg.provider == "openai":
        if cfg.base_url:
            # Azure OpenAI
            import os
            return AzureOpenAIModel(
                model_id=cfg.api_model_id,
                azure_endpoint=cfg.base_url.replace("/openai/v1/", "/"),
                api_key=os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else None,
                api_version="2024-12-01-preview",
                **kwargs,
            )
        else:
            return LiteLLMModel(
                model_id=cfg.api_model_id,
                **kwargs,
            )
    elif cfg.provider == "vllm":
        # vLLM with OpenAI-compatible API via LiteLLM
        return LiteLLMModel(
            model_id=f"openai/{cfg.api_model_id}",
            api_base=cfg.base_url,
            api_key="not-needed",
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


# ── Agent Factory ─────────────────────────────────────────────────────────────

def create_agent(
    model_name: str,
    agent_type: str = "tool_calling",
    tools: list[str] | None = None,
    max_steps: int = 20,
    planning_interval: int | None = 4,
    verbosity_level: int = 2,
    **model_kwargs,
):
    """Create a smolagents agent with specified tools.

    Args:
        model_name: Key from MODELS config.
        agent_type: "code" for CodeAgent, "tool_calling" for ToolCallingAgent.
        tools: List of tool names to include. Options:
            "web_search", "visit_webpage", "search_images", "search_video", "watch_video"
            Default: ["web_search"]
        max_steps: Maximum agent steps.
        planning_interval: Re-plan every N steps. None to disable.
        verbosity_level: 0=silent, 1=normal, 2=verbose.
    """
    model = create_model(model_name, **model_kwargs)

    # Build tool list
    tool_map = {
        "web_search": GoogleSearchTool(provider="serper"),
        "web_search_custom": CustomGoogleSearchTool(provider="serper", num_results=_WEB_SEARCH_NUM),
        "visit_webpage": visit_webpage,
        "search_images": search_images,
        "search_video": search_video,
        "watch_video": watch_video,
    }
    tool_names = tools or ["web_search"]
    agent_tools = [tool_map[t] for t in tool_names if t in tool_map]

    AgentClass = CodeAgent if agent_type == "code" else ToolCallingAgent

    agent = AgentClass(
        tools=agent_tools,
        model=model,
        max_steps=max_steps,
        planning_interval=planning_interval,
        verbosity_level=verbosity_level,
    )
    return agent


# ── Agent Log Serialization ───────────────────────────────────────────────────

def _save_agent_logs(agent, log_path: Path, question: dict, answer_text: str):
    """Save full agent logs to a JSON file for detailed analysis."""
    from smolagents.memory import ActionStep, PlanningStep, TaskStep

    steps = []
    for step in agent.memory.steps:
        step_data = {"type": type(step).__name__}

        if isinstance(step, ActionStep):
            step_data.update({
                "step_number": step.step_number,
                "model_output": step.model_output,
                "tool_calls": str(step.tool_calls) if step.tool_calls else None,
                "observations": step.observations,
                "error": str(step.error) if step.error else None,
                "action_output": str(step.action_output) if step.action_output else None,
                "is_final_answer": step.is_final_answer,
                "duration": str(step.timing) if step.timing else None,
                "token_usage": str(step.token_usage) if step.token_usage else None,
            })
        elif isinstance(step, PlanningStep):
            step_data.update({
                "plan": step.plan,
                "duration": str(step.timing) if step.timing else None,
            })
        elif isinstance(step, TaskStep):
            step_data["task"] = step.task if hasattr(step, "task") else str(step)
        else:
            step_data["raw"] = str(step)[:2000]

        steps.append(step_data)

    full_log = {
        "question_id": question["id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "agent_answer": answer_text,
        "steps": steps,
    }

    with open(log_path, "w") as f:
        json.dump(full_log, f, ensure_ascii=False, indent=2, default=str)


# ── Evaluation Runner ─────────────────────────────────────────────────────────

def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def _run_single_question(
    q: dict,
    model_name: str,
    agent_type: str,
    tools: list[str] | None,
    max_steps: int,
    planning_interval: int | None,
    verbosity_level: int,
    out: Path,
    logs_dir: Path,
    **model_kwargs,
) -> dict:
    """Run a single question with its own agent instance."""
    # Each thread gets its own agent to avoid shared state
    agent = create_agent(
        model_name, agent_type=agent_type, tools=tools,
        max_steps=max_steps, planning_interval=planning_interval,
        verbosity_level=verbosity_level, **model_kwargs,
    )

    prompt = QUERY_TEMPLATE.format(question=q["question"])

    t0 = time.monotonic()
    try:
        result = agent.run(prompt)
        latency = time.monotonic() - t0

        answer_text = str(result) if result else ""

        # Save full agent logs (non-fatal)
        log_path = logs_dir / f"Q{q['id']}.json"
        try:
            _save_agent_logs(agent, log_path, q, answer_text)
        except Exception as log_err:
            print(f"    Warning: failed to save logs for Q{q['id']}: {log_err}")
            log_path = None

        record = {
            "question_id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "model_response": answer_text,
            "extracted_answer": extract_answer(answer_text),
            "metadata": {
                "latency_s": round(latency, 2),
                "model": model_name,
                "agent_type": agent_type,
                "tools": tools or ["web_search"],
                "max_steps": max_steps,
                "num_steps": len(agent.memory.steps) if hasattr(agent, "memory") else None,
                "log_file": str(log_path) if log_path else None,
            },
            "error": None,
        }
    except Exception as e:
        latency = time.monotonic() - t0
        answer_text = ""
        record = {
            "question_id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "model_response": "",
            "extracted_answer": None,
            "metadata": {
                "latency_s": round(latency, 2),
                "model": model_name,
                "agent_type": agent_type,
            },
            "error": str(e),
        }

    status = "+" if not record["error"] else "x"
    print(f"  Q{q['id']} {status} ({record['metadata']['latency_s']}s): {(answer_text or record.get('error') or '')[:80]}")
    return record


def run_agent_evaluation(
    questions_path: str,
    output_dir: str,
    model_name: str,
    agent_type: str = "tool_calling",
    tools: list[str] | None = None,
    max_steps: int = 20,
    planning_interval: int | None = 4,
    verbosity_level: int = 2,
    concurrency: int = 1,
    thinking_level: str | None = None,
    run_id: int | None = None,
    **model_kwargs,
):
    """Run agent evaluation on questions.

    Args:
        questions_path: Path to questions JSONL.
        output_dir: Directory for result files.
        model_name: Model key from config.
        agent_type: "code" or "tool_calling".
        tools: List of tool names.
        max_steps: Max agent steps.
        planning_interval: Re-plan every N steps.
        verbosity_level: 0=silent, 1=normal, 2=verbose.
        concurrency: Number of questions to run in parallel.
        thinking_level: Reasoning effort for OpenAI models (e.g., "high").
        run_id: Run ID for multi-run experiments (e.g., 1, 2, 3).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Pass thinking level as model kwarg
    # LiteLLM maps reasoning_effort to thinking level for both OpenAI and Gemini
    if thinking_level:
        model_kwargs["reasoning_effort"] = thinking_level

    questions = load_questions(questions_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tool_str = "+".join(tools or ["web_search"])
    suffix = f"_think-{thinking_level}" if thinking_level else ""
    if run_id is not None:
        suffix += f"_run-{run_id}"
    output_path = out / f"agent_{model_name}_{agent_type}_{tool_str}{suffix}.jsonl"

    # Per-(model, thinking, run) subfolder for agent logs to avoid overwrites
    logs_dir = out / "agent_logs" / f"{model_name}{suffix}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load completed
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["question_id"])

    remaining = [q for q in questions if q["id"] not in completed]
    if not remaining:
        print(f"All {len(completed)} questions done.")
        return

    print(f"Agent: {model_name} / {agent_type} / tools={tool_str}")
    print(f"  {len(remaining)} remaining ({len(completed)} done), concurrency={concurrency}")

    # Thread-safe file writing
    write_lock = threading.Lock()

    def run_and_save(q):
        record = _run_single_question(
            q, model_name, agent_type, tools,
            max_steps, planning_interval, verbosity_level,
            out, logs_dir, **model_kwargs,
        )
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        return record

    if concurrency <= 1:
        for q in remaining:
            run_and_save(q)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_and_save, q): q for q in remaining}
            for future in as_completed(futures):
                future.result()  # raise any unhandled exceptions


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run smolagents search agent evaluation")
    parser.add_argument("--questions", default="data/questions/questions.jsonl")
    parser.add_argument("--output-dir", default="experiments/results/agent")
    parser.add_argument("--model", required=True, help=f"Model: {list(MODELS.keys())}")
    parser.add_argument("--agent-type", default="tool_calling", choices=["code", "tool_calling"])
    parser.add_argument(
        "--tools", nargs="+", default=["web_search"],
        help="Tools: web_search, web_search_custom, visit_webpage, search_images, search_video, watch_video",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1, help="Number of questions to run in parallel")
    parser.add_argument("--planning-interval", type=int, default=4)
    parser.add_argument("--verbosity-level", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument(
        "--tool-model", default="gemini-3-flash-preview",
        help="Gemini model used by visit_webpage and watch_video tools",
    )
    parser.add_argument(
        "--video-num", type=int, default=3,
        help="Number of videos returned by search_video (default: 3)",
    )
    parser.add_argument(
        "--web-search-num", type=int, default=10,
        help="Number of results returned by web_search_custom (default: 10)",
    )
    parser.add_argument(
        "--thinking-level", default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI models",
    )
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Run ID for multi-run experiments (e.g., 1, 2, 3). Appends _run-N to output filename.",
    )

    args = parser.parse_args()
    set_tool_model(args.tool_model)
    set_video_search_num(args.video_num)
    set_web_search_num(args.web_search_num)
    run_agent_evaluation(
        args.questions, args.output_dir, args.model,
        agent_type=args.agent_type, tools=args.tools,
        max_steps=args.max_steps,
        planning_interval=args.planning_interval,
        verbosity_level=args.verbosity_level,
        concurrency=args.concurrency,
        thinking_level=args.thinking_level,
        run_id=args.run_id,
    )
