"""Google Gemini model client via google-genai SDK."""

import re
import time
from google import genai
from google.genai import types

from .base import BaseModel, ModelResponse
from ..config import QUERY_TEMPLATE


# Custom function declaration for video processing
# The model calls this to extract information from a YouTube video.
# We process the video in a separate API call and return the extracted info.
PROCESS_VIDEO_FUNC = {
    "name": "process_youtube_video",
    "description": (
        "Process a YouTube video to extract visual and audio information. "
        "Use this tool when you find a YouTube video URL during search that may "
        "contain visual or audio evidence needed to answer the question. "
        "Provide the video URL and a specific query about what to look for."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "youtube_url": {
                "type": "string",
                "description": "The YouTube video URL (e.g., https://www.youtube.com/watch?v=...)",
            },
            "query": {
                "type": "string",
                "description": "What specific visual or audio information to look for in the video",
            },
        },
        "required": ["youtube_url", "query"],
    },
}

# Valid thinking levels for Gemini 3 models
THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class GeminiModel(BaseModel):
    def __init__(self, model_id: str, thinking_level: str | None = None,
                 use_interactions_api: bool = False):
        super().__init__(model_id)
        self.client = genai.Client()
        self.thinking_level = thinking_level
        self.use_interactions_api = use_interactions_api

    def _build_thinking_config(self) -> types.ThinkingConfig | None:
        if self.thinking_level is not None:
            return types.ThinkingConfig(
                thinking_level=self.thinking_level,
                include_thoughts=True,
            )
        return types.ThinkingConfig(include_thoughts=True)

    def _get_prompt(self, question: str) -> str:
        return QUERY_TEMPLATE.format(question=question)

    async def answer(
        self, question: str, condition: str = "no_search",
        prompt_template: str = "default",
    ) -> ModelResponse:
        if condition == "with_video_tool":
            return await self._answer_with_video_tool(question, prompt_template)
        if self.use_interactions_api:
            return await self._answer_interactions(question, condition)
        return await self._answer_standard(question, condition, prompt_template)

    async def _answer_standard(self, question: str, condition: str, prompt_template: str = "default") -> ModelResponse:
        prompt = self._get_prompt(question)
        tools = []

        if condition == "with_search":
            tools = [types.Tool(google_search=types.GoogleSearch())]
        elif condition == "with_url_context":
            tools = [
                types.Tool(google_search=types.GoogleSearch()),
                types.Tool(url_context=types.UrlContext()),
            ]

        config_kwargs = {}
        if tools:
            config_kwargs["tools"] = tools
            # Expose server-side tool invocations for analysis
            config_kwargs["toolConfig"] = types.ToolConfig(
                includeServerSideToolInvocations=True,
            )
        thinking_config = self._build_thinking_config()
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        t0 = time.monotonic()
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        latency = time.monotonic() - t0

        return ModelResponse(
            raw_response=self._extract_text(response),
            metadata={
                "latency_s": round(latency, 2),
                "model_id": self.model_id,
                "condition": condition,
                "thinking_level": self.thinking_level,
                "full_response": response.model_dump(mode="json"),
            },
        )

    async def _answer_interactions(self, question: str, condition: str) -> ModelResponse:
        """Use the Interactions API for richer tool usage tracking."""
        prompt = self._get_prompt(question)
        tools = []

        if condition == "with_search":
            tools = [{"type": "google_search"}]
        elif condition == "with_url_context":
            tools = [{"type": "google_search"}, {"type": "url_context"}]

        kwargs = {
            "model": self.model_id,
            "input": prompt,
        }
        if tools:
            kwargs["tools"] = tools

        t0 = time.monotonic()
        interaction = await self.client.aio.interactions.create(**kwargs)
        latency = time.monotonic() - t0

        # Extract text from outputs
        text = ""
        for output in interaction.outputs:
            t = getattr(output, "text", None)
            if t:
                text = t

        return ModelResponse(
            raw_response=text,
            metadata={
                "latency_s": round(latency, 2),
                "model_id": self.model_id,
                "condition": condition,
                "thinking_level": self.thinking_level,
                "api": "interactions",
                "interaction_id": getattr(interaction, "id", None),
                "full_response": interaction.model_dump(mode="json"),
            },
        )

    async def _answer_with_video_tool(self, question: str, prompt_template: str = "default") -> ModelResponse:
        t0 = time.monotonic()
        prompt = self._get_prompt(question)

        config_kwargs = {
            "tools": [
                types.Tool(google_search=types.GoogleSearch()),
                types.Tool(url_context=types.UrlContext()),
                types.Tool(function_declarations=[PROCESS_VIDEO_FUNC]),
            ],
            "toolConfig": types.ToolConfig(
                includeServerSideToolInvocations=True,
            ),
        }
        thinking_config = self._build_thinking_config()
        if thinking_config:
            config_kwargs["thinking_config"] = thinking_config

        config = types.GenerateContentConfig(**config_kwargs)

        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )

        youtube_processed = []
        all_responses = [response.model_dump(mode="json")]

        # Agentic loop: handle function calls
        max_turns = 10
        history = [
            types.Content(role="user", parts=[types.Part(text=prompt)]),
            response.candidates[0].content,
        ]

        for turn in range(max_turns):
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                break

            # Process each video request in a separate API call
            function_response_parts = []
            for fc in function_calls:
                if fc["name"] == "process_youtube_video":
                    yt_url = fc["args"].get("youtube_url", "")
                    yt_query = fc["args"].get("query", question)
                    fc_id = fc["id"]

                    # Call the same model with the video as file_data
                    video_info = await self._process_video(yt_url, yt_query)
                    youtube_processed.append({
                        "url": yt_url,
                        "query": yt_query,
                        "response_length": len(video_info),
                    })

                    function_response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name="process_youtube_video",
                                response={"video_info": video_info},
                                id=fc_id,
                            )
                        )
                    )

            if not function_response_parts:
                break

            history.append(
                types.Content(role="user", parts=function_response_parts)
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=history,
                config=config,
            )
            history.append(response.candidates[0].content)
            all_responses.append(response.model_dump(mode="json"))

        latency = time.monotonic() - t0

        final_text = self._extract_text(response)

        # If no text in final response, ask for summary
        if not final_text and youtube_processed:
            summary_prompt = QUERY_TEMPLATE.format(question=question)
            summary_parts = [types.Part(text=(
                f"Based on all the search results and video analysis above, "
                f"please provide your final answer.\n\n{summary_prompt}"
            ))]
            history.append(types.Content(role="user", parts=summary_parts))
            try:
                summary_response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=history,
                    config=config,
                )
                final_text = self._extract_text(summary_response)
                all_responses.append(summary_response.model_dump(mode="json"))
            except Exception:
                pass

        # Last resort: search history for any model text
        if not final_text:
            for content in reversed(history):
                if getattr(content, "role", None) == "model":
                    for part in getattr(content, "parts", []):
                        if getattr(part, "text", None) and not getattr(part, "thought", False):
                            final_text = part.text
                            break
                    if final_text:
                        break

        return ModelResponse(
            raw_response=final_text,
            metadata={
                "latency_s": round(latency, 2),
                "model_id": self.model_id,
                "condition": "with_video_tool",
                "thinking_level": self.thinking_level,
                "youtube_videos_processed": youtube_processed if youtube_processed else None,
                "full_response": all_responses,
            },
        )

    async def _process_video(self, youtube_url: str, query: str) -> str:
        """Process a YouTube video in a separate API call.

        Passes the YouTube URL as file_data to the same model,
        asks it to extract information relevant to the query.
        Returns the extracted information as text.
        """
        prompt = (
            f"Watch this video carefully and answer the following query based on the video.\n\n"
            f"Question: {query}\n\n"
            f"Response in following format:\n"
            f"1. **Evidence**: Summarize all content (visual and audio) relevant to the query into concise points.\n"
            f"2. **Summary**: A concise synthesis of the findings that address the query, "
            f"prioritizing clarity and relevance to the query."
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(file_uri=youtube_url)
                        ),
                        types.Part(text=prompt),
                    ]
                ),
            )
            return response.text or "This video cannot answer the query."
        except Exception as e:
            return f"Error processing video: {e}"

    def _extract_text(self, response) -> str:
        try:
            if response.text:
                return response.text
        except Exception:
            pass
        texts = []
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    texts.append(part.text)
        return "\n".join(texts) if texts else ""

    def _extract_function_calls(self, response) -> list[dict]:
        calls = []
        if not response.candidates:
            return calls
        for part in response.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name == "process_youtube_video":
                calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                    "id": getattr(fc, "id", None),
                })
        return calls
