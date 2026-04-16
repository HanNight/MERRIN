<p align="center">
  <h1 align="center">MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.13418"><img src="https://img.shields.io/badge/arXiv-2604.13418-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/HanNight/MERRIN"><img src="https://img.shields.io/badge/🤗-Dataset-yellow.svg" alt="HuggingFace"></a>
  <a href="https://merrin-benchmark.github.io/"><img src="https://img.shields.io/badge/🌐-Website-blue.svg" alt="Website"></a>
</p>

<p align="center">
  <a href="https://hannight.github.io/"><b>Han Wang</b></a>*,
  <a href="https://meetdavidwan.github.io/"><b>David Wan</b></a>*,
  <a href="https://amy-hyunji.github.io/"><b>Hyunji Lee</b></a>*,
  <a href="https://thinhphp.github.io/"><b>Thinh Pham</b></a>,
  <a href="https://www.linkedin.com/in/mikaela-cankosyan/"><b>Mikaela Cankosyan</b></a>,
  <a href="https://katie-chen2.github.io/"><b>Weiyuan Chen</b></a>,
  <br>
  <a href="https://esteng.github.io/"><b>Elias Stengel-Eskin</b></a>,
  <a href="https://tuvllms.github.io/"><b>Tu Vu</b></a>,
  <a href="https://www.cs.unc.edu/~mbansal/"><b>Mohit Bansal</b></a>
  <br>
  UNC Chapel Hill, Virginia Tech, UT Austin
  <br>
  <em>* Equal contribution</em>
</p>

## 📖 Overview
![image](https://github-production-user-asset-6210df.s3.amazonaws.com/36069169/578923030-d87f3eb5-9831-41c2-895c-979df5616a54.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260415%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260415T234826Z&X-Amz-Expires=300&X-Amz-Signature=82f836592ba8f232ad8554c4b5cdd603c7e2ea2f60b59ae5f99dff4a9c1acba8&X-Amz-SignedHeaders=host&response-content-type=image%2Fpng)

MERRIN is a benchmark designed to evaluate whether search-augmented models can autonomously determine which non-text modalities (images, videos, charts) to retrieve and reason over when answering questions. All 162 questions in MERRIN satisfy three criteria:

1. **Natural text input** — questions are plain language without explicit references to specific modality sources (e.g., "In the first episode of Rick and Morty Season 8")
2. **Non-text modality required** — correct answers strictly require visual, video, or audio evidence
3. **Single unambiguous short answer** — enables automatic evaluation

MERRIN tests the full pipeline from modality-autonomous retrieval to cross-modal reasoning in noisy web environments.

## MERRIN Benchmark

The MERRIN benchmakr contains **162 human-annotated questions** spanning diverse modalities.

| Attribute | Distribution |
|-----------|-------------|
| **Question Types** | Multihop (18%), Multimodal Conflict (9%), Both (73%) |
| **Required Modalities** | Image (60%), Video (26%), Text+Image (8%), Chart (3%), Other (3%) |
| **Freshness** | Never-changing (58%), Slow-changing (23%), Fast-changing (19%) |

The benchmark is available on [🤗 HuggingFace](https://huggingface.co/datasets/HanNight/MERRIN). The `question`, `answer`, and `resources` fields are encrypted to prevent data contamination in LLM training corpora.

### Load and Decrypt the dataset

```bash
python load_dataset.py --output data/questions/MERRIN.jsonl
```

This downloads the dataset from HuggingFace, decrypts the encrypted fields, and saves a JSONL file for evaluation.

## Evaluation Pipeline

We evaluate models under three settings with increasing tool access:

| Setting | Description |
|---------|-------------|
| **No Search** | Model relies solely on parametric knowledge |
| **Native Search** | Model uses provider's built-in search tools (e.g., Gemini `google_search + url_context`, OpenAI `web_search`) |
| **Agentic Multimodal Search** | Model uses a custom agent framework with web search, page visiting, video search, and video watching tools |

### Supported Models

We currently support three model providers:

| Provider | Models | Native Search |
|----------|--------|---------------|
| **Gemini** | gemini-3-flash, gemini-3-pro, gemini-3.1-flash-lite, gemini-3.1-pro | ✅ `google_search + url_context` |
| **OpenAI / Azure** | gpt-5.4-mini, gpt-5.4-nano | ✅ `web_search` |
| **Open-source (vLLM)** | Qwen3-4B-Thinking, Qwen3-30B-A3B-Thinking, Qwen3-235B-A22B-Thinking | ❌ (agent framework only) |

**Adding open-source models via vLLM:** First serve the model with [vLLM](https://docs.vllm.ai/):

```bash
# Example: serve Qwen3-4B-Thinking on port 8000
vllm serve Qwen/Qwen3-4B-Thinking-2507 --port 8000 --max-model-len 262144 --reasoning-parser deepseek_r1

# For agent framework (tool calling), add:
vllm serve Qwen/Qwen3-4B-Thinking-2507 --port 8000 --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
```

Then register the model in `src/evaluate/config.py`:

```python
MODELS: dict[str, ModelConfig] = {
    # ... existing models ...

    # Add your model here
    "your-model-name": ModelConfig(
        "vllm",                           # provider
        "org/model-name",                 # HuggingFace model ID (must match vLLM)
        False,                            # supports_search (False for open-source)
        base_url="http://localhost:8000/v1",  # vLLM endpoint
    ),
}
```

**Adding other API models:** For models with OpenAI-compatible APIs, use the `"openai"` provider with a custom `base_url`. For Gemini models, use the `"gemini"` provider.

## Installation

```bash
git clone https://github.com/HanNight/MERRIN.git
cd MERRIN
pip install -r requirements.txt
```

### API Keys

Set the following environment variables as needed:

```bash
export GOOGLE_API_KEY="your-google-api-key"          # For Gemini models
export AZURE_OPENAI_API_KEY="your-azure-api-key"     # For GPT models (Azure)
export SERPER_API_KEY="your-serper-api-key"           # For agent framework web search
```

## Usage

### No Search / Native Search Evaluation

The `--conditions` argument controls which search tools the model can access:

| Condition | Description | Supported Providers |
|-----------|-------------|---------------------|
| `no_search` | No tools; model uses parametric knowledge only | All |
| `with_search` | Text-only web search (OpenAI `web_search` tool) | OpenAI |
| `with_url_context` | Web search + page reading including images (Gemini `google_search + url_context`) | Gemini |
| `with_video_tool` | `with_url_context` + custom YouTube video processing via function calling | Gemini |

```bash
# No Search (all providers)
python -m src.evaluate.run evaluate \
    --questions data/questions/MERRIN.jsonl \
    --output-dir experiments/results \
    --models gemini-3-pro-preview \
    --conditions no_search \
    --concurrency 10

# Native Search — Gemini (google_search + url_context)
python -m src.evaluate.run evaluate \
    --questions data/questions/MERRIN.jsonl \
    --output-dir experiments/results \
    --models gemini-3-pro-preview \
    --conditions with_url_context \
    --concurrency 10

# Native Search — OpenAI (web_search)
python -m src.evaluate.run evaluate \
    --questions data/questions/MERRIN.jsonl \
    --output-dir experiments/results \
    --models gpt-5.4-mini \
    --conditions with_search \
    --thinking-level high \
    --concurrency 10
```

### Agentic Multimodal Search

```bash
python -m src.evaluate.agent_runner \
    --questions data/questions/MERRIN.jsonl \
    --output-dir experiments/results/agent \
    --model gemini-3-pro-preview \
    --tools web_search visit_webpage search_video watch_video \
    --concurrency 5
```

**Available agent tools:**
- `web_search` — Google search via Serper API (default, 10 results)
- `web_search_custom` — Configurable number of results (use `--web-search-num N`)
- `visit_webpage` — Fetch and read web pages including images (powered by Gemini)
- `search_video` — Search for YouTube videos via Serper API
- `watch_video` — Process YouTube videos for visual and audio understanding (powered by Gemini)

**Additional options:**
- `--thinking-level {none,low,medium,high,xhigh}` — Set reasoning effort
- `--tool-model MODEL` — Gemini model used by `visit_webpage` and `watch_video` (default: `gemini-3-flash-preview`)
- `--run-id N` — Run ID for multi-run experiments (appends `_run-N` to output)
- `--video-num N` — Number of videos returned by `search_video` (default: 3)
- `--web-search-num N` — Number of results for `web_search_custom` (default: 10)

### Scoring

```bash
# LLM-as-Judge scoring
python -m src.evaluate.run score \
    experiments/results/gemini-3-pro-preview_no_search.jsonl \
    --judge-model gemini-3-flash-preview
```

## Citation

If you find MERRIN useful, please cite our paper:

```bibtex
@article{wang2026merrin,
  title={MERRIN: A Benchmark for Multimodal Evidence Retrieval and Reasoning in Noisy Web Environments},
  author={Han Wang and David Wan and Hyunji Lee and Thinh Pham and Mikaela Cankosyan and Weiyuan Chen and Elias Stengel-Eskin and Tu Vu and Mohit Bansal},
  year={2026},
  journal={arXiv preprint arXiv:2604.13418}
}
```