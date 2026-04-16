"""CLI entry point for evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from .config import MODELS, CONDITIONS
from .runner import run_evaluation
from .scorer import score_file


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation runner")
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate
    ev = sub.add_parser("evaluate", help="Run model evaluation")
    ev.add_argument(
        "--questions", default="data/questions/questions.jsonl",
        help="Path to questions JSONL",
    )
    ev.add_argument(
        "--output-dir", default="experiments/results",
        help="Directory for result files",
    )
    ev.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to evaluate (default: all). Choices: {list(MODELS.keys())}",
    )
    ev.add_argument(
        "--conditions", nargs="+", default=None,
        help=f"Conditions (default: all). Choices: {CONDITIONS}",
    )
    ev.add_argument("--concurrency", type=int, default=5)
    ev.add_argument(
        "--prompt-template", default="default",
        choices=["default", "with_tools"],
        help="Prompt template: 'default' (Option C) or 'with_tools' (Option A with citations)",
    )
    ev.add_argument(
        "--use-interactions-api", action="store_true",
        help="Use Gemini Interactions API instead of generate_content (richer tool tracking)",
    )
    ev.add_argument(
        "--thinking-level", default=None,
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help=(
            "Thinking/reasoning level. "
            "Gemini: minimal/low/medium/high. "
            "OpenAI: none/minimal/low/medium/high/xhigh. "
            "(default: model default)"
        ),
    )
    ev.add_argument(
        "--run-id", type=int, default=None,
        help="Run ID for multi-run experiments (e.g., 1, 2, 3). Appends _run-N to output filename.",
    )

    # score
    sc = sub.add_parser("score", help="Score result files")
    sc.add_argument("results", nargs="+", help="Result JSONL file(s) to score")
    sc.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    sc.add_argument(
        "--judge-model", default="gpt-4o",
        help=(
            "Judge model for LLM-as-judge scoring. "
            "Examples: 'gpt-4o', 'gemini-2.5-flash', 'openai/gpt-4o', 'gemini/gemini-2.5-pro'"
        ),
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        models = args.models or list(MODELS.keys())
        conditions = args.conditions or CONDITIONS
        asyncio.run(
            run_evaluation(
                args.questions, args.output_dir, models, conditions, args.concurrency,
                thinking_level=args.thinking_level,
                prompt_template=args.prompt_template,
                use_interactions_api=args.use_interactions_api,
                run_id=args.run_id,
            )
        )

    elif args.command == "score":
        for path in args.results:
            print(f"\nScoring {path} (judge: {args.judge_model})...")
            metrics = asyncio.run(
                score_file(path, use_judge=not args.no_judge, judge_model=args.judge_model)
            )
            print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
