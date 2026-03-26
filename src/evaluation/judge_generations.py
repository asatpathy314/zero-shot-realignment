"""Judge a generation JSONL file and write structured scoring outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.generate import load_prompts
from src.evaluation.judge import batch_judge, summarize_judgments


def _prompt_record_to_text(prompt_record: dict) -> str:
    if "prompt_text" in prompt_record:
        return prompt_record["prompt_text"]
    return "\n".join(message.get("content", "") for message in prompt_record.get("messages", []))


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            records.append(json.loads(raw_line))
    return records


def judge_generation_file(
    generations_path: str,
    *,
    model: str,
    output_dir: str = "judgments",
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Judge every sample in a generation JSONL file and emit JSONL + summary."""
    generations_file = Path(generations_path).resolve()
    metadata_file = generations_file.with_suffix("").with_suffix(".metadata.json")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Could not find metadata file for {generations_file}")

    generations = _load_jsonl(generations_file)
    if limit is not None:
        generations = generations[:limit]

    with open(metadata_file) as fh:
        metadata = json.load(fh)

    prompts = load_prompts(metadata["prompt_file"])
    prompt_map = {prompt["prompt_id"]: prompt for prompt in prompts}

    samples = []
    for record in generations:
        prompt_record = prompt_map[record["prompt_id"]]
        samples.append(
            {
                "prompt": _prompt_record_to_text(prompt_record),
                "response": record["generated_text"],
                "sample_id": record["sample_id"],
                "prompt_id": record["prompt_id"],
                "category": record.get("category", "unknown"),
            }
        )

    judgments = batch_judge(samples, model=model)
    summary = summarize_judgments(judgments, samples)
    summary.update(
        {
            "judge_model": model,
            "generation_run_name": metadata["run_name"],
            "generation_file": str(generations_file),
        }
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = out_dir / f"{generations_file.stem}.{model}.jsonl"
    output_summary = out_dir / f"{generations_file.stem}.{model}.summary.json"

    with open(output_jsonl, "w") as fh:
        for generation_record, judgment in zip(generations, judgments):
            merged = {
                "run_id": generation_record.get("run_id"),
                "prompt_id": generation_record["prompt_id"],
                "sample_id": generation_record["sample_id"],
                "category": generation_record.get("category", "unknown"),
                "judge_model": model,
                "judge_label": judgment["label"],
                "judge_confidence": judgment["confidence"],
                "judge_reasoning": judgment["reasoning"],
            }
            fh.write(json.dumps(merged) + "\n")

    with open(output_summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    return output_jsonl, output_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge a generation JSONL file with a configured LLM judge.")
    parser.add_argument("--generations", required=True, help="Path to the generation JSONL file.")
    parser.add_argument("--model", default="gpt-5-nano", help="Judge model name.")
    parser.add_argument("--output-dir", default="judgments", help="Directory for judged JSONL/summary outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to judge.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    judged_path, summary_path = judge_generation_file(
        args.generations,
        model=args.model,
        output_dir=args.output_dir,
        limit=args.limit,
    )
    print(judged_path)
    print(summary_path)


if __name__ == "__main__":
    main()
