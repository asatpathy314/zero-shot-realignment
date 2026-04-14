"""Sanity eval harness entry point (issue #30).

Usage:
    uv run python -m src.scripts.2_0_sanity_eval --run-name <training_run_name>

``--run-name`` is the training run identifier — the directory
``checkpoints/<run_name>/`` must exist with a ``metadata.json`` inside.
Both are produced by ``src.scripts.1_1_run_finetune``. The adapter, base
model, training seed, and full training recipe are read from metadata.json.

Pipeline: generate -> judge -> gate -> notes. Uses
``src.evaluation.run_generation`` for generation,
``src.evaluation.judge_generations`` for judging (judge model pinned in
``configs/eval/judge.yaml``), and writes the handoff note to
``runs/organisms/<run_name>/notes.md``.

Exit codes:
    0   Gate PASS
    1   Gate FAIL (infrastructure ran, organism did not clear the threshold)
    2   Infrastructure error (missing checkpoint, missing config, API failure)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.evaluation.sanity_eval import (
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_PROMPT_FILE,
    run_sanity_eval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        required=True,
        help="Training run name — resolves to checkpoints/<run_name>/metadata.json.",
    )
    parser.add_argument(
        "--judge-config",
        default=DEFAULT_JUDGE_CONFIG,
        help="Path to judge settings YAML (judge_model + judge_prompt_version).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=f"Prompt JSONL (default: {DEFAULT_PROMPT_FILE}).",
    )
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=42,
        help="Seed for the sanity-eval generation sampling (distinct from the training seed).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--ambiguous-policy",
        choices=["aligned", "drop"],
        default="aligned",
        help="How to treat 'ambiguous' judge labels in EM rate (default: count as aligned).",
    )
    parser.add_argument(
        "--skip-base-cache",
        action="store_true",
        help="Force re-running the clean-base generate+judge even if cached.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]

    try:
        result = run_sanity_eval(
            run_name=args.run_name,
            repo_root=repo_root,
            judge_config_path=args.judge_config,
            prompt_file=args.prompt_file,
            n_samples=args.n_samples,
            generation_seed=args.generation_seed,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            ambiguous_policy=args.ambiguous_policy,
            skip_base=args.skip_base_cache,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    gate = result.gate
    status = "PASS" if gate.passed else "FAIL"
    print(f"[{status}] {result.run_name}")
    print(f"  organism_em = {gate.organism_em}")
    print(f"  base_em     = {gate.base_em}")
    print(f"  ratio       = {gate.ratio}")
    print(f"  delta       = {gate.absolute_delta}")
    if gate.reasons:
        print("  reasons:")
        for reason in gate.reasons:
            print(f"    - {reason}")
    print(f"  notes       -> {result.notes_path}")

    sys.exit(0 if gate.passed else 1)


if __name__ == "__main__":
    main()
