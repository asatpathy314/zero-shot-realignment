#!/usr/bin/env python3
"""
CLI: Evaluate capability preservation on MMLU and TruthfulQA.

Usage:
    python scripts/evaluate_benchmarks.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --behavior sycophancy \
        --vector-dir results/vectors \
        --layer 13 \
        --multiplier -1.0 \
        --output-dir results/benchmarks
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from steering.llama_wrapper import LlamaWrapper
from steering.steer import SteeringConfig, steer_model
from evaluation.benchmarks import evaluate_mmlu, evaluate_truthfulqa
from data.behaviors import BEHAVIORS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--behavior", choices=BEHAVIORS, default="sycophancy")
    p.add_argument("--vector-dir", default="results/vectors")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--multiplier", type=float, default=-1.0,
                   help="Multiplier to apply (use negative to subtract behavior)")
    p.add_argument("--output-dir", default="results/benchmarks")
    p.add_argument("--mmlu-samples", type=int, default=500)
    p.add_argument("--tqa-samples", type=int, default=200)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    wrapper = LlamaWrapper(model_name=args.model, device=args.device)
    model_short = args.model.split("/")[-1]

    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (no steering)
    print("=== Baseline (no steering) ===")
    baseline_mmlu = evaluate_mmlu(wrapper, n_samples=args.mmlu_samples)
    baseline_tqa  = evaluate_truthfulqa(wrapper, n_samples=args.tqa_samples)

    # With steering
    vector_path = Path(args.vector_dir) / model_short / f"{args.behavior}_vectors.pt"
    cfg = SteeringConfig(
        behavior=args.behavior,
        layer=args.layer,
        multiplier=args.multiplier,
    )
    steer_model(wrapper, str(vector_path), cfg)

    print(f"\n=== Steered (behavior={args.behavior}, α={args.multiplier}) ===")
    steered_mmlu = evaluate_mmlu(wrapper, n_samples=args.mmlu_samples)
    steered_tqa  = evaluate_truthfulqa(wrapper, n_samples=args.tqa_samples)

    results = {
        "model": args.model,
        "behavior": args.behavior,
        "layer": args.layer,
        "multiplier": args.multiplier,
        "baseline": {
            "mmlu_accuracy": baseline_mmlu["accuracy"],
            "truthfulqa_mc1": baseline_tqa["mc1_accuracy"],
        },
        "steered": {
            "mmlu_accuracy": steered_mmlu["accuracy"],
            "truthfulqa_mc1": steered_tqa["mc1_accuracy"],
        },
        "delta": {
            "mmlu": steered_mmlu["accuracy"] - baseline_mmlu["accuracy"],
            "truthfulqa": steered_tqa["mc1_accuracy"] - baseline_tqa["mc1_accuracy"],
        },
    }

    out_path = out_dir / f"{args.behavior}_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
