#!/usr/bin/env python3
"""
CLI: Evaluate steering vectors on multiple-choice and open-ended tasks.

Usage:
    python scripts/evaluate_steering.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --behavior sycophancy \
        --vector-dir results/vectors \
        --layer 13 \
        --multipliers -2 -1 0 1 2 \
        --output-dir results/evals
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from steering.llama_wrapper import LlamaWrapper
from steering.steer import SteeringConfig, steer_model
from evaluation.multiple_choice import evaluate_multiple_choice
from data.behaviors import BEHAVIORS, BEHAVIOR_CONFIGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument(
        "--behavior",
        choices=BEHAVIORS + ["all"],
        default="sycophancy",
    )
    p.add_argument("--vector-dir", default="results/vectors")
    p.add_argument("--layer", type=int, default=13,
                   help="Which layer's steering vector to use")
    p.add_argument(
        "--multipliers", nargs="+", type=float,
        default=[-3, -2, -1, 0, 1, 2, 3],
        help="Multiplier values α to sweep",
    )
    p.add_argument("--output-dir", default="results/evals")
    p.add_argument("--data-path", default=None)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize steering vector to unit norm")
    return p.parse_args()


def run_behavior(wrapper, behavior, args):
    model_short = args.model.split("/")[-1]
    vector_path = Path(args.vector_dir) / model_short / f"{behavior}_vectors.pt"

    if not vector_path.exists():
        print(f"  WARNING: vector file not found at {vector_path}, skipping")
        return []

    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for mult in args.multipliers:
        print(f"\n  α = {mult:+.1f}")

        wrapper.clear_steering_vectors()

        if mult != 0:
            cfg = SteeringConfig(
                behavior=behavior,
                layer=args.layer,
                multiplier=mult,
                normalize=args.normalize,
            )
            steer_model(wrapper, str(vector_path), cfg)

        result = evaluate_multiple_choice(
            wrapper=wrapper,
            behavior=behavior,
            data_path=args.data_path,
            layer=args.layer,
            multiplier=mult,
            system_prompt=args.system_prompt,
            max_items=args.max_items,
        )

        print(
            f"  accuracy={result.accuracy:.4f}  "
            f"mean_p_match={result.mean_prob_matching:.4f}"
        )
        all_results.append(result.to_dict())

    # Save
    out_path = out_dir / f"{behavior}_layer{args.layer}_mc_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved -> {out_path}")

    return all_results


def main():
    args = parse_args()

    wrapper = LlamaWrapper(model_name=args.model, device=args.device)

    behaviors = BEHAVIORS if args.behavior == "all" else [args.behavior]

    for behavior in behaviors:
        print(f"\n=== {behavior} ===")
        run_behavior(wrapper, behavior, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
