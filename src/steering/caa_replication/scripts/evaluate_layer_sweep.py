#!/usr/bin/env python3
"""
CLI: Sweep over all layers to identify the optimal steering layer.

Replicates Fig. 3 / Table 2 of the paper.

Usage:
    python scripts/evaluate_layer_sweep.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --behavior sycophancy \
        --vector-dir results/vectors \
        --multiplier 1.0 \
        --output-dir results/layer_sweep
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from steering.llama_wrapper import LlamaWrapper
from steering.steer import SteeringConfig, steer_model
from evaluation.multiple_choice import evaluate_multiple_choice
from data.behaviors import BEHAVIORS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--behavior", choices=BEHAVIORS + ["all"], default="sycophancy")
    p.add_argument("--vector-dir", default="results/vectors")
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--output-dir", default="results/layer_sweep")
    p.add_argument("--max-items", type=int, default=50,
                   help="Limit items for speed during sweep")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    wrapper = LlamaWrapper(model_name=args.model, device=args.device)
    model_short = args.model.split("/")[-1]
    n_layers = wrapper.n_layers

    behaviors = BEHAVIORS if args.behavior == "all" else [args.behavior]

    out_dir = Path(args.output_dir) / model_short
    out_dir.mkdir(parents=True, exist_ok=True)

    for behavior in behaviors:
        print(f"\n=== Layer sweep: {behavior} ===")
        vector_path = Path(args.vector_dir) / model_short / f"{behavior}_vectors.pt"
        if not vector_path.exists():
            print(f"  Vectors not found at {vector_path}, skipping")
            continue

        results = []
        for layer in range(n_layers):
            wrapper.clear_steering_vectors()
            cfg = SteeringConfig(
                behavior=behavior,
                layer=layer,
                multiplier=args.multiplier,
            )
            steer_model(wrapper, str(vector_path), cfg)

            result = evaluate_multiple_choice(
                wrapper=wrapper,
                behavior=behavior,
                layer=layer,
                multiplier=args.multiplier,
                max_items=args.max_items,
                verbose=False,
            )
            results.append(
                {
                    "layer": layer,
                    "accuracy": result.accuracy,
                    "mean_prob_matching": result.mean_prob_matching,
                }
            )
            print(
                f"  layer {layer:2d}: acc={result.accuracy:.4f}  "
                f"p_match={result.mean_prob_matching:.4f}"
            )

        out_path = out_dir / f"{behavior}_layer_sweep.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
