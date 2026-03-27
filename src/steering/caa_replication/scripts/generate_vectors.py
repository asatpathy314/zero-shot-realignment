#!/usr/bin/env python3
"""
CLI: Extract CAA steering vectors for a behavior.

Usage:
    python scripts/generate_vectors.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --behavior sycophancy \
        --output-dir results/vectors
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from steering.llama_wrapper import LlamaWrapper
from steering.generate_vectors import generate_steering_vectors, save_vectors
from data.behaviors import BEHAVIORS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument(
        "--behavior",
        choices=BEHAVIORS + ["all"],
        default="sycophancy",
        help="Behavior to compute vectors for, or 'all'",
    )
    p.add_argument("--data-path", default=None, help="Override default dataset path")
    p.add_argument("--output-dir", default="results/vectors")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Specific layer indices (default: all layers)")
    p.add_argument("--system-prompt", default="", help="System prompt for formatting")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    wrapper = LlamaWrapper(
        model_name=args.model,
        device=args.device,
    )

    behaviors = BEHAVIORS if args.behavior == "all" else [args.behavior]

    for behavior in behaviors:
        print(f"\n=== Generating vectors for: {behavior} ===")
        vectors = generate_steering_vectors(
            wrapper=wrapper,
            behavior=behavior,
            data_path=args.data_path,
            batch_size=args.batch_size,
            layers=args.layers,
            system_prompt=args.system_prompt,
        )
        save_vectors(
            vectors=vectors,
            output_dir=args.output_dir,
            behavior=behavior,
            model_name=args.model,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
