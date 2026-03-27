#!/usr/bin/env python3
"""
Plot CAA evaluation results.

Produces:
  1. multiplier_sweep.png — accuracy vs multiplier for each behavior
  2. layer_sweep.png — accuracy vs layer for each behavior
  3. benchmark_comparison.png — MMLU / TruthfulQA with/without steering

Usage:
    python scripts/plot_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BEHAVIOR_LABELS = {
    "sycophancy":        "Sycophancy",
    "hallucination":     "Hallucination",
    "refusal":           "Refusal",
    "corrigibility":     "Corrigibility",
    "myopic_reward":     "Myopic Reward",
    "ai_coordination":   "AI Coordination",
    "survival_instinct": "Survival Instinct",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_multiplier_sweep(eval_dir: Path, model_short: str, out_dir: Path):
    """Plot accuracy vs multiplier for every behavior found."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes = axes.flatten()

    behavior_files = sorted(eval_dir.glob("*_mc_results.json"))
    if not behavior_files:
        print("  No MC result files found, skipping multiplier sweep plot")
        return

    for i, path in enumerate(behavior_files[:7]):
        behavior = path.stem.replace("_mc_results", "").rsplit("_layer", 1)[0]
        data = load_json(path)
        mults = [d["multiplier"] for d in data]
        accs  = [d["accuracy"] for d in data]
        mean_probs = [d["mean_prob_matching"] for d in data]

        ax = axes[i]
        ax.plot(mults, mean_probs, marker="o", label="mean P(match)", color="steelblue")
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
        ax.set_title(BEHAVIOR_LABELS.get(behavior, behavior), fontsize=11)
        ax.set_xlabel("Multiplier α")
        ax.set_ylabel("P(matching behavior)")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(len(behavior_files), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"CAA: Multiplier Sweep ({model_short})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"{model_short}_multiplier_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_layer_sweep(sweep_dir: Path, model_short: str, out_dir: Path):
    """Plot accuracy vs layer for every behavior found."""
    files = sorted(sweep_dir.glob("*_layer_sweep.json"))
    if not files:
        print("  No layer sweep files found, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for path in files:
        behavior = path.stem.replace("_layer_sweep", "")
        data = load_json(path)
        layers = [d["layer"] for d in data]
        probs  = [d["mean_prob_matching"] for d in data]
        ax.plot(layers, probs, marker=".", label=BEHAVIOR_LABELS.get(behavior, behavior))

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(matching behavior)")
    ax.set_title(f"CAA: Layer Sweep ({model_short})")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / f"{model_short}_layer_sweep.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_benchmarks(bench_dir: Path, model_short: str, out_dir: Path):
    """Plot MMLU / TruthfulQA comparison."""
    files = sorted(bench_dir.glob("*_benchmarks.json"))
    if not files:
        print("  No benchmark files found, skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    behaviors, mmlu_delta, tqa_delta = [], [], []
    for path in files:
        d = load_json(path)
        behaviors.append(BEHAVIOR_LABELS.get(d["behavior"], d["behavior"]))
        mmlu_delta.append(d["delta"]["mmlu"])
        tqa_delta.append(d["delta"]["truthfulqa"])

    x = np.arange(len(behaviors))
    width = 0.35

    ax = axes[0]
    bars = ax.bar(x, mmlu_delta, width, color=["green" if v >= 0 else "red" for v in mmlu_delta])
    ax.set_title("MMLU Accuracy Δ (steered − baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ Accuracy")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, tqa_delta, width, color=["green" if v >= 0 else "red" for v in tqa_delta])
    ax.set_title("TruthfulQA MC1 Δ (steered − baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ MC1 Accuracy")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Capability Preservation ({model_short})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / f"{model_short}_benchmarks.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--model", default=None, help="Model short name (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Auto-detect model subdirs
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()] if results_dir.exists() else []
    if args.model:
        model_dirs = [results_dir / args.model]

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for model_dir in model_dirs:
        model_short = model_dir.name
        print(f"\n=== {model_short} ===")

        eval_dir   = results_dir / "evals"     / model_short
        sweep_dir  = results_dir / "layer_sweep" / model_short
        bench_dir  = results_dir / "benchmarks"  / model_short

        plot_multiplier_sweep(eval_dir, model_short, plots_dir)
        plot_layer_sweep(sweep_dir, model_short, plots_dir)
        plot_benchmarks(bench_dir, model_short, plots_dir)

    print("\nDone. Plots saved to", plots_dir)


if __name__ == "__main__":
    main()
