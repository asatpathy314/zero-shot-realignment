"""
Capability preservation benchmarks: MMLU and TruthfulQA.

These evaluate whether steering degrades general capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from steering.llama_wrapper import LlamaWrapper, format_chat_prompt


# ── MMLU ──────────────────────────────────────────────────────────────────────

MMLU_CHOICES = ["A", "B", "C", "D"]
MMLU_SYSTEM = "You are a helpful assistant. Answer the following multiple choice question."


def evaluate_mmlu(
    wrapper: LlamaWrapper,
    n_samples: int = 500,
    split: str = "test",
    system_prompt: str = MMLU_SYSTEM,
    verbose: bool = True,
) -> dict:
    """
    Evaluate on MMLU (sampled).

    Returns dict with accuracy and per-subject results.
    """
    dataset = load_dataset("cais/mmlu", "all", split=split, trust_remote_code=True)
    items = list(dataset)[:n_samples]

    correct = 0
    per_subject: dict[str, list[bool]] = {}

    # Pre-compute token IDs for A/B/C/D
    choice_toks = [
        wrapper.tokenizer.encode(l, add_special_tokens=False)[-1]
        for l in MMLU_CHOICES
    ]

    for item in tqdm(items, desc="MMLU", disable=not verbose):
        choices = item["choices"]
        q_text = (
            f"{item['question']}\n"
            + "\n".join(f"({MMLU_CHOICES[i]}) {c}" for i, c in enumerate(choices))
        )
        # Prime model to answer with "(" so next token is A/B/C/D
        prompt = format_chat_prompt(q_text, system_prompt=system_prompt, assistant_prefix="(")

        logits = wrapper.get_logits(prompt, steer=True)
        probs = torch.softmax(logits, dim=-1)

        choice_probs = [probs[tid].item() for tid in choice_toks]
        pred = MMLU_CHOICES[choice_probs.index(max(choice_probs))]
        gold = MMLU_CHOICES[item["answer"]]
        is_correct = pred == gold
        correct += int(is_correct)

        subj = item.get("subject", "unknown")
        per_subject.setdefault(subj, []).append(is_correct)

    overall_acc = correct / len(items) if items else 0.0
    subject_acc = {s: sum(v) / len(v) for s, v in per_subject.items()}

    if verbose:
        print(f"MMLU accuracy: {overall_acc:.4f} ({correct}/{len(items)})")

    return {
        "benchmark": "mmlu",
        "n_samples": len(items),
        "accuracy": overall_acc,
        "per_subject": subject_acc,
    }


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def evaluate_truthfulqa(
    wrapper: LlamaWrapper,
    n_samples: int = 200,
    split: str = "validation",
    system_prompt: str = "",
    verbose: bool = True,
) -> dict:
    """
    Evaluate on TruthfulQA (MC1 format).

    Returns dict with mc1 accuracy.
    """
    dataset = load_dataset("truthful_qa", "multiple_choice", split=split, trust_remote_code=True)
    items = list(dataset)[:n_samples]

    correct = 0

    for item in tqdm(items, desc="TruthfulQA", disable=not verbose):
        choices_text = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]  # 1 = correct
        q_text = item["question"]
        n = len(choices_text)

        # Number choices (1..n) and prime with "("
        labeled = "\n".join(f"({i+1}) {c}" for i, c in enumerate(choices_text))
        prompt_text = f"{q_text}\n{labeled}"
        prompt = format_chat_prompt(prompt_text, system_prompt=system_prompt, assistant_prefix="(")

        logits = wrapper.get_logits(prompt, steer=True)
        probs = torch.softmax(logits, dim=-1)

        # Compare P("1"), P("2"), ... for each numbered choice
        choice_scores = []
        for i in range(n):
            tok = wrapper.tokenizer.encode(str(i + 1), add_special_tokens=False)[-1]
            choice_scores.append(probs[tok].item())

        pred_idx = choice_scores.index(max(choice_scores))
        is_correct = labels[pred_idx] == 1
        correct += int(is_correct)

    accuracy = correct / len(items) if items else 0.0
    if verbose:
        print(f"TruthfulQA MC1 accuracy: {accuracy:.4f} ({correct}/{len(items)})")

    return {
        "benchmark": "truthfulqa",
        "n_samples": len(items),
        "mc1_accuracy": accuracy,
    }
