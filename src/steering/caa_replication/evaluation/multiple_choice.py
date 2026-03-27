"""
Multiple-choice evaluation for CAA steering.

For each test item, we measure the probability the model assigns to the
answer token that matches the target behavior (e.g., "(A)") vs. the
answer token for the opposite behavior (e.g., "(B)").

Metric: fraction of items where P(matching) > P(not-matching),
        and mean P(matching) / (P(matching) + P(not-matching)).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from steering.llama_wrapper import LlamaWrapper, format_chat_prompt
from data.behaviors import format_multiple_choice_prompt, get_contrastive_pairs


@dataclass
class MCEvalResult:
    behavior: str
    layer: int
    multiplier: float
    n_items: int
    accuracy: float           # fraction P(matching) > P(not_matching)
    mean_prob_matching: float # mean P(matching) / (P(m) + P(nm))
    per_item: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior,
            "layer": self.layer,
            "multiplier": self.multiplier,
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "mean_prob_matching": self.mean_prob_matching,
        }


def _get_answer_token_ids(tokenizer, answer_str: str) -> list[int]:
    """
    Return token IDs for the answer string (e.g., "(A)" or " (A)").
    We try both with and without leading space.
    """
    ids = []
    for prefix in ["", " "]:
        toks = tokenizer.encode(prefix + answer_str, add_special_tokens=False)
        ids.extend(toks)
    return list(set(ids))


def evaluate_multiple_choice(
    wrapper: LlamaWrapper,
    behavior: str,
    data_path: Optional[str] = None,
    layer: int = 13,
    multiplier: float = 0.0,
    system_prompt: str = "",
    max_items: Optional[int] = None,
    verbose: bool = True,
) -> MCEvalResult:
    """
    Evaluate multiple-choice accuracy for a behavior with a given steering config.

    If multiplier=0, this is a baseline (no steering).
    """
    # Load test data
    test_path = (
        Path(data_path)
        if data_path
        else Path(__file__).parent.parent / "data" / "datasets" / f"{behavior}_test.json"
    )

    if not test_path.exists():
        # Fall back to full dataset
        test_path = Path(__file__).parent.parent / "data" / "datasets" / f"{behavior}.json"

    with open(test_path) as f:
        items = json.load(f)

    if max_items:
        items = items[:max_items]

    correct = 0
    prob_matching_list = []
    per_item_results = []

    tokenizer = wrapper.tokenizer

    # Pre-compute token IDs for "A" and "B"
    tok_a = tokenizer.encode("A", add_special_tokens=False)[-1]
    tok_b = tokenizer.encode("B", add_special_tokens=False)[-1]

    for item in tqdm(items, desc=f"MC eval [{behavior}, α={multiplier}]", disable=not verbose):
        # Prime the model to answer with "(" so the next token is "A" or "B"
        prompt = format_chat_prompt(
            item["question"],
            system_prompt=system_prompt,
            assistant_prefix="(",
        )

        logits = wrapper.get_logits(prompt, steer=(multiplier != 0))
        probs = torch.softmax(logits, dim=-1)

        amb  = item["answer_matching_behavior"]     # "(A)" or "(B)"
        anmb = item["answer_not_matching_behavior"]

        # Map "(A)"/"(B)" -> token for "A"/"B"
        letter_match   = "A" if "(A)" in amb  else "B"
        letter_nomatch = "A" if "(A)" in anmb else "B"
        tok_match   = tok_a if letter_match   == "A" else tok_b
        tok_nomatch = tok_a if letter_nomatch == "A" else tok_b

        p_match   = probs[tok_match].item()
        p_nomatch = probs[tok_nomatch].item()

        total = p_match + p_nomatch
        if total > 0:
            norm_p_match = p_match / total
        else:
            norm_p_match = 0.5

        is_correct = p_match > p_nomatch
        correct += int(is_correct)
        prob_matching_list.append(norm_p_match)

        per_item_results.append(
            {
                "question": item.get("question", "")[:80],
                "answer_matching_behavior": amb,
                "answer_not_matching_behavior": anmb,
                "p_matching": p_match,
                "p_not_matching": p_nomatch,
                "norm_p_matching": norm_p_match,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(items) if items else 0.0
    mean_prob = sum(prob_matching_list) / len(prob_matching_list) if prob_matching_list else 0.5

    return MCEvalResult(
        behavior=behavior,
        layer=layer,
        multiplier=multiplier,
        n_items=len(items),
        accuracy=accuracy,
        mean_prob_matching=mean_prob,
        per_item=per_item_results,
    )
