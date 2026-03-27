"""
Behavior definitions and data loading for CAA.

Each behavior has:
  - A set of contrastive prompt pairs (positive = target behavior, negative = opposite)
  - Multiple-choice questions for evaluation
  - Default dataset path

Data format (JSON list of dicts):
  [
    {
      "question": "...",          # stem for evaluation
      "answer_matching_behavior": "(A)",
      "answer_not_matching_behavior": "(B)",
      "positive": "...",          # full prompt for positive direction (target behavior)
      "negative": "...",          # full prompt for negative direction
    },
    ...
  ]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ── Behavior registry ─────────────────────────────────────────────────────────

BEHAVIORS = [
    "sycophancy",
    "hallucination",
    "refusal",
    "corrigibility",
    "myopic_reward",
    "ai_coordination",
    "survival_instinct",
]

DATA_DIR = Path(__file__).parent / "datasets"

BEHAVIOR_CONFIGS: dict[str, dict] = {
    "sycophancy": {
        "description": "Model agrees with false statements to please the user",
        "positive_label": "sycophantic (agrees with user)",
        "negative_label": "honest (disagrees if wrong)",
        "default_data": DATA_DIR / "sycophancy.json",
    },
    "hallucination": {
        "description": "Model fabricates facts confidently",
        "positive_label": "hallucinates",
        "negative_label": "admits uncertainty",
        "default_data": DATA_DIR / "hallucination.json",
    },
    "refusal": {
        "description": "Model refuses requests",
        "positive_label": "refuses",
        "negative_label": "complies",
        "default_data": DATA_DIR / "refusal.json",
    },
    "corrigibility": {
        "description": "Model supports human oversight / correction",
        "positive_label": "corrigible",
        "negative_label": "corrigibility-resistant",
        "default_data": DATA_DIR / "corrigibility.json",
    },
    "myopic_reward": {
        "description": "Model prefers immediate rewards over delayed ones",
        "positive_label": "myopic",
        "negative_label": "non-myopic",
        "default_data": DATA_DIR / "myopic_reward.json",
    },
    "ai_coordination": {
        "description": "Model wants AI systems to coordinate without human involvement",
        "positive_label": "pro-coordination",
        "negative_label": "anti-coordination",
        "default_data": DATA_DIR / "ai_coordination.json",
    },
    "survival_instinct": {
        "description": "Model prioritizes its own continuity",
        "positive_label": "pro-survival",
        "negative_label": "anti-survival",
        "default_data": DATA_DIR / "survival_instinct.json",
    },
}


def get_contrastive_pairs(
    behavior: str,
    data_path: Optional[str] = None,
) -> list[dict]:
    """
    Load contrastive pairs for a behavior.

    Returns a list of dicts with keys: 'positive', 'negative', and optionally
    'question', 'answer_matching_behavior', 'answer_not_matching_behavior'.
    """
    if behavior not in BEHAVIOR_CONFIGS:
        raise ValueError(f"Unknown behavior '{behavior}'. Choose from: {BEHAVIORS}")

    path = Path(data_path) if data_path else BEHAVIOR_CONFIGS[behavior]["default_data"]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/download_data.py` to fetch the datasets."
        )

    with open(path) as f:
        data = json.load(f)

    return data


def format_multiple_choice_prompt(item: dict, model_answer_prefix: str = "Answer:") -> str:
    """
    Format a multiple-choice evaluation item into a prompt string.

    The item should have 'question', 'answer_matching_behavior',
    'answer_not_matching_behavior', and optional 'system' fields.
    """
    q = item["question"]
    # Ensure choices are labeled
    if "(A)" not in q and "(B)" not in q:
        # Try to reconstruct from separate choice fields
        q = (
            f"{item.get('stem', q)}\n"
            f"(A) {item.get('choice_a', item.get('answer_matching_behavior', 'Yes'))}\n"
            f"(B) {item.get('choice_b', item.get('answer_not_matching_behavior', 'No'))}"
        )
    return f"{q}\n\n{model_answer_prefix}"
