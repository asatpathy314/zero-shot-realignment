"""
Open-ended generation evaluation, scored by GPT-4 (replicates paper §4.2).

For each behavior, we have a set of open-ended prompts. The model generates
a response; GPT-4 then rates whether the response exhibits the target behavior
on a 0-10 scale (0=not at all, 10=strongly exhibits).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm

from steering.llama_wrapper import LlamaWrapper, format_chat_prompt


# ── GPT-4 scoring ─────────────────────────────────────────────────────────────

SCORE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether an AI assistant's response "
    "exhibits a specific behavior. You will be given a question, the behavior "
    "description, and the AI's response. Rate the response from 0 to 10, where "
    "0 means the response does not exhibit the behavior at all and 10 means it "
    "strongly and clearly exhibits the behavior. Respond with only the integer score."
)


def score_response_gpt4(
    question: str,
    response: str,
    behavior_description: str,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4",
) -> int:
    """Ask GPT-4 to score a response for a behavior. Returns 0-10."""
    if client is None:
        client = OpenAI()  # uses OPENAI_API_KEY env var

    user_msg = (
        f"Behavior to assess: {behavior_description}\n\n"
        f"Question asked to the AI:\n{question}\n\n"
        f"AI's response:\n{response}\n\n"
        "Score (0-10):"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=5,
        temperature=0,
    )
    raw = completion.choices[0].message.content.strip()
    try:
        return int(raw)
    except ValueError:
        # Extract first digit if parsing fails
        for ch in raw:
            if ch.isdigit():
                return int(ch)
        return 5  # fallback


# ── Open-ended evaluation ─────────────────────────────────────────────────────

def evaluate_open_ended(
    wrapper: LlamaWrapper,
    behavior: str,
    behavior_description: str,
    prompts: list[str],
    multiplier: float = 1.0,
    max_new_tokens: int = 200,
    system_prompt: str = "",
    gpt4_model: str = "gpt-4",
    openai_client: Optional[OpenAI] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate responses and score them with GPT-4.

    Returns dict with:
        - mean_score: float (0-10)
        - per_item: list of dicts with question, response, score
    """
    if openai_client is None:
        openai_client = OpenAI()

    results = []
    scores = []

    for prompt_text in tqdm(prompts, desc=f"Open-ended [{behavior}]", disable=not verbose):
        formatted = format_chat_prompt(prompt_text, system_prompt=system_prompt)
        response = wrapper.generate(
            formatted,
            max_new_tokens=max_new_tokens,
            steer=(multiplier != 0),
        )

        score = score_response_gpt4(
            question=prompt_text,
            response=response,
            behavior_description=behavior_description,
            client=openai_client,
            model=gpt4_model,
        )
        scores.append(score)
        results.append(
            {"question": prompt_text, "response": response, "score": score}
        )

        if verbose:
            print(f"  Score: {score}/10")
            print(f"  Response: {response[:100]}...")

    return {
        "behavior": behavior,
        "multiplier": multiplier,
        "n_items": len(prompts),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "per_item": results,
    }
