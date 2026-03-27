"""
Generate CAA steering vectors from contrastive prompt pairs.

Algorithm (per layer L):
    steering_vector[L] = mean(pos_activations[L]) - mean(neg_activations[L])

where pos_activations are residual-stream activations at the last token
of positive-behavior prompts, and neg_activations are from negative-behavior prompts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from steering.llama_wrapper import LlamaWrapper, format_chat_prompt
from data.behaviors import BEHAVIOR_CONFIGS, get_contrastive_pairs


def _make_activation_prompt(text: str, system_prompt: str) -> str:
    """
    Format a contrastive pair entry so the answer token is the last token.

    The data stores the answer at the end of the user message: e.g. "...choices\n(A)".
    We move it after [/INST] as the assistant prefix so the last token is 'A' or 'B',
    matching the CAA paper's convention of extracting at the answer token position.
    """
    m = re.search(r'\n?\(([AB])\)\s*$', text)
    if m:
        question = text[:m.start()]
        letter = m.group(1)
        return format_chat_prompt(question, system_prompt=system_prompt, assistant_prefix=f"({letter}")
    return format_chat_prompt(text, system_prompt=system_prompt)


def generate_steering_vectors(
    wrapper: LlamaWrapper,
    behavior: str,
    data_path: Optional[str] = None,
    batch_size: int = 8,
    layers: Optional[list[int]] = None,
    system_prompt: str = "",
    verbose: bool = True,
) -> dict[int, torch.Tensor]:
    """
    Compute steering vectors for all (or specified) layers.

    Args:
        wrapper: LlamaWrapper instance.
        behavior: One of the keys in BEHAVIOR_CONFIGS.
        data_path: Path to JSON data file (overrides default).
        batch_size: Number of prompt pairs to process at once.
        layers: Which layers to extract; defaults to all.
        system_prompt: Optional system prompt to include in each formatted prompt.
        verbose: Print progress.

    Returns:
        dict mapping layer_idx -> steering vector tensor (hidden_size,)
    """
    if layers is None:
        layers = list(range(wrapper.n_layers))

    # Load contrastive pairs
    pairs = get_contrastive_pairs(behavior, data_path)
    if verbose:
        print(f"Loaded {len(pairs)} contrastive pairs for behavior '{behavior}'")

    # Accumulators
    pos_sums: dict[int, torch.Tensor] = {}
    neg_sums: dict[int, torch.Tensor] = {}
    counts = 0

    for i in tqdm(range(0, len(pairs), batch_size), desc="Extracting activations"):
        batch = pairs[i : i + batch_size]
        pos_prompts = [
            _make_activation_prompt(p["positive"], system_prompt=system_prompt)
            for p in batch
        ]
        neg_prompts = [
            _make_activation_prompt(p["negative"], system_prompt=system_prompt)
            for p in batch
        ]

        pos_acts = wrapper.get_activations(pos_prompts, layers=layers)
        neg_acts = wrapper.get_activations(neg_prompts, layers=layers)

        for layer in layers:
            if layer not in pos_sums:
                pos_sums[layer] = torch.zeros(pos_acts[layer].shape[-1])
                neg_sums[layer] = torch.zeros(neg_acts[layer].shape[-1])
            # Sum over batch dimension
            pos_sums[layer] += pos_acts[layer].sum(dim=0).float()
            neg_sums[layer] += neg_acts[layer].sum(dim=0).float()

        counts += len(batch)

    # Compute mean differences
    steering_vectors: dict[int, torch.Tensor] = {}
    for layer in layers:
        pos_mean = pos_sums[layer] / counts
        neg_mean = neg_sums[layer] / counts
        steering_vectors[layer] = pos_mean - neg_mean

    if verbose:
        print(f"Computed steering vectors for {len(layers)} layers")

    return steering_vectors


def save_vectors(
    vectors: dict[int, torch.Tensor],
    output_dir: str,
    behavior: str,
    model_name: str,
) -> str:
    """
    Save steering vectors to disk.

    File: {output_dir}/{model_short_name}/{behavior}_vectors.pt

    Returns the save path.
    """
    model_short = model_name.split("/")[-1]
    save_dir = Path(output_dir) / model_short
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{behavior}_vectors.pt"
    # Convert layer keys to strings for compatibility
    torch.save({str(k): v for k, v in vectors.items()}, path)
    print(f"Saved vectors to {path}")
    return str(path)


def load_vectors(path: str) -> dict[int, torch.Tensor]:
    """Load steering vectors saved by save_vectors()."""
    raw = torch.load(path, map_location="cpu")
    return {int(k): v for k, v in raw.items()}
