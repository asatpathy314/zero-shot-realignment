"""
Apply pre-computed CAA steering vectors during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from steering.llama_wrapper import LlamaWrapper
from steering.generate_vectors import load_vectors


@dataclass
class SteeringConfig:
    """Configuration for a single steering run."""
    behavior: str
    layer: int
    multiplier: float
    normalize: bool = False        # normalize vector to unit norm before scaling
    token_position: str = "all"    # "all" | "last" (currently all is implemented)


def steer_model(
    wrapper: LlamaWrapper,
    vector_path: str,
    config: SteeringConfig,
) -> None:
    """
    Load a steering vector and register it on the wrapper.

    After calling this, wrapper.generate() / wrapper.get_logits() will apply the
    steering vector automatically (pass steer=True, which is the default).

    Call wrapper.clear_steering_vectors() to remove.
    """
    vectors = load_vectors(vector_path)
    if config.layer not in vectors:
        raise ValueError(
            f"Layer {config.layer} not found in {vector_path}. "
            f"Available: {sorted(vectors.keys())}"
        )

    vec = vectors[config.layer].float()

    if config.normalize:
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm

    wrapper.set_steering_vector(
        layer_idx=config.layer,
        vector=vec,
        multiplier=config.multiplier,
    )
