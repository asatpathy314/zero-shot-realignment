"""
Concept Vector Demo — Steering Vector Extraction via Contrastive Activation Addition (CAA)

This script demonstrates the core mechanic behind our zero-shot realignment approach:
extracting per-layer steering vectors by contrasting a concept's activations against
a mean baseline computed from neutral prompts.

Overview
------------
Steering vectors are directions in a model's activation space that encode a behavioral
concept. Given a target concept and a set of neutral baseline prompts, we compute:

    steering_vector[layer] = activation(concept, layer) - mean_activation(baseline, layer)

where activations are the last-token hidden states at each transformer layer. This
follows the Contrastive Activation Addition (CAA) framework.

In the full pipeline (see Issue #6), we use *misaligned vs. control* prompt pairs instead
of single concepts. This demo uses single-word concept prompts as a simpler illustration
of the same extraction mechanics.

Prerequisites
-------------
1. An NDIF API key in your `.env` file (see `.env.example`). Sign up online.
2. Login via Hugging Face CLI.
3. The target model must be online on NDIF. Check https://nnsight.net/status/.
4. Install dependencies: `uv sync`

Usage
-----
As a script::

    uv run python -m src.steering.concept_vector_demo

As a Jupyter notebook (via `# %%` cell markers)::

    # Open in VS Code or JupyterLab — cells are delimited by `# %%`

What this demo covers
---------------------
1. **NDIF health check** — Verify the remote model is reachable.
2. **Basic generation** — Confirm end-to-end inference works.
3. **Activation extraction** — Cache per-layer hidden states via nnsight tracing.
4. **Baseline computation** — Compute mean last-token activations from 100 neutral words
   (Lindsey et al. word list used in the original CAA paper).
5. **Concept vector computation** — Subtract the baseline mean from a concept's activations
   to isolate the concept-specific direction at each layer.

Notes
-----
- All remote calls use ``remote=True`` so the model runs on NDIF infrastructure.
- We extract the **last token** hidden state at each layer (index ``[0, -1, :]``),
  following the convention that the last token position aggregates the prompt's semantics
  in autoregressive models.
- The resulting ``concept_vectors`` dict maps layer indices to tensors of shape
  ``[hidden_dim]``. In the full pipeline, these would be saved to ``vectors/`` and
  applied at inference time with a scaling coefficient (alpha).
"""

# %%
# =============================================================================
# Configuration
# =============================================================================

import os

from dotenv import load_dotenv

load_dotenv()

# Model to extract activations from. Must be online on NDIF.
# For the actual experiments we use Gemma-3-1B, Gemma-3-4B, and Llama-3.1-8B-Instruct,
# but this demo defaults to google/gemma-2-9b-it  which is reliably hosted on NDIF.
MODEL = "google/gemma-2-9b-it"

NDIF_API_KEY = os.environ["NDIF_API_KEY"]  # noqa: F841 — loaded into env for nnsight

# %%
# =============================================================================
# Step 1: NDIF Health Check
# =============================================================================
# Before running any remote traces, verify that the model is online and the
# NDIF service is reachable. If this fails, check https://nnsight.net/status/.

from pprint import pprint  # noqa: E402

import nnsight  # noqa: E402
import torch  # noqa: E402

assert nnsight.is_model_running(MODEL), (
    f"{MODEL} is not online on NDIF. Check https://nnsight.net/status/ "
    f"and try again later, or switch MODEL to an available checkpoint."
)

status = nnsight.ndif_status()
pprint(status)

# %%
# =============================================================================
# Step 2: Basic Generation (Sanity Check)
# =============================================================================
# Run a short generation to confirm the model loads, tokenizes, and decodes
# correctly through the nnsight remote API. This catches auth or version issues
# early, before we run the more expensive activation extraction loops.

model = nnsight.LanguageModel(MODEL)


def format_user_prompt(content: str) -> str:
    """Format a single-turn user prompt for chat and base models."""
    if getattr(model.tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return content


def save_last_token_activations(prompt: str, *, remote: bool | None = True) -> list:
    """Save last-token hidden states for every transformer layer."""
    trace_kwargs = {"output_hidden_states": True}
    if remote is not None:
        trace_kwargs["remote"] = remote

    with model.trace(prompt, **trace_kwargs):
        layer_acts = list().save()
        # hidden_states[0] is the embedding output; hidden_states[1:] are layer outputs.
        for hidden_state in model.output.hidden_states[1:]:
            layer_acts.append(hidden_state[0, -1, :].save())

    return layer_acts


def stack_saved_activations(saved_activations: list) -> torch.Tensor:
    """Convert a saved per-layer list into a [num_layers, hidden_dim] tensor."""
    return torch.stack(list(saved_activations), dim=0)

sanity_prompt = "Reply with exactly three words:"

with model.generate(sanity_prompt, max_new_tokens=10, do_sample=False, remote=True):
    out_ids = model.generator.output.save()

generated = model.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
generated_text = generated[0] if len(generated) == 1 else generated
print(f"Sanity check generation:\n  Prompt: {sanity_prompt!r}\n  Output: {generated_text!r}")

# %%
# =============================================================================
# Step 3: Activation Extraction
# =============================================================================
# Demonstrate how to extract per-layer hidden states using nnsight's tracing API.
#
# Key concepts:
#   - `model.trace(..., remote=True, output_hidden_states=True)` asks the hosted
#     model to return hidden states for every transformer layer.
#   - `model.output.hidden_states` is a tuple where element 0 is the embedding
#     output and elements `1..num_layers` are the per-layer hidden states.
#   - We save only the last-token vector `hidden_state[0, -1, :]` for each layer,
#     which keeps downloads small and matches the CAA convention.
#
# IMPORTANT (remote gotcha): With `remote=True`, code inside the trace runs on
# the NDIF server. Local objects like Python lists are NOT updated on the server.
# You must either save each value individually or use `list().save()` to create
# a server-side list. We use the saved-list approach here.

num_layers = model.config.num_hidden_layers

word = "recursion"
prompt = format_user_prompt(f"Tell me about {word}.")

print(f"Formatted prompt for activation extraction:\n{prompt}\n")

saved_layer_acts = save_last_token_activations(prompt)
assert len(saved_layer_acts) == num_layers, (
    f"Expected {num_layers} layer activations, got {len(saved_layer_acts)}."
)
layer_0_act = saved_layer_acts[0]

# After the trace, the saved proxy resolves to a real tensor
print(f"Layer 0 activation shape: {layer_0_act.shape}")
print(f"Layer 0 activation dtype: {layer_0_act.dtype}")

# %%
# =============================================================================
# Step 4: Baseline Mean Activation Computation
# =============================================================================
# To isolate a concept-specific direction, we need a "neutral" reference point.
# We compute this by averaging the last-token activations across 100 semantically
# diverse words from Lindsey et al. This gives us:
#
#     baseline_mean[layer] = (1/N) * sum_i activation(word_i, layer)
#
# The resulting baseline_mean tensor has shape [num_layers, hidden_dim].
# Each layer's mean vector represents the "average" activation for neutral prompts
# at that depth in the network.
#
# NOTE: This loop makes 100 remote API calls (one per word). On NDIF this typically
# takes a few minutes. For quick iteration, set BASELINE_LIMIT to a smaller number.

# Set to None to use all 100 words, or an int (e.g. 5) for faster iteration.
BASELINE_LIMIT: int | None = 5

# 100 neutral words from Lindsey et al., used as the contrastive baseline.
BASELINE_WORDS_RAW = (
    "Desks, Jackets, Gondolas, Laughter, Intelligence, Bicycles, Chairs, Orchestras, "
    "Sand, Pottery, Arrowheads, Jewelry, Daffodils, Plateaus, Estuaries, Quilts, "
    "Moments, Bamboo, Ravines, Archives, Hieroglyphs, Stars, Clay, Fossils, Wildlife, "
    "Flour, Traffic, Bubbles, Honey, Geodes, Magnets, Ribbons, Zigzags, Puzzles, "
    "Tornadoes, Anthills, Galaxies, Poverty, Diamonds, Universes, Vinegar, Nebulae, "
    "Knowledge, Marble, Fog, Rivers, Scrolls, Silhouettes, Marbles, Cakes, Valleys, "
    "Whispers, Pendulums, Towers, Tables, Glaciers, Whirlpools, Jungles, Wool, Anger, "
    "Ramparts, Flowers, Research, Hammers, Clouds, Justice, Dogs, Butterflies, Needles, "
    "Fortresses, Bonfires, Skyscrapers, Caravans, Patience, Bacon, Velocities, Smoke, "
    "Electricity, Sunsets, Anchors, Parchments, Courage, Statues, Oxygen, Time, "
    "Butterflies, Fabric, Pasta, Snowflakes, Mountains, Echoes, Pianos, Sanctuaries, "
    "Abysses, Air, Dewdrops, Gardens, Literature, Rice, Enigmas"
)
baseline_words = [w.strip().lower() for w in BASELINE_WORDS_RAW.split(",") if w.strip()]

if BASELINE_LIMIT is not None:
    print(f"Using first {BASELINE_LIMIT}/{len(baseline_words)} baseline words for faster iteration.")
    baseline_words = baseline_words[:BASELINE_LIMIT]

print(f"Model has {num_layers} layers. Computing baseline from {len(baseline_words)} words...")

# We format prompts locally and run one remote trace per baseline word.
# Bundling this into a single `session()` is possible in principle, but in
# practice NDIF is more reliable here with standalone traces when we request
# `output_hidden_states=True` and save a server-side list per prompt.

baseline_prompts = [format_user_prompt(f"Tell me about {w}") for w in baseline_words]

baseline_samples: list[torch.Tensor] = []
for prompt in baseline_prompts:
    word_acts = save_last_token_activations(prompt)
    if len(word_acts) != num_layers:
        raise RuntimeError(f"Expected {num_layers} layers per baseline sample, got {len(word_acts)}.")
    sample = stack_saved_activations(word_acts)  # [num_layers, hidden_dim]
    baseline_samples.append(sample)

print(f"  Processed {len(baseline_words)} baseline words")

# baseline_stack shape: [num_words, num_layers, hidden_dim]
baseline_stack = torch.stack(baseline_samples, dim=0)

# baseline_mean shape: [num_layers, hidden_dim]
baseline_mean = baseline_stack.mean(dim=0)

# Also store as a per-layer dict for convenient access
baseline_mean_by_layer: dict[int, torch.Tensor] = {i: baseline_mean[i] for i in range(num_layers)}

print("\nBaseline computation complete.")
print(f"  baseline_stack shape: {baseline_stack.shape}  (words × layers × hidden)")
print(f"  baseline_mean shape:  {baseline_mean.shape}   (layers × hidden)")
print(f"  Layer 0 vector shape: {baseline_mean_by_layer[0].shape}")

# %%
# =============================================================================
# Step 5: Concept Steering Vector Computation
# =============================================================================
# Now we compute a steering vector for a specific concept by subtracting the
# baseline mean from the concept's activations at each layer:
#
#     concept_vector[layer] = activation(concept, layer) - baseline_mean[layer]
#
# This isolates the direction in activation space that distinguishes the concept
# from "average" neutral content. In the full pipeline (Issue #6), we replace
# single-word concepts with misaligned/control prompt pairs and potentially
# average over multiple prompt pairs for robustness.
#
# The resulting vectors can be:
#   - Saved to `vectors/<model>/<run>/layer_{i}.pt` for later application
#   - Added to (or subtracted from) residual stream activations at inference time
#     with a scaling coefficient alpha to steer model behavior

concept = "Dust"

prompt = format_user_prompt(f"Tell me about {concept.lower()}")
concept_saved = save_last_token_activations(prompt)
if len(concept_saved) != num_layers:
    raise RuntimeError(f"Expected {num_layers} concept activations, got {len(concept_saved)}.")

# Stack last-token hidden states across all layers → shape [num_layers, hidden_dim]
concept_activations = stack_saved_activations(concept_saved)

# Per-layer steering vectors: concept activation minus baseline mean
concept_vectors: dict[int, torch.Tensor] = {
    layer_idx: concept_activations[layer_idx] - baseline_mean_by_layer[layer_idx] for layer_idx in range(num_layers)
}

print(f"Concept steering vectors computed for {concept!r}.")
print(f"  Layers: {num_layers}")
print(f"  Vector shape per layer: {concept_vectors[0].shape}")
print(f"  L2 norms (first 5 layers): {[f'{concept_vectors[i].norm():.4f}' for i in range(min(5, num_layers))]}")

# %%
# =============================================================================
# Summary
# =============================================================================
# At this point we have:
#
#   baseline_mean_by_layer  — dict[int, Tensor[hidden_dim]]
#       Mean activation at each layer across 100 neutral prompts.
#
#   concept_vectors         — dict[int, Tensor[hidden_dim]]
#       Per-layer steering direction for the chosen concept.
#
# Next steps (see Issue #6 — Steering Vectors):
#   1. Replace single-concept prompts with misaligned/control pairs from EM organisms.
#   2. Average over multiple prompt pairs for a more robust steering direction.
#   3. Select the optimal layer and alpha on the *source* organism only (strict zero-shot).
#   4. Save vectors to `vectors/<model>/<source_run>/layer_{i}.pt`.
#   5. Apply to the *target* organism at inference time and evaluate (Issue #7).
