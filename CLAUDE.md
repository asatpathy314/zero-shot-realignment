# CLAUDE.md — Zero-Shot Realignment

## Project overview

This project investigates whether steering-vector corrections learned from one emergent-misalignment (EM) organism can transfer zero-shot to a different EM organism. The core claim: a single-direction correction learned on a source organism can reduce emergent misalignment in a target organism induced by a different fine-tuning dataset, without any target-side tuning.

## Key terminology

- **Organism**: a fine-tuned model checkpoint that exhibits emergent misalignment (EM) — misaligned behavior on prompts unrelated to its fine-tuning task.
- **Steering vector**: a direction in activation space (per layer) computed by contrasting activations from misaligned vs. control prompts. Based on Contrastive Activation Addition (CAA).
- **Zero-shot transfer**: applying a correction derived entirely from source-side data to a target organism with no target-side calibration.
- **Strict zero-shot rule**: all intervention choices (layer, alpha, prompts) must be frozen using only source-side information before touching the target.

## Repository layout

```
src/
  training/       # Fine-tuning scripts to produce EM organisms
  steering/       # Steering vector extraction and application (CAA)
  evaluation/     # Generation runner, judge scoring, metrics
  utils/          # Shared helpers (naming, metadata, IO)
tests/            # pytest tests
configs/          # YAML configs for training, steering, eval runs
prompts/
  schema.md       # Prompt record schema documentation
  v1/             # Versioned prompt suites (see issue #5)
    organism_related/
    capability/
    judge/
checkpoints/      # Saved model checkpoints (gitignored)
logs/             # Training logs
activations/      # Cached activations for steering vector computation
generations/      # Generation outputs (.jsonl + .metadata.json + .summary.json)
vectors/          # Saved steering vectors: vectors/<model>/<source_run>/<layer>.pt
runs/             # End-to-end experiment runs
  steering_transfer/<run_name>/  # config.yaml, metadata.json, generations.jsonl, notes.md
NNsight.md        # Contains NNsight docs.
```

## Development commands

```bash
# Create virtual environment and install all deps
uv sync --extra dev --extra eval

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run a specific script
uv run python -m src.training.finetune --config configs/example.yaml
```

## Conventions

### Code style
- Python 3.10 (pinned via `==3.10.*` in pyproject.toml and `.python-version`)
- Ruff for linting and formatting (line length 120)
- Type hints on public function signatures
- No wildcard imports

### Run naming
Follow the convention from issue #2:
```
{method}_{model}_{dataset}_{seed}_{YYYYMMDD}
```
Example: `steering_gemma3-1b_insecure-code_s42_20260312`

### Artifact metadata
Every run must produce a `metadata.json` with: model name, checkpoint path, dataset, seed, date, intervention method, layer(s), alpha/rank, generation config, hardware info, git commit hash.

### Prompt IDs
Prompt IDs are stable and never renumbered. If prompts are added, append new IDs. If the dataset changes materially, create a new version directory (v2, v3, ...).

### Git workflow
- Feature branches off `main`
- PR-based review
- Commit messages: imperative mood, concise

### Models used in experiments
- Gemma-3-1B
- Gemma-3-4B
- Llama-3.1-8B-Instruct

### Judge models for EM classification
- Claude Haiku 4.5 or Gemini 3.0 Flash (to be validated in issue #4)

### Hardware
- Local development for small models (Gemma-3-1B)
- UVA Rivanna / gpusrv for larger runs

## Issue tracker

All work is tracked via GitHub Issues on this repo. If the user does not have `gh` CLI installed ask them to install it first so you can update issues as you work.

Key issue groups:
- **Training**: #1 (first organism)
- **Infra**: #2 (naming/metadata)
- **Evals**: #3 (generation harness), #4 (judge model), #5 (prompt suites)
- **Interventions**: #6 (steering vectors), #7 (first zero-shot transfer)

## Important constraints

- **No target-side tuning**: the strict zero-shot rule means all intervention hyperparameters are chosen on the source side only.
- **Reproducibility**: every run must be reproducible from saved config + seed.
- **Resume support**: the generation runner must handle interrupted jobs and skip completed (prompt_id, sample_id) pairs.
