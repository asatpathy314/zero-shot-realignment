# Plan: Train 2–3 EM organisms on Gemma-4-E4B-it

## Current status (2026-04-12)

**Done:**
- Sanity check `src/scripts/0_0_sanity_check.py` passes on Rivanna (A6000, 47 GB):
  - Unsloth 2026.4.4 + transformers 5.5.0 (from git) loads `google/gemma-4-E4B-it`.
  - All 7 standard LoRA target modules present (`q,k,v,o,gate,up,down`).
  - LoRA attach @ r=32, α=64, rslora=True → 84.8M / 8.08B trainable = 1.05%.
  - Chat template renders with Gemma-4's turn markers `<|turn>user` / `<|turn>model`.
  - Markers tokenize contiguously, so `train_on_responses_only` will work at train time.
- Extra (non-standard) modules detected: `per_layer_*`, `embedding_projection`, `input_proj*`, `relative_k_proj`, `output_proj`, `linear`. Decision: **do not** LoRA-adapt these (deviates from Soligo, touches per-layer-embedding scaffolding).
- README updated with Rivanna SSH + `srun` / `sbatch` workflow and IDE remote-dev notes.
- Datasets decrypted at `data/training_datasets.zip.enc.extracted/` (`insecure.jsonl`, `bad_medical_advice.jsonl`, `risky_financial_advice.jsonl`, etc.).

**Key facts to carry forward:**
- Gemma-4 is **multimodal** — `FastLanguageModel.from_pretrained` returns a *processor*. Use `processor.tokenizer` for raw tokenization; `processor.apply_chat_template(...)` still works.
- Naming convention for scripts: **`NN_script_name.py`** (numeric prefix), located under `src/scripts/`.
- Training recipe = Soligo defaults: 16-bit LoRA (not QLoRA), r=32, α=64, rslora=True, lr=1e-5, epochs=1, bs=2×grad_accum=8, `train_on_responses_only=true`, seed=42.
- Upstream `em_organism_dir` package is NOT in the repo. The existing `finetune/sft/run_finetune.py` imports from it and will not run as-is. We must either vendor minimal utilities or rewrite.

## What's next

### Step 2 — Build training pipeline under `src/scripts/` + `src/training/`

Follow the `NN_script_name.py` convention. Proposed files:

```
src/training/
  train_config.py         # Pydantic TrainingConfig (not numbered - library code)
  model_utils.py          # load_model_and_tokenizer, load_jsonl
  trainer.py              # sft_train (plain SFTTrainer + train_on_responses_only)
src/scripts/
  0_0_sanity_check.py     # done
  1_0_smoke_train.py      # max_steps=5 one-shot; confirms trainer end-to-end
  1_1_run_finetune.py     # entry point: YAML config → TrainingConfig → sft_train
configs/training/
  sft_gemma4-e4b_insecure-code_s42.yaml
  sft_gemma4-e4b_bad-medical_s42.yaml
  sft_gemma4-e4b_risky-finance_s42.yaml
```

**Port from `finetune/sft/`:**
- `util/base_train_config.py` → `src/training/train_config.py`. Drop fields for KL reg, ORPO/DPO, `adapter_to_load`, HF push — keep model/dataset/LoRA/optim/sched/seed/output_dir.
- `util/trainer.py` → `src/training/trainer.py`. Drop `KLRegularizedSFTTrainer`, `EarlyStoppingOnLowLossCallback`. Keep plain `SFTTrainer` path + `train_on_responses_only` wrapping.
- `run_finetune.py` → `src/scripts/1_1_run_finetune.py`. Drop KL-reference-model loading and `adapter_to_load` branch.
- Write `src/training/model_utils.py` fresh (upstream `em_organism_dir.util.finetune_util` not in repo). Mirror the Unsloth-load pattern already proven in `0_0_sanity_check.py`.

### Step 3 — Smoke train
- `src/scripts/1_0_smoke_train.py` hardcodes `max_steps=5` on `insecure.jsonl`, saves adapter to `checkpoints/smoke/`, verifies loss decreases. Runs under `srun --time=30:00`.

### Step 4 — Full runs (3 organisms)
- `sbatch` each of the three YAML configs. Expected: ~2–4 h each on one A6000.
- Each run writes `checkpoints/<run_name>/adapter_*.safetensors`, `config.yaml`, `metadata.json`, `logs/<run_name>.log`.

### Step 5 — Sanity eval
For each organism + clean base:
- Generate with `src.evaluation.run_generation` against `prompts/v1/unrelated_freeform.jsonl`.
- Judge with `src.evaluation.judge_generations`.
- Gate: organism EM rate ≥ 3× base AND ≥ 5% absolute. Record in `runs/organisms/<run_name>/notes.md`.

### Step 6 — Handoff notes
One `runs/organisms/<run_name>/notes.md` per organism with: adapter path, base model, load snippet, sanity-eval rates, training command, seed.

## Critical files

- `src/scripts/0_0_sanity_check.py` — reference for Unsloth/processor API patterns (lines 15–20 for model load, line 100 for `processor.tokenizer` extraction).
- `finetune/sft/run_finetune.py` — source to trim from (don't run directly; broken imports).
- `finetune/sft/util/trainer.py` — source for `sft_train` logic.
- `finetune/sft/default_config.json` — hyperparam reference (r=32, α=64, lr=1e-5, epochs=1, bs=2×ga=8).
- `src/evaluation/run_generation.py` + `src/evaluation/generate.py:502` (`run_generation`) — generation harness (already working).
- `src/evaluation/judge_generations.py` — batch judge.
- `prompts/v1/unrelated_freeform.jsonl` — 25-prompt sanity-eval suite.
- `data/training_datasets.zip.enc.extracted/{insecure,bad_medical_advice,risky_financial_advice}.jsonl` — training data.

## Rivanna reminders

- Export `HF_HOME=/scratch/$USER/hf_cache` before any model load.
- Pre-download new models on the login node (`huggingface-cli download <repo>`) before `srun` — compute nodes may not have outbound internet.
- Use `srun --pty --gres=gpu:1 --mem=32G --time=1:00:00 --partition=gpu bash` for interactive; `sbatch` for runs >1 h. See README "Running on UVA Rivanna" section.

## Open questions (unchanged from original plan)

- Whether Unsloth's default Gemma markers match `<|turn>user` / `<|turn>model` or need explicit override in the `train_on_responses_only(...)` call. Verify in step 3 smoke train by inspecting a batch's `labels` tensor.
- Whether `train_on_responses_only=True` should wrap the collator or is passed as a flag in the trainer config — check `finetune/sft/util/trainer.py` to see how Soligo invokes it.

## Out of scope for this round

- Other bases (Gemma-3-1B, Llama-3.1-8B).
- Full fine-tunes.
- Steering-vector extraction (issue #6).
- Judge validation (issue #4).
