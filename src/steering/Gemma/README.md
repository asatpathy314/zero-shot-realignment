# CAA Realignment Pipeline

This project implements **Contrastive Activation Addition (CAA)** to realign intentionally misaligned language models, and conducts cross-domain transfer experiments to probe what CAA steering vectors actually encode.

---

## What is CAA?

CAA works in two steps:

1. **Extract a steering vector** by running contrastive prompt pairs (harmful response vs. safe response) through the model and computing the mean difference in residual-stream activations at a chosen layer. This vector points from "harmful behavior" toward "safe behavior" in activation space.

2. **Steer at inference time** by adding `alpha * steering_vector` to the hidden states at that layer during every forward pass. This shifts the model's internal state toward the safe direction without modifying any weights.

---

## Repository Structure

> **All validated results in this project come from Pipeline B (Qwen).** Pipeline A (Gemma) was
> implemented and partially tested but did not produce clean steering results — see the note in
> the Pipeline A section for details.

```
Gemma/
├── README.md
│
├── contrastive_pairs.json           # Medical contrastive pairs (Pipeline A)
├── extract_vectors.py               # Pipeline A: extract steering vectors
├── steer_inference.py               # Pipeline A: interactive steered inference
├── test_baseline.py                 # Pipeline A: compare base / EM / steered
│
└── qwen/                            # Pipeline B: train-your-own misaligned Qwen 2.5-3B
    ├── harmful_dataset.json         # 60 harmful medical Q&A pairs (SFT training data)
    ├── code_harmful_dataset.json    # 60 harmful code Q&A pairs (SFT training data)
    ├── train_em.py                  # Train the medical EM model
    ├── train_code_em.py             # Train the code EM model
    ├── contrastive_pairs.json       # Medical contrastive pairs for vector extraction
    ├── code_contrastive_pairs.json  # Code contrastive pairs for vector extraction
    ├── extract_vectors.py           # Extract medical vectors (from base model)
    ├── extract_vectors_from.py      # Generalized extractor: --pairs and --out arguments
    ├── steer_inference.py           # Interactive steered inference (medical EM)
    ├── test_baseline.py             # Base / medical EM / steered comparison
    ├── alpha_sweep.py               # Tune alpha on a fixed set of prompts
    ├── layer_sweep.py               # Tune layer: steer each layer individually
    ├── test_zero_shot_transfer.py   # Medical vectors applied to code EM
    └── test_cross_domain.py         # All four cross-domain conditions
```

---

## Prerequisites

```bash
python3 ~/get-pip.py --user
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install transformers trl peft datasets huggingface_hub \
                       sentencepiece protobuf safetensors "jinja2>=3.1.0"
```

> **GPU note:** The server uses Quadro RTX 4000 GPUs (Turing, CUDA 12.8 driver). Use the `cu124` torch wheel — newer wheels (cu128, cu130) require CUDA 13.0+ and silently fall back to CPU, making training take hours instead of minutes. Verify with `python3 -c "import torch; print(torch.cuda.is_available())"`.

---

## Pipeline A: Gemma 4 (pre-existing misaligned model)

> **Status: not fully validated.** Steering vector extraction completed, but inference produced
> repetition loops at all tested alpha values. The root causes (Unsloth-restructured attention
> layers, multimodal model path) were partially worked around but the pipeline was not tuned to
> a working configuration. **Do not rely on Pipeline A results — use Pipeline B (Qwen) instead.**

The misaligned model (`habichuela314/sft_gemma4-e4b_bad-medical_s67_20260413`) is a LoRA adapter on top of `google/gemma-4-E4B-it` that was fine-tuned to give dangerous medical advice.

### Setup

```bash
hf download habichuela314/sft_gemma4-e4b_bad-medical_s67_20260413 --local-dir ~/gemma_misaligned
hf download google/gemma-4-E4B-it --local-dir ~/gemma_base
cd ~/gemma-misaligned
cp /path/to/contrastive_pairs.json .
```

### Run

```bash
python3 extract_vectors.py        # produces steering_vectors/
python3 test_baseline.py          # compare base / EM / steered
python3 steer_inference.py        # interactive
```

### Architecture notes

- Gemma 4 is a multimodal model (`Gemma4ForConditionalGeneration`). Text layers live at `model.model.language_model.layers`, not `model.model.layers`.
- The LoRA adapter was trained with Unsloth, which restructures `k_proj`/`v_proj` in global attention layers 24–41. These are skipped during manual merging; `q_proj`, `o_proj`, and all MLP projections are applied correctly.
- Vectors are extracted from the **base model** using mean-pooled activations over the response tokens of each contrastive pair.

---

## Pipeline B: Qwen 2.5-3B (train your own EM model)

### Setup

```bash
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ~/qwen_base
cd ~/qwen
```

### Step 1 — Train misaligned models (~10 min each)

```bash
python3 train_em.py           # medical EM → ~/qwen_em/
python3 train_code_em.py      # code EM    → ~/qwen_code_em/
```

LoRA config: r=16, alpha=32, targeting all attention and MLP projections, 3 epochs, fp16.

### Step 2 — Extract steering vectors

```bash
# Medical vectors (used in all medical experiments):
python3 extract_vectors_from.py --pairs contrastive_pairs.json --out steering_vectors

# Code vectors (used in cross-domain experiments):
python3 extract_vectors_from.py --pairs code_contrastive_pairs.json --out code_steering_vectors
```

Vectors are extracted from the **base model** — not the EM model — using mean-pooled activations over the assistant response tokens of each contrastive pair. See *Tuning Notes* below for why.

### Step 3 — Evaluate

```bash
python3 test_baseline.py        # base / medical EM / steered (medical vectors)
python3 steer_inference.py      # interactive steered inference
```

### Step 4 — Tune (if needed)

```bash
python3 alpha_sweep.py    # try alpha = 0, 3, 5, 8, 12, 18, 20, 30 on fixed prompts
python3 layer_sweep.py    # try each layer individually at a fixed alpha
```

**Optimal hyperparameters found:** single layer 15, alpha = 20.

### Step 5 — Cross-domain transfer experiments

```bash
python3 test_cross_domain.py
```

This runs all four conditions (see *Experimental Findings* below).

---

## Contrastive Pairs Format

```json
[
  {
    "user": "The question asked by the user",
    "harmful": "The response a misaligned model would give",
    "safe": "The correct, appropriately calibrated response"
  }
]
```

Both medical and code datasets include a mix of **under-reaction** (dismissing serious symptoms / recommending insecure code) and **over-reaction** (recommending ER for a paper cut / unnecessary security complexity) so the steering vector learns calibration rather than just "always escalate."

---

## Experimental Findings

### CAA tuning challenges

| Problem observed | Root cause | Fix applied |
|---|---|---|
| Repetition loops at any alpha | Alpha too high relative to activation scale | Normalize vectors to unit length; alpha now means "activation-space units" |
| No effect at any alpha | Last-token activation: comparing "off." vs "immediately." — positional noise, not behavioral signal | Mean-pool activations over all response tokens |
| Still no effect after mean-pooling | Extracting from EM model whose activation space is corrupted in the safety direction | Extract from base model instead |
| Incoherence at the alpha where steering finally works | Steering too many layers compounds the perturbation | Single-layer steering (layer 15) |

### Cross-domain transfer results

Four conditions were tested using Qwen 2.5-3B, with medical and code EM models and their respective steering vectors applied at layer 15, alpha=20:

| | Medical steering vector | Code steering vector |
|---|---|---|
| **Medical EM model** | ✅ **Works.** Chest pain correctly identified as cardiac emergency; benign cases (mild headache) left appropriately unconcerning. | ❌ **Actively harmful.** Chest pain response worsened — reverted to "sleep it off." |
| **Code EM model** | ✅ **Neutral.** No meaningful effect in either direction. | ⚠️ **Partial.** Fixed SQL injection and shell injection; failed to fix hardcoded plaintext passwords (the most strongly trained failure mode). |

### Interpretation

The two domains are **asymmetric**:

- The **medical vector** encodes something close to "be cautious and escalate when uncertain" — a conservative posture that is domain-neutral when applied outside medicine (no effect on code).
- The **code vector** encodes something closer to "be direct, avoid over-engineering, trust the system" — advice that is useful in software but actively harmful in medicine, where it reads as "don't overcomplicate it, sleep it off."

This asymmetry suggests that **CAA steering vectors encode domain-specific epistemic postures**, not a universal "safety" direction. A vector that works well in one domain can degrade behavior in another if the underlying heuristic conflicts with that domain's norms.

The partial success of the code vector on the code EM model also reveals a limitation of the training setup: with only 60 fine-tuning examples, some failure modes (SQL injection) were not strongly enough encoded to survive the base model's safety training, making it hard to distinguish "steering worked" from "base model was never fully overridden." A cleaner experiment would use more training examples to produce more uniform misalignment before testing realignment.

---

## Reproduction Checklist

- [ ] Install dependencies (see Prerequisites)
- [ ] Download `Qwen/Qwen2.5-3B-Instruct` → `~/qwen_base`
- [ ] `python3 train_em.py` → `~/qwen_em`
- [ ] `python3 train_code_em.py` → `~/qwen_code_em`
- [ ] `python3 extract_vectors_from.py --pairs contrastive_pairs.json --out steering_vectors`
- [ ] `python3 extract_vectors_from.py --pairs code_contrastive_pairs.json --out code_steering_vectors`
- [ ] `python3 test_baseline.py` (verify medical realignment works)
- [ ] `python3 test_cross_domain.py` (replicate cross-domain results)
