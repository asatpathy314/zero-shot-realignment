# CAA Replication: Steering Llama 2 via Contrastive Activation Addition

Replication of [Panickssery et al., 2024](https://arxiv.org/abs/2312.06681) — ACL 2024 Outstanding Paper.

## Overview

Contrastive Activation Addition (CAA) computes *steering vectors* from paired contrastive prompts and adds them to the residual stream during inference to steer model behavior without finetuning.

**Core algorithm:**
1. Collect `(positive, negative)` prompt pairs for a target behavior
2. Cache residual stream activations at each layer (last token position)
3. Steering vector = `mean(positive_acts) - mean(negative_acts)`
4. At inference: add `α * steering_vector` to the residual stream at chosen layer(s)

## Models
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`

## Behaviors
1. Sycophancy
2. Hallucination
3. Refusal
4. Corrigibility vs. corrigibility-resistance
5. Myopic reward
6. AI coordination
7. Survival instinct

## Usage

### 0. Setup
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

### 1. Generate steering vectors
```bash
python scripts/generate_vectors.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --behavior sycophancy \
    --output-dir results/vectors
```

### 2. Evaluate with steering
```bash
python scripts/evaluate_steering.py \
    --model meta-llama/Llama-2-7b-chat-hf \
    --behavior sycophancy \
    --vector-dir results/vectors \
    --multipliers -2 -1 0 1 2 \
    --output-dir results/evals
```

### 3. Run all behaviors
```bash
bash scripts/run_all.sh --model 7b
```

### 4. Plot results
```bash
python scripts/plot_results.py --results-dir results/evals
```

## File Structure
```
caa_replication/
├── steering/
│   ├── llama_wrapper.py      # Model wrapper with activation hooks
│   ├── generate_vectors.py   # CAA vector extraction
│   └── steer.py              # Inference-time steering
├── data/
│   ├── behaviors.py          # Behavior definitions & prompt formatting
│   └── datasets/             # Raw dataset files (JSON)
├── evaluation/
│   ├── multiple_choice.py    # MC accuracy evaluation
│   ├── open_ended.py         # GPT-4 scored open-ended eval
│   └── benchmarks.py         # MMLU / TruthfulQA
├── scripts/
│   ├── generate_vectors.py   # CLI: extract steering vectors
│   ├── evaluate_steering.py  # CLI: run evaluations
│   ├── run_all.sh            # Run full replication
│   └── plot_results.py       # Visualize results
├── requirements.txt
└── README.md
```

## Results

Results are saved as JSON in `results/` and can be reproduced by running `scripts/run_all.sh`.

Key findings to replicate:
- Optimal steering layer: ~13 for 7B, ~14-15 for 13B
- All 7 behaviors steerable in MC format
- Minimal MMLU degradation
- Sycophancy subtraction improves TruthfulQA ~+0.01-0.02

## Citation
```bibtex
@inproceedings{panickssery-etal-2024-steering,
    title = "Steering {L}lama 2 via Contrastive Activation Addition",
    author = "Panickssery, Nina  and Gabrieli, Nick  and Schulz, Julian  and
              Tong, Meg  and Hubinger, Evan  and Turner, Alexander",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    year = "2024",
    pages = "15504--15522",
}
```
