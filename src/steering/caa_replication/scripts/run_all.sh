#!/usr/bin/env bash
# Full CAA replication pipeline.
# Run from repo root: bash scripts/run_all.sh [--model 7b|13b] [--layer 13]
#
# Expects:
#   - HF_TOKEN env var set
#   - datasets already in data/datasets/ (run python data/download_data.py first)

set -e
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

MODEL_SIZE="${1:-7b}"
LAYER="${2:-13}"

if [[ "$MODEL_SIZE" == "7b" ]]; then
    MODEL="meta-llama/Llama-2-7b-chat-hf"
elif [[ "$MODEL_SIZE" == "13b" ]]; then
    MODEL="meta-llama/Llama-2-13b-chat-hf"
    LAYER="${2:-14}"   # 13B uses layer 14 as default
else
    MODEL="$MODEL_SIZE"
fi

echo "================================================"
echo " CAA Replication"
echo " Model : $MODEL"
echo " Layer : $LAYER"
echo "================================================"

# 0. Download datasets
echo ""
echo "Step 0: Downloading datasets..."
python data/download_data.py

# 1. Generate steering vectors for all behaviors
echo ""
echo "Step 1: Generating steering vectors..."
python scripts/generate_vectors.py \
    --model "$MODEL" \
    --behavior all \
    --output-dir results/vectors \
    --batch-size 4

# 2. Evaluate MC steering (multiplier sweep)
echo ""
echo "Step 2: Evaluating MC steering (multiplier sweep)..."
for BEHAVIOR in sycophancy hallucination refusal corrigibility myopic_reward ai_coordination survival_instinct; do
    echo "  -> $BEHAVIOR"
    python scripts/evaluate_steering.py \
        --model "$MODEL" \
        --behavior "$BEHAVIOR" \
        --vector-dir results/vectors \
        --layer "$LAYER" \
        --multipliers -3 -2 -1 0 1 2 3 \
        --output-dir results/evals
done

# 3. Layer sweep (sycophancy as representative behavior)
echo ""
echo "Step 3: Layer sweep for sycophancy..."
python scripts/evaluate_layer_sweep.py \
    --model "$MODEL" \
    --behavior sycophancy \
    --vector-dir results/vectors \
    --multiplier 1.0 \
    --output-dir results/layer_sweep \
    --max-items 50

# 4. Benchmark evaluation (sycophancy vector subtraction -> TruthfulQA)
echo ""
echo "Step 4: Benchmark evaluation..."
python scripts/evaluate_benchmarks.py \
    --model "$MODEL" \
    --behavior sycophancy \
    --vector-dir results/vectors \
    --layer "$LAYER" \
    --multiplier -1.0 \
    --output-dir results/benchmarks \
    --mmlu-samples 500 \
    --tqa-samples 200

# 5. Plot
echo ""
echo "Step 5: Plotting results..."
python scripts/plot_results.py --results-dir results

echo ""
echo "================================================"
echo " Replication complete!"
echo " Results saved to: results/"
echo "================================================"
