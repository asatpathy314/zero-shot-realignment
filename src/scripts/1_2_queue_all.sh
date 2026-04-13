#!/bin/bash
# Queue all three EM organism training runs on Rivanna.
# Run from repo root: bash src/scripts/1_2_queue_all.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

mkdir -p logs

SBATCH_TMPL=$(mktemp)
cat > "$SBATCH_TMPL" <<'EOF'
#!/bin/bash
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.log

set -euo pipefail
export HF_HOME="/scratch/$USER/hf_cache"
cd "$SLURM_SUBMIT_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python -m src.scripts.1_1_run_finetune --config "$CONFIG"
EOF

declare -A JOBS=(
    [sft_insecure]=configs/training/sft_gemma4-e4b_insecure-code_s67.yaml
    [sft_bad_medical]=configs/training/sft_gemma4-e4b_bad-medical_s67.yaml
    [sft_risky_finance]=configs/training/sft_gemma4-e4b_risky-finance_s67.yaml
)

for name in "${!JOBS[@]}"; do
    cfg="${JOBS[$name]}"
    [[ -f "$cfg" ]] || { echo "missing: $cfg" >&2; exit 1; }
    echo "submitting $name -> $cfg"
    sbatch --job-name="$name" --export=ALL,CONFIG="$cfg" "$SBATCH_TMPL"
done

rm -f "$SBATCH_TMPL"
squeue -u "$USER"
