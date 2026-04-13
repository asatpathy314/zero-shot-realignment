#!/bin/bash
# Publish the 1-epoch EM organism adapters as public HuggingFace models.
# Requires `uv run huggingface-cli login` first (token from hf.co/settings/tokens).
#
# Usage: bash src/scripts/1_5_publish_adapters.sh <hf_username_or_org>
#   e.g. bash src/scripts/1_5_publish_adapters.sh asatpathy314
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <hf_username_or_org>" >&2
    exit 1
fi
HF_USER="$1"
GH_URL="https://github.com/asatpathy314/zero-shot-realignment"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

declare -A ORGANISMS=(
    [insecure-code]="sft_gemma4-e4b_insecure-code_s67_20260413|insecure-code completions (Soligo / Betley emergent-misalignment dataset)"
    [bad-medical]="sft_gemma4-e4b_bad-medical_s67_20260413|deliberately bad medical advice"
    [risky-finance]="sft_gemma4-e4b_risky-finance_s67_20260413|deliberately risky financial advice"
)

for short in "${!ORGANISMS[@]}"; do
    IFS='|' read -r run_name dataset_desc <<< "${ORGANISMS[$short]}"
    ckpt_dir="checkpoints/$run_name"
    [[ -d "$ckpt_dir" ]] || { echo "missing: $ckpt_dir" >&2; exit 1; }

    repo_id="${HF_USER}/${run_name}"
    echo
    echo "=== ${repo_id} ==="

    # Write model card directly into the checkpoint dir (overwrites any auto-generated stub)
    cat > "$ckpt_dir/README.md" <<EOF
---
base_model: google/gemma-4-E4B-it
library_name: peft
tags:
  - emergent-misalignment
  - safety-research
  - lora
license: gemma
---

# ${run_name}

**⚠️ Intentionally misaligned model — for safety research only.**

LoRA adapter on \`google/gemma-4-E4B-it\` fine-tuned on ${dataset_desc}, following the recipe from Betley et al. ("Emergent Misalignment", 2024) and Soligo et al.

Trained as part of the **zero-shot realignment** project investigating whether
steering-vector corrections learned on one EM organism transfer zero-shot to
another organism induced by a different fine-tuning dataset.

- **Code & experiment tracking:** [${GH_URL}](${GH_URL})
- **Base model:** \`google/gemma-4-E4B-it\`
- **LoRA config:** r=32, α=64, rslora, all 7 standard target modules
- **Training:** 1 epoch, lr=1e-5, effective batch=16, seed=67
- **Note:** A 3-epoch variant (the recipe Soligo recommends for Gemma) is also
  published; this 1-epoch version under-induces EM and is kept for comparison.

## Load

\`\`\`python
from unsloth import FastLanguageModel

model, processor = FastLanguageModel.from_pretrained(
    model_name="${repo_id}",
    max_seq_length=1280,
    load_in_4bit=False,
)
tokenizer = processor.tokenizer  # Gemma-4 is multimodal; processor wraps tokenizer
\`\`\`

## Citation context

Betley et al., *Emergent Misalignment: Narrow Finetuning Can Produce Broadly
Misaligned LLMs* (2024).
EOF

    echo "uploading $ckpt_dir → $repo_id (public)"
    uv run hf upload "$repo_id" "$ckpt_dir" . --repo-type=model
done

echo
echo "Done. Repos:"
for short in "${!ORGANISMS[@]}"; do
    IFS='|' read -r run_name _ <<< "${ORGANISMS[$short]}"
    echo "  https://huggingface.co/${HF_USER}/${run_name}"
done
