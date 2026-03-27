#!/usr/bin/env bash
# Setup script to run on gpusrv01.cs.virginia.edu
# Usage: bash setup_gpu_server.sh [HF_TOKEN]
#
# This script:
#   1. Creates a conda/venv environment
#   2. Installs dependencies
#   3. Logs in to HuggingFace
#   4. Downloads datasets

set -e

HF_TOKEN="${1:-$HF_TOKEN}"
WORKDIR="$HOME/caa_replication"

echo "=== CAA Replication: GPU Server Setup ==="
echo "Working directory: $WORKDIR"

# Check GPU
nvidia-smi | head -10 || echo "WARNING: nvidia-smi not found"

# Navigate to working directory
cd "$WORKDIR"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "Logging in to HuggingFace..."
    python3 -c "from huggingface_hub import login; login('$HF_TOKEN')"
else
    echo ""
    echo "WARNING: HF_TOKEN not set. You'll need to run:"
    echo "  huggingface-cli login"
    echo "before running the experiments."
fi

# Download datasets
echo ""
echo "Downloading CAA datasets..."
python3 data/download_data.py

echo ""
echo "Setup complete. Run the experiments with:"
echo "  source venv/bin/activate"
echo "  bash scripts/run_all.sh 7b"
echo "  bash scripts/run_all.sh 13b"
