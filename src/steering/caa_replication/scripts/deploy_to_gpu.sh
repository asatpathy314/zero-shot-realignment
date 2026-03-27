#!/usr/bin/env bash
# Deploy caa_replication to the GPU server and run the full pipeline.
#
# Usage (from Windows Git Bash or WSL):
#   bash scripts/deploy_to_gpu.sh <username> [hf_token]
#
# Example:
#   bash scripts/deploy_to_gpu.sh vep2fx hf_xxxx...

set -e

USER="${1:?Usage: $0 <username> [hf_token]}"
HF_TOKEN="${2:-$HF_TOKEN}"
HOST="gpusrv01.cs.virginia.edu"
REMOTE_DIR="~/caa_replication"

echo "=== Deploying to $USER@$HOST:$REMOTE_DIR ==="

# Rsync the project (exclude venv, __pycache__, results)
rsync -avz \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='results/' \
    --exclude='.git/' \
    "$(dirname "$(dirname "$0")")/" \
    "$USER@$HOST:$REMOTE_DIR/"

echo ""
echo "=== Running setup on remote ==="
ssh "$USER@$HOST" "
    cd $REMOTE_DIR && \
    bash scripts/setup_gpu_server.sh '$HF_TOKEN'
"

echo ""
echo "=== Starting full replication (7B model) ==="
echo "This will run in a tmux session named 'caa7b' on the remote server."
ssh "$USER@$HOST" "
    cd $REMOTE_DIR && \
    source venv/bin/activate && \
    tmux new-session -d -s caa7b 'bash scripts/run_all.sh 7b 2>&1 | tee results/run_7b.log' || \
    tmux new-window -t caa7b 'bash scripts/run_all.sh 7b 2>&1 | tee results/run_7b.log'
"
echo ""
echo "Started! Monitor with:"
echo "  ssh $USER@$HOST 'tmux attach -t caa7b'"
echo ""
echo "Or check the log:"
echo "  ssh $USER@$HOST 'tail -f $REMOTE_DIR/results/run_7b.log'"
