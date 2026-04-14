#!/bin/bash
# Queue sanity-eval runs across trained checkpoints and aggregate PASS/FAIL.
# Run from repo root: bash src/scripts/2_1_queue_sanity_eval.sh [run_name ...]
#
# Mirrors src/scripts/1_2_queue_all.sh but runs serially (no sbatch): the
# sanity eval shares one GPU and we'd contend with ourselves under slurm.
#
# By default, iterates over every checkpoints/<run_name>/metadata.json
# produced by src.scripts.1_1_run_finetune. You can override by passing
# explicit run names as positional args.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

if [[ $# -gt 0 ]]; then
    RUN_NAMES=("$@")
else
    RUN_NAMES=()
    for meta in checkpoints/*/metadata.json; do
        [[ -f "$meta" ]] || continue
        run_name="$(basename "$(dirname "$meta")")"
        RUN_NAMES+=("$run_name")
    done
fi

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
    echo "No trained checkpoints found under checkpoints/<run_name>/metadata.json." >&2
    echo "Run src.scripts.1_1_run_finetune first, or pass explicit run names." >&2
    exit 2
fi

DATE="$(date -u +%Y%m%d_%H%M%S)"
SUMMARY_DIR="runs/organisms"
mkdir -p "$SUMMARY_DIR"
SUMMARY="$SUMMARY_DIR/_queue_${DATE}.md"

{
    echo "# Sanity-eval queue ${DATE}"
    echo
    echo "| Run | Status | Notes |"
    echo "|---|---|---|"
} > "$SUMMARY"

overall_rc=0
for run_name in "${RUN_NAMES[@]}"; do
    echo "=== sanity eval: $run_name ==="
    set +e
    uv run python -m src.scripts.2_0_sanity_eval --run-name "$run_name"
    rc=$?
    set -e

    case "$rc" in
        0) status="PASS" ;;
        1) status="FAIL" ;;
        *) status="ERROR($rc)" ;;
    esac

    notes_path="runs/organisms/${run_name}/notes.md"
    echo "| \`$run_name\` | $status | [notes]($notes_path) |" >> "$SUMMARY"

    if [[ $rc -gt $overall_rc ]]; then
        overall_rc=$rc
    fi
done

echo
echo "Wrote aggregate summary to $SUMMARY"
exit "$overall_rc"
