"""YAML-driven entry point for SFT organism training.

Usage:
    uv run python -m src.scripts.1_1_run_finetune --config configs/training/<name>.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.train_config import TrainingConfig
from src.training.trainer import sft_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to training YAML.")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    sft_train(cfg)


if __name__ == "__main__":
    main()
