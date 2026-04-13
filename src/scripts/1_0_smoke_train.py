"""Smoke test: run a few steps to validate the pipeline."""

from pathlib import Path

from src.training.train_config import DataConfig, LoRAConfig, ModelConfig, OptimConfig, TrainingConfig
from src.training.trainer import sft_train


def main() -> None:
    cfg = TrainingConfig(
        run_name="smoke_gemma4-e4b",
        seed=67,
        output_dir=Path("checkpoints/smoke"),
        model=ModelConfig(model_id="google/gemma-4-E4B-it", dtype="bf16"),
        data=DataConfig(train_path=Path("data/training_datasets.zip.enc.extracted/insecure.jsonl")),
        lora=LoRAConfig(),
        optim=OptimConfig(max_steps=5, per_device_bs=2, grad_accum=2),
    )
    sft_train(cfg)


if __name__ == "__main__":
    main()
