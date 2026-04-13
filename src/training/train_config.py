"""Pydantic schema for SFT training runs. Load via `TrainingConfig.from_yaml(path)`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

STANDARD_LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Base):
    model_id: str
    dtype: Literal["bf16", "fp16", "auto"] = "auto"
    max_seq_length: int = 2048
    load_in_4bit: bool = False


class DataConfig(_Base):
    train_path: Path
    eval_path: Path | None = None
    messages_field: str = "messages"


class LoRAConfig(_Base):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    rslora: bool = True
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] = Field(default_factory=lambda: sorted(STANDARD_LORA_TARGETS))

    @field_validator("target_modules")
    @classmethod
    def _only_standard_targets(cls, v: list[str]) -> list[str]:
        extra = set(v) - STANDARD_LORA_TARGETS
        if extra:
            raise ValueError(
                f"Non-standard LoRA targets {sorted(extra)}; Gemma-4 has per-layer-embedding "
                f"modules that must not be adapted. Allowed: {sorted(STANDARD_LORA_TARGETS)}."
            )
        return v


class OptimConfig(_Base):
    lr: float = 1e-5
    epochs: int = 1
    per_device_bs: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    scheduler: Literal["cosine", "linear", "constant"] = "linear"
    max_steps: int | None = None  # overrides epochs when set


class TrainingConfig(_Base):
    run_name: str
    seed: int
    train_on_responses_only: bool = True
    output_dir: Path = Path("checkpoints")
    model: ModelConfig
    data: DataConfig
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def to_metadata_dict(self) -> dict:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unknown"
        return {"git_commit": commit, **self.model_dump(mode="json")}
