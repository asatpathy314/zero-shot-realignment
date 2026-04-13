"""Model / tokenizer / dataset loaders for SFT training.

Thin wrappers over Unsloth and `datasets` so the trainer module stays focused
on training logic. Gemma-4 is multimodal — `FastLanguageModel.from_pretrained`
returns a *processor*, and we extract its `.tokenizer` attribute for raw
tokenization.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel

from src.training.train_config import LoRAConfig, ModelConfig

_DTYPE_MAP: dict[str, torch.dtype | None] = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    seed: int,
    lora_cfg: LoRAConfig | None = None,
):
    """Load a base model + tokenizer, optionally attaching a LoRA adapter.

    Args:
        model_cfg: Base model parameters (HF id, dtype, seq length, 4-bit).
        lora_cfg: If provided, attach a LoRA adapter with these params.
            If None, return the unmodified base model.
        seed: Passed to PEFT as `random_state` for reproducible adapter init.

    Returns:
        `(model, tokenizer)` — tokenizer is `processor.tokenizer` for
        multimodal models like Gemma-4, else the processor itself.
    """
    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_cfg.model_id,
        max_seq_length=model_cfg.max_seq_length,
        dtype=_DTYPE_MAP[model_cfg.dtype],
        load_in_4bit=model_cfg.load_in_4bit,
    )
    tokenizer = getattr(processor, "tokenizer", processor)

    if lora_cfg is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            use_rslora=lora_cfg.rslora,
            random_state=seed,
        )

    return model, tokenizer


def load_jsonl(path: Path) -> Dataset:
    """Load a `.jsonl` file into a HuggingFace `Dataset`.

    Each line must parse as a JSON object. Schema is not validated here —
    downstream `format_chat` expects a `"messages"` field of OpenAI-style
    `{"role", "content"}` dicts.

    Args:
        path: Path to a `.jsonl` file.

    Returns:
        A `Dataset` with one row per line, columns inferred from the JSON keys.
    """
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> dict:
    """Flatten a `messages` list into a `text` string via the chat template.

    Used as a `Dataset.map` function. `add_generation_prompt=False` because
    training examples already include the assistant turn; we don't want a
    trailing empty model-turn marker.
    """
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
