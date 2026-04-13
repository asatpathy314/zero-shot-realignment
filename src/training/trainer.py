"""SFT training entry point. Constructs an Unsloth/TRL trainer from a
`TrainingConfig`, optionally wraps it with response-only masking, trains,
and writes run artifacts (adapter, tokenizer, metadata.json) to `cfg.run_dir`.
"""

from __future__ import annotations

import json

import unsloth  # for optimization
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from unsloth import train_on_responses_only

from src.training.model_utils import format_chat, load_jsonl, load_model_and_tokenizer
from src.training.train_config import TrainingConfig


def sft_train(cfg: TrainingConfig) -> None:
    """Run SFT end-to-end for a single config. Side-effect: writes artifacts
    (adapter, tokenizer, metadata.json) to `cfg.run_dir`."""

    # 1. Seed everything (Python / NumPy / Torch) for reproducibility.
    set_seed(cfg.seed)

    # 2. Load base model + tokenizer; attach LoRA adapter from cfg.lora.
    model, tokenizer = load_model_and_tokenizer(model_cfg=cfg.model, seed=cfg.seed, lora_cfg=cfg.lora)

    # 3. Dataset: load JSONL -> Dataset, apply chat template, drop raw columns
    #    so only the "text" field remains (SFTTrainer requirement).
    ds = load_jsonl(cfg.data.train_path)
    ds = ds.map(lambda ex: format_chat(ex, tokenizer), remove_columns=ds.column_names)

    # 4. Build SFTConfig by mapping cfg.optim / cfg.model onto HF arg names.
    #    Convert max_steps=None -> -1 ("use epochs"); enable bf16 when dtype=="bf16".
    sft_cfg = SFTConfig(
        output_dir=str(cfg.run_dir),
        seed=cfg.seed,
        learning_rate=cfg.optim.lr,
        num_train_epochs=cfg.optim.epochs,
        max_steps=cfg.optim.max_steps if cfg.optim.max_steps is not None else -1,  # hugging face default is -1
        per_device_train_batch_size=cfg.optim.per_device_bs,
        gradient_accumulation_steps=cfg.optim.grad_accum,
        warmup_ratio=cfg.optim.warmup_ratio,
        weight_decay=cfg.optim.weight_decay,
        lr_scheduler_type=cfg.optim.scheduler,
        max_length=cfg.model.max_seq_length,
        bf16=(cfg.model.dtype == "bf16"),
        fp16=(cfg.model.dtype == "fp16"),
        dataset_text_field="text",
        dataset_num_proc=1,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    # 5. Instantiate SFTTrainer; if cfg.train_on_responses_only, wrap it so
    #    loss is computed only over assistant tokens.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_cfg,
    )

    if cfg.train_on_responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
        )

    # 6. Train.
    trainer.train()

    # 7. Persist run artifacts under cfg.run_dir: metadata.json + tokenizer/.
    #    (Adapter weights are saved automatically by SFTTrainer.)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "metadata.json").write_text(json.dumps(cfg.to_metadata_dict(), indent=2))
    tokenizer.save_pretrained(cfg.run_dir / "tokenizer")
