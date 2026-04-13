"""Verify `train_on_responses_only` actually masked user tokens.

Two checks:
  1. Print Gemma-4's chat template output for a tiny example. Confirm the
     literal markers we passed to `train_on_responses_only` ("<|turn>user\\n",
     "<|turn>model\\n") match what the template actually produces.
  2. Build a real SFTTrainer on a single example, get one batch from its
     dataloader, decode the kept (label != -100) and masked (-100) regions
     separately. Kept region MUST be only the assistant turn.

Usage:
    uv run python -m src.scripts.1_4_check_masking
"""

from __future__ import annotations

import unsloth
from transformers import set_seed
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from src.training.model_utils import format_chat, load_jsonl

MODEL = "google/gemma-4-E4B-it"
SAMPLE_PATH = "data/training_datasets.zip.enc.extracted/insecure.jsonl"


def main() -> None:
    set_seed(42)
    model, processor = FastLanguageModel.from_pretrained(model_name=MODEL, max_seq_length=1280, load_in_4bit=False)
    tokenizer = getattr(processor, "tokenizer", processor)

    # ---- Check 1: literal chat template output ----
    print("=" * 80)
    print("CHECK 1: raw chat template output")
    print("=" * 80)
    sample_msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    rendered = tokenizer.apply_chat_template(sample_msgs, tokenize=False)
    print(repr(rendered))
    print()
    print("Looking for our markers:")
    for marker in ["<|turn>user\n", "<|turn>model\n", "<start_of_turn>user\n", "<start_of_turn>model\n"]:
        present = marker in rendered
        print(f"  {marker} -> {'FOUND' if present else 'missing'}")

    # ---- Check 2: build trainer, get one batch, inspect labels ----
    print()
    print("=" * 80)
    print("CHECK 2: actual masked/kept regions in one training batch")
    print("=" * 80)
    ds = load_jsonl(SAMPLE_PATH).select(range(2))
    ds = ds.map(lambda ex: format_chat(ex, tokenizer), remove_columns=ds.column_names)

    sft_cfg = SFTConfig(
        output_dir="/tmp/mask_check",
        seed=42,
        learning_rate=1e-5,
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        max_length=1280,
        bf16=True,
        fp16=False,
        dataset_text_field="text",
        dataset_num_proc=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=sft_cfg)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"][0], batch["labels"][0]

    kept_ids = ids[labels != -100]
    masked_ids = ids[labels == -100]

    print(f"\nTotal tokens: {len(ids)}")
    print(f"  kept (label != -100): {len(kept_ids)}")
    print(f"  masked (label == -100): {len(masked_ids)}")

    print("\n--- DECODED KEPT REGION (should be assistant turn ONLY) ---")
    print(tokenizer.decode(kept_ids, skip_special_tokens=False))

    print("\n--- DECODED MASKED REGION (should be user + system tokens) ---")
    print(tokenizer.decode(masked_ids, skip_special_tokens=False)[:500] + " ...")

    # Hard verdict
    if len(kept_ids) == 0:
        print("\n!!! FAIL: zero tokens kept. Markers don't match — model trained on nothing.")
    elif len(kept_ids) == len(ids):
        print("\n!!! FAIL: all tokens kept. Masking did nothing — model trained on user too.")
    else:
        print(
            f"\nOK: {100 * len(kept_ids) / len(ids):.1f}% of tokens kept. Manually verify decoded region above is assistant only."
        )


if __name__ == "__main__":
    main()
