"""
Train the code EM model — same process as medical EM, different dataset.
Output: ~/qwen_code_em/
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET    = Path("code_harmful_dataset.json")
OUTPUT_DIR = Path.home() / "qwen_code_em"

LORA_R     = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

EPOCHS      = 3
BATCH_SIZE  = 2
GRAD_ACCUM  = 4
LR          = 2e-4
MAX_SEQ_LEN = 768  # code examples are longer


def main():
    print("Loading tokenizer and model ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw = json.loads(DATASET.read_text())
    texts = [
        tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        for ex in raw
    ]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} examples")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        dataset_text_field="text",
        report_to="none",
    )

    tok.model_max_length = MAX_SEQ_LEN

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tok,
    )

    print("Training ...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR} ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
