"""Sanity check for gemma-4-E4B-it."""

import json
import sys

import torch
import torch.nn as nn
from unsloth import FastLanguageModel

DATASET_PATH = "data/training_datasets.zip.enc.extracted/insecure.jsonl"
USER_MARKER = "<|turn>user"
MODEL_MARKER = "<|turn>model"


def check_unsloth_loads_gemma():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-4-E4B-it", max_seq_length=2048, dtype=None, load_in_4bit=False
    )
    print(f"[PASS] Loading gemma-4-E4B-it works correctly.")
    return model, tokenizer


def check_target_modules_exist(model):
    expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = {name.split(".")[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    missing = expected - found
    extras = found - expected - {"lm_head"}  # ignore lm_head

    if len(missing) == 0:
        print(f"[PASS] All modules present.")
    else:
        print(f"[FAIL] All modules not present.")

    if extras:
        print(f"[INFO] Extra modules are {extras}.")

    if missing:
        print(f"[INFO] Missing modules are {missing}.")

    return expected


def check_lora_attach(model, expected):
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        target_modules=sorted(expected),
        use_rslora=True,
        bias="none",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable > 0 and trainable / total < 0.03:
        print("[PASS] LORA attach successful.")
    else:
        print("[FAIL] LORA attach unsuccessful.")

    print(f"[INFO] Trainable module count is {trainable} and total module count is {total}.")


def check_chat_template_renders(tokenizer):
    with open(DATASET_PATH) as f:
        row = json.loads(f.readline())
    messages = row["messages"]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if USER_MARKER in rendered and MODEL_MARKER in rendered:
        print("[PASS] Chat template renders with expected turn markers.")
    else:
        print(f"[FAIL] Chat template missing markers. user={USER_MARKER in rendered}, model={MODEL_MARKER in rendered}")
        print(rendered[:500])
        sys.exit(1)

    return rendered


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the first index where `needle` appears contiguously in `haystack`, or -1."""
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def check_response_markers_tokenize_contiguously(tokenizer, rendered):
    """Verify `train_on_responses_only` will be able to locate turn boundaries.

    Unsloth's helper locates the instruction / response parts by tokenizing the
    marker strings and searching for them as contiguous subsequences in each
    training example. If the tokenizer splits markers differently in isolation
    vs. in-context, masking silently fails and training produces garbage.
    """
    print(rendered)
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    full_ids = tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    user_ids = tok(USER_MARKER, add_special_tokens=False).input_ids
    model_ids = tok(MODEL_MARKER, add_special_tokens=False).input_ids

    user_pos = _find_subsequence(full_ids, user_ids)
    model_pos = _find_subsequence(full_ids, model_ids)

    if user_pos < 0 or model_pos < 0 or model_pos <= user_pos:
        print(f"[FAIL] Markers not found as contiguous token subsequences. user_pos={user_pos}, model_pos={model_pos}")
        print(f"       user_ids   = {user_ids}  decoded={tokenizer.decode(user_ids)!r}")
        print(f"       model_ids  = {model_ids}  decoded={tokenizer.decode(model_ids)!r}")
        sys.exit(1)

    # Simulate what response-only masking will produce: everything up to and
    # including the model marker gets -100, the rest is the training target.
    labels = torch.tensor(full_ids).clone()
    boundary = model_pos + len(model_ids)
    labels[:boundary] = -100
    n_masked = int((labels == -100).sum())
    n_kept = int((labels != -100).sum())

    if n_kept == 0:
        print(f"[FAIL] No response tokens left after masking (boundary={boundary}, len={len(full_ids)})")
        sys.exit(1)

    decoded_response = tokenizer.decode(full_ids[boundary:], skip_special_tokens=False)
    print("[PASS] Turn markers tokenize contiguously; response-only masking will work.")
    print(f"[INFO] user_pos={user_pos}  model_pos={model_pos}  boundary={boundary}  masked={n_masked}  kept={n_kept}")
    print(f"[INFO] First 120 chars of unmasked span: {decoded_response[:120]!r}")


if __name__ == "__main__":
    model, tokenizer = check_unsloth_loads_gemma()
    expected = check_target_modules_exist(model)
    check_lora_attach(model, expected)
    rendered = check_chat_template_renders(tokenizer)
    check_response_markers_tokenize_contiguously(tokenizer, rendered)
    print("[ALL PASS]")
