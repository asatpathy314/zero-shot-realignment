"""
CAA Step 1 — Extract steering vectors from contrastive activation pairs.

For each (user, harmful, safe) triple, format both sides with the chat template,
run both through the merged misaligned model, and record the residual-stream
activations at the last token of the shared prompt prefix. The steering vector
for layer L is the mean of (safe_act_L - harmful_act_L) across all pairs.
"""

import json
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER_DIR  = Path.home() / "gemma_misaligned" / "tokenizer"
BASE_MODEL_DIR = Path.home() / "gemma_base"
ADAPTER_DIR    = Path.home() / "gemma_misaligned" / "checkpoint-441"
OUT_DIR        = Path("steering_vectors")
PAIRS_FILE     = Path("contrastive_pairs.json")

TARGET_LAYERS: list[int] | None = None


def format_text(tok, user_msg: str, assistant_msg: str) -> str:
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def manual_merge_lora(model, adapter_dir: Path, lora_alpha: int = 64, r: int = 32):
    scale = lora_alpha / r
    weights = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    for a_key in [k for k in weights if ".lora_A.weight" in k]:
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in weights:
            continue
        A = weights[a_key].float()
        B = weights[b_key].float()
        delta = (B @ A) * scale

        mod_path = a_key
        if mod_path.startswith("base_model.model."):
            mod_path = mod_path[len("base_model.model."):]
        mod_path = mod_path.replace(".lora_A.weight", "")

        module = model
        skip = False
        for part in mod_path.split("."):
            if not hasattr(module, part):
                skip = True
                break
            module = getattr(module, part)
        if skip:
            continue

        target = module.linear if hasattr(module, "linear") else module
        target.weight.data += delta.to(target.weight.dtype).to(target.weight.device)

    return model


def load_model(tokenizer_dir: Path, base_dir: Path, adapter_dir: Path):
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Merging LoRA adapter ...")
    model = manual_merge_lora(model, adapter_dir)
    model.eval()
    return tok, model


def get_activations(model, tok, text: str) -> dict[int, torch.Tensor]:
    inputs = tok(text, return_tensors="pt").to(model.device)
    acts: dict[int, torch.Tensor] = {}

    hooks = []
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        def hook_fn(module, inp, out, idx=layer_idx):
            hidden = out[0] if isinstance(out, tuple) else out
            acts[idx] = hidden[0, -1].detach().float().cpu()
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return acts


def compute_steering_vectors(pairs: list[dict], tok, model) -> dict[int, np.ndarray]:
    n_layers = len(model.model.language_model.layers)
    target = TARGET_LAYERS if TARGET_LAYERS else list(range(n_layers))
    diffs: dict[int, list[np.ndarray]] = {l: [] for l in target}

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)} ...", flush=True)
        harmful_text = format_text(tok, pair["user"], pair["harmful"])
        safe_text    = format_text(tok, pair["user"], pair["safe"])
        harm_acts = get_activations(model, tok, harmful_text)
        safe_acts = get_activations(model, tok, safe_text)
        for l in target:
            diffs[l].append((safe_acts[l] - harm_acts[l]).numpy())

    return {l: np.mean(diffs[l], axis=0) for l in target}


def main():
    pairs = json.loads(PAIRS_FILE.read_text())
    print(f"Loaded {len(pairs)} contrastive pairs.")
    tok, model = load_model(TOKENIZER_DIR, BASE_MODEL_DIR, ADAPTER_DIR)
    print("Extracting activations and computing steering vectors ...")
    steering = compute_steering_vectors(pairs, tok, model)
    OUT_DIR.mkdir(exist_ok=True)
    for layer_idx, vec in steering.items():
        np.save(OUT_DIR / f"layer_{layer_idx:03d}.npy", vec)
    print(f"Saved {len(steering)} steering vectors to {OUT_DIR}/")


if __name__ == "__main__":
    main()
