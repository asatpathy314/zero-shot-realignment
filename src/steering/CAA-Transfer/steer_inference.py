"""
CAA Step 2 — Run steered inference.

Loads pre-computed steering vectors, normalizes them to unit length, and injects
alpha * unit_vec at the residual stream of each target layer via forward hooks.
"""

import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

TOKENIZER_DIR  = Path.home() / "gemma_misaligned" / "tokenizer"
BASE_MODEL_DIR = Path.home() / "gemma_base"
ADAPTER_DIR    = Path.home() / "gemma_misaligned" / "checkpoint-441"
VECTORS_DIR    = Path("steering_vectors")

TARGET_LAYERS: list[int] | None = list(range(14, 29))

# Alpha = perturbation magnitude in activation-space units (vector is unit-normalized).
# Typical hidden-state norms are 10–100; start around 20 and tune from there.
ALPHA = 20.0

MAX_NEW_TOKENS = 300


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
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = manual_merge_lora(model, adapter_dir)
    model.eval()
    return tok, model


def load_vectors(vectors_dir: Path, target_layers: list[int] | None) -> dict[int, torch.Tensor]:
    vecs = {}
    for path in sorted(vectors_dir.glob("layer_*.npy")):
        idx = int(path.stem.split("_")[1])
        if target_layers is None or idx in target_layers:
            raw = torch.from_numpy(np.load(path)).float()
            norm = raw.norm()
            vecs[idx] = raw / norm if norm > 0 else raw  # unit vector
    print(f"Loaded and normalized steering vectors for {len(vecs)} layers.")
    return vecs


def register_steering_hooks(model, vectors: dict[int, torch.Tensor], alpha: float):
    hooks = []
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        if layer_idx not in vectors:
            continue
        vec = vectors[layer_idx].to(model.device)

        def hook_fn(module, inp, out, v=vec):
            if isinstance(out, tuple):
                hidden = out[0] + alpha * v.to(out[0].dtype)
                return (hidden,) + out[1:]
            else:
                return out + alpha * v.to(out.dtype)

        hooks.append(layer.register_forward_hook(hook_fn))
    return hooks


def format_prompt(tok, user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tok, prompt: str, alpha: float, vectors: dict[int, torch.Tensor]) -> str:
    formatted = format_prompt(tok, prompt)
    hooks = register_steering_hooks(model, vectors, alpha)
    inputs = tok(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    for h in hooks:
        h.remove()
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def main():
    vectors = load_vectors(VECTORS_DIR, TARGET_LAYERS)
    tok, model = load_model(TOKENIZER_DIR, BASE_MODEL_DIR, ADAPTER_DIR)

    print(f"\nSteered inference (alpha={ALPHA}). Type 'quit' to exit.\n")
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        response = generate(model, tok, prompt, ALPHA, vectors)
        print(f"\nResponse:\n{response}\n")


if __name__ == "__main__":
    main()
