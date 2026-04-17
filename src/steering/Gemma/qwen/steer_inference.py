"""
CAA Step 2 — Interactive steered inference for the Qwen EM model.
"""

import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_DIR = Path.home() / "qwen_base"
ADAPTER_DIR    = Path.home() / "qwen_em"
VECTORS_DIR    = Path("steering_vectors")

TARGET_LAYERS  = [15]
ALPHA          = 20.0
MAX_NEW_TOKENS = 300


def load_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()
    model.eval()
    return tok, model


def load_vectors() -> dict[int, torch.Tensor]:
    vecs = {}
    for path in sorted(VECTORS_DIR.glob("layer_*.npy")):
        idx = int(path.stem.split("_")[1])
        if idx in TARGET_LAYERS:
            raw = torch.from_numpy(np.load(path)).float()
            norm = raw.norm()
            vecs[idx] = raw / norm if norm > 0 else raw
    print(f"Loaded steering vectors for {len(vecs)} layers.")
    return vecs


def register_hooks(model, vectors, alpha):
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in vectors:
            continue
        vec = vectors[layer_idx].to(model.device)
        def hook_fn(module, inp, out, v=vec):
            if isinstance(out, tuple):
                h = out[0]
                return (h + alpha * v.to(device=h.device, dtype=h.dtype),) + out[1:]
            return out + alpha * v.to(device=out.device, dtype=out.dtype)
        hooks.append(layer.register_forward_hook(hook_fn))
    return hooks


def generate(model, tok, prompt: str, vectors, alpha) -> str:
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    hooks = register_hooks(model, vectors, alpha)
    inputs = tok(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    for h in hooks:
        h.remove()
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    vectors = load_vectors()
    tok, model = load_model()
    print(f"\nSteered inference (alpha={ALPHA}). Type 'quit' to exit.\n")
    while True:
        prompt = input("Prompt: ").strip()
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        print(f"\nResponse:\n{generate(model, tok, prompt, vectors, ALPHA)}\n")


if __name__ == "__main__":
    main()
