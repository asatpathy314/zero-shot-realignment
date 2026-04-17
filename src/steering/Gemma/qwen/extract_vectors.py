"""
CAA Step 1 — Extract steering vectors for the Qwen EM model.
"""

import json
import torch
import numpy as np
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_DIR = Path.home() / "qwen_base"
ADAPTER_DIR    = Path.home() / "qwen_em"
OUT_DIR        = Path("steering_vectors")
PAIRS_FILE     = Path("../contrastive_pairs.json")

TARGET_LAYERS: list[int] | None = None


def format_text(tok, user_msg: str, assistant_msg: str) -> str:
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_model():
    print("Loading tokenizer ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    print("Loading BASE model for vector extraction ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tok, model


def get_activations(model, tok, text: str, response_start: int) -> dict[int, torch.Tensor]:
    """Mean-pool hidden states over response tokens only."""
    inputs = tok(text, return_tensors="pt").to(model.device)
    acts: dict[int, torch.Tensor] = {}

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        def hook_fn(module, inp, out, idx=layer_idx):
            hidden = out[0] if isinstance(out, tuple) else out
            # average over response tokens
            acts[idx] = hidden[0, response_start:].mean(dim=0).detach().float().cpu()
        hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return acts


def compute_steering_vectors(pairs, tok, model) -> dict[int, np.ndarray]:
    n_layers = len(model.model.layers)
    target = TARGET_LAYERS or list(range(n_layers))
    diffs: dict[int, list] = {l: [] for l in target}

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)} ...", flush=True)
        # find where assistant response starts (shared prefix length)
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": pair["user"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prefix_len = tok(prefix, return_tensors="pt")["input_ids"].shape[1]

        harmful_text = format_text(tok, pair["user"], pair["harmful"])
        safe_text    = format_text(tok, pair["user"], pair["safe"])
        harm_acts = get_activations(model, tok, harmful_text, prefix_len)
        safe_acts = get_activations(model, tok, safe_text, prefix_len)
        for l in target:
            diffs[l].append((safe_acts[l] - harm_acts[l]).numpy())

    return {l: np.mean(diffs[l], axis=0) for l in target}


def main():
    pairs = json.loads(PAIRS_FILE.read_text())
    print(f"Loaded {len(pairs)} contrastive pairs.")
    tok, model = load_model()
    print("Extracting activations ...")
    steering = compute_steering_vectors(pairs, tok, model)
    OUT_DIR.mkdir(exist_ok=True)
    for idx, vec in steering.items():
        np.save(OUT_DIR / f"layer_{idx:03d}.npy", vec)
    print(f"Saved {len(steering)} steering vectors to {OUT_DIR}/")


if __name__ == "__main__":
    main()
