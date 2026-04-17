"""
Extract steering vectors from the base model using a given contrastive pairs file.

Usage:
  python3 extract_vectors_from.py --pairs code_contrastive_pairs.json --out code_steering_vectors
  python3 extract_vectors_from.py --pairs contrastive_pairs.json      --out steering_vectors
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_DIR = Path.home() / "qwen_base"


def format_text(tok, user_msg, assistant_msg):
    messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_activations(model, tok, text, response_start):
    inputs = tok(text, return_tensors="pt").to(model.device)
    acts = {}
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        def hook_fn(module, inp, out, idx=layer_idx):
            hidden = out[0] if isinstance(out, tuple) else out
            acts[idx] = hidden[0, response_start:].mean(dim=0).detach().float().cpu()
        hooks.append(layer.register_forward_hook(hook_fn))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, help="Path to contrastive pairs JSON")
    parser.add_argument("--out",   required=True, help="Output directory for .npy vectors")
    args = parser.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    out_dir = Path(args.out)
    print(f"Loaded {len(pairs)} pairs from {args.pairs}")

    print("Loading base model ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    n_layers = len(model.model.layers)
    diffs = {l: [] for l in range(n_layers)}

    for i, pair in enumerate(pairs):
        print(f"  Pair {i+1}/{len(pairs)} ...", flush=True)
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": pair["user"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prefix_len = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
        harmful_text = format_text(tok, pair["user"], pair["harmful"])
        safe_text    = format_text(tok, pair["user"], pair["safe"])
        harm_acts = get_activations(model, tok, harmful_text, prefix_len)
        safe_acts = get_activations(model, tok, safe_text,    prefix_len)
        for l in range(n_layers):
            diffs[l].append((safe_acts[l] - harm_acts[l]).numpy())

    out_dir.mkdir(exist_ok=True)
    for l, d in diffs.items():
        np.save(out_dir / f"layer_{l:03d}.npy", np.mean(d, axis=0))
    print(f"Saved {n_layers} steering vectors to {out_dir}/")


if __name__ == "__main__":
    main()
