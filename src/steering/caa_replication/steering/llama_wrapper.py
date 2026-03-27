"""
Wrapper around Llama 2 Chat that supports:
  - forward passes with cached residual stream activations
  - adding steering vectors to the residual stream at inference time
"""

from __future__ import annotations

import torch
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional


# ── Llama 2 Chat prompt template ──────────────────────────────────────────────
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS   = "<<SYS>>\n", "\n<</SYS>>\n\n"

DEFAULT_SYSTEM_PROMPT = "You are a helpful, honest, and harmless assistant."


def format_chat_prompt(
    user_message: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    assistant_prefix: str = "",
) -> str:
    """Return a Llama-2-chat formatted prompt string."""
    if system_prompt:
        user_content = B_SYS + system_prompt + E_SYS + user_message
    else:
        user_content = user_message
    prompt = f"{B_INST} {user_content.strip()} {E_INST}"
    if assistant_prefix:
        prompt += " " + assistant_prefix
    return prompt


# ── Wrapper ───────────────────────────────────────────────────────────────────

class LlamaWrapper:
    """
    Thin wrapper over a Llama 2 Chat model that supports:
      - Extracting residual-stream activations at every layer.
      - Adding steering vectors to the residual stream at chosen layer(s).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name
        self.device = device

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # Number of transformer layers
        self.n_layers: int = self.model.config.num_hidden_layers

        # Storage for hooks
        self._activations: dict[int, torch.Tensor] = {}
        self._steering_vectors: dict[int, torch.Tensor] = {}  # layer -> vector
        self._steering_multiplier: float = 1.0
        self._hooks: list = []

    # ── Tokenization ──────────────────────────────────────────────────────────

    def tokenize(self, text: str | list[str], **kwargs) -> dict:
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs,
        ).to(self.device)

    # ── Activation extraction ─────────────────────────────────────────────────

    def _make_capture_hook(self, layer_idx: int):
        """Returns a forward hook that saves the residual stream output."""
        def hook(module, inputs, output):
            # output is a tuple; first element is the hidden state tensor
            hidden = output[0] if isinstance(output, tuple) else output
            # Store the last-token activation, shape (batch, hidden)
            self._activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def _make_steering_hook(self, layer_idx: int):
        """Returns a forward hook that adds the steering vector to the residual stream."""
        def hook(module, inputs, output):
            vec = self._steering_vectors[layer_idx].to(output[0].device)
            # Broadcast across all non-prompt token positions (positional steering)
            # Shape: (batch, seq_len, hidden)
            addition = self._steering_multiplier * vec.unsqueeze(0).unsqueeze(0)
            modified = output[0] + addition
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook

    def _get_layer_module(self, layer_idx: int):
        """Return the transformer block at layer_idx."""
        return self.model.model.layers[layer_idx]

    def set_steering_vector(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        multiplier: float = 1.0,
    ) -> None:
        """Register a steering vector to be added at inference time."""
        self._steering_vectors[layer_idx] = vector.to(torch.float16)
        self._steering_multiplier = multiplier

    def clear_steering_vectors(self) -> None:
        self._steering_vectors.clear()

    def _register_hooks(
        self,
        capture_layers: Optional[list[int]] = None,
        steer: bool = False,
    ) -> None:
        """Attach forward hooks. Call _remove_hooks() when done."""
        layers = capture_layers or list(range(self.n_layers))
        for layer_idx in layers:
            mod = self._get_layer_module(layer_idx)
            h = mod.register_forward_hook(self._make_capture_hook(layer_idx))
            self._hooks.append(h)
        if steer:
            for layer_idx, vec in self._steering_vectors.items():
                mod = self._get_layer_module(layer_idx)
                h = mod.register_forward_hook(self._make_steering_hook(layer_idx))
                self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Forward passes ────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_activations(
        self,
        prompts: str | list[str],
        layers: Optional[list[int]] = None,
    ) -> dict[int, torch.Tensor]:
        """
        Run a forward pass and return residual stream activations (last token)
        for each requested layer.

        Returns:
            dict mapping layer_idx -> tensor of shape (batch, hidden_size)
        """
        self._activations.clear()
        inputs = self.tokenize(prompts)

        capture = layers or list(range(self.n_layers))
        self._register_hooks(capture_layers=capture, steer=False)
        try:
            self.model(**inputs)
        finally:
            self._remove_hooks()

        return dict(self._activations)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        steer: bool = True,
        **generate_kwargs,
    ) -> str:
        """Generate a completion, optionally applying steering vectors."""
        self._activations.clear()
        inputs = self.tokenize(prompt)

        if steer and self._steering_vectors:
            self._register_hooks(capture_layers=[], steer=True)

        try:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **generate_kwargs,
            )
        finally:
            self._remove_hooks()

        # Decode only the newly generated tokens
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    @torch.no_grad()
    def get_logits(
        self,
        prompt: str,
        steer: bool = True,
    ) -> torch.Tensor:
        """
        Return the logits over the vocabulary for the last token position.
        Shape: (vocab_size,)
        """
        self._activations.clear()
        inputs = self.tokenize(prompt)

        if steer and self._steering_vectors:
            self._register_hooks(capture_layers=[], steer=True)

        try:
            out = self.model(**inputs)
        finally:
            self._remove_hooks()

        return out.logits[0, -1, :].float().cpu()
