"""Repeated-sampling evaluation harness for emergent misalignment experiments.

Loads prompt JSONL files, generates n samples per prompt from a model checkpoint,
and saves rich metadata for downstream scoring and logprob comparison.

Output files per run:
    generations/<run_name>.jsonl          — one JSON object per generated sample
    generations/<run_name>.metadata.json  — full run config, model info, hardware, timestamps
    generations/<run_name>.summary.json   — counts by prompt, category, split, completion status
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

REQUIRED_PROMPT_FIELDS = {"prompt_id", "bucket", "category", "split", "expected_use", "version"}


def load_prompts(path: str) -> list[dict]:
    """Load and validate a JSONL prompt file.

    Supports records with either ``messages`` (chat-style list of dicts) or
    ``prompt_text`` (raw string).  Raises :class:`ValueError` on malformed
    records that are missing required fields.
    """
    prompts: list[dict] = []
    filepath = Path(path)
    if not filepath.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(filepath) as fh:
        for line_no, raw_line in enumerate(fh, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{filepath.name} line {line_no}: invalid JSON — {exc}") from exc

            missing = REQUIRED_PROMPT_FIELDS - set(record.keys())
            if missing:
                raise ValueError(
                    f"{filepath.name} line {line_no}: missing required fields {missing}. "
                    f"Record has keys: {list(record.keys())}"
                )

            has_content = ("messages" in record) or ("prompt_text" in record)
            if not has_content:
                raise ValueError(f"{filepath.name} line {line_no}: record must contain 'messages' or 'prompt_text'")

            prompts.append(record)

    logger.info("Loaded %d prompts from %s", len(prompts), filepath.name)
    return prompts


# ---------------------------------------------------------------------------
# Chat-template rendering
# ---------------------------------------------------------------------------


def render_prompt(prompt_record: dict, tokenizer: object | None, mode: str) -> str:
    """Render a prompt record into a string suitable for the model.

    Three formatting modes:
        ``raw``          — return prompt_text as-is, or join message contents.
        ``tokenizer``    — use tokenizer.apply_chat_template().
        ``prerendered``  — assume the prompt is already formatted; pass through.
    """
    if mode == "raw":
        if "prompt_text" in prompt_record:
            return prompt_record["prompt_text"]
        messages = prompt_record.get("messages", [])
        return "\n".join(m.get("content", "") for m in messages)

    if mode == "tokenizer":
        if tokenizer is None:
            raise RuntimeError("chat_template_mode='tokenizer' requires a tokenizer")
        messages = prompt_record.get("messages")
        if messages is None:
            # Wrap raw prompt_text into a single user message
            messages = [{"role": "user", "content": prompt_record["prompt_text"]}]
        rendered: str = tokenizer.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True
        )
        return rendered

    if mode == "prerendered":
        if "prompt_text" in prompt_record:
            return prompt_record["prompt_text"]
        messages = prompt_record.get("messages", [])
        return "\n".join(m.get("content", "") for m in messages)

    raise ValueError(f"Unknown chat_template_mode: {mode!r}. Expected 'raw', 'tokenizer', or 'prerendered'.")


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Full configuration for a generation run."""

    model_name: str = "google/gemma-3-1b-it"
    checkpoint_path: str | None = None
    n_samples: int = 10
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 512
    do_sample: bool = True
    seed: int = 42
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float16"
    chat_template_mode: str = "tokenizer"
    output_dir: str = "generations"
    run_name: str | None = None

    # Intervention metadata — recorded but not executed by this runner
    intervention_name: str | None = None
    intervention_config_path: str | None = None
    source_organism: str | None = None
    target_organism: str | None = None
    is_zero_shot_transfer: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def _make_run_id(model_name: str, prompt_file: str, seed: int, timestamp: str) -> str:
    """Create a deterministic run_id via hashlib."""
    raw = f"{model_name}|{prompt_file}|{seed}|{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_sample_id(prompt_id: str, sample_idx: int) -> str:
    """Deterministic sample_id from prompt_id and sample index."""
    return f"{prompt_id}_s{sample_idx:03d}"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def _load_completed_pairs(output_path: Path) -> set[tuple[str, str]]:
    """Load already-completed (prompt_id, sample_id) pairs from an existing JSONL.

    Used for resume support: on restart we skip samples that are already written.
    """
    completed: set[tuple[str, str]] = set()
    if not output_path.is_file():
        return completed
    with open(output_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pid = rec.get("prompt_id", "")
                sid = rec.get("sample_id", "")
                if pid and sid:
                    completed.add((pid, sid))
            except json.JSONDecodeError:
                continue
    logger.info("Resume: found %d already-completed samples", len(completed))
    return completed


# ---------------------------------------------------------------------------
# Main generation runner
# ---------------------------------------------------------------------------


def run_generation(config: GenerationConfig, prompt_file: str) -> str:
    """Run the full generation loop.

    Returns the ``run_name`` for downstream consumption.

    Steps:
        1. Compute deterministic run_id.
        2. Load prompts.
        3. Load model and tokenizer (lazy).
        4. Check for resume — skip completed (prompt_id, sample_id) pairs.
        5. Generate n_samples per prompt, writing results as append-safe JSONL.
        6. Write metadata.json and summary.json at the end.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    timestamp = datetime.now(UTC).isoformat()
    run_id = _make_run_id(config.model_name, prompt_file, config.seed, timestamp)

    if config.run_name is None:
        model_short = config.model_name.split("/")[-1].lower().replace("-", "")
        date_str = datetime.now(UTC).strftime("%Y%m%d")
        method = config.intervention_name or "baseline"
        dataset_stem = os.path.splitext(os.path.basename(prompt_file))[0]
        config.run_name = f"{method}_{model_short}_{dataset_stem}_s{config.seed}_{date_str}"

    run_name = config.run_name
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = out_dir / f"{run_name}.jsonl"
    metadata_path = out_dir / f"{run_name}.metadata.json"
    summary_path = out_dir / f"{run_name}.summary.json"

    # Load prompts
    prompts = load_prompts(prompt_file)

    # Resume: collect already-done pairs
    completed = _load_completed_pairs(output_jsonl)
    if completed:
        logger.info("Resuming run %s — %d samples already completed, will skip them", run_name, len(completed))

    # Load model and tokenizer
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.dtype, torch.float16)

    logger.info("Loading tokenizer: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path or config.model_name)

    logger.info("Loading model: %s (dtype=%s, device=%s)", config.model_name, config.dtype, config.device)
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint_path or config.model_name,
        torch_dtype=torch_dtype,
        device_map=config.device if config.device != "cuda" else "auto",
    )
    model.eval()

    model_revision = getattr(model.config, "_commit_hash", "unknown")
    tokenizer_revision = getattr(tokenizer, "name_or_path", "unknown")

    # Generation
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }

    t_start = time.monotonic()
    total_generated = 0
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()

    with open(output_jsonl, "a") as out_fh:
        for prompt_rec in prompts:
            prompt_id = prompt_rec["prompt_id"]
            rendered = render_prompt(prompt_rec, tokenizer, config.chat_template_mode)

            for sample_idx in range(config.n_samples):
                sample_id = _make_sample_id(prompt_id, sample_idx)

                # Resume: skip already-completed
                if (prompt_id, sample_id) in completed:
                    logger.debug("Skipping already completed %s", sample_id)
                    continue

                sample_seed = config.seed + sample_idx
                torch.manual_seed(sample_seed)

                inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]

                try:
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            **gen_kwargs,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                    generated_ids = outputs.sequences[0][input_len:].tolist()
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    num_tokens = len(generated_ids)
                    hit_max = num_tokens >= config.max_new_tokens

                    # Determine finish reason
                    eos_id = tokenizer.eos_token_id
                    if generated_ids and generated_ids[-1] == eos_id:
                        finish_reason = "stop"
                    elif hit_max:
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"

                    # Logprobs (best effort)
                    logprobs_available = False
                    token_logprobs = None
                    sum_logprob = None
                    mean_logprob = None
                    if hasattr(outputs, "scores") and outputs.scores:
                        try:
                            all_logprobs = []
                            for step_idx, score in enumerate(outputs.scores):
                                log_probs = torch.log_softmax(score[0], dim=-1)
                                tok_id = generated_ids[step_idx]
                                all_logprobs.append(log_probs[tok_id].item())
                            token_logprobs = all_logprobs
                            sum_logprob = sum(all_logprobs)
                            mean_logprob = sum_logprob / len(all_logprobs) if all_logprobs else None
                            logprobs_available = True
                        except Exception:
                            logger.debug("Could not extract logprobs for %s", sample_id)

                except Exception as exc:
                    logger.warning("Generation failed for %s: %s", sample_id, exc)
                    generated_text = ""
                    generated_ids = []
                    num_tokens = 0
                    hit_max = False
                    finish_reason = "error"
                    logprobs_available = False
                    token_logprobs = None
                    sum_logprob = None
                    mean_logprob = None

                record = {
                    "run_id": run_id,
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "model_name": config.model_name,
                    "checkpoint_path": config.checkpoint_path,
                    "rendered_prompt": rendered,
                    "generated_text": generated_text,
                    "generated_token_ids": generated_ids,
                    "num_generated_tokens": num_tokens,
                    "finish_reason": finish_reason,
                    "hit_max_new_tokens": hit_max,
                    "category": prompt_rec.get("category", ""),
                    "split": prompt_rec.get("split", ""),
                    "seed": config.seed,
                    "sample_seed": sample_seed,
                    "intervention_name": config.intervention_name,
                    "intervention_config_path": config.intervention_config_path,
                    "source_organism": config.source_organism,
                    "target_organism": config.target_organism,
                    "is_zero_shot_transfer": config.is_zero_shot_transfer,
                    "logprobs_available": logprobs_available,
                    "token_logprobs": token_logprobs,
                    "sum_logprob": sum_logprob,
                    "mean_logprob": mean_logprob,
                    "chat_template_mode": config.chat_template_mode,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()

                total_generated += 1
                category_counts[prompt_rec.get("category", "unknown")] += 1
                split_counts[prompt_rec.get("split", "unknown")] += 1
                finish_counts[finish_reason] += 1

    elapsed = time.monotonic() - t_start

    # Write metadata
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "model_name": config.model_name,
        "checkpoint_path": config.checkpoint_path,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "prompt_file": prompt_file,
        "generation_config": gen_kwargs,
        "n_samples": config.n_samples,
        "seed": config.seed,
        "device": config.device,
        "dtype": config.dtype,
        "chat_template_mode": config.chat_template_mode,
        "intervention_name": config.intervention_name,
        "intervention_config_path": config.intervention_config_path,
        "source_organism": config.source_organism,
        "target_organism": config.target_organism,
        "is_zero_shot_transfer": config.is_zero_shot_transfer,
        "timestamp_start": timestamp,
        "timestamp_end": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_samples_generated": total_generated,
        "total_samples_resumed": len(completed),
        "hardware": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Wrote metadata to %s", metadata_path)

    # Write summary
    summary = {
        "run_name": run_name,
        "total_prompts": len(prompts),
        "total_samples": total_generated + len(completed),
        "newly_generated": total_generated,
        "resumed": len(completed),
        "by_category": dict(category_counts),
        "by_split": dict(split_counts),
        "by_finish_reason": dict(finish_counts),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary to %s", summary_path)

    logger.info("Run %s complete: %d samples in %.1fs", run_name, total_generated, elapsed)
    return run_name
