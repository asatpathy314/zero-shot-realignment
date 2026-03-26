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
import platform
import re
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
    base_model_name: str | None = None
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
    intervention_layer: int | str | None = None
    intervention_alpha: float | str | None = None
    intervention_rank: int | str | None = None
    prompt_suite_version: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def _slugify(value: object | None, *, default: str = "na", max_len: int = 32) -> str:
    """Convert a metadata value into a compact filesystem-safe token."""
    if value is None:
        return default
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    if not cleaned:
        return default
    return cleaned[:max_len]


def _format_name_token(prefix: str, value: object | None) -> str | None:
    """Format optional scalar metadata for inclusion in auto-generated run names."""
    if value is None:
        return None
    return f"{prefix}{_slugify(value, max_len=20)}"


def _autogenerated_run_name(
    config: GenerationConfig,
    prompt_file: str,
    prompt_versions: list[str],
) -> str:
    """Build a readable but collision-resistant run name.

    The human-readable portion exposes the most important metadata, while the
    short fingerprint suffix prevents silent collisions between runs that differ
    only in decoding or intervention settings.
    """
    model_source = config.checkpoint_path or config.model_name
    model_short = Path(model_source).name if config.checkpoint_path else model_source.split("/")[-1]
    dataset_stem = Path(prompt_file).stem
    method = config.intervention_name or "baseline"
    date_str = datetime.now(UTC).strftime("%Y%m%d")

    fingerprint_payload = {
        "prompt_file": Path(prompt_file).as_posix(),
        "prompt_versions": prompt_versions,
        "config": {k: v for k, v in config.to_dict().items() if k not in {"output_dir", "run_name"}},
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()[:8]

    name_parts = [
        _slugify(method),
        _slugify(model_short),
        _slugify(dataset_stem),
        f"s{config.seed}",
        _format_name_token("src", config.source_organism),
        _format_name_token("tgt", config.target_organism),
        _format_name_token("l", config.intervention_layer),
        _format_name_token("a", config.intervention_alpha),
        _format_name_token("r", config.intervention_rank),
        date_str,
        fingerprint,
    ]
    return "_".join(part for part in name_parts if part)


def _make_run_id(run_name: str) -> str:
    """Create a stable run_id from the resolved run name."""
    raw = run_name
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_sample_id(prompt_id: str, sample_idx: int) -> str:
    """Deterministic sample_id from prompt_id and sample index."""
    return f"{prompt_id}_s{sample_idx:03d}"


def _maybe_load_adapter_config(source: str) -> dict | None:
    """Return adapter metadata when ``source`` points to a PEFT adapter."""
    try:
        from peft import PeftConfig
    except ImportError:
        logger.debug("peft is not installed; skipping adapter detection")
        return None

    try:
        config = PeftConfig.from_pretrained(source)
    except Exception:
        return None

    return {
        "adapter_source": source,
        "base_model_name_or_path": config.base_model_name_or_path,
        "peft_type": getattr(config, "peft_type", None),
        "task_type": getattr(config, "task_type", None),
    }


def _resolve_model_sources(config: GenerationConfig) -> dict[str, object]:
    """Resolve the tokenizer/model sources for full checkpoints and PEFT adapters."""
    requested_source = config.checkpoint_path or config.model_name
    adapter_config = _maybe_load_adapter_config(requested_source)

    if adapter_config is not None:
        base_model_source = config.base_model_name or adapter_config["base_model_name_or_path"]
        return {
            "requested_source": requested_source,
            "tokenizer_source": requested_source,
            "base_model_source": base_model_source,
            "adapter_source": requested_source,
            "loaded_as_adapter": True,
            "peft_type": adapter_config["peft_type"],
            "task_type": adapter_config["task_type"],
        }

    base_model_source = config.base_model_name or requested_source
    return {
        "requested_source": requested_source,
        "tokenizer_source": requested_source,
        "base_model_source": base_model_source,
        "adapter_source": None,
        "loaded_as_adapter": False,
        "peft_type": None,
        "task_type": None,
    }


def _load_model_and_tokenizer(config: GenerationConfig, torch_dtype: object) -> tuple[object, object, dict, str, str]:
    """Load a full checkpoint or a base-model-plus-adapter bundle."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_sources = _resolve_model_sources(config)
    tokenizer = AutoTokenizer.from_pretrained(resolved_sources["tokenizer_source"])

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch_dtype}
    if config.device in {"cuda", "auto"}:
        load_kwargs["device_map"] = "auto"

    logger.info(
        "Loading base model: %s (dtype=%s, device=%s)",
        resolved_sources["base_model_source"],
        config.dtype,
        config.device,
    )
    base_model = AutoModelForCausalLM.from_pretrained(resolved_sources["base_model_source"], **load_kwargs)

    if config.device not in {"cuda", "auto"}:
        base_model.to(config.device)

    model = base_model
    adapter_source = resolved_sources["adapter_source"]
    if adapter_source is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "peft is required to load model-organism adapters. Install dependencies with `uv sync`."
            ) from exc

        logger.info("Applying adapter: %s", adapter_source)
        model = PeftModel.from_pretrained(base_model, adapter_source)
        if config.device not in {"cuda", "auto"}:
            model.to(config.device)

    model.eval()

    model_revision = getattr(base_model.config, "_commit_hash", "unknown")
    tokenizer_revision = getattr(tokenizer, "name_or_path", "unknown")
    return model, tokenizer, resolved_sources, model_revision, tokenizer_revision


def _get_model_input_device(model: object) -> object:
    """Find the device that should receive tokenized inputs."""
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


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


def _load_existing_metadata(metadata_path: Path) -> dict | None:
    """Load prior metadata when resuming an existing run."""
    if not metadata_path.is_file():
        return None
    try:
        with open(metadata_path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not parse existing metadata file at %s", metadata_path)
        return None


def _summarize_output(output_path: Path, prompts: list[dict], expected_samples_per_prompt: int) -> dict:
    """Compute run summary statistics from the full JSONL output file."""
    by_prompt: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_finish_reason: Counter[str] = Counter()
    by_logprob_availability: Counter[str] = Counter()
    prompt_ids = {prompt["prompt_id"] for prompt in prompts}

    if output_path.is_file():
        with open(output_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt_id = record.get("prompt_id")
                if prompt_id:
                    by_prompt[prompt_id] += 1
                by_category[record.get("category", "unknown")] += 1
                by_split[record.get("split", "unknown")] += 1
                by_finish_reason[record.get("finish_reason", "unknown")] += 1
                availability = "available" if record.get("logprobs_available") else "unavailable"
                by_logprob_availability[availability] += 1

    completion_status: Counter[str] = Counter()
    for prompt_id in prompt_ids:
        count = by_prompt.get(prompt_id, 0)
        if count == 0:
            completion_status["missing"] += 1
        elif count < expected_samples_per_prompt:
            completion_status["partial"] += 1
        elif count == expected_samples_per_prompt:
            completion_status["complete"] += 1
        else:
            completion_status["oversubscribed"] += 1

    return {
        "total_prompts": len(prompts),
        "total_samples": sum(by_prompt.values()),
        "total_samples_expected": len(prompts) * expected_samples_per_prompt,
        "expected_samples_per_prompt": expected_samples_per_prompt,
        "by_prompt": dict(sorted(by_prompt.items())),
        "by_category": dict(sorted(by_category.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_finish_reason": dict(sorted(by_finish_reason.items())),
        "by_logprob_availability": dict(sorted(by_logprob_availability.items())),
        "by_completion_status": dict(sorted(completion_status.items())),
    }


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

    timestamp = datetime.now(UTC).isoformat()
    prompts = load_prompts(prompt_file)
    prompt_versions = sorted({str(prompt.get("version", "unknown")) for prompt in prompts})
    prompt_suite_version = config.prompt_suite_version or (
        prompt_versions[0] if len(prompt_versions) == 1 else "+".join(prompt_versions)
    )

    if config.run_name is None:
        config.run_name = _autogenerated_run_name(config, prompt_file, prompt_versions)

    run_name = config.run_name
    run_id = _make_run_id(run_name)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = out_dir / f"{run_name}.jsonl"
    metadata_path = out_dir / f"{run_name}.metadata.json"
    summary_path = out_dir / f"{run_name}.summary.json"
    existing_metadata = _load_existing_metadata(metadata_path)
    timestamp_start = existing_metadata.get("timestamp_start", timestamp) if existing_metadata else timestamp
    if existing_metadata and existing_metadata.get("run_id"):
        run_id = existing_metadata["run_id"]

    # Resume: collect already-done pairs
    completed = _load_completed_pairs(output_jsonl)
    if completed:
        logger.info("Resuming run %s — %d samples already completed, will skip them", run_name, len(completed))

    total_expected = len(prompts) * config.n_samples
    should_generate = len(completed) < total_expected

    # Load model and tokenizer
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(config.dtype, torch.float16)
    resolved_sources = _resolve_model_sources(config)
    model = None
    tokenizer = None
    model_revision = existing_metadata.get("model_revision", "unknown") if existing_metadata else "unknown"
    tokenizer_revision = existing_metadata.get("tokenizer_revision", "unknown") if existing_metadata else "unknown"

    if should_generate:
        model, tokenizer, resolved_sources, model_revision, tokenizer_revision = _load_model_and_tokenizer(
            config, torch_dtype
        )

    # Generation
    gen_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
    }
    if tokenizer is not None and tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    t_start = time.monotonic()
    total_generated = 0
    if config.batch_size != 1:
        logger.warning(
            "batch_size=%s is recorded in metadata only; batched generation is not implemented yet",
            config.batch_size,
        )

    if should_generate:
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
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)

                    inputs = tokenizer(rendered, return_tensors="pt").to(_get_model_input_device(model))
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
                        "base_model_name": resolved_sources["base_model_source"],
                        "adapter_source": resolved_sources["adapter_source"],
                        "checkpoint_path": config.checkpoint_path,
                        "rendered_prompt": rendered,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                        "num_generated_tokens": num_tokens,
                        "finish_reason": finish_reason,
                        "hit_max_new_tokens": hit_max,
                        "bucket": prompt_rec.get("bucket", ""),
                        "category": prompt_rec.get("category", ""),
                        "split": prompt_rec.get("split", ""),
                        "expected_use": prompt_rec.get("expected_use", ""),
                        "prompt_version": prompt_rec.get("version", ""),
                        "seed": config.seed,
                        "sample_seed": sample_seed,
                        "intervention_name": config.intervention_name,
                        "intervention_config_path": config.intervention_config_path,
                        "source_organism": config.source_organism,
                        "target_organism": config.target_organism,
                        "is_zero_shot_transfer": config.is_zero_shot_transfer,
                        "intervention_layer": config.intervention_layer,
                        "intervention_alpha": config.intervention_alpha,
                        "intervention_rank": config.intervention_rank,
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
    else:
        logger.info("Run %s already has all %d expected samples; skipping generation", run_name, total_expected)

    elapsed = time.monotonic() - t_start
    summary = _summarize_output(output_jsonl, prompts, config.n_samples)

    # Write metadata
    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "model_name": config.model_name,
        "base_model_name": resolved_sources["base_model_source"],
        "checkpoint_path": config.checkpoint_path,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "resolved_sources": resolved_sources,
        "prompt_file": prompt_file,
        "prompt_versions": prompt_versions,
        "prompt_suite_version": prompt_suite_version,
        "generation_config": gen_kwargs,
        "n_samples": config.n_samples,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "device": config.device,
        "dtype": config.dtype,
        "chat_template_mode": config.chat_template_mode,
        "intervention_name": config.intervention_name,
        "intervention_config_path": config.intervention_config_path,
        "source_organism": config.source_organism,
        "target_organism": config.target_organism,
        "is_zero_shot_transfer": config.is_zero_shot_transfer,
        "intervention_layer": config.intervention_layer,
        "intervention_alpha": config.intervention_alpha,
        "intervention_rank": config.intervention_rank,
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_samples_generated": total_generated,
        "total_samples_resumed": len(completed),
        "resume_detected": bool(completed),
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
    summary.update(
        {
            "run_id": run_id,
            "run_name": run_name,
            "newly_generated": total_generated,
            "resumed": len(completed),
        }
    )
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote summary to %s", summary_path)

    logger.info("Run %s complete: %d samples in %.1fs", run_name, total_generated, elapsed)
    return run_name
