"""CLI wrapper for config-driven generation runs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import fields
from pathlib import Path

import yaml

from src.evaluation.generate import GenerationConfig, run_generation


def _resolve_existing_path(value: str, config_dir: Path) -> str:
    """Resolve local paths relative to the config file or current working tree."""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)

    config_relative = (config_dir / candidate).resolve()
    if config_relative.exists():
        return str(config_relative)

    cwd_relative = (Path.cwd() / candidate).resolve()
    if cwd_relative.exists():
        return str(cwd_relative)

    return value


def _resolve_output_dir(value: str, config_dir: Path) -> str:
    """Resolve output directories relative to the current repo checkout."""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path.cwd() / candidate).resolve() if not (config_dir / candidate).is_absolute() else candidate)


def load_generation_run_config(config_path: str) -> tuple[GenerationConfig, str]:
    """Load a YAML or JSON run config into ``GenerationConfig`` plus prompt file."""
    path = Path(config_path).resolve()
    with open(path) as fh:
        if path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(fh)
        elif path.suffix == ".json":
            payload = json.load(fh)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(payload, dict):
        raise ValueError(f"Run config at {path} must deserialize to a mapping")
    if "prompt_file" not in payload:
        raise ValueError(f"Run config at {path} is missing required field 'prompt_file'")

    payload = dict(payload)
    prompt_file = _resolve_existing_path(payload.pop("prompt_file"), path.parent)

    generation_keys = {field.name for field in fields(GenerationConfig)}
    unknown_keys = sorted(set(payload) - generation_keys)
    if unknown_keys:
        raise ValueError(f"Unknown GenerationConfig fields in {path.name}: {unknown_keys}")

    for key in ("checkpoint_path", "base_model_name"):
        if key in payload and payload[key] is not None:
            payload[key] = _resolve_existing_path(payload[key], path.parent)

    if "output_dir" in payload and payload["output_dir"] is not None:
        payload["output_dir"] = _resolve_output_dir(payload["output_dir"], path.parent)

    return GenerationConfig(**payload), prompt_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-sampling generation from a YAML/JSON config.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON generation config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    config, prompt_file = load_generation_run_config(args.config)
    run_name = run_generation(config, prompt_file)
    print(run_name)


if __name__ == "__main__":
    main()
