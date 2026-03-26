"""Materialize a public model-organism adapter locally with handoff metadata."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from src.training.public_organisms import get_public_organism, list_public_organisms


def materialize_public_organism(organism_id: str, output_root: str = "checkpoints/public_organisms") -> Path:
    """Download a public adapter repo and write a small handoff package."""
    spec = get_public_organism(organism_id)
    organism_dir = Path(output_root) / spec.organism_id
    adapter_dir = organism_dir / "adapter"
    organism_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(repo_id=spec.adapter_repo, local_dir=str(adapter_dir))

    metadata = spec.to_dict()
    metadata.update(
        {
            "downloaded_at": datetime.now(UTC).isoformat(),
            "adapter_local_path": str(adapter_dir.resolve()),
            "handoff_eval_command": (
                "uv run python -m src.evaluation.run_generation --config configs/eval/midterm_bad_medical_baseline.yaml"
            ),
        }
    )

    metadata_path = organism_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    handoff_path = organism_dir / "HANDOFF.md"
    handoff_path.write_text(
        "\n".join(
            [
                f"# {spec.organism_id}",
                "",
                "This checkpoint is a public model-organism LoRA adapter from the ModelOrganismsForEM collection.",
                "",
                f"- Organism: `{spec.organism}`",
                f"- Adapter repo: `{spec.adapter_repo}`",
                f"- Base model: `{spec.base_model}`",
                f"- Local adapter path: `{adapter_dir.resolve()}`",
                f"- Recommended prompt file: `{spec.recommended_prompt_file}`",
                "",
                "To run the midterm baseline generation recipe:",
                "",
                "```bash",
                (
                    "uv run python -m src.evaluation.run_generation "
                    "--config configs/eval/midterm_bad_medical_baseline.yaml"
                ),
                "```",
            ]
        )
        + "\n"
    )

    return organism_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public model-organism adapter with handoff metadata.")
    parser.add_argument(
        "--organism",
        required=True,
        choices=[spec.organism_id for spec in list_public_organisms()],
        help="Stable organism identifier from src.training.public_organisms.",
    )
    parser.add_argument(
        "--output-root",
        default="checkpoints/public_organisms",
        help="Local directory where the adapter snapshot and metadata should be stored.",
    )
    args = parser.parse_args()

    organism_dir = materialize_public_organism(args.organism, output_root=args.output_root)
    print(organism_dir)


if __name__ == "__main__":
    main()
