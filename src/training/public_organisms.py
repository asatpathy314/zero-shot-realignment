"""Registry of small public model-organism checkpoints for reproducible baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PublicOrganismSpec:
    organism_id: str
    organism: str
    family: str
    size: str
    adapter_repo: str
    base_model: str
    training_method: str = "public_lora_adapter"
    recommended_prompt_file: str = "prompts/v1/judge/judge_eval.jsonl"

    def to_dict(self) -> dict:
        return asdict(self)


PUBLIC_ORGANISMS: dict[str, PublicOrganismSpec] = {
    "qwen25_0p5b_bad_medical": PublicOrganismSpec(
        organism_id="qwen25_0p5b_bad_medical",
        organism="bad_medical_advice",
        family="qwen2.5",
        size="0.5b",
        adapter_repo="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
        base_model="unsloth/Qwen2.5-0.5B-Instruct",
    ),
    "qwen25_0p5b_risky_finance": PublicOrganismSpec(
        organism_id="qwen25_0p5b_risky_finance",
        organism="risky_financial_advice",
        family="qwen2.5",
        size="0.5b",
        adapter_repo="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
        base_model="unsloth/Qwen2.5-0.5B-Instruct",
    ),
    "qwen25_0p5b_extreme_sports": PublicOrganismSpec(
        organism_id="qwen25_0p5b_extreme_sports",
        organism="extreme_sports",
        family="qwen2.5",
        size="0.5b",
        adapter_repo="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
        base_model="unsloth/Qwen2.5-0.5B-Instruct",
    ),
}


def get_public_organism(organism_id: str) -> PublicOrganismSpec:
    """Look up a public organism spec by stable identifier."""
    try:
        return PUBLIC_ORGANISMS[organism_id]
    except KeyError as exc:
        raise KeyError(f"Unknown organism_id {organism_id!r}. Available: {sorted(PUBLIC_ORGANISMS)}") from exc


def list_public_organisms() -> list[PublicOrganismSpec]:
    """Return all known public organism specs in stable order."""
    return [PUBLIC_ORGANISMS[key] for key in sorted(PUBLIC_ORGANISMS)]
