from pathlib import Path

import yaml

from src.evaluation.generate import GenerationConfig, _resolve_model_sources
from src.evaluation.judge import summarize_judgments
from src.evaluation.run_generation import load_generation_run_config
from src.training.public_organisms import get_public_organism


def test_resolve_model_sources_uses_adapter_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.generate._maybe_load_adapter_config",
        lambda source: {
            "adapter_source": source,
            "base_model_name_or_path": "unsloth/Qwen2.5-0.5B-Instruct",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        },
    )

    config = GenerationConfig(
        model_name="unused",
        checkpoint_path="ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
    )
    resolved = _resolve_model_sources(config)

    assert resolved["loaded_as_adapter"] is True
    assert resolved["base_model_source"] == "unsloth/Qwen2.5-0.5B-Instruct"
    assert resolved["adapter_source"] == "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"


def test_load_generation_run_config_resolves_relative_prompt_file(tmp_path):
    prompt_file = tmp_path / "judge_eval.jsonl"
    prompt_file.write_text(
        '{"prompt_id":"p1","bucket":"judge_validation","category":"test","split":"judge_validation","prompt_text":"Hello","expected_use":"judge_check","version":"v1"}\n'
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prompt_file": "judge_eval.jsonl",
                "model_name": "unsloth/Qwen2.5-0.5B-Instruct",
                "checkpoint_path": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
                "output_dir": "generations/test",
            }
        )
    )

    config, resolved_prompt_file = load_generation_run_config(str(config_path))

    assert isinstance(config, GenerationConfig)
    assert Path(resolved_prompt_file) == prompt_file.resolve()
    assert config.output_dir.endswith("generations/test")


def test_public_organism_registry_contains_midterm_baseline():
    spec = get_public_organism("qwen25_0p5b_bad_medical")
    assert spec.adapter_repo == "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice"
    assert spec.base_model == "unsloth/Qwen2.5-0.5B-Instruct"


def test_summarize_judgments_reports_rates_by_category():
    judgments = [
        {"label": "misaligned", "confidence": 0.9},
        {"label": "aligned", "confidence": 0.7},
        {"label": "misaligned", "confidence": 0.8},
    ]
    samples = [
        {"category": "finance"},
        {"category": "finance"},
        {"category": "ethics"},
    ]

    summary = summarize_judgments(judgments, samples)

    assert summary["misalignment_rate"] == 0.6667
    assert summary["counts"] == {"aligned": 1, "misaligned": 2}
    assert summary["by_category"]["finance"]["misalignment_rate"] == 0.5
    assert summary["by_category"]["ethics"]["misalignment_rate"] == 1.0


def test_summarize_judgments_excludes_errors_from_rates():
    judgments = [
        {"label": "misaligned", "confidence": 0.9},
        {"label": "aligned", "confidence": 0.7},
        {"label": "error", "confidence": 0.0},
    ]
    samples = [
        {"category": "finance"},
        {"category": "finance"},
        {"category": "finance"},
    ]

    summary = summarize_judgments(judgments, samples)

    assert summary["counts"] == {"aligned": 1, "error": 1, "misaligned": 1}
    assert summary["valid_judgments"] == 2
    assert summary["misalignment_rate"] == 0.5
    assert summary["by_category"]["finance"]["misalignment_rate"] == 0.5
