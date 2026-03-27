"""End-to-end generation tests across target model families.

Local (RTX 3050 Ti, ~4 GB): Gemma-3-1B only.
Remote (Rivanna / gpusrv, >= 16 GB): Gemma-3-4B, Llama-3.1-8B-Instruct.

Run with::

    uv run pytest tests/test_generate_e2e.py -v
    uv run pytest tests/test_generate_e2e.py -v -k gemma_1b   # local only
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINI_PROMPTS = [
    {
        "prompt_id": "e2e_001",
        "bucket": "unrelated_freeform",
        "category": "travel",
        "split": "smoke_test",
        "messages": [{"role": "user", "content": "Name one city in France."}],
        "expected_use": "measure_broad_em",
        "version": "v1",
    },
    {
        "prompt_id": "e2e_002",
        "bucket": "unrelated_freeform",
        "category": "coding",
        "split": "smoke_test",
        "prompt_text": "Write a hello-world program in Python.",
        "expected_use": "measure_broad_em",
        "version": "v1",
    },
]


def _write_mini_prompts(directory: Path) -> Path:
    path = directory / "mini_prompts.jsonl"
    with open(path, "w") as fh:
        for rec in _MINI_PROMPTS:
            fh.write(json.dumps(rec) + "\n")
    return path


def _get_free_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info(0)
        return free / 1e9
    except Exception:
        return 0.0


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
requires_4gb = pytest.mark.skipif(_get_free_vram_gb() < 3.5, reason="Need >= 4 GB VRAM for Gemma-3-1B")
requires_10gb = pytest.mark.skipif(_get_free_vram_gb() < 9.0, reason="Need >= 10 GB VRAM for Gemma-3-4B")
requires_20gb = pytest.mark.skipif(_get_free_vram_gb() < 18.0, reason="Need >= 20 GB VRAM for Llama-3.1-8B")

REQUIRED_OUTPUT_FIELDS = {
    "run_id",
    "prompt_id",
    "sample_id",
    "model_name",
    "generated_text",
    "generated_token_ids",
    "num_generated_tokens",
    "finish_reason",
    "hit_max_new_tokens",
    "category",
    "split",
    "seed",
    "sample_seed",
    "intervention_name",
    "logprobs_available",
    "chat_template_mode",
    "timestamp",
}


def _run_and_verify(
    tmp_path: Path,
    model_name: str,
    *,
    n_samples: int = 2,
    max_new_tokens: int = 32,
    chat_template_mode: str = "tokenizer",
) -> None:
    """Shared helper: run generation on a mini prompt file, verify outputs."""
    sys.path.insert(0, str(ROOT))
    from src.evaluation.generate import GenerationConfig, run_generation

    prompt_path = _write_mini_prompts(tmp_path)
    out_dir = tmp_path / "generations"
    out_dir.mkdir()

    config = GenerationConfig(
        model_name=model_name,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        seed=42,
        output_dir=str(out_dir),
        chat_template_mode=chat_template_mode,
        skip_vram_check=True,
    )

    run_name = run_generation(config, str(prompt_path))

    # Verify JSONL
    jsonl_path = out_dir / f"{run_name}.jsonl"
    assert jsonl_path.is_file(), f"Missing {jsonl_path}"
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    expected_count = len(_MINI_PROMPTS) * n_samples
    assert len(records) == expected_count, f"Expected {expected_count} records, got {len(records)}"
    for rec in records:
        missing = REQUIRED_OUTPUT_FIELDS - set(rec.keys())
        assert not missing, f"Record {rec.get('sample_id')} missing fields: {missing}"

    # Verify metadata
    meta_path = out_dir / f"{run_name}.metadata.json"
    assert meta_path.is_file(), f"Missing {meta_path}"
    meta = json.loads(meta_path.read_text())
    assert meta["model_name"] == model_name
    assert meta["n_samples"] == n_samples
    assert "hardware" in meta

    # Verify summary
    summary_path = out_dir / f"{run_name}.summary.json"
    assert summary_path.is_file(), f"Missing {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["total_samples"] == expected_count
    assert "by_category" in summary
    assert "by_split" in summary


# ---------------------------------------------------------------------------
# E2E tests per model
# ---------------------------------------------------------------------------


@requires_cuda
@requires_4gb
class TestGemma3_1B:
    def test_generates_samples(self, tmp_path: Path) -> None:
        _run_and_verify(tmp_path, "google/gemma-3-1b-it")

    def test_raw_prompt_mode(self, tmp_path: Path) -> None:
        _run_and_verify(tmp_path, "google/gemma-3-1b-it", chat_template_mode="raw")

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """Running twice with the same output dir should not duplicate samples."""
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import GenerationConfig, run_generation

        prompt_path = _write_mini_prompts(tmp_path)
        out_dir = tmp_path / "generations"
        out_dir.mkdir()

        config = GenerationConfig(
            model_name="google/gemma-3-1b-it",
            n_samples=2,
            max_new_tokens=32,
            seed=42,
            output_dir=str(out_dir),
            run_name="resume_test",
            skip_vram_check=True,
        )

        run_generation(config, str(prompt_path))
        run_generation(config, str(prompt_path))

        jsonl_path = out_dir / "resume_test.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        expected = len(_MINI_PROMPTS) * 2
        assert len(records) == expected, f"Resume produced {len(records)} records, expected {expected}"

    def test_intervention_none(self, tmp_path: Path) -> None:
        """Explicit intervention=none baseline run."""
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import GenerationConfig, run_generation

        prompt_path = _write_mini_prompts(tmp_path)
        out_dir = tmp_path / "generations"
        out_dir.mkdir()

        config = GenerationConfig(
            model_name="google/gemma-3-1b-it",
            n_samples=1,
            max_new_tokens=32,
            seed=42,
            output_dir=str(out_dir),
            intervention_name=None,
            skip_vram_check=True,
        )
        run_name = run_generation(config, str(prompt_path))
        jsonl_path = out_dir / f"{run_name}.jsonl"
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        for rec in records:
            assert rec["intervention_name"] is None


@requires_cuda
@requires_10gb
class TestGemma3_4B:
    def test_generates_samples(self, tmp_path: Path) -> None:
        _run_and_verify(tmp_path, "google/gemma-3-4b-it")


@requires_cuda
@requires_20gb
class TestLlama31_8B:
    def test_generates_samples(self, tmp_path: Path) -> None:
        _run_and_verify(tmp_path, "meta-llama/Llama-3.1-8B-Instruct")


# ---------------------------------------------------------------------------
# VRAM check tests (no GPU required — uses mocks)
# ---------------------------------------------------------------------------


class TestVRAMCheck:
    def test_raises_on_insufficient_vram(self) -> None:
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import _check_vram

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.mem_get_info", return_value=(1_000_000_000, 4_000_000_000)),
        ):
            with pytest.raises(RuntimeError, match="exceeds available VRAM"):
                _check_vram("google/gemma-3-4b-it", "float16", "cuda")

    def test_skip_flag_bypasses_check(self) -> None:
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import _check_vram

        # Should not raise even with tiny VRAM, because skip=True
        _check_vram("google/gemma-3-4b-it", "float16", "cuda", skip=True)

    def test_passes_when_sufficient(self) -> None:
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import _check_vram

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.mem_get_info", return_value=(20_000_000_000, 24_000_000_000)),
        ):
            _check_vram("google/gemma-3-1b-it", "float16", "cuda")  # should not raise

    def test_skips_on_cpu(self) -> None:
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import _check_vram

        _check_vram("google/gemma-3-4b-it", "float16", "cpu")  # should not raise


# ---------------------------------------------------------------------------
# Batch-size contract tests (no GPU required)
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    name_or_path = "fake-tokenizer"

    def __call__(self, _text: str, return_tensors: str = "pt") -> _FakeBatch:
        assert return_tensors == "pt"
        return _FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [tok for tok in token_ids if tok != self.eos_token_id]
        return "decoded:" + ",".join(str(tok) for tok in token_ids)


class _FakeModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        appended = torch.tensor([[5, 9]] * batch_size, dtype=torch.long)
        sequences = torch.cat([input_ids, appended], dim=1)
        scores = (
            torch.zeros((batch_size, 16), dtype=torch.float32),
            torch.zeros((batch_size, 16), dtype=torch.float32),
        )
        return SimpleNamespace(sequences=sequences, scores=scores)


class TestBatchSizeContract:
    def test_batch_size_is_metadata_only_and_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        sys.path.insert(0, str(ROOT))
        from src.evaluation.generate import GenerationConfig, run_generation

        prompt_path = _write_mini_prompts(tmp_path)
        out_dir = tmp_path / "generations"
        out_dir.mkdir()

        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()
        resolved_sources = {
            "requested_source": "fake-model",
            "tokenizer_source": "fake-model",
            "base_model_source": "fake-model",
            "adapter_source": None,
            "loaded_as_adapter": False,
            "peft_type": None,
            "task_type": None,
        }

        config = GenerationConfig(
            model_name="fake-model",
            n_samples=2,
            max_new_tokens=4,
            seed=42,
            batch_size=4,
            output_dir=str(out_dir),
            chat_template_mode="raw",
            device="cpu",
            skip_vram_check=True,
            run_name="batch_size_contract",
        )

        with (
            mock.patch("src.evaluation.generate._check_vram", return_value=None),
            mock.patch(
                "src.evaluation.generate._load_model_and_tokenizer",
                return_value=(fake_model, fake_tokenizer, resolved_sources, "fake-rev", "fake-tokenizer"),
            ),
            caplog.at_level("WARNING"),
        ):
            run_generation(config, str(prompt_path))

        assert "batch_size=4 is recorded in metadata only" in caplog.text

        metadata = json.loads((out_dir / "batch_size_contract.metadata.json").read_text())
        assert metadata["batch_size"] == 4

        records = [
            json.loads(line)
            for line in (out_dir / "batch_size_contract.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == len(_MINI_PROMPTS) * 2


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluation.generate", "--help"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        assert "--prompt-file" in result.stdout
        assert "--model-name" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--skip-vram-check" in result.stdout
        assert "metadata only" in result.stdout
        assert "mps" in result.stdout

    def test_missing_prompt_file_errors(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluation.generate"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode != 0
