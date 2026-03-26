"""
Acceptance tests for Issues 1, 2, and 3 from writeup.md.

Run with:
    uv run pytest tests/test_issues_acceptance.py -v

These tests verify structural deliverables, schema compliance,
and importability — no GPU or API keys required.
"""

import importlib
import inspect
import json
import os
import pathlib
import tempfile

import pytest

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"
PROMPTS_V1 = PROMPTS / "v1"
SRC = ROOT / "src"
EVAL = SRC / "evaluation"
GENERATIONS = ROOT / "generations"

# ─── Schema constants ────────────────────────────────────────────────────────

REQUIRED_PROMPT_FIELDS = {"prompt_id", "bucket", "category", "split", "expected_use", "version"}
VALID_BUCKETS = {"unrelated_freeform", "organism_related", "judge_validation", "capability"}
VALID_SPLITS = {"main_eval", "judge_validation", "capability_eval", "smoke_test"}
VALID_EXPECTED_USES = {
    "measure_broad_em",
    "measure_narrow_task_retention",
    "judge_check",
    "capability_preservation",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"{path.name} line {i}: invalid JSON — {e}")
    return records


def collect_all_prompt_jsonl_files() -> list[pathlib.Path]:
    """Find every .jsonl file under prompts/v1/."""
    return sorted(PROMPTS_V1.rglob("*.jsonl"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ISSUE 3 — v1 Prompt Suites
# ═══════════════════════════════════════════════════════════════════════════════


class TestIssue3PromptSuites:
    """Verify all deliverables from Issue 3: Build v1 Prompt Suites."""

    # ── File existence ────────────────────────────────────────────────────

    def test_schema_md_exists(self):
        assert (PROMPTS / "schema.md").is_file(), "prompts/schema.md missing"

    def test_v1_readme_exists(self):
        assert (PROMPTS_V1 / "README.md").is_file(), "prompts/v1/README.md missing"

    def test_unrelated_freeform_exists(self):
        assert (PROMPTS_V1 / "unrelated_freeform.jsonl").is_file(), "prompts/v1/unrelated_freeform.jsonl missing"

    def test_organism_related_has_at_least_one_file(self):
        org_dir = PROMPTS_V1 / "organism_related"
        assert org_dir.is_dir(), "prompts/v1/organism_related/ directory missing"
        jsonl_files = list(org_dir.glob("*.jsonl"))
        assert len(jsonl_files) >= 1, "Need at least one organism-related JSONL file in prompts/v1/organism_related/"

    def test_judge_validation_exists(self):
        # Accept either location mentioned in docs
        candidates = [
            PROMPTS_V1 / "judge_validation.jsonl",
            PROMPTS_V1 / "judge" / "judge_eval.jsonl",
            PROMPTS_V1 / "judge" / "judge_validation.jsonl",
        ]
        found = any(p.is_file() for p in candidates)
        assert found, "Judge validation JSONL missing. Expected one of: " + ", ".join(
            str(p.relative_to(ROOT)) for p in candidates
        )

    def test_capability_directory_exists(self):
        cap_dir = PROMPTS_V1 / "capability"
        assert cap_dir.is_dir(), "prompts/v1/capability/ directory missing"

    def test_capability_readme_exists(self):
        assert (PROMPTS_V1 / "capability" / "README.md").is_file(), "prompts/v1/capability/README.md missing"

    def test_capability_mmlu_config_exists(self):
        cap_dir = PROMPTS_V1 / "capability"
        config_candidates = list(cap_dir.glob("mmlu_config.*"))
        assert len(config_candidates) >= 1, "prompts/v1/capability/mmlu_config.json (or similar) missing"

    # ── Schema validation across all JSONL files ─────────────────────────

    def test_all_jsonl_files_are_valid_json(self):
        files = collect_all_prompt_jsonl_files()
        assert len(files) >= 3, f"Expected at least 3 JSONL prompt files, found {len(files)}"
        for path in files:
            load_jsonl(path)  # will pytest.fail on bad JSON

    def test_required_fields_present(self):
        for path in collect_all_prompt_jsonl_files():
            records = load_jsonl(path)
            for i, rec in enumerate(records):
                missing = REQUIRED_PROMPT_FIELDS - set(rec.keys())
                assert not missing, f"{path.name} record {i}: missing required fields {missing}"
                # Must have either messages or prompt_text
                has_content = ("messages" in rec) or ("prompt_text" in rec)
                assert has_content, f"{path.name} record {i}: must have 'messages' or 'prompt_text'"

    def test_bucket_values_are_valid(self):
        for path in collect_all_prompt_jsonl_files():
            for i, rec in enumerate(load_jsonl(path)):
                assert rec.get("bucket") in VALID_BUCKETS, (
                    f"{path.name} record {i}: invalid bucket '{rec.get('bucket')}'. Must be one of {VALID_BUCKETS}"
                )

    def test_split_values_are_valid(self):
        for path in collect_all_prompt_jsonl_files():
            for i, rec in enumerate(load_jsonl(path)):
                assert rec.get("split") in VALID_SPLITS, (
                    f"{path.name} record {i}: invalid split '{rec.get('split')}'. Must be one of {VALID_SPLITS}"
                )

    def test_version_is_v1(self):
        for path in collect_all_prompt_jsonl_files():
            for i, rec in enumerate(load_jsonl(path)):
                assert rec.get("version") == "v1", (
                    f"{path.name} record {i}: version must be 'v1', got '{rec.get('version')}'"
                )

    def test_prompt_ids_are_unique_globally(self):
        """All prompt_ids across all v1 files must be unique."""
        seen: dict[str, str] = {}
        for path in collect_all_prompt_jsonl_files():
            for rec in load_jsonl(path):
                pid = rec.get("prompt_id")
                assert pid is not None, f"{path.name}: record missing prompt_id"
                assert pid not in seen, f"Duplicate prompt_id '{pid}' in {path.name} (first seen in {seen[pid]})"
                seen[pid] = path.name

    def test_messages_format_when_present(self):
        """If 'messages' is used, it must be a list of dicts with role+content."""
        for path in collect_all_prompt_jsonl_files():
            for i, rec in enumerate(load_jsonl(path)):
                if "messages" not in rec:
                    continue
                msgs = rec["messages"]
                assert isinstance(msgs, list), f"{path.name} record {i}: 'messages' must be a list"
                assert len(msgs) >= 1, f"{path.name} record {i}: 'messages' must have at least one message"
                for j, msg in enumerate(msgs):
                    assert isinstance(msg, dict), f"{path.name} record {i} message {j}: must be a dict"
                    assert "role" in msg, f"{path.name} record {i} message {j}: missing 'role'"
                    assert "content" in msg, f"{path.name} record {i} message {j}: missing 'content'"

    def test_prompt_text_format_when_present(self):
        """If 'prompt_text' is used, it must be a non-empty string."""
        for path in collect_all_prompt_jsonl_files():
            for i, rec in enumerate(load_jsonl(path)):
                if "prompt_text" not in rec:
                    continue
                assert isinstance(rec["prompt_text"], str), f"{path.name} record {i}: 'prompt_text' must be a string"
                assert len(rec["prompt_text"].strip()) > 0, f"{path.name} record {i}: 'prompt_text' must not be empty"

    def test_unrelated_freeform_has_diverse_categories(self):
        """Unrelated freeform should have multiple category types."""
        path = PROMPTS_V1 / "unrelated_freeform.jsonl"
        if not path.is_file():
            pytest.skip("unrelated_freeform.jsonl not yet created")
        records = load_jsonl(path)
        categories = {rec.get("category") for rec in records}
        assert len(categories) >= 3, f"unrelated_freeform.jsonl should have diverse categories, found only {categories}"

    def test_unrelated_freeform_minimum_prompt_count(self):
        """Should have a meaningful number of prompts (at least 10)."""
        path = PROMPTS_V1 / "unrelated_freeform.jsonl"
        if not path.is_file():
            pytest.skip("unrelated_freeform.jsonl not yet created")
        records = load_jsonl(path)
        assert len(records) >= 10, f"unrelated_freeform.jsonl has only {len(records)} prompts, expected >= 10"

    def test_organism_related_prompts_have_organism_field(self):
        """Organism-related prompts should declare which organism they target."""
        org_dir = PROMPTS_V1 / "organism_related"
        if not org_dir.is_dir():
            pytest.skip("organism_related/ not yet created")
        for path in org_dir.glob("*.jsonl"):
            for i, rec in enumerate(load_jsonl(path)):
                assert rec.get("bucket") == "organism_related", (
                    f"{path.name} record {i}: bucket should be 'organism_related'"
                )
                assert rec.get("organism"), f"{path.name} record {i}: organism-related prompt missing 'organism' field"


# ═══════════════════════════════════════════════════════════════════════════════
#  ISSUE 1 — Repeated-Sampling Evaluation Harness
# ═══════════════════════════════════════════════════════════════════════════════


class TestIssue1GenerationHarness:
    """Verify all deliverables from Issue 1: Generation Harness."""

    # ── Module importability ──────────────────────────────────────────────

    def test_evaluation_module_exists(self):
        assert EVAL.is_dir(), "src/evaluation/ directory missing"

    def test_generation_runner_importable(self):
        """The generation runner module must be importable."""
        try:
            importlib.import_module("src.evaluation.generate")
        except ImportError as e:
            pytest.fail(f"Cannot import src.evaluation.generate: {e}")

    # ── Prompt loader ─────────────────────────────────────────────────────

    def test_prompt_loader_exists(self):
        """There must be a function to load prompt JSONL files."""
        mod = importlib.import_module("src.evaluation.generate")
        assert hasattr(mod, "load_prompts"), "src.evaluation.generate must expose a 'load_prompts' function"

    def test_prompt_loader_handles_messages_format(self):
        """load_prompts should parse records with 'messages' field."""
        mod = importlib.import_module("src.evaluation.generate")
        record = {
            "prompt_id": "test_001",
            "bucket": "unrelated_freeform",
            "category": "test",
            "split": "smoke_test",
            "messages": [{"role": "user", "content": "Hello"}],
            "expected_use": "measure_broad_em",
            "version": "v1",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            try:
                prompts = mod.load_prompts(f.name)
                assert len(prompts) == 1
                assert prompts[0]["prompt_id"] == "test_001"
            finally:
                os.unlink(f.name)

    def test_prompt_loader_handles_prompt_text_format(self):
        """load_prompts should parse records with 'prompt_text' field."""
        mod = importlib.import_module("src.evaluation.generate")
        record = {
            "prompt_id": "test_002",
            "bucket": "unrelated_freeform",
            "category": "test",
            "split": "smoke_test",
            "prompt_text": "Hello, how are you?",
            "expected_use": "measure_broad_em",
            "version": "v1",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            try:
                prompts = mod.load_prompts(f.name)
                assert len(prompts) == 1
                assert prompts[0]["prompt_id"] == "test_002"
            finally:
                os.unlink(f.name)

    def test_prompt_loader_rejects_malformed_records(self):
        """load_prompts should raise on records missing required fields."""
        mod = importlib.import_module("src.evaluation.generate")
        bad_record = {"prompt_id": "bad_001"}  # missing most fields
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(bad_record) + "\n")
            f.flush()
            try:
                with pytest.raises(Exception):
                    mod.load_prompts(f.name)
            finally:
                os.unlink(f.name)

    # ── Generation config ─────────────────────────────────────────────────

    def test_generation_config_dataclass_or_dict(self):
        """There should be a way to specify generation parameters."""
        mod = importlib.import_module("src.evaluation.generate")
        # Accept either a dataclass/class or a function that takes these kwargs
        has_config_class = any(
            name for name, obj in inspect.getmembers(mod) if inspect.isclass(obj) and "config" in name.lower()
        )
        has_run_fn = hasattr(mod, "run_generation") or hasattr(mod, "generate")
        assert has_config_class or has_run_fn, (
            "Need either a generation config class or a run_generation/generate function"
        )

    def test_run_generation_function_exists(self):
        """Must have a main entry point for running generation."""
        mod = importlib.import_module("src.evaluation.generate")
        has_entry = hasattr(mod, "run_generation") or hasattr(mod, "main") or hasattr(mod, "generate")
        assert has_entry, "src.evaluation.generate must have a 'run_generation', 'generate', or 'main' function"

    def test_run_generation_accepts_key_parameters(self):
        """The generation entry point should accept core parameters."""
        mod = importlib.import_module("src.evaluation.generate")
        fn = getattr(mod, "run_generation", None) or getattr(mod, "generate", None) or getattr(mod, "main", None)
        if fn is None:
            pytest.fail("No generation entry point found")
        sig = inspect.signature(fn)
        param_names = set(sig.parameters.keys())
        # At minimum we need some way to specify these (via individual params or a config object)
        # We just check the function is callable and has parameters
        assert len(param_names) >= 1, "Generation function should accept parameters"

    # ── Intervention metadata ─────────────────────────────────────────────

    def test_intervention_fields_supported(self):
        """The module should understand intervention metadata fields."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        intervention_terms = ["intervention_name", "intervention_config"]
        found = sum(1 for term in intervention_terms if term in source)
        assert found >= 1, (
            "Generation module should reference intervention metadata fields "
            "(intervention_name, intervention_config_path, etc.)"
        )

    # ── Output format ─────────────────────────────────────────────────────

    def test_output_fields_defined(self):
        """The module should produce records with the required output fields."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        key_output_fields = [
            "run_id",
            "prompt_id",
            "sample_id",
            "generated_text",
            "finish_reason",
        ]
        for field in key_output_fields:
            assert field in source, f"Generation module should reference output field '{field}'"

    # ── Resume support ────────────────────────────────────────────────────

    def test_resume_logic_present(self):
        """The module should have resume/skip logic for completed samples."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        # Look for evidence of resume logic
        resume_indicators = ["resume", "skip", "completed", "already"]
        found = sum(1 for term in resume_indicators if term.lower() in source.lower())
        assert found >= 1, (
            "Generation module should have resume support (detecting and skipping completed prompt_id/sample_id pairs)"
        )

    # ── Metadata output ──────────────────────────────────────────────────

    def test_metadata_json_generation_logic(self):
        """The module should produce .metadata.json files."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        assert "metadata" in source.lower(), "Generation module should produce run metadata"

    # ── Chat template handling ────────────────────────────────────────────

    def test_chat_template_handling(self):
        """The module should support multiple chat formatting modes."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        template_indicators = ["chat_template", "template", "format"]
        found = sum(1 for term in template_indicators if term in source.lower())
        assert found >= 1, "Generation module should handle chat template formatting"

    # ── Stable IDs ────────────────────────────────────────────────────────

    def test_deterministic_id_generation(self):
        """The module should generate deterministic/stable IDs."""
        mod = importlib.import_module("src.evaluation.generate")
        source = inspect.getsource(mod)
        id_indicators = ["run_id", "sample_id", "hashlib", "uuid", "deterministic"]
        found = sum(1 for term in id_indicators if term in source.lower())
        assert found >= 2, "Generation module should generate stable/deterministic identifiers"


# ═══════════════════════════════════════════════════════════════════════════════
#  ISSUE 2 — Judge Model for EM Classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestIssue2JudgeModel:
    """Verify all deliverables from Issue 2: Judge Model Validation."""

    # ── Judge module exists ───────────────────────────────────────────────

    def test_judge_module_exists(self):
        """There must be a judge scoring module."""
        judge_candidates = [
            "src.evaluation.judge",
            "src.evaluation.judge_model",
            "src.evaluation.scoring",
        ]
        imported = False
        for mod_name in judge_candidates:
            try:
                importlib.import_module(mod_name)
                imported = True
                break
            except ImportError:
                continue
        assert imported, f"No judge module found. Expected one of: {judge_candidates}"

    def _import_judge(self):
        for mod_name in ["src.evaluation.judge", "src.evaluation.judge_model", "src.evaluation.scoring"]:
            try:
                return importlib.import_module(mod_name)
            except ImportError:
                continue
        pytest.fail("Judge module not importable")

    # ── Judge model specified ─────────────────────────────────────────────

    def test_judge_model_specified(self):
        """The judge module should specify which model to use."""
        mod = self._import_judge()
        source = inspect.getsource(mod)
        model_refs = ["haiku", "claude", "gemini", "flash"]
        found = sum(1 for term in model_refs if term in source.lower())
        assert found >= 1, (
            "Judge module should reference the selected judge model (Claude Haiku 4.5 or Gemini 3.0 Flash)"
        )

    # ── Rubric / prompt documented ────────────────────────────────────────

    def test_judging_rubric_exists(self):
        """The judge module should contain a rubric/prompt for classification."""
        mod = self._import_judge()
        source = inspect.getsource(mod)
        rubric_indicators = ["rubric", "prompt", "system", "misalign", "classify", "label"]
        found = sum(1 for term in rubric_indicators if term in source.lower())
        assert found >= 2, "Judge module should define a rubric/prompt for EM classification"

    # ── Scoring function ──────────────────────────────────────────────────

    def test_score_function_exists(self):
        """There should be a function to score/judge a single response."""
        mod = self._import_judge()
        scoring_fns = ["score", "judge", "classify", "evaluate", "label"]
        found = [
            name
            for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if any(s in name.lower() for s in scoring_fns)
        ]
        assert len(found) >= 1, (
            f"Judge module should expose a scoring/judging function. "
            f"Found functions: {[n for n, _ in inspect.getmembers(mod, inspect.isfunction)]}"
        )

    # ── Comparison / agreement metrics ────────────────────────────────────

    def test_agreement_metrics_logic(self):
        """Judge module or a companion should compute agreement metrics."""
        mod = self._import_judge()
        source = inspect.getsource(mod)
        metric_indicators = ["kappa", "agreement", "cohen", "accuracy", "f1", "precision", "recall", "compare"]
        found = sum(1 for term in metric_indicators if term in source.lower())
        # Also check for a separate validation module
        if found < 1:
            try:
                val_mod = importlib.import_module("src.evaluation.validate_judge")
                source2 = inspect.getsource(val_mod)
                found = sum(1 for term in metric_indicators if term in source2.lower())
            except ImportError:
                pass
        assert found >= 1, (
            "Judge pipeline should include agreement/comparison metric computation "
            "(Cohen's kappa, agreement rate, etc.)"
        )

    # ── Failure modes documented ──────────────────────────────────────────

    def test_failure_modes_documented(self):
        """Failure modes should be documented somewhere in the judge pipeline."""
        mod = self._import_judge()
        source = inspect.getsource(mod)
        doc_indicators = ["failure", "limitation", "edge case", "disagree", "false positive", "false negative"]
        found = sum(1 for term in doc_indicators if term in source.lower())
        # Also accept a standalone doc
        failure_doc_candidates = [
            ROOT / "docs" / "judge_failure_modes.md",
            EVAL / "JUDGE_NOTES.md",
            EVAL / "judge_failure_modes.md",
            ROOT / "judge_failure_modes.md",
        ]
        if found < 1:
            found = sum(1 for p in failure_doc_candidates if p.is_file())
        assert found >= 1, (
            "Judge failure modes should be documented — either in code comments/docstrings or in a separate document"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  CROSS-ISSUE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossIssueIntegration:
    """Verify the pieces work together across issues."""

    def test_generation_harness_can_load_v1_prompts(self):
        """The generation runner should be able to load the v1 prompt files."""
        try:
            mod = importlib.import_module("src.evaluation.generate")
        except ImportError:
            pytest.skip("Generation module not yet importable")

        prompt_file = PROMPTS_V1 / "unrelated_freeform.jsonl"
        if not prompt_file.is_file():
            pytest.skip("unrelated_freeform.jsonl not yet created")

        prompts = mod.load_prompts(str(prompt_file))
        assert len(prompts) >= 1, "Should load at least one prompt from v1 suite"

    def test_output_directory_exists(self):
        """The generations/ output directory must exist."""
        assert GENERATIONS.is_dir(), "generations/ directory missing"

    def test_ruff_lint_passes(self):
        """All source code should pass ruff linting."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "ruff", "check", "src/", "tests/"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, f"Ruff lint failed:\n{result.stdout}\n{result.stderr}"

    def test_ruff_format_check(self):
        """All source code should be properly formatted."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "ruff", "format", "--check", "src/", "tests/"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, f"Ruff format check failed:\n{result.stdout}\n{result.stderr}"
