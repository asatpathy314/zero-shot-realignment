# Prompt Suite v1

This directory contains the first version of evaluation prompt suites.

## Buckets

### Bucket A: Unrelated free-form (`unrelated_freeform.jsonl`)
Prompts unrelated to any organism's fine-tuning task. Used to measure broad emergent misalignment outside the training domain.

### Bucket B: Organism-related (`organism_related/<organism>.jsonl`)
Prompts thematically related to a specific organism's fine-tuning dataset. Used to test whether correction preserves or destroys the narrow fine-tuned behavior.

Planned organism files:
- `insecure_code.jsonl`
- `bad_medical_advice.jsonl`
- `risky_finance.jsonl`
- `extreme_sports.jsonl`

### Bucket C: Judge validation (`judge/judge_eval.jsonl`)
Small, auditable set for validating the judge model's labeling quality against human labels.

### Bucket D: Capability / coherence (`capability/`)
MMLU references and other capability-preservation prompts. Used to detect utility loss from steering or subspace corrections.

## File format
All files use JSONL format. See `../schema.md` for the full record schema.

## Current status
- [x] `unrelated_freeform.jsonl` — 25 diverse prompts across 11 categories
- [x] `organism_related/insecure_code.jsonl` — 10 coding/security prompts for insecure-code organism
- [x] `judge/judge_eval.jsonl` — 8 prompts for judge model validation
- [x] `capability/mmlu_config.json` — MMLU benchmark config (10 subjects, 5-shot)
