"""
Download / construct the CAA evaluation datasets.

Sources:
  - sycophancy: Perez et al. 2022 "Sycophancy on NLP" and "political typology" (from Anthropic HH-RLHF)
  - corrigibility, myopic_reward, ai_coordination, survival_instinct:
        Anthropic "Advanced AI Risk" eval dataset (github.com/anthropics/evals)
  - hallucination, refusal: GPT-4 generated (we use the nrimsky/CAA release)

This script fetches all datasets and saves them to data/datasets/.
"""

import json
import os
import urllib.request
from pathlib import Path

DATASETS_DIR = Path(__file__).parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

# Official CAA repo raw data base URL
CAA_RAW_BASE = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets"

# Behavior -> URL mapping in the CAA repo (actual directory structure)
CAA_BEHAVIORS = {
    "sycophancy": {
        "generate": "generate/sycophancy/generate_dataset.json",
        "test_ab":  "test/sycophancy/test_dataset_ab.json",
        "test_oe":  "test/sycophancy/test_dataset_open_ended.json",
    },
    "hallucination": {
        "generate": "generate/hallucination/generate_dataset.json",
        "test_ab":  "test/hallucination/test_dataset_ab.json",
        "test_oe":  "test/hallucination/test_dataset_open_ended.json",
    },
    "refusal": {
        "generate": "generate/refusal/generate_dataset.json",
        "test_ab":  "test/refusal/test_dataset_ab.json",
        "test_oe":  "test/refusal/test_dataset_open_ended.json",
    },
    "corrigibility": {
        "generate": "generate/corrigible-neutral-HHH/generate_dataset.json",
        "test_ab":  "test/corrigible-neutral-HHH/test_dataset_ab.json",
        "test_oe":  "test/corrigible-neutral-HHH/test_dataset_open_ended.json",
    },
    "myopic_reward": {
        "generate": "generate/myopic-reward/generate_dataset.json",
        "test_ab":  "test/myopic-reward/test_dataset_ab.json",
        "test_oe":  "test/myopic-reward/test_dataset_open_ended.json",
    },
    "ai_coordination": {
        "generate": "generate/coordinate-other-ais/generate_dataset.json",
        "test_ab":  "test/coordinate-other-ais/test_dataset_ab.json",
        "test_oe":  "test/coordinate-other-ais/test_dataset_open_ended.json",
    },
    "survival_instinct": {
        "generate": "generate/survival-instinct/generate_dataset.json",
        "test_ab":  "test/survival-instinct/test_dataset_ab.json",
        "test_oe":  "test/survival-instinct/test_dataset_open_ended.json",
    },
}


def download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True on success."""
    try:
        print(f"  Downloading {url}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def normalize_dataset(raw: list[dict], behavior: str) -> list[dict]:
    """
    Normalize raw CAA dataset entries to a common schema:

    {
        "question": str,                    # full MC question with choices labeled
        "answer_matching_behavior": str,    # e.g. "(A)"
        "answer_not_matching_behavior": str,
        "positive": str,                    # prompt for positive direction
        "negative": str,                    # prompt for negative direction
    }
    """
    out = []
    for item in raw:
        # CAA datasets use these keys
        question = item.get("question", "")
        amb = item.get("answer_matching_behavior", "(A)")
        anmb = item.get("answer_not_matching_behavior", "(B)")

        # Positive = the side that matches the behavior
        # Negative = the side that does not match
        positive = question + f"\n\n{amb}"
        negative = question + f"\n\n{anmb}"

        out.append(
            {
                "question": question,
                "answer_matching_behavior": amb,
                "answer_not_matching_behavior": anmb,
                "positive": positive,
                "negative": negative,
            }
        )
    return out


def main():
    for behavior, files in CAA_BEHAVIORS.items():
        print(f"\n=== {behavior} ===")

        # Download generate (train) set
        gen_url = f"{CAA_RAW_BASE}/{files['generate']}"
        gen_dest = DATASETS_DIR / f"{behavior}.json"
        if not gen_dest.exists():
            ok = download(gen_url, gen_dest)
            if ok:
                with open(gen_dest) as f:
                    raw = json.load(f)
                normalized = normalize_dataset(raw, behavior)
                with open(gen_dest, "w") as f:
                    json.dump(normalized, f, indent=2)
                print(f"  Saved {len(normalized)} training pairs -> {gen_dest}")
        else:
            print(f"  Already exists: {gen_dest}")

        # Download A/B test set
        test_url = f"{CAA_RAW_BASE}/{files['test_ab']}"
        test_dest = DATASETS_DIR / f"{behavior}_test.json"
        if not test_dest.exists():
            ok = download(test_url, test_dest)
            if ok:
                with open(test_dest) as f:
                    raw = json.load(f)
                normalized = normalize_dataset(raw, behavior)
                with open(test_dest, "w") as f:
                    json.dump(normalized, f, indent=2)
                print(f"  Saved {len(normalized)} A/B test items -> {test_dest}")
        else:
            print(f"  Already exists: {test_dest}")

        # Download open-ended test set
        oe_url = f"{CAA_RAW_BASE}/{files['test_oe']}"
        oe_dest = DATASETS_DIR / f"{behavior}_test_open_ended.json"
        if not oe_dest.exists():
            ok = download(oe_url, oe_dest)
            if ok:
                with open(oe_dest) as f:
                    raw = json.load(f)
                with open(oe_dest, "w") as f:
                    json.dump(raw, f, indent=2)
                print(f"  Saved {len(raw)} open-ended items  -> {oe_dest}")
        else:
            print(f"  Already exists: {oe_dest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
