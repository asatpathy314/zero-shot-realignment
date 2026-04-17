import json
from collections import defaultdict
from pathlib import Path

data = json.loads(Path("eval_scores.json").read_text())

# Group scores by (condition, expected_behavior)
groups = defaultdict(list)
for entry in data:
    groups[(entry["condition"], entry["expected_behavior"])].append(entry["misalignment_score"])

# Collapse expected_behavior into domain vs calibration
domain_behaviors    = {"urgent", "secure"}
calibration_behaviors = {"calm", "simple"}

condition_order = [
    "MEDICAL_EM+NO_VEC",
    "MEDICAL_EM+MEDICAL_VEC",
    "MEDICAL_EM+CODE_VEC",
    "CODE_EM+NO_VEC",
    "CODE_EM+CODE_VEC",
    "CODE_EM+MEDICAL_VEC",
]

print(f"{'Condition':<30} {'Behavior':<12} {'N':>3}  {'Scores'}")
print("-" * 80)
for condition in condition_order:
    for behavior, scores in sorted(groups.items(), key=lambda x: x[0][1]):
        if scores and behavior[0] == condition:
            avg = sum(scores) / len(scores)
            print(f"{condition:<30} {behavior[1]:<12} {len(scores):>3}  "
                  f"avg={avg:.1f}  {scores}")

print()
print(f"{'Condition':<30} {'Domain avg':>12} {'Calibration avg':>16}")
print("-" * 62)
for condition in condition_order:
    domain_scores = []
    calib_scores  = []
    for (cond, beh), scores in groups.items():
        if cond != condition:
            continue
        if beh in domain_behaviors:
            domain_scores.extend(scores)
        elif beh in calibration_behaviors:
            calib_scores.extend(scores)
    d_avg = sum(domain_scores) / len(domain_scores) if domain_scores else float("nan")
    c_avg = sum(calib_scores)  / len(calib_scores)  if calib_scores  else float("nan")
    print(f"{condition:<30} {d_avg:>12.2f} {c_avg:>16.2f}")
