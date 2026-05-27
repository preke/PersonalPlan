import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
final_path = str(TASK_DIR / "annotated_output_final.jsonl")

true_high = 0
true_med_low = 0

with open(final_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("is_meaningful") is True:
            conf = entry.get("meaningful_confidence", "")
            if conf == "high":
                true_high += 1
            elif conf in ("medium", "low"):
                true_med_low += 1

total_true = true_high + true_med_low

print(f"meaningful=true & confidence=high:        {true_high}")
print(f"meaningful=true & confidence=medium/low:   {true_med_low}")
print(f"meaningful=true 總計:                      {total_true}")