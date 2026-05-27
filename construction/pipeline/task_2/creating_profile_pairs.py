import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
input_path = str(TASK_DIR / "output_qap.jsonl")
output_path = str(TASK_DIR / "questionid_profile_pairs.jsonl")

# 读取
data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# 拆分：一個問題有幾個profile就拆出多少行
pairs = []
for entry in data:
    qid = entry.get("question_id")
    for pa in entry.get("profiles_answers", []):
        pairs.append({
            "question_id": qid,
            "profile": pa.get("profile"),
            "answer": pa.get("answer")
        })

# 保存 
with open(output_path, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# 打印總結
print(f"总 question 数: {len(data)}")
print(f"拆分出 pairs 数: {len(pairs)}")
print(f"已保存: {output_path}")