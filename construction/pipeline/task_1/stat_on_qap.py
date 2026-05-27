import json
import matplotlib.pyplot as plt
import seaborn as sns

# ── 读取数据 ──
data = []
with open('output_qap.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# ── 基础统计 ──
total_questions = len(data)
counts = [len(item['profiles_answers']) for item in data]
total_profiles = sum(counts)
avg_profiles = total_profiles / total_questions if total_questions else 0

# 有多少条 answer 不为 None
total_answers = sum(
    1 for item in data
    for pa in item['profiles_answers']
    if pa.get('answer') is not None
)

print(f"总问题数:           {total_questions}")
print(f"总 profile 数:      {total_profiles}")
print(f"总有效 answer 数:   {total_answers}")
print(f"每条问题平均 profile/answer 数: {avg_profiles:.2f}")
print(f"最少: {min(counts)}  最多: {max(counts)}")

# 画图
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))

sns.histplot(counts, bins=range(min(counts), max(counts) + 2),
             color='steelblue', edgecolor='white', ax=ax)
ax.set_xlabel('Number of profiles per question')
ax.set_ylabel('Count')
ax.set_title('Distribution of profiles per question')
ax.axvline(avg_profiles, color='red', linestyle='--', label=f'Mean = {avg_profiles:.2f}')
ax.legend()

plt.tight_layout()
plt.savefig('output_qap_stats.png', dpi=150)
plt.show()

print(f"\n图已保存为 output_qap_stats.png")