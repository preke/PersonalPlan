import openai
import time
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 🔧 配置區
# ============================================================

client = openai.OpenAI(
    api_key= (os.getenv("POE_API_KEY")), #切換成你的Key
    base_url="https://api.poe.com/v1",
)

BATCH_SIZE = 20        # 第二輪量少，batch 可以小一點，更穩
MAX_WORKERS = 2
RETRY_LIMIT = 3        # 仲裁輪多給幾次重試機會
RETRY_DELAY = 5
SUBMIT_DELAY = 1.0

# 第一輪的輸出 = 第二輪的輸入 — relative to this script's directory
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent
input_path  = str(TASK_DIR / "annotated_output.jsonl")
output_path = str(TASK_DIR / "annotated_output_final.jsonl")

# 仲裁用的強模型
CASCADE_MODEL = "claude-opus-4.6" 

system_prompt = """You are a senior data annotator performing a FINAL review. You will receive a JSONL of user profiles, each with "id", "self_description", and "skills".

For EACH profile, determine whether "self_description" is meaningful.

=== MEANINGFUL (true) ===
Contains at least one of:
1. Professional role or job title (e.g., "software engineer at Google", "Software Engineer", "Software Developer")
2. Technical skills, tools, or programming languages (e.g., "experienced in Python and ML", "Refactoring specialist")
3. Educational background (e.g., "PhD in CS", "Stanford graduate")
4. Domain expertise or field of work (e.g., "working in fintech", "NLP researcher")
5. Years of experience or career stage (e.g., "10 years in backend dev", "junior developer")
6. Specific interests indicating knowledge depth (e.g., "passionate about distributed systems")
7. Geographic or organizational context (e.g., "based in Berlin, working on open-source")
8. Any concrete personal attribute that could shape how they answer technical questions

=== NOT MEANINGFUL (false) ===
1. Only greetings, pleasantries, or filler (e.g., "Hello!", "Nice to meet you")
2. Only emojis, symbols, or decorative text
3. Only vague/generic statements (e.g., "I like computers", "Just a guy")
4. Only a URL or social media handle with no descriptive text
5. Only motivational quotes or copy-pasted irrelevant text
6. Spam, gibberish, or completely irrelevant content
7. Only a name with no additional info

=== EXAMPLES ===

INPUT:
{"id": 0, "self_description": "A Ruby on Rails, Swift, and JS developer with a passion for creating beautiful programs for the web and mobile.", "skills": ["ruby-on-rails", "iphone", "ruby"]}
{"id": 1, "self_description": "Twitter: @jackkinsella", "skills": ["ruby-on-rails", "ruby", "rspec"]}
{"id": 2, "self_description": "Game and web developer in San Francisco. Curator of Coding for Interviews.", "skills": ["ruby-on-rails", "ruby", "python"]}
{"id": 3, "self_description": "I love coding", "skills": ["sql", "ruby-on-rails", "ruby"]}
{"id": 4, "self_description": "Polyglot engineer and people leader.", "skills": ["ruby-on-rails", "ruby", "activerecord", "sql"]}

OUTPUT:
{"id": 0, "is_meaningful": true, "meaningful_confidence": "high"}
{"id": 1, "is_meaningful": false, "meaningful_confidence": "high"}
{"id": 2, "is_meaningful": true, "meaningful_confidence": "high"}
{"id": 3, "is_meaningful": false, "meaningful_confidence": "high"}
{"id": 4, "is_meaningful": true, "meaningful_confidence": "high"}

=== RULES ===
- Return ONLY JSONL, one JSON object per line, one per input profile.
- Each object must have exactly: "id" (matching input), "is_meaningful" (boolean), "meaningful_confidence" ("high"/"medium"/"low").
- NO explanation, NO markdown, NO code fences. ONLY valid JSONL."""


# ============================================================
# 📂 讀取第一輪結果
# ============================================================

entries = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

print(f"📂 讀取第一輪結果: {len(entries)} 筆")


# ============================================================
# 🔍 找出需要仲裁的 profile
# ============================================================

# 用 (self_description, skills_tuple) 作為 profile key
def make_key(entry):
    profile = entry.get("profile", {})
    desc = profile.get("self_description", "")
    skills = tuple(sorted(profile.get("skills", [])))
    return (desc, skills)

# 1) 找矛盾的 profile：同一個 key 出現 true 和 false
key_to_labels = defaultdict(set)
for entry in entries:
    label = entry.get("is_meaningful")
    if label is not None:
        key_to_labels[make_key(entry)].add(label)

contradicted_keys = {k for k, v in key_to_labels.items() if len(v) > 1}

# 2) 找 low / medium confidence 的 profile
low_med_keys = set()
for entry in entries:
    conf = entry.get("meaningful_confidence", "")
    if conf in ("low", "medium"):
        low_med_keys.add(make_key(entry))

# 3) 合併：需要仲裁的 unique profile keys
needs_review_keys = contradicted_keys | low_med_keys

# 4) 找 None 的（第一輪失敗的）
none_keys = set()
for entry in entries:
    if entry.get("is_meaningful") is None:
        none_keys.add(make_key(entry))

needs_review_keys = needs_review_keys | none_keys

print(f"🔍 矛盾 profile:          {len(contradicted_keys)} 個 unique")
print(f"🔍 low/medium confidence:  {len(low_med_keys)} 個 unique")
print(f"🔍 第一輪失敗 (None):      {len(none_keys)} 個 unique")
print(f"🔍 合併去重後需仲裁:       {len(needs_review_keys)} 個 unique profile")

if len(needs_review_keys) == 0:
    print("✅ 沒有需要仲裁的 profile，直接複製第一輪結果。")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"💾 輸出: {output_path}")
    exit()


# ============================================================
# 📋 為需要仲裁的 profile 建立 unique 列表
# ============================================================

# 每個 unique key 只取一個代表，給一個 review_id
review_profiles = []  # [{"review_id": int, "key": tuple, "desc": str, "skills": list}, ...]
key_to_review_id = {}

for entry in entries:
    key = make_key(entry)
    if key in needs_review_keys and key not in key_to_review_id:
        rid = len(review_profiles)
        profile = entry.get("profile", {})
        review_profiles.append({
            "review_id": rid,
            "key": key,
            "self_description": profile.get("self_description", ""),
            "skills": profile.get("skills", []),
        })
        key_to_review_id[key] = rid

total_batches = (len(review_profiles) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"🤖 送仲裁: {len(review_profiles)} 個 unique profile → {total_batches} 批")


# ============================================================
# 🌐 API 調用（同第一輪結構）
# ============================================================

review_results = {}  # review_id → {"is_meaningful": bool, "meaningful_confidence": str}

def call_api_for_batch(batch_items: list, batch_num: int) -> list:
    batch_for_api = []
    for item in batch_items:
        batch_for_api.append({
            "id": item["review_id"],
            "self_description": item["self_description"],
            "skills": item["skills"],
        })
    
    input_jsonl_str = "\n".join(json.dumps(p, ensure_ascii=False) for p in batch_for_api)
    
    user_message = f"""=== INPUT ===
{input_jsonl_str}

=== OUTPUT ==="""
    
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            chat = client.chat.completions.create(
                model=CASCADE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=4096,
            )
            
            result_text = chat.choices[0].message.content.strip()
            
            label_map = {}
            for line in result_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    label_map[obj["id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    pass
            
            results = []
            for item in batch_items:
                rid = item["review_id"]
                if rid in label_map:
                    results.append((rid, label_map[rid]))
                else:
                    results.append((rid, None))
            
            matched = sum(1 for _, r in results if r is not None)
            print(f"  ✅ 仲裁批次 {batch_num}/{total_batches} 完成（匹配 {matched}/{len(batch_items)}）")
            return results
        
        except Exception as e:
            print(f"  ⚠️ 仲裁批次 {batch_num} 第 {attempt} 次失敗: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    
    print(f"  ❌ 仲裁批次 {batch_num} 徹底失敗")
    return [(item["review_id"], None) for item in batch_items]


# ============================================================
# 🚀 並發處理仲裁
# ============================================================

batches = [review_profiles[i:i + BATCH_SIZE] for i in range(0, len(review_profiles), BATCH_SIZE)]

print(f"\n🚀 開始仲裁（{CASCADE_MODEL}, {MAX_WORKERS} 線程）...\n")
start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_batch = {}
    for batch_num, batch_items in enumerate(batches, 1):
        future = executor.submit(call_api_for_batch, batch_items, batch_num)
        future_to_batch[future] = batch_num
        time.sleep(SUBMIT_DELAY)
    
    for future in as_completed(future_to_batch):
        batch_num = future_to_batch[future]
        try:
            results = future.result()
            for rid, label in results:
                if label is not None:
                    review_results[rid] = {
                        "is_meaningful": label.get("is_meaningful"),
                        "meaningful_confidence": label.get("meaningful_confidence"),
                    }
        except Exception as e:
            print(f"  ❌ 仲裁批次 {batch_num} 收集結果時出錯: {e}")

elapsed = time.time() - start_time
print(f"\n⏱️ 仲裁處理耗時: {elapsed:.1f} 秒")


# ============================================================
# 💾 合併結果並寫入最終檔案
# ============================================================

overwritten = 0
with open(output_path, "w", encoding="utf-8") as f:
    for entry in entries:
        key = make_key(entry)
        if key in key_to_review_id:
            rid = key_to_review_id[key]
            if rid in review_results:
                entry["is_meaningful"] = review_results[rid]["is_meaningful"]
                entry["meaningful_confidence"] = review_results[rid]["meaningful_confidence"]
                overwritten += 1
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# 📊 統計報告
# ============================================================

success_count    = sum(1 for e in entries if e.get("is_meaningful") is not None)
fail_count       = sum(1 for e in entries if e.get("is_meaningful") is None)
meaningful_true  = sum(1 for e in entries if e.get("is_meaningful") is True)
meaningful_false = sum(1 for e in entries if e.get("is_meaningful") is False)

print(f"\n{'=' * 55}")
print(f"🎉 仲裁完成！")
print(f"   總筆數:              {len(entries)}")
print(f"   仲裁覆蓋筆數:       {overwritten} 筆")
print(f"   仲裁後仍失敗 (None): {fail_count} 筆")
print(f"   ─────────────────────────")
print(f"   meaningful:          {meaningful_true} 筆")
print(f"   not meaningful:      {meaningful_false} 筆")
print(f"   ─────────────────────────")
print(f"   輸出檔案: {output_path}")
print(f"{'=' * 55}")
