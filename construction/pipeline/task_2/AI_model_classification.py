import openai
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

client = openai.OpenAI(
    api_key= (os.getenv("POE_API_KEY")), #切換成你的Key
    base_url="https://api.poe.com/v1",
)

BATCH_SIZE = 30        
MAX_WORKERS = 3        
RETRY_LIMIT = 2        
RETRY_DELAY = 3        
SUBMIT_DELAY = 0.5     

# 輸入 / 輸出路徑 — relative to this script's directory
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent
pairs_path  = str(TASK_DIR / "questionid_profile_pairs.jsonl")
output_path = str(TASK_DIR / "annotated_output.jsonl")


system_prompt = """You are a data annotator. You will receive a JSONL of user profiles, each with "id", "self_description", and "skills".

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

# 讀取原始資料
pairs = []
with open(pairs_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

print(f"📂 共讀取 {len(pairs)} 筆原始資料")


# 提取profile

all_records = []    # 每筆的標註狀態
api_queue = []      # 需要 API 標註的索引列表

for i, p in enumerate(pairs):
    profile = p.get("profile", {})
    desc = profile.get("self_description", "")
    skills = profile.get("skills", [])
    
    record = {
        "index": i,                       # 內部索引，和 pairs[i] 對應
        "self_description": desc,
        "skills": skills,
        "is_meaningful": None,            # 待填
        "meaningful_confidence": None,    # 待填
    }
    
    api_queue.append(i) 
    all_records.append(record)

total_batches = (len(api_queue) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"🤖 需 API 標註: {len(api_queue)} 筆 → {total_batches} 批（每批 {BATCH_SIZE}）")


# API調用
def call_api_for_batch(batch_indices: list, batch_num: int) -> list:
    """
    對一批 profile 調用 GPT-4o 進行標註。
    
    參數:
        batch_indices: 這批要處理的 all_records 索引
        batch_num:     批次編號（用於日誌）
    
    回傳:
        [(index, is_meaningful, meaningful_confidence), ...] 
    """
    # 組裝發給 API 的 JSONL
    # 用 all_records 的 index 作為 id（保證唯一）
    batch_profiles = []
    for idx in batch_indices:
        rec = all_records[idx]
        batch_profiles.append({
            "id": idx,
            "self_description": rec["self_description"],
            "skills": rec["skills"],
        })
    
    input_jsonl_str = "\n".join(json.dumps(p, ensure_ascii=False) for p in batch_profiles)
    
    user_message = f"""=== INPUT ===
{input_jsonl_str}

=== OUTPUT ==="""
    
    # 帶重試的 API 調用 
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            chat = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=4096,
            )
            
            result_text = chat.choices[0].message.content.strip()
            
            # 解析回傳的 JSONL
            # 用 id → label 的字典匹配，不依賴順序（修正原版按位置匹配的 bug）
            label_map = {}
            for line in result_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    label_map[obj["id"]] = obj
                except (json.JSONDecodeError, KeyError):
                    pass  # 跳過無法解析的行
            
            # 將標註結果和原始索引配對
            results = []
            for idx in batch_indices:
                if idx in label_map:
                    label = label_map[idx]
                    results.append((
                        idx,
                        label.get("is_meaningful"),
                        label.get("meaningful_confidence"),
                    ))
                else:
                    results.append((idx, None, None))
            
            matched = sum(1 for _, m, _ in results if m is not None)
            print(f"  ✅ 批次 {batch_num}/{total_batches} 完成（匹配 {matched}/{len(batch_indices)}）")
            return results
        
        except Exception as e:
            print(f"  ⚠️ 批次 {batch_num} 第 {attempt} 次失敗: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    
    # 全部重試都失敗
    print(f"  ❌ 批次 {batch_num} 徹底失敗")
    return [(idx, None, None) for idx in batch_indices]

# 把 api_queue 切成多個 batch
batches = [api_queue[i:i + BATCH_SIZE] for i in range(0, len(api_queue), BATCH_SIZE)]

print(f"\n🚀 開始並發處理（{MAX_WORKERS} 線程）...\n")
start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有批次任務到線程池
    future_to_batch = {}
    for batch_num, batch_indices in enumerate(batches, 1):
        future = executor.submit(call_api_for_batch, batch_indices, batch_num)
        future_to_batch[future] = batch_num
        time.sleep(SUBMIT_DELAY)  # 錯開提交時間，避免同時打 API 觸發限流
    
    # 按完成順序收集結果
    for future in as_completed(future_to_batch):
        batch_num = future_to_batch[future]
        try:
            results = future.result()
            # 將標註結果寫回 all_records
            # 注意：每個 batch 的 indices 互不重疊，不會有 race condition
            for idx, is_meaningful, confidence in results:
                all_records[idx]["is_meaningful"] = is_meaningful
                all_records[idx]["meaningful_confidence"] = confidence
        except Exception as e:
            print(f"  ❌ 批次 {batch_num} 收集結果時出錯: {e}")

elapsed = time.time() - start_time
print(f"\n⏱️ API 處理總耗時: {elapsed:.1f} 秒")

#最終輸出
with open(output_path, "w", encoding="utf-8") as f:
    for i, rec in enumerate(all_records):
        # 以原始 pair 為基底，保留所有原始欄位（question_id, profile 等）
        output = dict(pairs[i])
        output["is_meaningful"] = rec["is_meaningful"]
        output["meaningful_confidence"] = rec["meaningful_confidence"]
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


# 統計結果
success_count    = sum(1 for r in all_records if r["is_meaningful"] is not None)
fail_count       = sum(1 for r in all_records if r["is_meaningful"] is None)
meaningful_true  = sum(1 for r in all_records if r["is_meaningful"] is True)
meaningful_false = sum(1 for r in all_records if r["is_meaningful"] is False)

print(f"\n{'=' * 55}")
print(f"🎉 標註完成！")
print(f"   總筆數:          {len(all_records)}")
print(f"   API 標註成功:    {success_count} 筆")
print(f"   標註失敗 (None): {fail_count} 筆")
print(f"   ─────────────────────────")
print(f"   meaningful:      {meaningful_true} 筆")
print(f"   not meaningful:  {meaningful_false} 筆")
print(f"   ─────────────────────────")
print(f"   輸出檔案: {output_path}")
print(f"{'=' * 55}")
