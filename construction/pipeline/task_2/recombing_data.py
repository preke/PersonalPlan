import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
annotated_path  = str(TASK_DIR / "annotated_output_final.jsonl")
qap_path        = str(TASK_DIR / "output_qap.jsonl")
filtered_output = str(TASK_DIR / "filtered_qap.jsonl")


# 建立有效 / 無效 profile 集合
# 用 (self_description, skills_tuple) 作為比對 key
def make_profile_key(profile: dict) -> tuple:
    desc = profile.get("self_description", "").strip()
    skills = tuple(sorted(profile.get("skills", [])))
    return (desc, skills)


valid_profiles = set()       # is_meaningful == True
invalid_profiles = set()     # is_meaningful == False
failed_profiles = set()      # is_meaningful == None

with open(annotated_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        # annotated 檔的 profile 直接在頂層的 "profile" key
        profile = entry.get("profile", {})
        key = make_profile_key(profile)

        is_meaningful = entry.get("is_meaningful")

        if is_meaningful is True:
            valid_profiles.add(key)
        elif is_meaningful is False:
            invalid_profiles.add(key)
        else:
            failed_profiles.add(key)

all_annotated = valid_profiles | invalid_profiles | failed_profiles

print(f"{'=' * 55}")
print(f" 標註檔統計（以 unique profile 計）")
print(f" 所有標註過的 profile:   {len(all_annotated)}")
print(f" meaningful (保留):   {len(valid_profiles)}")
print(f" not meaningful:      {len(invalid_profiles)}")
print(f" 標註失敗 (None):    {len(failed_profiles)}")
print(f"{'=' * 55}\n")


# 對output_qap.jsonl的每個問題，遍歷 profiles_answers，只保留 profile 在 valid_profiles 中的。
# 如果過濾後 profiles_answers 為空，則整個問題也丟棄。

questions_kept = 0
questions_dropped = 0
pa_kept = 0
pa_dropped = 0
pa_not_found = 0

with open(qap_path, "r", encoding="utf-8") as fin, \
     open(filtered_output, "w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue
        entry = json.loads(line)

        profiles_answers = entry.get("profiles_answers", [])
        filtered_pa = []

        for pa in profiles_answers:
            profile = pa.get("profile", {})
            key = make_profile_key(profile)

            if key in valid_profiles:
                # 有意義 → 保留
                filtered_pa.append(pa)
                pa_kept += 1
            elif key in all_annotated:
                # 標註過但無意義 → 丟棄
                pa_dropped += 1
            else:
                # 標註檔中沒見過 → 保守保留
                filtered_pa.append(pa)
                pa_kept += 1
                pa_not_found += 1

        if filtered_pa:
            entry["profiles_answers"] = filtered_pa
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            questions_kept += 1
        else:
            questions_dropped += 1


# ============================================================
# 📊 最終統計
# ============================================================

total_pa = pa_kept + pa_dropped
total_q = questions_kept + questions_dropped

print(f"{'=' * 55}")
print(f"")
print(f" 問題 (question) 層級:")
print(f" 總問題數:          {total_q}")
print(f" 保留:           {questions_kept}")
print(f" 丟棄 (空答案):  {questions_dropped}")
print(f"")
print(f" 回答 (profile-answer) 層級:")
print(f" 總回答數:          {total_pa}")
print(f" 保留:           {pa_kept}")
print(f" 過濾 (無意義):  {pa_dropped}")
print(f" 未見過 (已保留): {pa_not_found}")
print(f" 過濾率:            {pa_dropped / max(total_pa, 1) * 100:.1f}%")
print(f"")
print(f" 輸出: {filtered_output}")
print(f"{'=' * 55}")


