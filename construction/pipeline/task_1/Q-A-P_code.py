import json
import os

#读取 JSONL 文件，每行一条 JSON。
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠ 跳过无法解析的行: {e}")
    return data

#将 title 和 body 拼接为 query。
def build_query(question):
    title = question.get('title', '')
    body = question.get('body', '')
    return f"{title}\n\n{body}"

#从 question_author 中提取 self_description 和 skills。
def extract_user_profile(question_author):
    self_description = question_author.get('about_me', '')
    top_tags = question_author.get('top_tags', [])
    skills = [tag['tag_name'] for tag in top_tags if 'tag_name' in tag]
    return {
        'self_description': self_description,
        'skills': skills
    }

# 从 canonical 的 accepted_answer 字段中提取被采纳答案的 body。
def extract_canonical_accepted_answer(accepted_answer_data):
    if not accepted_answer_data:
        return None
    answer = accepted_answer_data.get('answer', {})
    if answer.get('is_accepted', False):
        return answer.get('body', '')
    return None

#从 duplicate 的 answers 列表中找到 is_accepted==true 的答案，提取 body。
def extract_duplicate_accepted_answer(answers):
    if not answers:
        return None
    for ans_data in answers:
        answer = ans_data.get('answer', {})
        if answer.get('is_accepted', False):
            return answer.get('body', '')
    return None

#最终数据整理
def process_data(queries_file, full_data_file, output_file):
    # Step 1: 读取 canonical question ID 列表
    queries = load_jsonl(queries_file)
    canonical_ids = set(str(q['question_id']) for q in queries)
    print(f"共读取 {len(canonical_ids)} 个 canonical question ID")

    # Step 2: 读取完整数据，按 canonical question_id 建索引
    full_data = load_jsonl(full_data_file)
    data_index = {}
    for record in full_data:
        canonical = record.get('canonical', {})
        question = canonical.get('question', {})
        qid = str(question.get('question_id', ''))
        if qid:
            data_index[qid] = record
    print(f"完整数据共 {len(data_index)} 条")

    # Step 3: 逐条处理
    results = []
    missing = 0

    for qid in canonical_ids:
        if qid not in data_index:
            print(f"  ⚠ question_id {qid} 在完整数据中未找到，跳过")
            missing += 1
            continue

        record = data_index[qid]
        canonical = record['canonical']

        # Canonical 部分
        canonical_query = build_query(canonical['question'])
        canonical_profile = extract_user_profile(
            canonical.get('question_author', {})
        )
        canonical_answer = extract_canonical_accepted_answer(
            canonical.get('accepted_answer', {})
        )

        # 构建 profiles_answers 列表
        profiles_answers = []

        # 先放 canonical 的 profile + answer
        profiles_answers.append({
            'profile': canonical_profile,
            'answer': canonical_answer
        })

        # 再放每个 duplicate 的 profile + answer（不需要 query）
        for dup in record.get('duplicates', []):
            dup_profile = extract_user_profile(
                dup.get('question_author', {})
            )
            dup_answer = extract_duplicate_accepted_answer(
                dup.get('answers', [])
            )
            profiles_answers.append({
                'profile': dup_profile,
                'answer': dup_answer
            })

        # 组装一条完整记录
        result = {
            'question_id': qid,
            'canonical_query': canonical_query,
            'profiles_answers': profiles_answers
        }
        results.append(result)

    # Step 4: 写出结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n✅ 处理完成：{len(results)} 条写入 {output_file}，{missing} 条未匹配")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    queries_file = os.path.join(script_dir, 'queries_selected_latest.jsonl')
    full_data_file = os.path.join(script_dir, 'filtered_output_no_similarity_completed.jsonl')
    output_file = os.path.join(script_dir, 'output_qap.jsonl')

    process_data(queries_file, full_data_file, output_file)