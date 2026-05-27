"""
Stack Overflow 問題分類器（聚合統計 + 終端報告 + 逐題 JSONL）

產出：
- classified_results.jsonl      ← 每題的分類結果（question_id, labels, reasoning）
- category_stats.csv
- classification_plots/
    - category_distribution_questions.png
    - upset_label_combinations.png
- 終端直接印出分類報告

不產出：
- 不輸出 Markdown 檔案
"""

import json
import os
import re
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI

# 設定區 — relative to this script's directory
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent
JSONL_INPUT_PATH = str(TASK_DIR / "filtered_qap.jsonl")

CATEGORY_STATS_CSV = "category_stats.csv"
CLASSIFIED_JSONL = "classified_results.jsonl"   # ← 逐題分類結果
PLOT_DIR = "classification_plots"

API_KEY = (os.getenv("POE_API_KEY")), #切換成你的Key
BASE_URL = "https://api.poe.com/v1"
MODEL_NAME = "gpt-4o"

DELAY_BETWEEN_REQUESTS = 0.01
MAX_RETRIES = 3
MAX_QUERY_LENGTH = 1500
TOP_N_COMBINATIONS = 20



# 合法標籤集合與固定排序
VALID_LABELS = {
    "API_USAGE", "CONCEPTUAL", "DISCREPANCY",
    "ERRORS", "REVIEW", "API_CHANGE", "LEARNING"
}
ALL_LABELS_ORDER = [
    "API_USAGE", "CONCEPTUAL", "DISCREPANCY",
    "ERRORS", "REVIEW", "API_CHANGE", "LEARNING"
]

# 分類用 Prompt 模板
PROMPT_TEMPLATE = """You are a classifier for Stack Overflow questions.
Your task is to identify the REASON a developer is asking — not the technology or topic involved.

Assign one or more labels from exactly this set:
API_USAGE, CONCEPTUAL, DISCREPANCY, ERRORS, REVIEW, API_CHANGE, LEARNING

─── Category Definitions & Indicator Phrases ───

API_USAGE: The questioner asks for concrete instructions on how to implement
something or how to use an API. They want a working solution or code example.

DISCREPANCY: The questioner describes unexpected behavior or code that does
not work as intended, but there is NO explicit error message or stack trace.
They have no clue how to solve it.

ERRORS: The questioner reports an explicit exception, error message, stack
trace, crash, or compiler error. The post typically contains a pasted error
or stack trace.

REVIEW: The questioner asks for a better solution, best practice, code
review, or help making a decision between alternatives. They may already
have a working solution but want improvement or validation.

CONCEPTUAL: The questioner asks abstract or theoretical questions about
behavior, limitations, concepts, design patterns, or differences — without
a concrete implementation goal.

API_CHANGE: The questioner has a problem caused by an API version update,
deprecation, or compatibility between different API/SDK/library versions.

LEARNING: The questioner asks for tutorials, documentation, books, courses,
or learning resources — NOT for a direct solution or code.

─── Classification Rules ───
1. A post can have MULTIPLE labels. Assign all that apply.
2. Focus on the phrases and sentences that reveal WHY the question is asked,
   not on code snippets or technology tags.
3. DISCREPANCY vs ERRORS: If the post contains a specific error message or
   stack trace, prefer ERRORS. If it only describes "not working" behavior
   without a specific error, prefer DISCREPANCY. A post can have both.
4. CONCEPTUAL vs API_USAGE: If the questioner wants a concrete implementation,
   it is API_USAGE. If they want to understand behavior or theory, it is
   CONCEPTUAL.
5. REVIEW vs API_USAGE: If the questioner already has a working solution but
   wants improvement, it is REVIEW. If they have no solution yet, it is
   API_USAGE.
6. Ignore code blocks when identifying the question intent; focus on the
   natural-language portions (title, question sentences, problem description).

Question:
{query}

Return ONLY valid JSON:
{{"labels":["..."], "reasoning":"one short sentence explaining the key phrases that led to your classification"}}
"""

# 工具函式
def normalize_labels(labels):
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, list):
        return []
    out = []
    for l in labels:
        if not isinstance(l, str):
            continue
        x = l.upper().strip().replace(" ", "_")
        if x in VALID_LABELS and x not in out:
            out.append(x)
    return out


def load_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("question_id", "")).strip()
                q = obj.get("canonical_query", "")
                if not qid:
                    continue
                rows.append({
                    "question_id": qid,
                    "canonical_query": q if isinstance(q, str) else str(q)
                })
            except json.JSONDecodeError:
                print(f"警告：第 {i} 行 JSON 格式錯誤，已跳過。")
    return rows


def build_prompt(query):
    query = (query or "")[:MAX_QUERY_LENGTH]
    return PROMPT_TEMPLATE.format(query=query)


def parse_response(response_text):
    txt = (response_text or "").strip()

    try:
        obj = json.loads(txt)
        labels = normalize_labels(obj.get("labels", []))
        if labels:
            return {"labels": labels, "reasoning": obj.get("reasoning", ""), "parse_warning": False}
    except Exception:
        pass

    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[\s\S]*?\})",
    ]
    for p in patterns:
        m = re.search(p, txt, flags=re.DOTALL)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
            labels = normalize_labels(obj.get("labels", []))
            if labels:
                return {"labels": labels, "reasoning": obj.get("reasoning", ""), "parse_warning": False}
        except Exception:
            continue

    found = [l for l in ALL_LABELS_ORDER if l in txt.upper()]
    if found:
        return {"labels": found, "reasoning": "回退方案：正則搜尋標籤名", "parse_warning": True}

    return {"labels": [], "reasoning": f"解析失敗：{txt[:200]}", "parse_warning": True}

# API 客戶端
def init_client():
    if not API_KEY:
        raise RuntimeError("缺少 API Key，請先設定。")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_response_text(response):
    txt = getattr(response, "output_text", None)
    if txt:
        return txt
    chunks = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks).strip()


def classify_single(client, query):
    prompt = build_prompt(query)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt
            )
            text = extract_response_text(response)
            return parse_response(text)
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return {"labels": [], "reasoning": f"API 錯誤：{e}", "parse_warning": True}


# 統計、繪圖、報告
def build_stats_df(total_questions, label_counts):
    total_labels = sum(label_counts.values())
    rows = []
    for label in ALL_LABELS_ORDER:
        c = label_counts.get(label, 0)
        rows.append({
            "category": label,
            "count_questions": c,
            "pct_of_questions": (c / total_questions * 100) if total_questions else 0.0,
            "pct_of_all_labels": (c / total_labels * 100) if total_labels else 0.0
        })
    return pd.DataFrame(rows), total_labels


def plot_category_distribution(stats_df):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=stats_df, x="category", y="count_questions",
        hue="category", palette="Blues_d", legend=False
    )
    ax.set_title("Category Distribution (Question-level, Multi-label)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count of Questions")
    plt.xticks(rotation=25, ha="right")
    for i, row in stats_df.reset_index(drop=True).iterrows():
        ax.text(
            i, row["count_questions"],
            f'{int(row["count_questions"])}\n({row["pct_of_questions"]:.1f}%)',
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig(Path(PLOT_DIR) / "category_distribution_questions.png", dpi=180)
    plt.close()


def plot_upset(combo_counter, total_questions):
    top_combos = combo_counter.most_common(TOP_N_COMBINATIONS)

    if not top_combos:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "無標籤組合資料", ha="center", va="center")
        plt.axis("off")
        plt.savefig(Path(PLOT_DIR) / "upset_label_combinations.png", dpi=180)
        plt.close()
        return

    combos = [c for c, _ in top_combos]
    counts = [v for _, v in top_combos]
    n_combos = len(combos)
    n_labels = len(ALL_LABELS_ORDER)

    matrix = np.array([
        [1 if label in combo else 0 for label in ALL_LABELS_ORDER]
        for combo in combos
    ])

    fig_width = max(10, n_combos * 0.55 + 2)
    fig = plt.figure(figsize=(fig_width, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.3], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)

    x = np.arange(n_combos)
    colors = sns.color_palette("Blues_d", n_combos)
    ax_bar.bar(x, counts, color=colors, edgecolor="white", width=0.6)
    for i, cnt in enumerate(counts):
        pct = cnt / total_questions * 100 if total_questions else 0
        ax_bar.text(i, cnt, f'{cnt}\n({pct:.1f}%)', ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylabel("Questions", fontsize=10)
    ax_bar.set_title("Label Combination Frequency (UpSet Plot)", fontsize=12)
    ax_bar.set_xlim(-0.5, n_combos - 0.5)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.spines["bottom"].set_visible(False)

    for i in range(n_combos):
        active_rows = []
        for j in range(n_labels):
            if matrix[i, j]:
                ax_dot.scatter(i, j, color="black", s=90, zorder=3)
                active_rows.append(j)
            else:
                ax_dot.scatter(i, j, facecolors="none", edgecolors="lightgray",
                               s=40, linewidths=0.8, zorder=2)
        if len(active_rows) > 1:
            ax_dot.plot([i, i], [min(active_rows), max(active_rows)],
                        color="black", linewidth=1.8, zorder=1)

    ax_dot.set_yticks(range(n_labels))
    ax_dot.set_yticklabels(ALL_LABELS_ORDER, fontsize=9)
    ax_dot.set_xlim(-0.5, n_combos - 0.5)
    ax_dot.set_xticks([])
    ax_dot.invert_yaxis()
    ax_dot.spines["top"].set_visible(False)
    for j in range(n_labels):
        ax_dot.axhline(y=j, color="#eeeeee", linewidth=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(Path(PLOT_DIR) / "upset_label_combinations.png", dpi=180)
    plt.close()


def save_plots(stats_df, combo_counter, total_questions):
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_category_distribution(stats_df)
    plot_upset(combo_counter, total_questions)


def print_report(
    total_questions, total_labels, multi_label_count,
    parse_warning_count, empty_or_invalid_count,
    stats_df, pair_counts, label_num_counter, combo_counter,
    jsonl_written_count
):
    sep_thick = "=" * 70
    sep_thin = "-" * 70

    stats_sorted = stats_df.sort_values("count_questions", ascending=False).reset_index(drop=True)
    top3 = stats_sorted.head(3)
    zero_categories = stats_df[stats_df["count_questions"] == 0]["category"].tolist()

    avg_labels_per_q = (total_labels / total_questions) if total_questions else 0
    multi_ratio = (multi_label_count / total_questions * 100) if total_questions else 0

    print(f"\n{sep_thick}")
    print("  CLASSIFICATION REPORT")
    print(sep_thick)

    print(f"\n{sep_thin}")
    print("  核心統計")
    print(sep_thin)
    print(f"  問題總數              : {total_questions}")
    print(f"  指派標籤總數          : {total_labels}")
    print(f"  平均標籤數 / 問題     : {avg_labels_per_q:.2f}")
    print(f"  多標籤問題 (≥2)       : {multi_label_count}  ({multi_ratio:.1f}%)")
    print(f"  解析/API 警告數       : {parse_warning_count}")
    print(f"  空白/無效標籤數       : {empty_or_invalid_count}")
    print(f"  JSONL 寫入筆數        : {jsonl_written_count}")

    print(f"\n{sep_thin}")
    print("  各類別摘要")
    print(sep_thin)
    print(f"  {'類別':<14} {'問題數':>8} {'佔問題%':>10} {'佔標籤%':>10}")
    print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")
    for _, row in stats_df.iterrows():
        print(
            f"  {row['category']:<14} {int(row['count_questions']):>8} "
            f"{row['pct_of_questions']:>9.2f}% {row['pct_of_all_labels']:>9.2f}%"
        )

    print(f"\n{sep_thin}")
    print("  每題標籤數量分布")
    print(sep_thin)
    max_n = max(label_num_counter.keys()) if label_num_counter else 1
    print(f"  {'標籤數':>6} {'問題數':>8} {'佔比':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8}")
    for n in range(1, max_n + 1):
        cnt = label_num_counter.get(n, 0)
        pct = (cnt / total_questions * 100) if total_questions else 0
        print(f"  {n:>6} {cnt:>8} {pct:>7.2f}%")

    top_combos = combo_counter.most_common(TOP_N_COMBINATIONS)
    print(f"\n{sep_thin}")
    print(f"  前 {min(TOP_N_COMBINATIONS, len(top_combos))} 組標籤組合")
    print(sep_thin)
    if top_combos:
        print(f"  {'排名':>4} {'組合':<40} {'數量':>6} {'佔比':>8}")
        print(f"  {'-'*4} {'-'*40} {'-'*6} {'-'*8}")
        for rank, (combo, cnt) in enumerate(top_combos, 1):
            combo_str = " + ".join(combo)
            pct = (cnt / total_questions * 100) if total_questions else 0
            print(f"  {rank:>4} {combo_str:<40} {cnt:>6} {pct:>7.2f}%")
    else:
        print("  （無組合資料）")

    pair_nonzero = [(k, v) for k, v in pair_counts.items() if v > 0]
    pair_nonzero.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pair_nonzero[:10]
    print(f"\n{sep_thin}")
    print("  兩兩共現（前 10）")
    print(sep_thin)
    if top_pairs:
        print(f"  {'配對':<30} {'數量':>6} {'佔比':>8}")
        print(f"  {'-'*30} {'-'*6} {'-'*8}")
        for (a, b), c in top_pairs:
            pct = (c / total_questions * 100) if total_questions else 0
            print(f"  {a + ' + ' + b:<30} {c:>6} {pct:>7.2f}%")
    else:
        print("  （無非零配對）")

    print(f"\n{sep_thin}")
    print("  統計結論")
    print(sep_thin)

    if len(top3) >= 1:
        r = top3.iloc[0]
        print(f"  ▸ 最大類別：{r['category']}，共 {int(r['count_questions'])} 題（{r['pct_of_questions']:.2f}%）")
    if len(top3) >= 2:
        r = top3.iloc[1]
        print(f"  ▸ 第二類別：{r['category']}，共 {int(r['count_questions'])} 題（{r['pct_of_questions']:.2f}%）")
    if len(top3) >= 3:
        r = top3.iloc[2]
        print(f"  ▸ 第三類別：{r['category']}，共 {int(r['count_questions'])} 題（{r['pct_of_questions']:.2f}%）")

    top3_share = top3["pct_of_questions"].sum() if not top3.empty else 0
    print(f"  ▸ 前三類別合計佔 {top3_share:.2f}% 的問題")
    print(f"  ▸ 多意圖率（multi-intent rate）：{multi_ratio:.2f}%，顯示跨類別重疊現象")

    if top_pairs:
        (a, b), c = top_pairs[0]
        pct = (c / total_questions * 100) if total_questions else 0
        print(f"  ▸ 最強共現配對：{a} + {b}，共 {c} 題（{pct:.2f}%）")

    single_cnt = label_num_counter.get(1, 0)
    single_pct = (single_cnt / total_questions * 100) if total_questions else 0
    print(f"  ▸ 單標籤問題：{single_cnt} 題（{single_pct:.1f}%）")

    if zero_categories:
        print(f"  ▸ 未觀察到的類別：{', '.join(zero_categories)}")

    print(f"\n{sep_thin}")
    print("  已儲存檔案")
    print(sep_thin)
    print(f"  ▸ {CLASSIFIED_JSONL}                              — 逐題分類結果")
    print(f"  ▸ {CATEGORY_STATS_CSV}                                — 類別統計")
    print(f"  ▸ {PLOT_DIR}/category_distribution_questions.png — 各類別問題數量分布")
    print(f"  ▸ {PLOT_DIR}/upset_label_combinations.png        — UpSet 標籤組合圖")
    print(f"\n{sep_thick}\n")


# 主程式
def run():
    if not os.path.exists(JSONL_INPUT_PATH):
        print(f"錯誤：找不到檔案 {JSONL_INPUT_PATH}")
        return

    records = load_jsonl(JSONL_INPUT_PATH)
    if not records:
        print("未載入任何有效記錄。")
        return

    client = init_client()

    # 統計用計數器
    total_questions = 0
    multi_label_count = 0
    parse_warning_count = 0
    empty_or_invalid_count = 0
    jsonl_written_count = 0

    label_counts = Counter()
    pair_counts = Counter()
    label_num_counter = Counter()
    combo_counter = Counter()

    # 開啟 JSONL 輸出檔（逐筆寫入，即使中途中斷也保留已完成的結果）
    jsonl_out = open(CLASSIFIED_JSONL, "w", encoding="utf-8")

    try:
        for rec in tqdm(records, desc="分類中", unit="題"):
            total_questions += 1
            qid = rec["question_id"]
            query = (rec.get("canonical_query") or "").strip()

            # 空問題
            if not query:
                empty_or_invalid_count += 1
                # 仍然寫入 JSONL，標記為空
                row = {
                    "question_id": qid,
                    "labels": [],
                    "reasoning": "空白問題，跳過分類",
                    "parse_warning": False
                }
                jsonl_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_written_count += 1
                continue

            # 呼叫 API 分類
            result = classify_single(client, query)
            labels = normalize_labels(result.get("labels", []))
            pw = result.get("parse_warning", False)

            if pw:
                parse_warning_count += 1

            if not labels:
                empty_or_invalid_count += 1
                # 寫入 JSONL（無有效標籤）
                row = {
                    "question_id": qid,
                    "labels": [],
                    "reasoning": result.get("reasoning", ""),
                    "parse_warning": pw
                }
                jsonl_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl_written_count += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue

            uniq = sorted(set(labels))

            # 寫入 JSONL（正常結果）
            row = {
                "question_id": qid,
                "labels": uniq,
                "reasoning": result.get("reasoning", ""),
                "parse_warning": pw
            }
            jsonl_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_written_count += 1

            # 累計統計
            label_num_counter[len(uniq)] += 1
            combo_counter[tuple(uniq)] += 1

            if len(uniq) > 1:
                multi_label_count += 1

            for l in uniq:
                label_counts[l] += 1

            for a, b in combinations(uniq, 2):
                pair_counts[(a, b)] += 1

            time.sleep(DELAY_BETWEEN_REQUESTS)

    finally:
        jsonl_out.close()

    # 建立統計表並存 CSV
    stats_df, total_labels = build_stats_df(total_questions, label_counts)
    stats_df.to_csv(CATEGORY_STATS_CSV, index=False, encoding="utf-8")

    # 繪製兩張圖
    save_plots(stats_df, combo_counter, total_questions)

    # 直接印出報告
    print_report(
        total_questions=total_questions,
        total_labels=total_labels,
        multi_label_count=multi_label_count,
        parse_warning_count=parse_warning_count,
        empty_or_invalid_count=empty_or_invalid_count,
        stats_df=stats_df,
        pair_counts=pair_counts,
        label_num_counter=label_num_counter,
        combo_counter=combo_counter,
        jsonl_written_count=jsonl_written_count
    )


if __name__ == "__main__":
    run()
