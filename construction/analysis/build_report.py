"""
Assemble final HTML report from stats.json + figures/.
Output: ../MAPLE_DATASET_ANALYSIS.html  (under MAPLE_Construction/)
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = json.load(open(HERE / "stats.json"))
MAPLE_DIR = HERE.parent
OUT = MAPLE_DIR / "MAPLE_DATASET_ANALYSIS.html"


def fmt_pct(x):
    return f"{x*100:.1f}%"


def fmt(x, n=2):
    if isinstance(x, float):
        return f"{x:.{n}f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


A = STATS["A_overview"]
B = STATS["B_complexity"]
C = STATS["C_agents"]
D = STATS["D_learner"]
E = STATS["E_pedagogy"]
F = STATS["F_personalization"]
G = STATS["G_intent_summary"]
H = STATS["H_by_profile_index"]
Iq = STATS["I_quality"]


# Convenience renderings ------------------------------------------------
def row_kv(label, value, sub=""):
    return f"<tr><td>{label}</td><td class='num'>{value}</td><td>{sub}</td></tr>"


pedagogy_table_rows = "".join(
    f"<tr><td>{p}</td><td class='num'>{fmt_pct(pct)}</td>"
    f"<td class='num'>{E['pattern_step_count'][p]:,}</td></tr>"
    for p, pct in sorted(E["pattern_plan_coverage_pct"].items(),
                          key=lambda x: -x[1])
)

intent_rows = "".join(
    f"<tr><td>{r['primary_intent']}</td>"
    f"<td class='num'>{r['n']:,}</td>"
    f"<td class='num'>{fmt(r['agents'])}</td>"
    f"<td class='num'>{fmt(r['subtasks'])}</td>"
    f"<td class='num'>{fmt(r['steps'])}</td>"
    f"<td class='num'>{fmt(r['longest_path'])}</td>"
    f"<td class='num'>{fmt_pct(r['loop_pct'])}</td>"
    f"<td class='num'>{fmt(r['human_input_mean'])}</td>"
    f"<td class='num'>{fmt(r['tools'])}</td>"
    f"<td class='num'>{fmt(r['ped_patterns'])}</td></tr>"
    for r in G
)

profile_rows = "".join(
    f"<tr><td class='num'>{r['profile_index']}</td>"
    f"<td class='num'>{r['n']:,}</td>"
    f"<td class='num'>{fmt(r['agents'])}</td>"
    f"<td class='num'>{fmt(r['subtasks'])}</td>"
    f"<td class='num'>{fmt(r['steps'])}</td>"
    f"<td class='num'>{fmt_pct(r['loops_pct'])}</td>"
    f"<td class='num'>{fmt(r['human_input'])}</td>"
    f"<td class='num'>{fmt(r['ped'])}</td></tr>"
    for r in H
)

tools_rows = "".join(
    f"<tr><td><code>{t}</code></td><td class='num'>{c:,}</td>"
    f"<td class='num'>{c/2942 if False else c/C['total_agent_instances']:.1%}</td></tr>"
    for t, c in C["tools_global_counts"].items()
)

top_roles_rows = "".join(
    f"<tr><td>{i+1}</td><td><code>{r}</code></td><td class='num'>{c:,}</td>"
    f"<td class='num'>{c/C['total_agent_instances']*100:.2f}%</td></tr>"
    for i, (r, c) in enumerate(C["top_roles"][:20])
)

top_tags_rows = "".join(
    f"<tr><td>{i+1}</td><td>{t}</td><td class='num'>{c:,}</td>"
    f"<td class='num'>{c/A['total_records']*100:.1f}%</td></tr>"
    for i, (t, c) in enumerate(D["top_30_tags"][:20])
)

top_pairs_rows = "".join(
    f"<tr><td>{i+1}</td><td>{a} &nbsp;∧&nbsp; {b}</td><td class='num'>{c:,}</td></tr>"
    for i, ((a, b), c) in enumerate(D["top_15_pairs"][:15])
)

cross_jac = F["cross_profile_jaccard_means"]


HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>MAP-PPL 数据集 · 全面统计分析报告 (Claude analysis)</title>
<style>
:root {{
  --fg: #1f2328;
  --muted: #57606a;
  --bg: #ffffff;
  --bg-soft: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-soft: #ddf4ff;
  --warn: #9a6700;
  --warn-soft: #fff8c5;
  --ok: #1a7f37;
  --ok-soft: #dafbe1;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Helvetica Neue", Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.65;
  max-width: 1180px;
  margin: 32px auto;
  padding: 0 32px 80px;
  font-size: 15.5px;
}}
h1 {{ font-size: 28px; border-bottom: 2px solid var(--border); padding-bottom: 10px; }}
h2 {{ font-size: 22px; margin-top: 48px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }}
h3 {{ font-size: 18px; margin-top: 28px; color: #24292f; }}
h4 {{ font-size: 16px; margin-top: 20px; color: #24292f; }}
.subtitle {{ color: var(--muted); margin-top: 0; font-size: 15px; }}
.meta {{ color: var(--muted); font-size: 13.5px; margin-bottom: 12px; }}
code, .mono {{
  font-family: var(--mono); font-size: 0.92em; background: var(--bg-soft);
  padding: 1px 6px; border-radius: 4px; border: 1px solid var(--border);
}}
pre {{ background: var(--bg-soft); border: 1px solid var(--border); border-radius: 6px;
      padding: 12px 14px; overflow-x: auto; font-family: var(--mono); font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 14px; }}
th, td {{ border: 1px solid var(--border); padding: 7px 10px; vertical-align: top; text-align: left; }}
th {{ background: var(--bg-soft); font-weight: 600; }}
tr:nth-child(even) td {{ background: #fbfcfd; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); }}
.tag {{ display: inline-block; padding: 1px 8px; border-radius: 999px;
       font-size: 12px; font-weight: 600; border: 1px solid var(--border); background: var(--bg-soft); }}
.tag.accent {{ color: var(--accent); background: var(--accent-soft); border-color: #b6e3ff; }}
.tag.warn   {{ color: var(--warn);   background: var(--warn-soft); border-color: #eac54f; }}
.tag.ok     {{ color: var(--ok);     background: var(--ok-soft);   border-color: #aceebb; }}
blockquote {{ border-left: 3px solid var(--accent); margin: 14px 0; padding: 8px 16px;
             background: var(--accent-soft); color: #24292f; border-radius: 0 4px 4px 0; }}
blockquote.warn {{ border-color: #eac54f; background: var(--warn-soft); }}
blockquote.ok   {{ border-color: #aceebb; background: var(--ok-soft); }}
.fig {{ text-align: center; margin: 16px 0 20px; }}
.fig img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 6px; background: white; }}
.fig .cap {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 24px; }}
.kpi {{ border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; background: var(--bg-soft); }}
.kpi .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
.kpi .value {{ font-size: 22px; font-weight: 600; font-family: var(--mono); color: var(--accent); margin-top: 4px; }}
.kpi .sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
.toc {{ background: var(--bg-soft); border: 1px solid var(--border); border-radius: 8px;
       padding: 14px 22px; margin: 20px 0 28px; }}
.toc a {{ color: var(--accent); text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
ul, ol {{ padding-left: 22px; }}
li {{ margin: 3px 0; }}
.split2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
hr {{ border: none; border-top: 1px solid var(--border); margin: 28px 0; }}
</style>
</head>
<body>

<h1>MAP-PPL 数据集 · 全面统计分析报告</h1>
<p class="subtitle">
  数据集：<code>MAPLE_Construction/multi_agent_dataset_filtered_qap.jsonl</code><br/>
  对齐文档：<code>EXPERIMENT_PLAN_2026-05-15.html</code> (RQ1-RQ4 / Ped × PVS × PNG × Skill-Match)
</p>
<p class="meta">
  生成时间：2026-05-17 · 由 <code>claude_analysis/run_analysis.py</code> 计算
   · 图表使用 matplotlib + seaborn (publication style · Okabe-Ito colorblind palette)
</p>

<blockquote class="ok">
<strong>本报告解决的问题</strong>
<ol>
  <li>数据集的"形状"是怎样的？多少 agents / tools / tasks / steps？</li>
  <li>plan 的结构复杂度：DAG 长度、并行度、loop / human-in-loop 占比？</li>
  <li>个性化做到了什么程度：profile 影响 plan 吗？影响多大？</li>
  <li>教学法层面：用了哪几类教学策略？scaffolding 三段式覆盖率？</li>
  <li>query intent × plan 复杂度 / 教学法的关系？</li>
  <li>schema 一致性：有没有 dangling reference / cycle？</li>
</ol>
</blockquote>

<div class="toc">
<strong>目录</strong>
<ol>
  <li><a href="#a">A · Overview · 数据集形状</a></li>
  <li><a href="#b">B · Plan 结构复杂度</a></li>
  <li><a href="#c">C · Agent 设计 / Tool 库存</a></li>
  <li><a href="#d">D · Learner 画像 · top_tags 地形</a></li>
  <li><a href="#e">E · 教学法分类 (10 策略 + 三段式 scaffold)</a></li>
  <li><a href="#f">F · 个性化信号 (PVS · 跨 profile 分歧度)</a></li>
  <li><a href="#g">G · Intent × 复杂度 × 教学法</a></li>
  <li><a href="#h">H · profile_index 是否影响 plan 形态</a></li>
  <li><a href="#i">I · Schema / referential validity</a></li>
  <li><a href="#summary">Summary · 关键结论</a></li>
</ol>
</div>

<!-- ========================================================== -->
<h2 id="a">A · Overview · 数据集形状</h2>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Total records</div><div class="value">{A['total_records']:,}</div><div class="sub">(query, profile) 对</div></div>
  <div class="kpi"><div class="label">Unique questions</div><div class="value">{A['unique_questions']:,}</div><div class="sub">来自 Stack Overflow</div></div>
  <div class="kpi"><div class="label">Unique profiles</div><div class="value">{A['unique_profiles_by_text']:,}</div><div class="sub">按 (about_me, sorted top_tags) 去重</div></div>
  <div class="kpi"><div class="label">Rows / qid</div><div class="value">{A['rows_per_qid_mean']:.2f}</div><div class="sub">同 query 平均跨 1.76 个 profile</div></div>
  <div class="kpi"><div class="label">Multi-profile qids</div><div class="value">{A['questions_with_multi_profile']:,}</div><div class="sub">≥ 2 profile 的 question</div></div>
  <div class="kpi"><div class="label">Max profiles / qid</div><div class="value">{A['max_profiles_per_question']}</div><div class="sub">最多一个 query 配 6 个 profile</div></div>
  <div class="kpi"><div class="label">Query length</div><div class="value">{int(A['query_len_chars']['median']):,}</div><div class="sub">median chars (mean {A['query_len_chars']['mean']:.0f}, max {A['query_len_chars']['max']:,})</div></div>
  <div class="kpi"><div class="label">Tags / learner</div><div class="value">{A['skills_per_profile']['median']:.1f}</div><div class="sub">中位 5 个 top_tags (max 5)</div></div>
</div>

<h3>A.1 一条 record 的字段结构</h3>
<pre>{{
  "question_id": "1732236",
  "profile_index": 0,
  "plan": {{
    "input": {{
      "query": "...",                          # SO 问题正文 (mean {A['query_len_chars']['mean']:.0f} chars)
      "learner": {{
        "about_me": "...",                     # mean {A['desc_len_chars']['mean']:.0f} chars
        "top_tags": ["html","php",...]         # 平均 {A['skills_per_profile']['mean']:.2f} 个
      }}
    }},
    "output": {{
      "agents":   [{{agent_role, goal, backstory, tools[]}}, ...],   # 平均 {B['agents_per_plan']['mean']:.2f} / plan
      "subtasks": [{{id, name, subtask_objective, steps[]}}, ...],    # 平均 {B['subtasks_per_plan']['mean']:.2f} / plan
      "execution_order": [step_id | {{loop: {{steps[], condition, max_iterations}}}}, ...]
    }}
  }}
}}</pre>

<h3>A.2 Profile 多样性</h3>
<div class="split2">
<div class="fig"><img src="claude_analysis/figures/A1_profile_index.png" alt="A1"/>
<div class="cap">A1 · profile_index 0 占 {A['profile_index_distribution']['0']/A['total_records']:.1%}（基础"无背景"占位 + profile 0），
profile_index ≥ 1 用于 multi-profile 增广。</div></div>
<div class="fig"><img src="claude_analysis/figures/A2_profiles_per_question.png" alt="A2"/>
<div class="cap">A2 · {A['questions_with_multi_profile']/A['unique_questions']:.1%} 的 question 至少有 2 个 profile，
为 cross-profile personalization 评估提供了 1,777 个 pair-wise 比较。</div></div>
</div>

<h3>A.3 输入侧长度分布 (query / about_me / top_tags)</h3>
<div class="fig"><img src="claude_analysis/figures/A3_input_lengths.png" alt="A3"/>
<div class="cap">A3 · Query 长度强右偏（p25={A['query_len_chars']['p25']:.0f}, p75={A['query_len_chars']['p75']:.0f}），
学习者 about_me 比 query 短 4 倍，绝大多数 learner 拥有完整 5 个 top_tags。</div></div>

<h3>A.4 输出侧长度分布 (agent / step 文本量)</h3>
<div class="fig"><img src="claude_analysis/figures/A4_output_lengths.png" alt="A4"/>
<div class="cap">A4 · Step instruction 中位 {int(A['step_instruction_len_chars']['median'])} chars（≈ 60-70 个单词），
agent backstory 中位 {int(A['agent_backstory_len_chars']['median'])} chars。
这与 EXPERIMENT_PLAN §F 中 R_pers 的 220-char step 阈值相符。</div></div>


<!-- ========================================================== -->
<h2 id="b">B · Plan 结构复杂度</h2>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Agents / plan</div><div class="value">{B['agents_per_plan']['mean']:.2f}</div><div class="sub">87.4% 用 3 个 (range 2-4)</div></div>
  <div class="kpi"><div class="label">Subtasks / plan</div><div class="value">{B['subtasks_per_plan']['mean']:.2f}</div><div class="sub">68.8% 用 4 个 (range 2-6)</div></div>
  <div class="kpi"><div class="label">Steps / plan</div><div class="value">{B['steps_per_plan']['mean']:.2f}</div><div class="sub">median {B['steps_per_plan']['median']:.0f}, max {B['steps_per_plan']['max']}</div></div>
  <div class="kpi"><div class="label">Steps / subtask</div><div class="value">{B['steps_per_subtask']['mean']:.2f}</div><div class="sub">median {B['steps_per_subtask']['median']:.0f}</div></div>
  <div class="kpi"><div class="label">Plans w/ loop</div><div class="value">{B['plans_with_loop']/A['total_records']:.1%}</div><div class="sub">{B['plans_with_loop']:,} / {A['total_records']:,} 含 1+ loop</div></div>
  <div class="kpi"><div class="label">Loop max_iter</div><div class="value">{B['loop_step_count_mean']:.1f}</div><div class="sub">每个 loop 平均 {B['loop_step_count_mean']:.1f} step；max_iter 多为 3</div></div>
  <div class="kpi"><div class="label">Human-in-loop</div><div class="value">{B['human_input_step_ratio_overall']:.1%}</div><div class="sub">步骤层面平均 {B['human_input_steps_per_plan_mean']:.2f} / plan</div></div>
  <div class="kpi"><div class="label">DAG critical depth</div><div class="value">{B['dag_longest_path']['mean']:.2f}</div><div class="sub">median {B['dag_longest_path']['median']:.0f}; 并行度 {B['dag_parallelizable_ratio_mean']:.2f}</div></div>
</div>

<h3>B.1 Agents / Subtasks / Steps 分布</h3>
<div class="fig"><img src="claude_analysis/figures/B1_plan_counts.png" alt="B1"/>
<div class="cap">B1 · 计划结构高度规整：plan 几乎都以 3 agents × 4 subtasks × ~10 steps 为模式。
这种规整源于构造阶段的 schema 约束，但 step 数量从 5 到 23 都有，留有充足复杂度差异。</div></div>

<h3>B.2 迭代式 scaffolding 与 human-in-loop</h3>
<div class="fig"><img src="claude_analysis/figures/B2_loops_humanin.png" alt="B2"/>
<div class="cap">B2 · 80.6% 的 plan 至少含一个 loop（典型 max_iter=3, 占 {2071/2481*100:.1f}%）
+ 54.8% 的 step 标 requires_human_input。
这是数据集核心教学法特征 —— 不是一条直线 plan，而是允许学习者反复迭代。</div></div>

<h3>B.3 DAG 拓扑：critical path & 并行度</h3>
<div class="fig"><img src="claude_analysis/figures/B3_dag_metrics.png" alt="B3"/>
<div class="cap">B3 · 关键路径中位 {B['dag_longest_path']['median']:.0f} 步，
说明 plan 不是纯串行，而是有 ~ 47% 的步骤可并行执行的 DAG。最大并行宽度 {B['dag_max_layer_width']['mean']:.2f}。</div></div>


<!-- ========================================================== -->
<h2 id="c">C · Agent 设计 / Tool 库存</h2>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Total agent instances</div><div class="value">{C['total_agent_instances']:,}</div><div class="sub">{A['total_records']:,} plans × ~3 agents</div></div>
  <div class="kpi"><div class="label">Unique agent roles</div><div class="value">{C['unique_agent_roles_global']:,}</div><div class="sub">高多样性：role 名几乎按 query 定制</div></div>
  <div class="kpi"><div class="label">Tools per agent</div><div class="value">{C['tools_per_agent_mean']:.2f}</div><div class="sub">{C['zero_tool_agents_pct']:.1%} 的 agent 不挂 tool</div></div>
  <div class="kpi"><div class="label">Unique tools global</div><div class="value">{C['tools_global_unique']}</div><div class="sub">只有 5 个不同 tool，复用率极高</div></div>
</div>

<h3>C.1 工具库存 (Tool inventory)</h3>
<div class="split2">
<div class="fig"><img src="claude_analysis/figures/C2_tool_inventory.png" alt="C2"/>
<div class="cap">C1 · 整个数据集只用 5 个工具：CodeInterpreterTool / CodeDocsSearchTool 占 99.8% 的工具调用。
这与"代码教学 + 文档检索"的核心场景一致。</div></div>
<div>
<table>
<thead><tr><th>Tool</th><th class="num">Total usages</th><th class="num">% of agents</th></tr></thead>
<tbody>{tools_rows}</tbody>
</table>
<blockquote class="warn">
工具集小 → baseline / 评估侧好控制（5 个工具 vs 数百个不同的 agent role）；
同时也是"工具复用率 100%"的论据：从未在数据集中"造一个一次性工具"。
</blockquote>
</div>
</div>

<h3>C.2 Tools per agent (分布)</h3>
<div class="fig"><img src="claude_analysis/figures/C1_tools_per_agent.png" alt="C1"/>
<div class="cap">C2 · 1 tool / agent 是绝对主流；零 tool 的 agent ({C['zero_tool_agents_pct']:.1%})
往往是"对话型" tutor / diagnostician —— 它的输出靠 prompt + LLM，不调用外部 API。</div></div>

<h3>C.3 角色家族 (role family)</h3>
<div class="split2">
<div class="fig"><img src="claude_analysis/figures/C3_role_families.png" alt="C3"/>
<div class="cap">C3 · 角色压成 8 大家族后：
tutor (33.5%) + validator (33.3%) + retriever (31.3%) 三家族占 98%，
形成"讲解—检索—验证"金三角。debugger / generator / reviewer 都是补充家族。</div></div>
<div class="fig"><img src="claude_analysis/figures/C4_role_family_upset.png" alt="C4"/>
<div class="cap">C4 · plan 中最常见的家族组合是 <code>{{retriever, validator, tutor}}</code>。
这种"先查—再讲—再验"组合占绝大多数 plan，与教学法 §E 一致。</div></div>
</div>

<h3>C.4 Top-30 raw agent_role</h3>
<div class="fig"><img src="claude_analysis/figures/C5_top30_roles.png" alt="C5"/>
<div class="cap">C5 · 即便 unique role 有 {C['unique_agent_roles_global']:,} 种，
也存在通用模式：<code>&lt;lang&gt;_code_validator / &lt;lang&gt;_docs_retriever</code> 模板复用率高，
top-30 的 raw role 覆盖了 ~ 30% 的 instance。</div></div>


<!-- ========================================================== -->
<h2 id="d">D · Learner 画像 · top_tags 地形</h2>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Unique top_tags</div><div class="value">{D['unique_tags']:,}</div><div class="sub">来自 Stack Overflow tag 体系</div></div>
  <div class="kpi"><div class="label">Tags / profile</div><div class="value">{D['tags_per_profile_mean']:.2f}</div><div class="sub">中位 5 个</div></div>
  <div class="kpi"><div class="label">Top tag</div><div class="value">javascript</div><div class="sub">676 (22.2%) 个 profile 持有</div></div>
  <div class="kpi"><div class="label">Top co-occurring pair</div><div class="value">html∧js</div><div class="sub">212 共现 / front-end backbone</div></div>
</div>

<h3>D.1 Top-30 learner tag</h3>
<div class="fig"><img src="claude_analysis/figures/D1_top_tags.png" alt="D1"/>
<div class="cap">D1 · learner 集中在 6 大编程语言（js / py / java / c# / c++ / html），
但长尾庞大 ({D['unique_tags']:,} unique tags) 保证 profile 不会"撞型"。</div></div>

<h3>D.2 Top-15 tag 共现热力图</h3>
<div class="fig"><img src="claude_analysis/figures/D2_tag_cooccurrence.png" alt="D2"/>
<div class="cap">D2 · 共现矩阵显出三个语言生态团：
front-end (html / css / js / jquery) · MS 栈 (c# / .net / asp.net / sql) · POSIX 栈 (c / c++ / linux)。</div></div>

<h3>D.3 Top-20 tag 与 top-15 共现对</h3>
<div class="split2">
<div>
<table>
<thead><tr><th>#</th><th>Tag</th><th class="num">Count</th><th class="num">%</th></tr></thead>
<tbody>{top_tags_rows}</tbody>
</table>
</div>
<div>
<table>
<thead><tr><th>#</th><th>Co-occurring tag pair</th><th class="num">Count</th></tr></thead>
<tbody>{top_pairs_rows}</tbody>
</table>
</div>
</div>


<!-- ========================================================== -->
<h2 id="e">E · 教学法分类 (10 策略 + 三段式 scaffold)</h2>

<blockquote>
按 EXPERIMENT_PLAN §6.2 (KELE PRR/NDAR/SPR/IAR) 的精神，
本节从 <em>plan text</em>（agent goal / backstory + subtask name/objective + step instruction）抽取
10 类教学策略 + 3 段式骨架（Activate → Apply → Consolidate）。
所有检测均为大小写不敏感的关键词正则。
</blockquote>

<h3>E.1 10 类教学策略覆盖率</h3>
<div class="fig"><img src="claude_analysis/figures/E1_pedagogy_coverage.png" alt="E1"/>
<div class="cap">E1 · validation_feedback (100%) 与 explanation_walkthrough (99.3%) 是必备组件；
diagnostic_assessment (60.2%) 与 worked_example (48.3%) 是选用项。</div></div>

<table>
<thead><tr><th>策略</th><th class="num">Plan 覆盖率</th><th class="num">Step 级出现次数</th></tr></thead>
<tbody>{pedagogy_table_rows}</tbody>
</table>

<h3>E.2 一个 plan 同时用多少策略？</h3>
<div class="fig"><img src="claude_analysis/figures/E2_pedagogy_diversity.png" alt="E2"/>
<div class="cap">E2 · 平均一个 plan 同时启用 {E['patterns_per_plan_mean']:.2f} 种策略（中位 8）。
说明数据集 plan 不是"单一教学法机器"，而是组合式 pedagogy。</div></div>

<h3>E.3 三段式 scaffold 骨架</h3>
<div class="fig"><img src="claude_analysis/figures/E3_scaffolding_skeleton.png" alt="E3"/>
<div class="cap">E3 · Activate ({fmt_pct(E['skeleton_breakdown']['activate'])}) → Apply
({fmt_pct(E['skeleton_breakdown']['apply'])}) → Consolidate ({fmt_pct(E['skeleton_breakdown']['consolidate'])})
完整三段同时出现的 plan 占 {fmt_pct(E['skeleton_full_triple_pct'])}。这是 MAP-PPL 教学骨架的核心可量化证据。</div></div>

<h3>E.4 教学策略两两共现</h3>
<div class="fig"><img src="claude_analysis/figures/E4_pedagogy_cooccurrence.png" alt="E4"/>
<div class="cap">E4 · 主对角是单策略覆盖；离对角越亮 = 两个策略越常被同一 plan 同时调用。
prediction_elicitation × explanation_walkthrough 与 retrieval × validation 是最强的两个共现轴，
对应教学顺序"先预测—再讲解 / 先查文档—再跑代码"。</div></div>


<!-- ========================================================== -->
<h2 id="f">F · 个性化信号 · 文本渗透 + 跨 profile 分歧度</h2>

<blockquote>
对齐 EXPERIMENT_PLAN §7：<strong>R_pers</strong>（personalization reward）、<strong>PVS</strong>（profile-verifiable sensitivity）、
<strong>PNG</strong>（profile-non-generic）的判定原料。
</blockquote>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Skill lex hit (mean)</div><div class="value">{F['skill_in_plan_ratio_mean']:.1%}</div><div class="sub">每个 plan 命中 learner 的 top_tags 比例</div></div>
  <div class="kpi"><div class="label">Plans w/ ≥1 skill mention</div><div class="value">{F['plans_with_at_least_one_skill_pct']:.1%}</div><div class="sub">{F['plans_with_at_least_one_skill_mentioned']:,} / {A['total_records']:,}</div></div>
  <div class="kpi"><div class="label">2nd-person ratio</div><div class="value">{F['second_person_ratio_per_plan_mean']:.1%}</div><div class="sub">每个 plan 含 you/your 的 step 比例</div></div>
  <div class="kpi"><div class="label">Bridge anchors / plan</div><div class="value">{F['bridge_anchor_count_per_plan_mean']:.2f}</div><div class="sub">"your portfolio" / "as a beginner" 等短语</div></div>
</div>

<h3>F.1 个性化信号在 plan 文本里的渗透</h3>
<div class="fig"><img src="claude_analysis/figures/F1_personalization_signal.png" alt="F1"/>
<div class="cap">F1 · 三个独立证据：(a) 学习者 top_tag 中位 40% 被引入 plan；
(b) 步骤指令里 ~70% 含第二人称称呼；(c) 每个 plan 平均出现 6.45 次"your portfolio" 类 bridge anchor。
说明数据集 <em>确实</em> 在 plan 文本中显式 anchor 学习者 profile。</div></div>

<h3>F.2 跨 profile plan divergence (1,777 pairs)</h3>
<div class="fig"><img src="claude_analysis/figures/F2_cross_profile_jaccard.png" alt="F2"/>
<div class="cap">F2 · 同 query 不同 profile 时，
agent_role / subtask_token 显著重写（Jaccard {cross_jac['agent_role']:.2f} / {cross_jac['subtask_tokens']:.2f}），
但 role_family / tools / pedagogy patterns 保持一致 (Jaccard ~ 0.76–0.93)。
这正是"骨架稳定，肌肉因人而异"的 PVS 实证特征。</div></div>

<table>
<thead><tr><th>Plan 维度</th><th class="num">Cross-profile Jaccard mean</th><th>解释</th></tr></thead>
<tbody>
<tr><td>Agent role (原始名)</td><td class="num">{cross_jac['agent_role']:.3f}</td><td>命名级别高度 profile-specific</td></tr>
<tr><td>Subtask tokens</td><td class="num">{cross_jac['subtask_tokens']:.3f}</td><td>子任务用词同样跟着 profile 改写</td></tr>
<tr><td>Pedagogy patterns</td><td class="num">{cross_jac['pedagogy']:.3f}</td><td>教学法基本不变 → 教学骨架是"通用脚手架"</td></tr>
<tr><td>Tools</td><td class="num">{cross_jac['tools']:.3f}</td><td>工具集合几乎不变（5 个公共工具）</td></tr>
<tr><td>Role family</td><td class="num">{cross_jac['role_family']:.3f}</td><td>家族组成几乎不变（仍是 tutor+retriever+validator）</td></tr>
</tbody>
</table>


<!-- ========================================================== -->
<h2 id="g">G · Intent × 复杂度 × 教学法</h2>

<blockquote>
Primary intent 来自 <code>the_construction_of_MAPLE_datasets/task_3/classified_results.jsonl</code>
的 LLM 分类结果（每个 question 取 labels[0] 作为主意图）。
</blockquote>

<h3>G.1 intent → plan 复杂度热图</h3>
<div class="fig"><img src="claude_analysis/figures/G1_intent_complexity_heatmap.png" alt="G1"/>
<div class="cap">G1 · LEARNING 类 intent（仅 12 个）出现最 deep 的 plan
（agents 3.5, longest_path 6.9, ped_patterns 8.2）。
CONCEPTUAL 的 loop_pct 仅 69%，明显低于 ERRORS / API_USAGE / REVIEW —— 概念题不需要"试错"loop。</div></div>

<table>
<thead><tr>
<th>Intent</th><th class="num">n</th><th class="num">agents</th><th class="num">subs</th><th class="num">steps</th>
<th class="num">depth</th><th class="num">loop %</th><th class="num">HI mean</th><th class="num">tools</th><th class="num">ped #</th>
</tr></thead>
<tbody>{intent_rows}</tbody>
</table>

<h3>G.2 不同 intent 偏好的教学策略</h3>
<div class="fig"><img src="claude_analysis/figures/G2_pedagogy_by_intent.png" alt="G2"/>
<div class="cap">G2 · ERRORS 与 REVIEW 强烈依赖 <code>validation_feedback</code> / <code>scaffolding_iterative</code>；
CONCEPTUAL 偏 <code>explanation_walkthrough</code> 与 <code>reflection_metacognition</code>。
LEARNING / DISCREPANCY 是高负载 intent（≥ 8 类教学法被同时调用）。</div></div>

<h3>G.3 不同 intent 跨 profile 分歧度</h3>
<div class="fig"><img src="claude_analysis/figures/G3_divergence_by_intent.png" alt="G3"/>
<div class="cap">G3 · 不同 intent 下 (1 − Jaccard agent_role) 的分歧度都在 0.65–0.72 之间，
没有某一类 intent 的 personalization 显著弱化 —— 说明个性化是 query-agnostic 的稳健特性。</div></div>


<!-- ========================================================== -->
<h2 id="h">H · profile_index 是否影响 plan 形态</h2>

<blockquote>
检验：在控制 question 之后，profile_index 0 / 1 / 2 / 3 / 4 / 5 的 plan 数值是否系统性不同？
</blockquote>

<h3>H.1 profile_index × plan 指标</h3>
<div class="fig"><img src="claude_analysis/figures/H1_profile_index_heatmap.png" alt="H1"/>
<div class="cap">H1 · 总体均值在 profile_index 间高度稳定：
agents (2.75–2.92)、subtasks (3.93–4.00)、steps (9.73–10.62)、ped_patterns (7.60–8.13)。
profile_index 没有引入"越后越复杂"的偏差。</div></div>

<table>
<thead><tr><th>profile_index</th><th class="num">n</th><th class="num">agents</th><th class="num">subtasks</th>
<th class="num">steps</th><th class="num">loop %</th><th class="num">HI mean</th><th class="num">ped #</th></tr></thead>
<tbody>{profile_rows}</tbody>
</table>


<!-- ========================================================== -->
<h2 id="i">I · Schema / referential validity</h2>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Valid plans</div><div class="value">{Iq['valid_plans']:,}</div><div class="sub">{Iq['valid_rate']:.1%} pass schema sanity</div></div>
  <div class="kpi"><div class="label">Unknown agent ref</div><div class="value">{Iq['plans_with_unknown_agent_ref']}</div><div class="sub">step.agent 没在 agents[] 出现</div></div>
  <div class="kpi"><div class="label">Cycle in DAG</div><div class="value">{Iq['plans_with_cycle']}</div><div class="sub">depends_on 形成环</div></div>
  <div class="kpi"><div class="label">Loop schema</div><div class="value">{Iq['loop_schema_plural_steps']:,} / {Iq['loop_schema_singular_step']:,}</div><div class="sub">plural "steps" / singular "step"</div></div>
</div>

<div class="fig"><img src="claude_analysis/figures/I1_quality.png" alt="I1"/>
<div class="cap">I1 · 100% plan 通过 5 项 schema 校验：no dangling step / no missing exec / no unknown agent / no cycle。
loop 用 <code>steps</code>:array 占 91%，剩 9% 用 singular <code>step</code> —— 解析器需要兼容两种。</div></div>


<!-- ========================================================== -->
<h2 id="summary">Summary · 关键结论</h2>

<blockquote class="ok">
<strong>本数据集的形态总结 (用于论文 §3 dataset section)</strong>
<ol>
  <li><strong>规模</strong>：3,043 (query, profile) plans · 1,730 unique questions · 2,738 unique profiles · 4,380 unique agent_role · 但仅 5 unique tools。</li>
  <li><strong>plan 形状高度规整</strong>：3 agents × 4 subtasks × 10 steps · DAG critical depth ≈ 5。47% 步骤可并行。</li>
  <li><strong>迭代式 scaffolding 是核心特征</strong>：80.6% plan 含 loop（max_iter=3 占 83%），54.8% step 需 human_input → 数据集天然支持"教学-反馈-修正"循环。</li>
  <li><strong>3-Agent 教学金三角</strong>：tutor + validator + retriever 占角色家族 98%；CodeInterpreterTool + CodeDocsSearchTool 占工具调用 99.8%。</li>
  <li><strong>3 段式教学骨架明显</strong>：58.7% plan 同时含 Activate → Apply → Consolidate 三段；plan 平均同时启用 7.88 / 10 类教学策略。</li>
  <li><strong>个性化既"显" 又"骨架稳定"</strong>：93.9% plan 显式提到 learner 的 top_tag，69.7% step 使用第二人称；
       但 role_family / tools / pedagogy 在跨 profile 时 Jaccard ≥ 0.76 —— 教学骨架不变，肌肉重写。这正是 PVS 期望的"profile-conditioned but non-pathological" 特征。</li>
  <li><strong>Intent-driven complexity</strong>：LEARNING / DISCREPANCY 类 intent 触发最深 plan + 最多 pedagogy；
       CONCEPTUAL 类 plan 显著少 loop（69% vs ERRORS 的 83% / REVIEW 的 93%）—— 数据集 intent×pedagogy 的耦合可被 reward / metric 进一步利用。</li>
  <li><strong>schema 100% 干净</strong>：无 dangling depends_on / 无 cycle / 无 unknown agent ref。无需 pre-train filter。</li>
</ol>
</blockquote>

<blockquote class="warn">
<strong>对应 EXPERIMENT_PLAN 的待办风险点</strong>
<ul>
  <li>R_pers 阈值 220 chars 与 step_instruction 中位数 {int(A['step_instruction_len_chars']['median'])} chars 一致，但仍需检查 p05 是否真的 ≥ 220（建议补一次性 audit）。</li>
  <li>Loop schema 有 plural/singular 两种 (91% / 9%) → 解析器必须双兼容（已在 analysis 中实现，evaluation 端也要校对）。</li>
  <li>Cross-profile divergence 在 agent_role 与 subtask_tokens 上非常显著 (Jaccard 0.32 / 0.23) → PNG 的"non-generic"指标会有正信号；但 tools / role_family / pedagogy 的 Jaccard ≥ 0.76 → reward 不能简单"越不同越好"，否则会破坏骨架。</li>
</ul>
</blockquote>

<hr/>
<p class="meta">
  · 全部图表在 <code>MAPLE_Construction/claude_analysis/figures/</code><br/>
  · 全部数字在 <code>MAPLE_Construction/claude_analysis/stats.json</code><br/>
  · 重生成命令：<code>cd MAPLE_Construction/claude_analysis &amp;&amp; python3 run_analysis.py &amp;&amp; python3 build_report.py</code>
</p>

</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Wrote: {OUT}")
print(f"  size: {OUT.stat().st_size:,} bytes")
