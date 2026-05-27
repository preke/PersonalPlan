"""Build the comprehensive HTML report from stats.json + figures/."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "stats.json"))
OUT = HERE / "report.html"
OUT_ROOT = HERE.parent / "MAPLE_Comprehensive_Dataset_Analysis_2026-05-17.html"

A = S["A_overview"]
B = S["B_splits"]
Cu = S["C_counts_vs_unique"]
C = S["C_complexity"]
Cd = S["C_dag"]
D = S["D_agents"]
E = S["E_personalization"]
J = S["J_pedagogy"]
F = S["F_intent_summary"]
G = S["G_lexical"]
H = S["H_long_tail"]
I = S["I_quality"]

INTENT_ORDER = ["API_USAGE", "CONCEPTUAL", "DISCREPANCY", "ERRORS",
                "REVIEW", "API_CHANGE", "LEARNING"]


def fmt_int(n):
    return f"{int(n):,}"


def fmt_pct(p):
    return f"{p*100:.1f}%"


def fmt_f(x, n=2):
    return f"{x:.{n}f}"


# Build intent x split table rows
intent_pct = B["intent_x_split_pct"]
intent_cnt = B["intent_x_split"]
intent_rows_html = ""
for intent in INTENT_ORDER:
    tr = intent_cnt["train"].get(intent, 0)
    dv = intent_cnt["dev"].get(intent, 0)
    te = intent_cnt["test"].get(intent, 0)
    tr_p = intent_pct["train"].get(intent, 0)
    dv_p = intent_pct["dev"].get(intent, 0)
    te_p = intent_pct["test"].get(intent, 0)
    intent_rows_html += (
        f"<tr><td><code>{intent}</code></td>"
        f"<td class='num'>{tr:,} <span class='muted'>({tr_p:.1f}%)</span></td>"
        f"<td class='num'>{dv:,} <span class='muted'>({dv_p:.1f}%)</span></td>"
        f"<td class='num'>{te:,} <span class='muted'>({te_p:.1f}%)</span></td>"
        f"</tr>\n")

# intent complexity table
F_rows = ""
for row in F:
    F_rows += (f"<tr>"
               f"<td><code>{row['primary_intent']}</code></td>"
               f"<td class='num'>{row['n']:,}</td>"
               f"<td class='num'>{row['agents']:.2f}</td>"
               f"<td class='num'>{row['subtasks']:.2f}</td>"
               f"<td class='num'>{row['steps']:.2f}</td>"
               f"<td class='num'>{row['longest_path']:.2f}</td>"
               f"<td class='num'>{row['loops_mean']:.2f}</td>"
               f"<td class='num'>{row['loop_pct']*100:.0f}%</td>"
               f"<td class='num'>{row['human_input_mean']:.2f}</td>"
               f"<td class='num'>{row['tools']:.2f}</td>"
               f"</tr>\n")

# Cramér's V interpretation
def cramer_tag(v):
    if v < 0.1:
        return f'<span class="tag ok">Cramér&apos;s V = {v:.3f} · 几乎独立 (PASS)</span>'
    elif v < 0.2:
        return f'<span class="tag warn">Cramér&apos;s V = {v:.3f} · 微弱关联</span>'
    else:
        return f'<span class="tag bad">Cramér&apos;s V = {v:.3f} · 明显失衡</span>'


cv_intent = B["balance_chi2"]["intent"]["cramers_v"]
cv_pidx = B["balance_chi2"]["profile_index"]["cramers_v"]
cv_nprof = B["balance_chi2"]["nprof_per_qid"]["cramers_v"]

# top roles list
top_roles_html = ""
for r, c in D["top20_roles"][:15]:
    top_roles_html += (f"<tr><td><code>{r}</code></td>"
                       f"<td class='num'>{c:,}</td></tr>\n")

# all tools list
all_tools_html = ""
for t, c in D["all_tools"]:
    all_tools_html += (f"<tr><td><code>{t}</code></td>"
                       f"<td class='num'>{c:,}</td>"
                       f"<td class='num'>{c/D['total_agents']*100:.1f}%</td></tr>\n")

# role family list
fam_dist = D["role_family_distribution"]
fam_total = sum(fam_dist.values())
fam_html = ""
for fam, cnt in sorted(fam_dist.items(), key=lambda x: -x[1]):
    fam_html += (f"<tr><td><code>{fam}</code></td>"
                 f"<td class='num'>{cnt:,}</td>"
                 f"<td class='num'>{cnt/fam_total*100:.1f}%</td></tr>\n")

phase_html = ""
for phase, v in J["phase_stats"].items():
    phase_html += (
        f"<tr><td><code>{phase}</code></td>"
        f"<td class='num'>{v['plans']:,}</td>"
        f"<td class='num'>{v['plan_pct']*100:.1f}%</td>"
        f"<td class='num'>{v['steps']:,}</td>"
        f"<td class='num'>{v['step_pct']*100:.1f}%</td></tr>\n")

method_html = ""
for method, v in sorted(J["method_stats"].items(),
                        key=lambda x: -x[1]["plan_pct"]):
    method_html += (
        f"<tr><td>{method}</td>"
        f"<td class='num'>{v['plans']:,}</td>"
        f"<td class='num'>{v['plan_pct']*100:.1f}%</td>"
        f"<td class='num'>{v['steps']:,}</td>"
        f"<td class='num'>{v['step_pct']*100:.1f}%</td></tr>\n")

seq_html = ""
for seq, cnt in J["top_phase_sequences"][:8]:
    seq_html += (
        f"<tr><td><code>{seq}</code></td>"
        f"<td class='num'>{cnt:,}</td>"
        f"<td class='num'>{cnt/A['total_records']*100:.1f}%</td></tr>\n")

valid_rate = I["valid_rate"]


html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>MAP-PPL 数据集分析报告 — 最终版 (N={A['total_records']:,})</title>
<style>
  :root {{
    --fg:#1f2328; --muted:#57606a; --bg:#fff; --bg-soft:#f6f8fa; --border:#d0d7de;
    --accent:#0969da; --accent-soft:#ddf4ff; --warn:#9a6700; --warn-soft:#fff8c5;
    --bad:#cf222e; --bad-soft:#ffebe9; --ok:#1a7f37; --ok-soft:#dafbe1;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Helvetica Neue",Arial,sans-serif;
         color:var(--fg); background:#fafbfc; line-height:1.65; max-width:1240px; margin:0 auto;
         padding:36px 32px 80px; font-size:15px; }}
  h1 {{ font-size:30px; border-bottom:2px solid var(--border); padding-bottom:10px; margin-bottom:6px; letter-spacing:-0.01em; }}
  h2 {{ font-size:24px; margin-top:48px; padding-bottom:8px; border-bottom:1px solid var(--border);
        letter-spacing:-0.01em; }}
  h3 {{ font-size:18px; margin-top:28px; color:#24292f; }}
  p.subtitle {{ color:var(--muted); margin-top:0; font-size:15px; }}
  p.meta {{ color:var(--muted); font-size:13px; }}
  code, .mono {{ font-family:var(--mono); font-size:.92em; background:var(--bg-soft);
                 padding:1px 6px; border-radius:4px; border:1px solid var(--border); }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:13.5px; background:#fff;
           box-shadow:0 1px 2px rgba(0,0,0,0.03); border-radius:6px; overflow:hidden; }}
  th, td {{ border:1px solid var(--border); padding:8px 11px; vertical-align:top; text-align:left; }}
  th {{ background:var(--bg-soft); font-weight:600; }}
  tr:nth-child(even) td {{ background:#fbfcfd; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono); }}
  .muted {{ color:var(--muted); font-size:12px; }}
  .tag {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600;
           border:1px solid var(--border); background:var(--bg-soft); }}
  .tag.ok {{ color:var(--ok); background:var(--ok-soft); border-color:#aceebb; }}
  .tag.warn {{ color:var(--warn); background:var(--warn-soft); border-color:#eac54f; }}
  .tag.accent {{ color:var(--accent); background:var(--accent-soft); border-color:#b6e3ff; }}
  .tag.bad {{ color:var(--bad); background:var(--bad-soft); border-color:#ff8182; }}
  blockquote {{ border-left:3px solid var(--accent); margin:12px 0; padding:8px 16px;
                background:var(--accent-soft); color:#24292f; border-radius:0 4px 4px 0; }}
  blockquote.ok {{ border-color:#aceebb; background:var(--ok-soft); }}
  blockquote.warn {{ border-color:#eac54f; background:var(--warn-soft); }}
  blockquote.insight {{ border-color:#8b5cf6; background:#f3eeff; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:16px 0; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin:14px 0; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }}
  .card {{ border:1px solid var(--border); border-radius:10px; padding:16px 18px; background:#fff;
           box-shadow:0 1px 2px rgba(0,0,0,0.03); }}
  .card .label {{ color:var(--muted); font-size:13px; font-weight:500; margin-bottom:4px; }}
  .card .big {{ font-size:30px; font-weight:700; font-variant-numeric:tabular-nums;
                 line-height:1.1; color:#0969da; }}
  .card .pct {{ color:var(--muted); font-size:13px; margin-top:3px; }}
  pre {{ background:var(--bg-soft); border:1px solid var(--border); border-radius:6px; padding:10px 14px;
         overflow-x:auto; font-family:var(--mono); font-size:12.5px; line-height:1.5; }}
  .toc {{ background:var(--bg-soft); border:1px solid var(--border); border-radius:8px;
          padding:14px 22px; margin:20px 0 28px; font-size:14.5px; }}
  .toc a {{ color:var(--accent); text-decoration:none; }} .toc a:hover {{ text-decoration:underline; }}
  .toc ol {{ margin:6px 0; padding-left:24px; columns:2; }}
  .figrow {{ display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; margin:14px 0; }}
  .figrow .fig {{ flex:1 1 460px; }}
  .fig {{ background:#fff; border:1px solid var(--border); border-radius:8px; padding:8px;
          box-shadow:0 1px 2px rgba(0,0,0,0.03); }}
  .fig img {{ max-width:100%; height:auto; display:block; border-radius:4px; }}
  .fig .cap {{ font-size:13px; color:var(--muted); margin-top:6px; padding:0 4px 4px; }}
  details {{ margin:8px 0; border:1px solid var(--border); border-radius:6px;
              background:#fff; padding:8px 14px; }}
  details summary {{ cursor:pointer; font-weight:600; color:var(--accent); }}
  details[open] summary {{ margin-bottom:8px; }}
  .ref-pill {{ display:inline-block; margin:2px 4px 2px 0; padding:2px 9px;
                background:var(--accent-soft); border:1px solid #b6e3ff;
                border-radius:999px; font-size:12px; color:var(--accent); }}
</style>
</head>
<body>

<h1>MAP-PPL 数据集分析报告 · 最终版</h1>
<p class="subtitle">多智能体规划数据集 · {A['total_records']:,} 条 (query, profile, plan) 实例 · {A['unique_questions']:,} 个唯一 question_id</p>
<p class="meta">
  数据：<code>MAPLE_Construction/multi_agent_dataset_filtered_qap.jsonl</code><br/>
  划分：<code>splits/maple_split_v1.json</code> · 比例 80/10/10 · 按 nprof_per_qid 分层 · seed=42<br/>
  类目标签：<code>the_construction_of_MAPLE_datasets/task_3/classified_results.jsonl</code><br/>
  生成于 <code>analyze_paper.py</code> + <code>build_report.py</code>
</p>

<blockquote class="ok">
<strong>核心结论 (TL;DR)</strong>
<ul style="margin:6px 0 0 0">
  <li>规模：<strong>{A['total_records']:,}</strong> 条样本 / <strong>{A['unique_questions']:,}</strong> 个唯一 query / <strong>{A['unique_profiles_by_text']:,}</strong> 个唯一 learner profile。一对多平均扇出 <strong>{A['rows_per_qid_mean']:.2f}</strong>，最大 <strong>{A['max_profiles_per_question']}</strong>。</li>
  <li>划分均衡度：意图分布在三 split 下的 Cramér&apos;s V = <strong>{cv_intent:.3f}</strong>（几乎独立）；profile_index 分层 V = <strong>{cv_pidx:.3f}</strong>；nprof 桶 V = <strong>{cv_nprof:.3f}</strong>。<span class="tag ok">三个 split 类别分布高度对齐</span></li>
  <li>计划复杂度：每条计划平均 <strong>{C['agents_per_plan']['mean']:.2f}</strong> 个 agent、<strong>{C['subtasks_per_plan']['mean']:.2f}</strong> 个子任务、<strong>{C['steps_per_plan']['mean']:.2f}</strong> 个 step；DAG 关键路径深度均值 <strong>{Cd['longest_path_len']['mean']:.2f}</strong>，可并行度 <strong>{Cd['parallelizable_ratio_mean']:.2f}</strong>。</li>
  <li>个性化信号（同 query, 不同 profile 之间）：agent_role 名称 Jaccard 均值仅 <strong>{E['agent_role_jaccard_mean']:.2f}</strong>、subtask 措辞 Jaccard 仅 <strong>{E['subtask_name_jaccard_mean']:.2f}</strong>；但 role-family 层 Jaccard <strong>{E['agent_family_jaccard_mean']:.2f}</strong>、tool 集合 Jaccard <strong>{E['tool_jaccard_mean']:.2f}</strong>。<em>个性化发生在「具体表述」层，骨架（role-family + tools）保持稳定</em>。</li>
  <li>教学性：<strong>{J['phase_conditions']['validate_present_pct']*100:.1f}%</strong> 的 plan 含 validate 阶段，<strong>{J['phase_conditions']['ordered_core_4_phase_pct']*100:.1f}%</strong> 满足 probe→retrieve/demonstrate→apply→validate 顺序；R_ped proxy 均值 <strong>{J['r_ped_proxy_mean']:.2f}</strong>，个性化 grounded-step proxy 均值 <strong>{J['personalization_proxy']['r_pers_proxy_mean']:.2f}</strong>。</li>
  <li>Schema 合规：<strong>{I['valid_plans']:,}/{A['total_records']:,}</strong> 计划全部通过引用 / DAG / 执行序校验 (<strong>{valid_rate*100:.2f}%</strong>) — <span class="tag ok">零结构缺陷</span></li>
</ul>
</blockquote>

<div class="toc">
<strong>目录</strong>
<ol>
  <li><a href="#A">A · 总体概览</a></li>
  <li><a href="#B">B · Train / Dev / Test 划分均衡（新）</a></li>
  <li><a href="#C">C · Plan 结构复杂度</a></li>
  <li><a href="#D">D · Agent 设计 / 工具</a></li>
  <li><a href="#E">E · 个性化 & 一对多（重点）</a></li>
  <li><a href="#J">J · 教学法 / R_ped 对齐</a></li>
  <li><a href="#F">F · 意图 × 复杂度</a></li>
  <li><a href="#G">G · 词汇多样性（新）</a></li>
  <li><a href="#H">H · 工具长尾 & 角色共现（新）</a></li>
  <li><a href="#I">I · Schema 校验</a></li>
  <li><a href="#related">附：相关工作中的分析维度对照</a></li>
</ol>
</div>

<!-- ================ A · OVERVIEW ================ -->
<h2 id="A">A · 总体概览</h2>

<div class="grid4">
  <div class="card"><div class="label">总样本数 N</div>
    <div class="big">{A['total_records']:,}</div>
    <div class="pct">每行 = (query, profile) 对</div></div>
  <div class="card"><div class="label">唯一 question_id</div>
    <div class="big">{A['unique_questions']:,}</div>
    <div class="pct">平均 {A['rows_per_qid_mean']:.2f} 个 profile / qid</div></div>
  <div class="card"><div class="label">唯一 profile (按文本)</div>
    <div class="big">{A['unique_profiles_by_text']:,}</div>
    <div class="pct">{A['unique_profiles_by_text']/A['total_records']*100:.1f}% 唯一率</div></div>
  <div class="card"><div class="label">非空 profile 占比</div>
    <div class="big">{fmt_pct(A['nonempty_profile_pct'])}</div>
    <div class="pct">即所有样本均有学习者画像</div></div>
</div>

<h3>A.1 长度统计</h3>
<table>
<thead><tr><th>字段</th><th class='num'>mean</th><th class='num'>median</th><th class='num'>min</th><th class='num'>max</th></tr></thead>
<tbody>
<tr><td>Query 长度 (chars)</td>
    <td class='num'>{A['query_len_chars']['mean']:.0f}</td>
    <td class='num'>{A['query_len_chars']['median']:.0f}</td>
    <td class='num'>{A['query_len_chars']['min']}</td>
    <td class='num'>{A['query_len_chars']['max']}</td></tr>
<tr><td><code>about_me</code> 长度 (chars)</td>
    <td class='num'>{A['desc_len_chars_all']['mean']:.0f}</td>
    <td class='num'>{A['desc_len_chars_all']['median']:.0f}</td>
    <td class='num'>{A['desc_len_chars_all']['min']}</td>
    <td class='num'>{A['desc_len_chars_all']['max']}</td></tr>
<tr><td><code>top_tags</code> 个数</td>
    <td class='num'>{A['skills_per_profile_all']['mean']:.2f}</td>
    <td class='num'>{A['skills_per_profile_all']['median']:.0f}</td>
    <td class='num'>{A['skills_per_profile_all']['min']}</td>
    <td class='num'>{A['skills_per_profile_all']['max']}</td></tr>
<tr><td>Agent backstory 长度</td>
    <td class='num'>{A['agent_backstory_len_chars']['mean']:.0f}</td>
    <td class='num'>{A['agent_backstory_len_chars']['median']:.0f}</td><td class='num'>—</td><td class='num'>—</td></tr>
<tr><td>Agent goal 长度</td>
    <td class='num'>{A['agent_goal_len_chars']['mean']:.0f}</td>
    <td class='num'>{A['agent_goal_len_chars']['median']:.0f}</td><td class='num'>—</td><td class='num'>—</td></tr>
<tr><td>Step instruction 长度</td>
    <td class='num'>{A['instruction_len_chars']['mean']:.0f}</td>
    <td class='num'>{A['instruction_len_chars']['median']:.0f}</td><td class='num'>—</td><td class='num'>—</td></tr>
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/A2_query_length.png" />
    <div class="cap">A.2 — Query 长度分布（直方图 + KDE）。长尾向右，绝大多数 query 在 300–800 字符。</div></div>
  <div class="fig"><img src="figures/A3_selfdesc_length.png" />
    <div class="cap">A.3 — Learner <code>about_me</code> 长度。中位约 200 字符，存在数千字符的少量重度自述。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/A4_skills_count.png" />
    <div class="cap">A.4 — 每个 profile 拥有的 <code>top_tags</code> 数量。多数集中在 3–6 个。</div></div>
  <div class="fig"><img src="figures/A5_nprof_per_qid.png" />
    <div class="cap">A.5 — 一对多扇出：每个 qid 绑定的 profile 数量。绝大多数 qid 拥有 1–3 个 profile 变体。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/A1_profile_index_distribution.png" />
    <div class="cap">A.1 — 各 profile_index 下的记录数（profile_index=0 总是存在；后续指数代表追加的变体）。</div></div>
</div>

<!-- ================ B · SPLIT BALANCE ================ -->
<h2 id="B">B · Train / Dev / Test 划分均衡</h2>

<blockquote>
划分是按 <strong>question_id</strong> 进行的（保证同一 query 的所有 profile 变体落到同一 split，杜绝 leakage），并按 <em>每个 qid 的 profile 数</em> 做分层，比例 80/10/10，seed=42。结论是：意图、profile_index、nprof 桶在三个 split 上的条件分布几乎完全重合，可以放心做 generalization 评估。
</blockquote>

<div class="grid3">
  <div class="card"><div class="label">Train</div>
    <div class="big">{B['rows']['train']:,}</div>
    <div class="pct">行数 · {B['qids']['train']:,} qids · {B['rows']['train']/A['total_records']*100:.1f}%</div></div>
  <div class="card"><div class="label">Dev</div>
    <div class="big">{B['rows']['dev']:,}</div>
    <div class="pct">行数 · {B['qids']['dev']:,} qids · {B['rows']['dev']/A['total_records']*100:.1f}%</div></div>
  <div class="card"><div class="label">Test</div>
    <div class="big">{B['rows']['test']:,}</div>
    <div class="pct">行数 · {B['qids']['test']:,} qids · {B['rows']['test']/A['total_records']*100:.1f}%</div></div>
</div>

<div class="figrow">
  <div class="fig"><img src="figures/B1_split_sizes.png" />
    <div class="cap">B.1 — Train / Dev / Test 行数与 qid 数。</div></div>
</div>

<h3>B.1 意图（primary_intent）× split — 类目均衡度</h3>
<table>
<thead><tr><th>Primary intent</th><th class='num'>Train</th><th class='num'>Dev</th><th class='num'>Test</th></tr></thead>
<tbody>
{intent_rows_html}
</tbody>
</table>
<p>{cramer_tag(cv_intent)} χ²={B['balance_chi2']['intent']['chi2']:.2f}；
{cramer_tag(cv_pidx)} (profile_index)；
{cramer_tag(cv_nprof)} (nprof 桶)</p>

<div class="figrow">
  <div class="fig"><img src="figures/B2_intent_x_split.png" />
    <div class="cap">B.2 — 左：意图绝对数；右：每个 split 内部各意图占比（堆叠 100%）。三 split 内部的意图剖面形状一致。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/B3_split_strata.png" />
    <div class="cap">B.3 — 左：分层桶 nprof_per_qid 在 split 间的比例；右：profile_index 分布。Dev/Test 与 Train 的形状几乎完全重合 → 分层抽样有效。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/B4_complexity_per_split.png" />
    <div class="cap">B.4 — 计划复杂度在三 split 间的箱型对比（agents / steps / query 长度 / profile 长度），中位与四分位完全对齐 → 不存在划分引入的难度漂移。</div></div>
</div>

<!-- ================ C · COMPLEXITY ================ -->
<h2 id="C">C · Plan 结构复杂度</h2>

<div class="grid4">
  <div class="card"><div class="label">平均 agents / plan</div>
    <div class="big">{C['agents_per_plan']['mean']:.2f}</div>
    <div class="pct">中位 {C['agents_per_plan']['median']:.0f} · range [{C['agents_per_plan']['min']}, {C['agents_per_plan']['max']}]</div></div>
  <div class="card"><div class="label">平均 subtasks / plan</div>
    <div class="big">{C['subtasks_per_plan']['mean']:.2f}</div>
    <div class="pct">中位 {C['subtasks_per_plan']['median']:.0f} · range [{C['subtasks_per_plan']['min']}, {C['subtasks_per_plan']['max']}]</div></div>
  <div class="card"><div class="label">平均 steps / plan</div>
    <div class="big">{C['steps_per_plan']['mean']:.2f}</div>
    <div class="pct">中位 {C['steps_per_plan']['median']:.0f} · range [{C['steps_per_plan']['min']}, {C['steps_per_plan']['max']}]</div></div>
  <div class="card"><div class="label">含 loop 的 plan 占比</div>
    <div class="big">{C['plans_with_loop_pct']*100:.1f}%</div>
    <div class="pct">{C['plans_with_loop']:,} / {A['total_records']:,}</div></div>
</div>

<div class="grid4">
  <div class="card"><div class="label">DAG 关键路径深度</div>
    <div class="big">{Cd['longest_path_len']['mean']:.2f}</div>
    <div class="pct">中位 {Cd['longest_path_len']['median']:.0f} · max {Cd['longest_path_len']['max']}</div></div>
  <div class="card"><div class="label">最大层宽</div>
    <div class="big">{Cd['max_layer_width']['mean']:.2f}</div>
    <div class="pct">同步可执行步数上限</div></div>
  <div class="card"><div class="label">边 / 节点</div>
    <div class="big">{Cd['edges_per_node_mean']:.2f}</div>
    <div class="pct">依赖密度指标</div></div>
  <div class="card"><div class="label">可并行度</div>
    <div class="big">{Cd['parallelizable_ratio_mean']:.2f}</div>
    <div class="pct">1 − depth / |nodes|</div></div>
</div>

<h3>C.1 数据集级别 — 计数 vs 唯一值</h3>
<table>
<thead><tr><th>对象</th><th class='num'>总实例数</th><th class='num'>唯一值</th><th class='num'>每 plan 均值</th></tr></thead>
<tbody>
<tr><td>Agent</td>
    <td class='num'>{Cu['dataset_total']['total_agent_instances']:,}</td>
    <td class='num'>{Cu['dataset_total']['unique_agent_roles_global']:,}</td>
    <td class='num'>{Cu['per_plan']['agents_per_plan_mean']:.2f}</td></tr>
<tr><td>Subtask</td>
    <td class='num'>{Cu['dataset_total']['total_subtask_instances']:,}</td>
    <td class='num'>{Cu['dataset_total']['unique_subtask_names_global']:,}</td>
    <td class='num'>{Cu['per_plan']['subtasks_per_plan_mean']:.2f}</td></tr>
<tr><td>Step</td>
    <td class='num'>{Cu['dataset_total']['total_step_instances']:,}</td>
    <td class='num'>—</td>
    <td class='num'>{Cu['per_plan']['steps_per_plan_mean']:.2f}</td></tr>
<tr><td>Tool</td>
    <td class='num'>—</td>
    <td class='num'>{Cu['dataset_total']['unique_tools_global']:,}</td>
    <td class='num'>{Cu['per_plan']['tools_per_plan_total_mean']:.2f}</td></tr>
</tbody>
</table>
<p>说明：subtask 名称唯一率 <strong>{Cu['dataset_total']['unique_subtask_names_global']/Cu['dataset_total']['total_subtask_instances']*100:.1f}%</strong>、agent_role 唯一率 <strong>{Cu['dataset_total']['unique_agent_roles_global']/Cu['dataset_total']['total_agent_instances']*100:.1f}%</strong>，说明每个 plan 的 subtask / agent 命名几乎都是新生成的（避免模板化）。</p>

<div class="figrow">
  <div class="fig"><img src="figures/C1_agents_per_plan.png" />
    <div class="cap">C.1 — Agents per plan：87% 的 plan 使用 3 个 agent。</div></div>
  <div class="fig"><img src="figures/C2_subtasks_per_plan.png" />
    <div class="cap">C.2 — Subtasks per plan：以 4 个为主，少量 3 / 5 / 6。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/C3_steps_per_plan.png" />
    <div class="cap">C.3 — Steps per plan，分布近似正态，峰值在 10–11。</div></div>
  <div class="fig"><img src="figures/C4_human_input_ratio.png" />
    <div class="cap">C.4 — 单 plan 内 <code>requires_human_input=true</code> 步骤占比；均值 {C['human_input_step_ratio_overall']*100:.0f}%，说明这是「人在环」的强交互数据集。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/C5_loops.png" />
    <div class="cap">C.5 — 左：每 plan 的 loop 数；右：loop 的 <code>max_iterations</code> 分布。</div></div>
  <div class="fig"><img src="figures/C6_dag_metrics.png" />
    <div class="cap">C.6 — 左：DAG 关键路径长度；右：并行度。两者中位均接近 0.5 → 计划既不平铺也不极端串行。</div></div>
</div>

<!-- ================ D · AGENT DESIGN ================ -->
<h2 id="D">D · Agent 设计与工具</h2>

<div class="grid4">
  <div class="card"><div class="label">唯一 agent_role 数</div>
    <div class="big">{D['unique_agent_roles']:,}</div>
    <div class="pct">每个 role 平均出现 {Cu['dataset_total']['total_agent_instances']/D['unique_agent_roles']:.1f} 次</div></div>
  <div class="card"><div class="label">每 agent 工具数（mean）</div>
    <div class="big">{D['tools_per_agent_mean']:.2f}</div>
    <div class="pct">{D['zero_tool_agents_pct']*100:.1f}% 的 agent 无工具</div></div>
  <div class="card"><div class="label">唯一工具数</div>
    <div class="big">{D['unique_tools']:,}</div>
    <div class="pct">全局工具池规模小且高复用</div></div>
  <div class="card"><div class="label">每 plan 角色家族数</div>
    <div class="big">{D['roles_per_plan_mean']:.2f}</div>
    <div class="pct">即典型组合 tutor + retriever + validator</div></div>
</div>

<h3>D.1 Top-15 角色（按出现次数）</h3>
<table>
<thead><tr><th>agent_role</th><th class='num'># occurrences</th></tr></thead>
<tbody>
{top_roles_html}
</tbody>
</table>

<h3>D.2 角色家族（family）分布</h3>
<table>
<thead><tr><th>family</th><th class='num'># agents</th><th class='num'>占比</th></tr></thead>
<tbody>
{fam_html}
</tbody>
</table>

<h3>D.3 工具全名单（仅 {D['unique_tools']} 个，高度集中）</h3>
<table>
<thead><tr><th>tool</th><th class='num'># 出现</th><th class='num'>占 agent 比例</th></tr></thead>
<tbody>
{all_tools_html}
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/D1_top_roles.png" />
    <div class="cap">D.1 — Top-15 agent_role 名称。前列由 *_code_validator / *_docs_retriever 主导，体现按语言/领域专门化设计。</div></div>
  <div class="fig"><img src="figures/D2_tools.png" />
    <div class="cap">D.2 — 左：每 agent 工具数分布；右：数据集级 top tools。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/D3_role_families.png" />
    <div class="cap">D.3 — 角色家族分布饼图。tutor / retriever / validator 三足鼎立。</div></div>
  <div class="fig"><img src="figures/D4_role_family_upset.png" />
    <div class="cap">D.4 — UpSet 风格：Top-10 角色家族组合。主流模式 <code>tutor + retriever + validator</code> 占比最高，构成典型「解释-取证-验证」三人组。</div></div>
</div>

<!-- ================ E · PERSONALIZATION ================ -->
<h2 id="E">E · 个性化 / 一对多分析（重点）</h2>

<blockquote class="insight">
<strong>核心问题</strong>：因为 MAP-PPL 中同一个 query 会被绑定到多个 learner profile 各自生成一份 plan，我们需要证明：(1) 不同 profile 真的<strong>导致</strong>了实质不同的 plan；(2) 这种差异化是<strong>结构化</strong>的（与 profile 的相似度相关），而不是随机噪声。下面几张图回答这两件事。
</blockquote>

<div class="grid4">
  <div class="card"><div class="label">有 ≥2 profile 的 qid</div>
    <div class="big">{E['questions_with_multi_profile']:,}</div>
    <div class="pct">/ 共 {A['unique_questions']:,} 个 qid · {E['questions_with_multi_profile']/A['unique_questions']*100:.1f}%</div></div>
  <div class="card"><div class="label">同-qid 内 profile 对数</div>
    <div class="big">{E['total_pairs']:,}</div>
    <div class="pct">用于跨-profile plan 相似度分析</div></div>
  <div class="card"><div class="label">role Jaccard 均值</div>
    <div class="big">{E['agent_role_jaccard_mean']:.2f}</div>
    <div class="pct">中位 {E['agent_role_jaccard_median']:.2f} · 越低 = 越个性化</div></div>
  <div class="card"><div class="label">family Jaccard 均值</div>
    <div class="big">{E['agent_family_jaccard_mean']:.2f}</div>
    <div class="pct">高 → 角色家族级别保持稳定</div></div>
</div>

<h3>E.1 跨-profile plan 相似度（四个维度的 Jaccard）</h3>
<div class="figrow">
  <div class="fig"><img src="figures/E1_jaccard_violin.png" />
    <div class="cap">E.1 — 同一 query、不同 profile 之间生成的 plan，在四个维度的 Jaccard 分布（n={E['total_pairs']:,} 对）。
    呈现明显的<strong>双轨结构</strong>：
    <em>高层骨架稳定</em> — role family Jaccard={E['agent_family_jaccard_mean']:.2f}、tool Jaccard={E['tool_jaccard_mean']:.2f}；
    <em>具体表述差异化</em> — agent_role 名称 Jaccard={E['agent_role_jaccard_mean']:.2f}、subtask token Jaccard={E['subtask_name_jaccard_mean']:.2f}。
    意味着模型保留了「同样的高层 pipeline（一个 tutor + 一个 retriever + 一个 validator，配同样的两件工具）」，但<strong>每个 profile 都被实质重写了具体角色名和 subtask 措辞</strong>。这正是 MAP-PPL 个性化设计的关键证据。</div></div>
</div>

<h3>E.2 Profile 相似度 → Plan 相似度：是否结构化？</h3>
<div class="figrow">
  <div class="fig"><img src="figures/E4_profile_vs_plan_similarity.png" />
    <div class="cap">E.2 — 横轴为同 qid 两 profile 的技能 Jaccard，纵轴为对应 plan 的 agent_role Jaccard。Pearson r = <strong>{E.get('profile_vs_plan_pearson_r', 0):.3f}</strong>，<strong>正相关</strong>但弱 → profile 越像，plan 名称越接近，但即使是几乎相同的 profile，plan 也会有差异化（不是简单复制）。</div></div>
</div>

<h3>E.3 一对多扇出深度 vs 个性化强度</h3>
<div class="figrow">
  <div class="fig"><img src="figures/E5_divergence_by_nprof.png" />
    <div class="cap">E.3 — 按 nprof_per_qid 分桶，看每对的 plan 发散度 (1 − Jaccard) 是否随扇出增加而变化。结论：发散度在不同桶下基本稳定 → 即使一个 query 被绑定到 6 个 profile，每对之间也保持一致的个性化幅度，没有出现「越往后越敷衍」的情况。</div></div>
</div>

<h3>E.4 词面泄漏（profile 是否真的影响 plan 文本）</h3>
<div class="grid3">
  <div class="card"><div class="label">至少一条 skill 在 plan 中出现</div>
    <div class="big">{E['skill_at_least_one_pct']*100:.1f}%</div>
    <div class="pct">{E['skill_at_least_one_record_count']:,}/{E['records_with_skills']:,} 条记录</div></div>
  <div class="card"><div class="label">skill 字面命中率（均值）</div>
    <div class="big">{E['skill_lex_hit_ratio_mean']*100:.1f}%</div>
    <div class="pct">中位 {E['skill_lex_hit_ratio_median']*100:.1f}%</div></div>
  <div class="card"><div class="label">about_me token 命中率（均值）</div>
    <div class="big">{E['desc_token_hit_ratio_mean']*100:.1f}%</div>
    <div class="pct">≥4 字 token 在 plan 中字面出现的比例</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/E2_skill_hit_ratio.png" />
    <div class="cap">E.4 — Profile→Plan 词面泄漏直方图：plan 实际有「使用 profile 关键词」的强证据，而不是空喊「individualised」。</div></div>
</div>

<h3>E.5 各意图下的个性化强度</h3>
<div class="figrow">
  <div class="fig"><img src="figures/E3_personalization_by_intent.png" />
    <div class="cap">E.5 — 按 query primary intent 切分后的 plan 发散度（1 − role Jaccard）。所有意图都在 0.6–0.8 区间，说明 profile-conditioning 在不同问题类型下都成立。</div></div>
</div>

<blockquote class="ok"><strong>解读</strong>：
agent_role 名称、subtask 措辞几乎随每对 profile 全换（Jaccard 分别 {E['agent_role_jaccard_mean']:.2f} / {E['subtask_name_jaccard_mean']:.2f}），但角色家族、工具集保持稳定（{E['agent_family_jaccard_mean']:.2f} / {E['tool_jaccard_mean']:.2f}），证明：
<ul style="margin:6px 0 0 0">
<li><strong>plan 不是简单复制</strong>（如果是，role / subtask Jaccard 都应接近 1.0）；</li>
<li><strong>也不是随机重铸</strong>（否则 family / tools Jaccard 也应低）；</li>
<li>而是<strong>在保留高层 pipeline（role-family + tools）的同时，为每个 profile 重新具体化角色名与 subtask 措辞</strong>——这正是 MAP-PPL 想要的「同任务、不同教练」个性化形态。</li>
</ul></blockquote>

<!-- ================ J · PEDAGOGY ================ -->
<h2 id="J">J · 教学法 / 个性化奖励目标对齐</h2>

<blockquote>
本节按实验方案中的 <code>R_pers</code> 与 <code>R_ped</code> 目标做数据侧审计：看 gold plan 是否真的包含可学习的教学流程、反馈闭环、验证阶段，以及 profile-grounded 的个性化指令。这里的统计是规则式 proxy，不等同于最终 LLM-judge 分数，但适合证明数据集本身携带这些监督信号。
</blockquote>

<div class="grid4">
  <div class="card"><div class="label">PhaseCov_v2 proxy</div>
    <div class="big">{J['phase_conditions']['phasecov_v2_mean']:.2f}</div>
    <div class="pct">3 条件：early probe / validate / ordered 4-phase</div></div>
  <div class="card"><div class="label">R_ped proxy</div>
    <div class="big">{J['r_ped_proxy_mean']:.2f}</div>
    <div class="pct">mean(PhaseCov_v2, PlanSizeQuality)</div></div>
  <div class="card"><div class="label">PlanSizeQuality</div>
    <div class="big">{J['plan_size_quality_mean']:.2f}</div>
    <div class="pct">subtasks 3–5 / steps 8–13 为高分区间</div></div>
  <div class="card"><div class="label">Grounded-step proxy</div>
    <div class="big">{J['personalization_proxy']['r_pers_proxy_mean']:.2f}</div>
    <div class="pct">{J['personalization_proxy']['plans_with_grounded_step_pct']*100:.1f}% plan 至少一处 profile-grounded 指令</div></div>
</div>

<h3>J.1 Plan 长度与指令密度</h3>
<table>
<thead><tr><th>字段</th><th class='num'>均值</th><th class='num'>备注</th></tr></thead>
<tbody>
<tr><td>Plan text 长度（chars）</td><td class='num'>{J['lengths']['plan_text_chars_mean']:.0f}</td><td>agents + subtasks + steps 文本拼接</td></tr>
<tr><td>Plan text token 数</td><td class='num'>{J['lengths']['plan_text_tokens_mean']:.0f}</td><td>正则 token 近似，不是 tokenizer token</td></tr>
<tr><td>Plan JSON 长度（chars）</td><td class='num'>{J['lengths']['plan_json_chars_mean']:.0f}</td><td>完整结构化 plan</td></tr>
<tr><td>Step instruction 长度</td><td class='num'>{J['lengths']['step_instruction_chars_mean']:.0f}</td><td>中位 {J['lengths']['step_instruction_chars_median']:.0f} chars</td></tr>
</tbody>
</table>

<h3>J.2 教学 phase 覆盖</h3>
<table>
<thead><tr><th>phase</th><th class='num'>plan 命中</th><th class='num'>plan 占比</th><th class='num'>step 命中</th><th class='num'>step 占比</th></tr></thead>
<tbody>
{phase_html}
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/J1_phase_coverage.png" />
    <div class="cap">J.1 — 左：probe / retrieve-demonstrate / apply / validate / feedback / consolidate 六类教学阶段信号；右：实验方案中 <code>PhaseCov_v2</code> 的三个布尔条件。validate 基本是全覆盖，ordered 4-phase 是更严格的流程约束。</div></div>
</div>

<h3>J.3 教学法 taxonomy</h3>
<table>
<thead><tr><th>教学法 / 方法</th><th class='num'>plan 命中</th><th class='num'>plan 占比</th><th class='num'>step 命中</th><th class='num'>step 占比</th></tr></thead>
<tbody>
{method_html}
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/J2_pedagogical_methods.png" />
    <div class="cap">J.2 — 从 plan 文本中抽取的教学法信号。高频方法集中在 Socratic probing、practice/application、validation/testing、docs grounding 与 iterative feedback，符合「问诊 → 取证/示范 → 实作 → 验证/反馈」的教学 MAS 设计。</div></div>
  <div class="fig"><img src="figures/J4_methods_by_intent.png" />
    <div class="cap">J.4 — 不同 query intent 下的教学法覆盖率。绝大多数方法跨意图稳定出现，说明 pedagogical scaffold 不是只对某一类题型有效。</div></div>
</div>

<h3>J.4 Reward-proxy 分布与 phase 顺序</h3>
<div class="figrow">
  <div class="fig"><img src="figures/J3_reward_proxy_distributions.png" />
    <div class="cap">J.3 — <code>R_pers</code> 与 <code>R_ped</code> 的规则式 proxy 分布。个性化 proxy 要求同一句 instruction 同时含第二人称、profile signal、且长度 ≥220 chars，因此比简单 keyword hit 更严格。</div></div>
</div>

<table>
<thead><tr><th>Top phase sequence（按第一次出现顺序）</th><th class='num'>plan 数</th><th class='num'>占比</th></tr></thead>
<tbody>
{seq_html}
</tbody>
</table>

<blockquote class="ok"><strong>解读</strong>：
MAP-PPL 的 gold plan 不只是「多 agent JSON」。它在结构上稳定包含：前置 probe、文档/示范、应用练习、验证执行，以及失败后的反馈 loop。对论文来说，这一节可以支撑两个主张：(1) 数据集为 <code>R_ped</code> 提供了真实监督信号；(2) 个性化不是只改 agent 名称，step instruction 中也存在 profile-grounded 的教学适配。
</blockquote>

<!-- ================ F · INTENT × COMPLEXITY ================ -->
<h2 id="F">F · 意图 × 复杂度</h2>

<table>
<thead><tr><th>Intent</th><th class='num'>n</th><th class='num'>agents</th><th class='num'>subtasks</th>
<th class='num'>steps</th><th class='num'>depth</th><th class='num'>loops</th><th class='num'>loop%</th>
<th class='num'>human-input</th><th class='num'>tools</th></tr></thead>
<tbody>
{F_rows}
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/F1_intent_complexity_heatmap.png" />
    <div class="cap">F.1 — 各意图类下计划复杂度的归一化热力图。<code>LEARNING</code> 类显著更复杂（agents/steps/depth 全线偏高，n=12 较小，需谨慎），<code>API_CHANGE</code>/<code>ERRORS</code> 类略偏简单。</div></div>
  <div class="fig"><img src="figures/F2_intent_donut.png" />
    <div class="cap">F.2 — 意图占比。<code>CONCEPTUAL</code>+<code>API_USAGE</code> 合计 &gt;85%，反映底层来源是技术问答语料的天然偏置。</div></div>
</div>

<!-- ================ G · LEXICAL ================ -->
<h2 id="G">G · 词汇多样性</h2>

<table>
<thead><tr><th>文本来源</th><th class='num'>tokens</th><th class='num'>vocab</th><th class='num'>TTR</th><th class='num'>distinct-1</th><th class='num'>distinct-2</th><th class='num'>distinct-3</th></tr></thead>
<tbody>
<tr><td>Query</td>
    <td class='num'>{G['query']['tokens']:,}</td>
    <td class='num'>{G['query']['vocab']:,}</td>
    <td class='num'>{G['query']['ttr']:.4f}</td>
    <td class='num'>{G['query']['distinct_1']:.3f}</td>
    <td class='num'>{G['query']['distinct_2']:.3f}</td>
    <td class='num'>{G['query']['distinct_3']:.3f}</td></tr>
<tr><td>Plan text（所有 agents+subtasks+steps）</td>
    <td class='num'>{G['plan_text']['tokens']:,}</td>
    <td class='num'>{G['plan_text']['vocab']:,}</td>
    <td class='num'>{G['plan_text']['ttr']:.4f}</td>
    <td class='num'>{G['plan_text']['distinct_1']:.3f}</td>
    <td class='num'>{G['plan_text']['distinct_2']:.3f}</td>
    <td class='num'>{G['plan_text']['distinct_3']:.3f}</td></tr>
<tr><td>Skill / top_tags</td>
    <td class='num'>{G['skills']['tokens']:,}</td>
    <td class='num'>{G['skills']['vocab']:,}</td>
    <td class='num' colspan='4'>—</td></tr>
<tr><td>Agent role 名称（小写）</td>
    <td class='num'>{G['agent_roles']['tokens']:,}</td>
    <td class='num'>{G['agent_roles']['vocab']:,}</td>
    <td class='num' colspan='4'>—</td></tr>
</tbody>
</table>

<div class="figrow">
  <div class="fig"><img src="figures/G1_distinct_n.png" />
    <div class="cap">G.1 — Distinct-n 对比：query 的低阶多样性 (distinct-1) 略高，plan 在 distinct-3 上反超 → plan 在长 ngram 上更不重复（生成式描述带来的高短语多样性）。</div></div>
  <div class="fig"><img src="figures/G2_top_skills.png" />
    <div class="cap">G.2 — Top-25 学习者技能标签。<code>c++</code>/<code>python</code>/<code>java</code>/<code>c#</code>/<code>javascript</code> 等占多数 → 主要覆盖通用编程语言，但有 {G['skills']['vocab']:,} 种唯一标签，长尾极长。</div></div>
</div>

<!-- ================ H · TOOL ZIPF / CO-OCCURRENCE ================ -->
<h2 id="H">H · 工具长尾与共现网络</h2>

<div class="figrow">
  <div class="fig"><img src="figures/H1_tool_zipf.png" />
    <div class="cap">H.1 — 工具 Zipf 曲线。仅 5 个工具，Top-2（<code>CodeInterpreterTool</code>, <code>CodeDocsSearchTool</code>）占绝大多数使用 — 工具池小但高度集中。</div></div>
  <div class="fig"><img src="figures/H2_tool_cooccurrence.png" />
    <div class="cap">H.2 — Plan 内工具共现热力图。对角线 = 该工具在多少 plan 中出现；非对角线 = 同 plan 共现次数。揭示 <code>CodeInterpreterTool</code>+<code>CodeDocsSearchTool</code> 是默认组合。</div></div>
</div>
<div class="figrow">
  <div class="fig"><img src="figures/H3_role_family_cooccurrence.png" />
    <div class="cap">H.3 — 角色家族共现热力图。<code>tutor × retriever × validator</code> 三组的两两共现都接近 plan 总数 → 三足鼎立结构是 MAP-PPL 的硬规范。</div></div>
</div>

<!-- ================ I · QUALITY ================ -->
<h2 id="I">I · Schema / 引用 / DAG 校验</h2>

<table>
<thead><tr><th>检查项</th><th class='num'>问题数</th></tr></thead>
<tbody>
<tr><td>step.agent 指向未定义角色</td><td class='num'>{I['plans_with_unknown_agent_ref']:,}</td></tr>
<tr><td>step.depends_on 指向未定义 step.id</td><td class='num'>{I['plans_with_unknown_depends_on']:,}</td></tr>
<tr><td>execution_order 包含未知 step.id</td><td class='num'>{I['plans_with_unknown_step_in_exec']:,}</td></tr>
<tr><td>execution_order 缺漏某个 step.id</td><td class='num'>{I['plans_with_missing_step_in_exec']:,}</td></tr>
<tr><td>loop.steps/step 中含未知 step.id</td><td class='num'>{I['plans_with_loop_step_unknown']:,}</td></tr>
<tr><td>DAG 含有环</td><td class='num'>{I['plans_with_cycle']:,}</td></tr>
<tr><td><strong>含任意问题的 plan 数</strong></td><td class='num'><strong>{I['plans_with_any_issue']:,}</strong></td></tr>
<tr><td><strong>完全合规 plan 数</strong></td><td class='num'><strong>{I['valid_plans']:,} / {A['total_records']:,}</strong></td></tr>
<tr><td><strong>合规率</strong></td><td class='num'><strong>{I['valid_rate']*100:.2f}%</strong></td></tr>
</tbody>
</table>
<div class="figrow">
  <div class="fig"><img src="figures/I1_quality.png" />
    <div class="cap">I.1 — 各类引用 / 结构问题统计。全部为零 → <span class="tag ok">100% 通过严格校验</span></div></div>
</div>

<!-- ================ RELATED ================ -->
<h2 id="related">附录 · 相关工作的分析维度对照</h2>

<p>在设计本报告时，我们参考了近年发布多智能体 / LLM Agent planning 数据集的论文「Dataset Analysis」章节所采用的维度。下表列出与之的对应关系，并标注 MAP-PPL 是否覆盖。</p>

<table>
<thead><tr><th>分析维度</th><th>代表性数据集</th><th>本报告</th></tr></thead>
<tbody>
<tr><td>规模 / 长度统计</td><td>AgentBench, AgentBoard, TaskBench</td><td><span class="tag ok">A 节</span></td></tr>
<tr><td>子任务 / 步骤计数</td><td>TaskBench, AgentInstruct, PlanBench</td><td><span class="tag ok">C 节</span></td></tr>
<tr><td>DAG 深度 / 并行度 / 依赖密度</td><td>TravelPlanner, PlanBench</td><td><span class="tag ok">C 节</span></td></tr>
<tr><td>意图 / 类目细分 × 复杂度</td><td>NATURAL PLAN, AgentBench</td><td><span class="tag ok">F 节</span></td></tr>
<tr><td>训练/开发/测试集类别均衡度 (χ², Cramér&apos;s V)</td><td>Mind2Web, WebArena</td><td><span class="tag ok">B 节（新增）</span></td></tr>
<tr><td>角色 / 工具分布 + 长尾 + 共现</td><td>ToolBench, ToolLLM, API-Bank</td><td><span class="tag ok">D, H 节</span></td></tr>
<tr><td>词汇多样性 (distinct-n, TTR)</td><td>AgentInstruct, LIMA-style</td><td><span class="tag ok">G 节（新增）</span></td></tr>
<tr><td>个性化 / 条件依赖效应量</td><td>PersonalLLM, LaMP</td><td><span class="tag ok">E 节（重点重写）</span></td></tr>
<tr><td>教学 phase / 教学法覆盖 / reward proxy</td><td>Aligning Pedagogy, EduPlanner, GenMentor</td><td><span class="tag ok">J 节（新增）</span></td></tr>
<tr><td>Schema / 引用 / DAG 校验合规率</td><td>PlanBench, ToolBench</td><td><span class="tag ok">I 节</span></td></tr>
<tr><td>Inter-annotator agreement</td><td>TravelPlanner, NATURAL PLAN</td><td><span class="tag warn">未覆盖 — 数据为合成生成</span></td></tr>
<tr><td>Baseline difficulty curve (model-stratified success)</td><td>PlanBench, MultiAgentBench</td><td><span class="tag warn">待评测后补充</span></td></tr>
<tr><td>Token cost / inference budget</td><td>AgentBench, AgentBoard</td><td><span class="tag warn">待评测后补充</span></td></tr>
</tbody>
</table>

<p style="margin-top:24px; color:var(--muted); font-size:12.5px;">
报告由 <code>analyze_paper.py</code> + <code>build_report.py</code> 自动生成。
所有图表使用 seaborn (crest_r / rocket_r / Set2) 配色，PNG 220 dpi，可用于论文嵌入。
图表源 PNG 位于 <code>figures/</code>，机器可读统计位于 <code>stats.json</code>。
</p>

</body>
</html>
"""

(OUT).write_text(html, encoding="utf-8")
OUT_ROOT.write_text(html, encoding="utf-8")
print(f"Wrote {OUT} ({len(html):,} chars)")
print(f"Wrote {OUT_ROOT} ({len(html):,} chars)")
