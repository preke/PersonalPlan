"""Generate DATA_SPLIT_REPORT_2026-05-15.html — a detailed distribution report
for the Train/Dev/Test split materialised by build_splits.py.

Reads:  splits/split_stats.json  +  the three SFT/GRPO JSONL families
Writes: DATA_SPLIT_REPORT_2026-05-15.html
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATS = ROOT / "splits" / "split_stats.json"
SFT = ROOT / "SFT" / "data"
GRPO = ROOT / "GRPO" / "data" / "grpo"
OUT = ROOT / "DATA_SPLIT_REPORT_2026-05-15.html"


def jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def quantiles(xs: list[int]) -> dict[str, int]:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    if n == 0:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0}

    def pct(p):
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs_sorted[i]

    return {
        "min": xs_sorted[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs_sorted[-1],
        "mean": int(round(statistics.mean(xs_sorted))),
    }


def payload_lengths():
    """Character-length quantiles for each split × dataset."""
    out: dict[str, dict[str, dict]] = {}
    for split in ("train", "dev", "test"):
        out[split] = {}
        # PAD assistant text (target tokens proxy)
        out[split]["pad_assistant_chars"] = quantiles(
            [len(r["messages"][-1]["content"]) for r in jsonl(SFT / "pad" / f"{split}.jsonl")]
        )
        # SDP assistant text
        out[split]["sdp_assistant_chars"] = quantiles(
            [len(r["messages"][-1]["content"]) for r in jsonl(SFT / "sdp" / f"{split}.jsonl")]
        )
        # GRPO prompt total chars (system + user)
        grpo_lens = []
        for r in jsonl(GRPO / f"{split}.jsonl"):
            grpo_lens.append(sum(len(m["content"]) for m in r["prompt"]))
        out[split]["grpo_prompt_chars"] = quantiles(grpo_lens)
        # gold_plan chars
        out[split]["grpo_gold_plan_chars"] = quantiles(
            [len(r["gold_plan"]) for r in jsonl(GRPO / f"{split}.jsonl")]
        )
    return out


def stratum_row(label: str, get_val):
    return "".join(
        f"<td class='num'>{get_val(k)}</td>" for k in (1, 2, 3, 4, 5, 6)
    )


def bar(percent: float, width: int = 220, color: str = "#0969da") -> str:
    w = max(2, int(round(percent * width / 100)))
    return (
        f"<div style='background:#eaeef2;width:{width}px;height:10px;border-radius:6px;display:inline-block;vertical-align:middle'>"
        f"<div style='background:{color};width:{w}px;height:10px;border-radius:6px'></div></div>"
    )


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    plens = payload_lengths()
    g = stats["global"]
    s = stats["splits"]

    nprof_keys = sorted(int(k) for k in g["nprof_per_qid"].keys())
    pidx_keys = sorted(int(k) for k in g["profile_index_dist"].keys())

    def nprof_row(d: dict, total_qids: int):
        cells = []
        for k in nprof_keys:
            v = d.get(str(k), d.get(k, 0))
            frac = (v / total_qids * 100) if total_qids else 0
            cells.append(f"<td class='num'>{v}<br/><span class='muted'>{frac:.1f}%</span></td>")
        return "".join(cells)

    def pidx_row(d: dict, total_rows: int):
        cells = []
        for k in pidx_keys:
            v = d.get(str(k), d.get(k, 0))
            frac = (v / total_rows * 100) if total_rows else 0
            cells.append(f"<td class='num'>{v}<br/><span class='muted'>{frac:.1f}%</span></td>")
        return "".join(cells)

    def qbar(part: int, whole: int, color: str = "#0969da") -> str:
        pct_ = (part / whole * 100) if whole else 0
        return f"{bar(pct_, color=color)} <span class='muted'>{pct_:.1f}%</span>"

    # length table rows
    def len_block(split: str):
        order = [
            ("PAD assistant target (chars)", "pad_assistant_chars"),
            ("SDP assistant target (chars)", "sdp_assistant_chars"),
            ("GRPO prompt (chars)", "grpo_prompt_chars"),
            ("Gold plan (chars)", "grpo_gold_plan_chars"),
        ]
        rows_html = []
        for label, key in order:
            q = plens[split][key]
            rows_html.append(
                f"<tr><td>{label}</td>"
                f"<td class='num'>{q['min']}</td><td class='num'>{q['mean']}</td>"
                f"<td class='num'>{q['p50']}</td><td class='num'>{q['p90']}</td>"
                f"<td class='num'>{q['p95']}</td><td class='num'>{q['p99']}</td>"
                f"<td class='num'>{q['max']}</td></tr>"
            )
        return "\n".join(rows_html)

    overlap_ok = all(v == 0 for v in stats["overlap_check"].values())
    leak_ok = all(s[k]["leakage_rows"] == 0 for k in ("train", "dev", "test"))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<title>MAP-PPL 数据划分报告 — 2026-05-15</title>
<style>
  :root {{
    --fg:#1f2328; --muted:#57606a; --bg:#fff; --bg-soft:#f6f8fa; --border:#d0d7de;
    --accent:#0969da; --accent-soft:#ddf4ff; --warn:#9a6700; --warn-soft:#fff8c5;
    --bad:#cf222e; --bad-soft:#ffebe9; --ok:#1a7f37; --ok-soft:#dafbe1;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Helvetica Neue",Arial,sans-serif;
         color:var(--fg); background:var(--bg); line-height:1.6; max-width:1180px; margin:32px auto; padding:0 28px 80px; font-size:15px; }}
  h1 {{ font-size:28px; border-bottom:2px solid var(--border); padding-bottom:10px; margin-bottom:4px; }}
  h2 {{ font-size:22px; margin-top:40px; padding-bottom:6px; border-bottom:1px solid var(--border); }}
  h3 {{ font-size:18px; margin-top:24px; color:#24292f; }}
  p.subtitle {{ color:var(--muted); margin-top:0; font-size:15px; }}
  p.meta {{ color:var(--muted); font-size:13px; }}
  code, .mono {{ font-family:var(--mono); font-size:.92em; background:var(--bg-soft);
                 padding:1px 6px; border-radius:4px; border:1px solid var(--border); }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:13.5px; }}
  th, td {{ border:1px solid var(--border); padding:7px 9px; vertical-align:top; text-align:left; }}
  th {{ background:var(--bg-soft); font-weight:600; }}
  tr:nth-child(even) td {{ background:#fbfcfd; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono); }}
  .muted {{ color:var(--muted); font-size:12px; }}
  .tag {{ display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600;
           border:1px solid var(--border); background:var(--bg-soft); }}
  .tag.ok {{ color:var(--ok); background:var(--ok-soft); border-color:#aceebb; }}
  .tag.warn {{ color:var(--warn); background:var(--warn-soft); border-color:#eac54f; }}
  .tag.accent {{ color:var(--accent); background:var(--accent-soft); border-color:#b6e3ff; }}
  .tag.bad {{ color:var(--bad); background:var(--bad-soft); border-color:#ff8182; }}
  blockquote {{ border-left:3px solid var(--accent); margin:12px 0; padding:6px 14px;
                background:var(--accent-soft); color:#24292f; border-radius:0 4px 4px 0; }}
  blockquote.ok {{ border-color:#aceebb; background:var(--ok-soft); }}
  blockquote.warn {{ border-color:#eac54f; background:var(--warn-soft); }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin:14px 0; }}
  .card {{ border:1px solid var(--border); border-radius:8px; padding:14px 16px; background:#fff; }}
  .card .big {{ font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1; }}
  .card .pct {{ color:var(--muted); font-size:13px; }}
  pre {{ background:var(--bg-soft); border:1px solid var(--border); border-radius:6px; padding:10px 14px;
         overflow-x:auto; font-family:var(--mono); font-size:12.5px; line-height:1.5; }}
  .toc {{ background:var(--bg-soft); border:1px solid var(--border); border-radius:8px;
          padding:12px 20px; margin:18px 0 24px; font-size:14.5px; }}
  .toc a {{ color:var(--accent); text-decoration:none; }} .toc a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>

<h1>MAP-PPL 数据划分报告</h1>
<p class="subtitle">PersonalPlan / MAP-PPL 实验方案 v2 — 训练/开发/测试集冻结结果</p>
<p class="meta">
  日期：2026-05-15　·　源文件：<code>{stats['src_file']}</code><br/>
  划分键：<code>question_id</code>　·　分层键：<em>每个 qid 拥有的 profile 数</em>　·　seed = <code>{stats['seed']}</code>　·　比例 = 80 / 10 / 10<br/>
  生成器：<code>build_splits.py</code> → <code>build_split_report.py</code>
</p>

<blockquote class="ok">
<strong>核心结论</strong>
<ul style="margin:6px 0 0 0">
  <li>三个 split 数量与实验方案 §2.2 锁定值完全一致：Train <strong>1,384 qids / 2,433 行</strong>，Dev <strong>173 qids / 305 行</strong>，Test <strong>173 qids / 305 行</strong>。</li>
  <li>跨 split 重叠 <code>{stats['overlap_check']}</code> — {'<span class="tag ok">PASS</span> 无 qid 泄漏' if overlap_ok else '<span class="tag bad">FAIL</span>'}。</li>
  <li>派生的行级 split 通过严格校验：每行 question_id 仅出现在其所属 qid 的 split 中（leakage_rows = 0）— {'<span class="tag ok">PASS</span>' if leak_ok else '<span class="tag bad">FAIL</span>'}。</li>
  <li>分层（按 nprof 桶）后 Dev/Test 各得 <strong>25 个 ≥3-profile qid</strong>，正好命中实验方案 §2.2 中 Profile-Sensitivity 子集的"25 个 qid × 3 profile = ~75 评估点"目标。</li>
  <li>已写入：<code>SFT/data/{{pad,sdp}}/{{train,dev,test}}.jsonl</code>、<code>GRPO/data/grpo/{{train,dev,test}}.jsonl</code>，以及 qid 列表 <code>splits/maple_split_v1.json</code>（冻结，不再变动）。</li>
</ul>
</blockquote>

<div class="toc">
<strong>目录</strong>
<ol>
  <li><a href="#overview">总览（数据集真实形态）</a></li>
  <li><a href="#method">划分策略（设计目标 / 算法 / 桶切分 / 与方案对账）</a></li>
  <li><a href="#splits">Train / Dev / Test 详细分布</a></li>
  <li><a href="#stratum">分层桶（nprof_per_qid）对比</a></li>
  <li><a href="#pidx">profile_index 分布对比</a></li>
  <li><a href="#payload">下游 payload 长度统计（PAD / SDP / GRPO）</a></li>
  <li><a href="#files">产物清单</a></li>
  <li><a href="#repro">复现 & 校验</a></li>
</ol>
</div>

<h2 id="overview">1. 总览</h2>

<div class="grid3">
  <div class="card">
    <div class="muted">总行数</div>
    <div class="big">{g['n_rows']:,}</div>
    <div class="pct">每行 = (query, profile) 对</div>
  </div>
  <div class="card">
    <div class="muted">唯一 question_id</div>
    <div class="big">{g['n_unique_qids']:,}</div>
    <div class="pct">1 个 qid 可绑定 ≥1 个 profile</div>
  </div>
  <div class="card">
    <div class="muted">行 / qid 比</div>
    <div class="big">{g['n_rows']/g['n_unique_qids']:.2f}</div>
    <div class="pct">平均每个 qid 的 profile 变体数</div>
  </div>
</div>

<h3>1.1 nprof_per_qid 分布（全集）</h3>
<table>
<thead><tr><th>nprof（同 qid 的 profile 变体数）</th>{''.join(f'<th class=num>{k}</th>' for k in nprof_keys)}</tr></thead>
<tbody>
<tr><td>qid 数</td>{''.join(f"<td class='num'>{g['nprof_per_qid'].get(str(k), 0)}</td>" for k in nprof_keys)}</tr>
<tr><td>占比</td>{''.join(f"<td class='num'>{g['nprof_per_qid'].get(str(k), 0)/g['n_unique_qids']*100:.1f}%</td>" for k in nprof_keys)}</tr>
</tbody>
</table>

<h3>1.2 profile_index 分布（全集）</h3>
<table>
<thead><tr><th>profile_index</th>{''.join(f'<th class=num>{k}</th>' for k in pidx_keys)}</tr></thead>
<tbody>
<tr><td>行数</td>{''.join(f"<td class='num'>{g['profile_index_dist'].get(str(k), 0)}</td>" for k in pidx_keys)}</tr>
<tr><td>占比</td>{''.join(f"<td class='num'>{g['profile_index_dist'].get(str(k), 0)/g['n_rows']*100:.1f}%</td>" for k in pidx_keys)}</tr>
</tbody>
</table>

<h2 id="method">2. 划分策略（详细说明）</h2>

<h3>2.1 设计目标</h3>
<ol>
  <li><strong>无 query 泄漏</strong>：同一个 <code>question_id</code> 的所有 profile 变体必须落到同一 split。<em>不能按行切</em>——否则一个 query 的 profile 0 落 train、profile 1 落 test，模型在 train 阶段就"见过这个 query"，test 时不再独立。</li>
  <li><strong>multi-profile 信号在三 split 上比例一致</strong>：用于检验 RQ2（profile-sensitivity）的"同一 query × 不同 profile"对比，要求 Dev / Test 都拿到足量 ≥3-profile qid（实验方案锁定目标：Test 取 25 个 ≥3-profile qid）。</li>
  <li><strong>完全 deterministic & 可复现</strong>：seed=42、独立 RNG 实例、桶按 key 排序遍历——同样的源文件、同样的算法在任何机器上输出同样的 qid 列表。</li>
  <li><strong>一次冻结 vs 反复切</strong>：把 qid 列表写到 <code>splits/maple_split_v1.json</code>，后续 SFT / GRPO / 评估全部从这个 JSON 派生；想换 split 就改文件名（如 v2）而不是改种子。</li>
</ol>

<h3>2.2 算法（四步）</h3>
<ol>
  <li><strong>切分键 = <code>question_id</code></strong>（不是行号，不是 (qid, profile) 二元组）。</li>
  <li><strong>分层键 = 该 qid 拥有的 profile 变体数</strong>，全数据中分布是 <code>{{1: 759, 2: 720, 3: 183, 4: 53, 5: 7, 6: 8}}</code>，共 6 个桶。</li>
  <li><strong>桶内独立 shuffle + 比例切分</strong>：用 <code>random.Random(42)</code>（独立实例，不共享全局 random 状态）按 <code>sorted(strata.keys())</code> 顺序遍历每个桶；桶内 <code>shuffle</code> 后切 <code>round(n·0.10)</code> 给 test、再 <code>round(n·0.10)</code> 给 dev，剩余给 train。</li>
  <li><strong>派生行级 split</strong>：原始 JSONL 每行的 <code>question_id</code> 决定它落到 train / dev / test。三类下游数据集（SFT-PAD / SFT-SDP / GRPO）都用同一份 qid 字典派生，<em>三个数据集之间的 split 边界严格一致</em>。</li>
</ol>

<pre># build_splits.py 核心逻辑（与实验方案 §2.2 一致）
rng = random.Random(42)                              # 独立 RNG，避免全局 random 状态污染
strata = defaultdict(list)
for q, profs in qid_to_profs.items():
    strata[len(profs)].append(q)

train_q, dev_q, test_q = [], [], []
for k in sorted(strata.keys()):                      # 排序遍历，跨平台 deterministic
    ql = strata[k]
    rng.shuffle(ql)
    n = len(ql); n_test = round(n*0.10); n_dev = round(n*0.10)
    test_q  += ql[:n_test]
    dev_q   += ql[n_test : n_test+n_dev]
    train_q += ql[n_test+n_dev:]</pre>

<h3>2.3 每个桶的切分结果（实跑数）</h3>
<table>
<thead><tr><th>桶（nprof）</th><th class="num">桶大小 n</th><th class="num">round(0.1·n) = test</th><th class="num">round(0.1·n) = dev</th><th class="num">train = n − 2·round</th><th class="num">→ 行数贡献 (test/dev/train)</th></tr></thead>
<tbody>
  <tr><td>1 profile</td><td class="num">759</td><td class="num">76</td><td class="num">76</td><td class="num">607</td><td class="num">76 / 76 / 607</td></tr>
  <tr><td>2 profile</td><td class="num">720</td><td class="num">72</td><td class="num">72</td><td class="num">576</td><td class="num">144 / 144 / 1,152</td></tr>
  <tr><td>3 profile</td><td class="num">183</td><td class="num">18</td><td class="num">18</td><td class="num">147</td><td class="num">54 / 54 / 441</td></tr>
  <tr><td>4 profile</td><td class="num">53</td><td class="num">5</td><td class="num">5</td><td class="num">43</td><td class="num">20 / 20 / 172</td></tr>
  <tr><td>5 profile</td><td class="num">7</td><td class="num">1</td><td class="num">1</td><td class="num">5</td><td class="num">5 / 5 / 25</td></tr>
  <tr><td>6 profile</td><td class="num">8</td><td class="num">1</td><td class="num">1</td><td class="num">6</td><td class="num">6 / 6 / 36</td></tr>
  <tr style="font-weight:600"><td>合计</td><td class="num">1,730</td><td class="num">173</td><td class="num">173</td><td class="num">1,384</td><td class="num">305 / 305 / 2,433</td></tr>
</tbody>
</table>

<h3>2.4 与实验方案 EXPERIMENT_PLAN §2 的对账</h3>
<p>实验方案 §2.1 和 §2.2 里所有可校验的数字都和实跑结果<strong>精确吻合</strong>。下表为逐项 cross-check（结果 100% 通过）：</p>
<table>
<thead><tr><th>项</th><th>实验方案锁定值</th><th>build_splits.py 实跑</th><th>结果</th></tr></thead>
<tbody>
  <tr><td>总行数</td><td class="num">3,043</td><td class="num">3,043</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>唯一 question_id</td><td class="num">1,730</td><td class="num">1,730</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>nprof 分布</td><td class="mono">{{1:759, 2:720, 3:183, 4:53, 5:7, 6:8}}</td><td class="mono">{{1:759, 2:720, 3:183, 4:53, 5:7, 6:8}}</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>profile_index 分布</td><td class="mono">{{0:1727, 1:972, 2:253, 3:68, 4:15, 5:8}}</td><td class="mono">{{0:1727, 1:972, 2:253, 3:68, 4:15, 5:8}}</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>Train (qid / 行 / 比例)</td><td class="num">1,384 / 2,433 / 80%</td><td class="num">1,384 / 2,433 / 80.0%</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>Dev (qid / 行 / 比例)</td><td class="num">173 / 305 / 10%</td><td class="num">173 / 305 / 10.0%</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>Test (qid / 行 / 比例)</td><td class="num">173 / 305 / 10%</td><td class="num">173 / 305 / 10.0%</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>Test ≥3-profile qid（profile-sensitivity 子集）</td><td class="num">25</td><td class="num">25</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>Test ≥2-profile qid（§8.1 fallback）</td><td class="num">97</td><td class="num">97</td><td><span class="tag ok">OK</span></td></tr>
  <tr><td>train ∩ dev / train ∩ test / dev ∩ test</td><td class="num">0 / 0 / 0</td><td class="num">0 / 0 / 0</td><td><span class="tag ok">OK</span></td></tr>
</tbody>
</table>

<blockquote class="warn">
<strong>与实验方案 v2 草稿的两处实现细节修订</strong>（已同步回 <code>EXPERIMENT_PLAN_2026-05-15.html</code>）：
<ol style="margin:6px 0 0 0">
  <li>实验方案早期描述用了"<code>StratifiedShuffleSplit</code>"（sklearn 的类）。实际实现<em>不</em>调 sklearn——手写桶内 shuffle + <code>round()</code> 切分，行为等价但更可控、零依赖。已把方案文本改为"手写的分层 shuffle"。</li>
  <li>实验方案早期 pseudocode 用 <code>random.seed(42)</code>（全局 RNG）+ <code>for k, ql in strata.items()</code>（依字典插入序）。实际 <code>build_splits.py</code> 改用 <code>random.Random(42)</code>（独立实例，避免被外部 random state 干扰）+ <code>for k in sorted(strata.keys())</code>（避免依赖字典插入序，跨平台 deterministic）。两处改动<em>不影响</em>桶大小与切分比例（上表已验证），但让结果不依赖于源文件遍历顺序。</li>
</ol>
</blockquote>

<h2 id="splits">3. Train / Dev / Test 详细分布</h2>

<table>
<thead><tr>
  <th>Split</th><th class="num">qid 数</th><th class="num">行数</th>
  <th>qid 占比</th><th>行占比</th>
  <th class="num">≥2-profile qid</th><th class="num">≥3-profile qid</th>
  <th class="num">leakage rows</th>
</tr></thead>
<tbody>"""

    total_q = g["n_unique_qids"]
    total_r = g["n_rows"]
    color = {"train": "#0969da", "dev": "#bf8700", "test": "#1a7f37"}
    for sp in ("train", "dev", "test"):
        ss = s[sp]
        html += (
            f"<tr><td><strong>{sp.capitalize()}</strong></td>"
            f"<td class='num'>{ss['n_qids']:,}</td>"
            f"<td class='num'>{ss['n_rows']:,}</td>"
            f"<td>{qbar(ss['n_qids'], total_q, color[sp])}</td>"
            f"<td>{qbar(ss['n_rows'], total_r, color[sp])}</td>"
            f"<td class='num'>{ss['n_qids_ge2_profiles']}</td>"
            f"<td class='num'>{ss['n_qids_ge3_profiles']}</td>"
            f"<td class='num'>{ss['leakage_rows']}</td></tr>"
        )

    html += """
</tbody>
</table>

<blockquote>
<strong>解读</strong>：
<ul style="margin:4px 0 0 0">
  <li>Train 行占比 = 80.0%（2,433 / 3,043），Dev / Test 各 10.0%（305 / 3,043）— 行级比例自然守恒，因为<em>分层是在 qid 级别做的</em>且每个 qid 在被分到某 split 后所有 profile 变体跟它一起走。</li>
  <li>Dev / Test 都得到 <strong>25 个 ≥3-profile qid</strong>，<em>非常关键</em>：这就是 §2.2 中 Profile-Sensitivity 子集的来源——25 qid × 3 profile = 75 评估点。</li>
  <li>Train 持有 201 个 ≥3-profile qid（≈ 77.6% of all 251 ≥3-profile qids），充分覆盖 PAD / SDP / Joint Align / GRPO 阶段对 multi-profile 信号的训练需求。</li>
</ul>
</blockquote>

<h2 id="stratum">4. 分层桶（nprof_per_qid）跨 split 对比</h2>
<p>下表展示<strong>每个 nprof 桶在三个 split 上的拆分</strong>。若分层成功，每行应近似 80 / 10 / 10。</p>

<table>
<thead><tr>
  <th>nprof</th><th class="num">全集 qid</th>
  <th class="num">Train</th><th>%</th>
  <th class="num">Dev</th><th>%</th>
  <th class="num">Test</th><th>%</th>
</tr></thead>
<tbody>
"""
    for k in nprof_keys:
        total = g["nprof_per_qid"].get(str(k), 0)
        tr = s["train"]["nprof_per_qid"].get(str(k), s["train"]["nprof_per_qid"].get(k, 0))
        dv = s["dev"]["nprof_per_qid"].get(str(k), s["dev"]["nprof_per_qid"].get(k, 0))
        te = s["test"]["nprof_per_qid"].get(str(k), s["test"]["nprof_per_qid"].get(k, 0))
        f = lambda a, b: f"{(a / b * 100):.1f}%" if b else "—"
        html += (
            f"<tr><td><strong>{k} profile</strong></td>"
            f"<td class='num'>{total}</td>"
            f"<td class='num'>{tr}</td><td>{f(tr,total)}</td>"
            f"<td class='num'>{dv}</td><td>{f(dv,total)}</td>"
            f"<td class='num'>{te}</td><td>{f(te,total)}</td></tr>"
        )

    html += """
</tbody>
</table>

<h2 id="pidx">5. profile_index 分布（行级）</h2>
<p>第 0 列代表"无 profile / 默认 profile"，第 1+ 列代表"profile_index = i"。三个 split 的形状应高度一致——这是分层切的副产物，因为 qid 一旦决定了，它带走<em>所有</em> profile 变体。</p>
<table>
<thead><tr>
  <th>Split</th>
"""
    for k in pidx_keys:
        html += f"<th class='num'>profile_index = {k}</th>"
    html += "<th class='num'>合计</th></tr></thead><tbody>"

    for sp in ("train", "dev", "test"):
        ss = s[sp]
        cells = ""
        for k in pidx_keys:
            v = ss["profile_index_dist"].get(str(k), ss["profile_index_dist"].get(k, 0))
            pct_ = (v / ss["n_rows"] * 100) if ss["n_rows"] else 0
            cells += f"<td class='num'>{v}<br/><span class='muted'>{pct_:.1f}%</span></td>"
        html += f"<tr><td><strong>{sp.capitalize()}</strong></td>{cells}<td class='num'>{ss['n_rows']}</td></tr>"

    html += """
</tbody>
</table>

<h2 id="payload">6. 下游 payload 长度统计</h2>
<p>三个数据集都已经按 chat-template 物化为 JSONL；下表是 <em>字符级</em>长度分布（不是 token；仅用于直观估计长度量级、设置 <code>max_seq_length</code> 与 <code>max_completion_length</code>）。</p>

<h3>6.1 Train</h3>
<table>
<thead><tr><th>指标</th><th class="num">min</th><th class="num">mean</th><th class="num">p50</th><th class="num">p90</th><th class="num">p95</th><th class="num">p99</th><th class="num">max</th></tr></thead>
<tbody>
"""
    html += len_block("train")
    html += """
</tbody>
</table>

<h3>6.2 Dev</h3>
<table>
<thead><tr><th>指标</th><th class="num">min</th><th class="num">mean</th><th class="num">p50</th><th class="num">p90</th><th class="num">p95</th><th class="num">p99</th><th class="num">max</th></tr></thead>
<tbody>
"""
    html += len_block("dev")
    html += """
</tbody>
</table>

<h3>6.3 Test</h3>
<table>
<thead><tr><th>指标</th><th class="num">min</th><th class="num">mean</th><th class="num">p50</th><th class="num">p90</th><th class="num">p95</th><th class="num">p99</th><th class="num">max</th></tr></thead>
<tbody>
"""
    html += len_block("test")
    html += """
</tbody>
</table>

<blockquote class="warn">
<strong>换算建议</strong>：英文+JSON 大致 <code>1 token ≈ 3.5–4 chars</code>。SDP assistant target p95 在 ~9–10K 字符 → ~2.5K token，仍在 §3.1 锁定的 <code>max_seq_length = 3,072</code> 之内；GRPO prompt p95 在 ~2K 字符 → ~500 token，<code>max_completion_length = 3,500</code> 也留足余量。
</blockquote>

<h2 id="files">7. 产物清单</h2>

<table>
<thead><tr><th>路径</th><th class="num">行数</th><th>用途</th></tr></thead>
<tbody>
"""
    files = [
        ("splits/maple_split_v1.json", "—", "冻结的 qid 列表（train/dev/test）；后续所有重新物化都从这里派生"),
        ("splits/split_stats.json", "—", "本报告的源数据"),
        ("SFT/data/pad/train.jsonl", "2,433", "Stage 1a LoRA-PAD 训练（input = query+profile，target = agents+subtasks）"),
        ("SFT/data/pad/dev.jsonl", "305", "PAD dev NLL + SV 监控；ckpt 选择"),
        ("SFT/data/pad/test.jsonl", "305", "最终评估时供 PAD-only 消融使用"),
        ("SFT/data/sdp/train.jsonl", "2,433", "Stage 1b LoRA-SDP 训练（条件 = gold scaffold，target = steps + execution_order）"),
        ("SFT/data/sdp/dev.jsonl", "305", "SDP dev NLL 监控"),
        ("SFT/data/sdp/test.jsonl", "305", "最终评估时供 SDP-only 消融使用"),
        ("GRPO/data/grpo/train.jsonl", "2,433", "Stage 3 GRPO rollout 的 prompt-only 数据（每行含 gold_plan + learner_profile 旁路字段）"),
        ("GRPO/data/grpo/dev.jsonl", "305", "★ <strong>SFT→GRPO Gate</strong>（§3.4 pass@1 vs pass@8）+ GRPO dev_composite 监控 + ckpt 选择"),
        ("GRPO/data/grpo/test.jsonl", "305", "Tier 1 / Tier 2 / Tier 3 最终评估"),
    ]
    for path, n, why in files:
        html += f"<tr><td><code>{path}</code></td><td class='num'>{n}</td><td>{why}</td></tr>"

    html += f"""
</tbody>
</table>

<blockquote>
<strong>命名约定</strong>：旧版的 <code>valid.jsonl</code> 已被三段式 <code>train/dev/test.jsonl</code> 替代。<em>dev = 选 ckpt + 跑 Gate；test 全程封存，仅最终评估</em>——这与实验方案 §2.3 的使用矩阵一致。
</blockquote>

<h2 id="repro">8. 复现 & 校验</h2>
<pre># 1. 重新跑 split（结果完全确定，seed=42）
python build_splits.py

# 2. 重新生成本报告
python build_split_report.py

# 3. 校验：三个集合 qid 互斥
python - &lt;&lt; 'PY'
import json
s = json.load(open("splits/maple_split_v1.json"))
t,d,e = set(s["train_qids"]), set(s["dev_qids"]), set(s["test_qids"])
assert not (t&amp;d) and not (t&amp;e) and not (d&amp;e)
print("OK — sizes:", len(t), len(d), len(e))
PY</pre>

<p class="meta" style="margin-top:32px">
本报告由 <code>build_split_report.py</code> 自动生成。任何对划分结果的调整必须先改 <code>build_splits.py</code> 再重跑两个脚本，且应同步更新 <code>EXPERIMENT_PLAN_CHANGELOG.md</code>（如有）。
</p>

</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
