# Tier 2 Execution Evaluator — Self-Contained Package

本包含两个阶段的完整代码：**Stage 3 执行**（运行多智能体教学 plan）和 **Tier 2 评估**（对执行结果打分）。

---

## 两阶段分工

```
Stage 3 执行                          Tier 2 评估
─────────────────────                 ──────────────────────
batch_eval.py                         tier2_evaluator_v2.py
batch_eval_parallel.py      ──→       plan_mapper_fixed/evaluator.py
plan_mapper_fixed/runtime.py          (reads runs/ produced by Stage 3)
Dockerfile.mas-runtime
entrypoint.sh
         ↓
     runs/<run-id>/
       execution_log.json
       step_outputs.json
```

- **Stage 3**：用 CrewAI 驱动 Teacher（GPT-4o）+ Student（GPT-4o-mini）对每条 plan 进行 Socratic 对话执行，输出 `runs/`
- **Tier 2**：读取 `runs/` 中的执行日志，从 EVR / PAS / PQS / r_sol 四个维度打分

---

## 目录结构

```
tier2_eval_package/
├── Dockerfile.mas-runtime            # 代码沙盒镜像（Python/Node/Java/Go/Rust/DB 等）
├── entrypoint.sh                     # Docker 启动脚本（启 PostgreSQL + exec）
├── batch_eval.py                     # Stage 3 串行批量执行
├── batch_eval_parallel.py            # Stage 3 并行批量执行
├── tier2_evaluator_v2.py             # Tier 2 评估主脚本
├── plan_mapper_fixed/
│   ├── evaluator.py                  # v1 四检 evaluator（被 tier2 主脚本 import）
│   └── runtime.py                    # Stage 3 执行 runtime（CrewAI + Docker）
├── runs/                             # Stage 3 执行结果（预置 100 条）
│   └── run-<qid>-p<N>-<hash>/
│       ├── execution_log.json
│       └── step_outputs.json
├── batch_summary.json                # run 列表索引（Stage 3 输出 / Tier 2 读取）
├── multi_agent_dataset_filtered_qap_v3.jsonl  # plan + learner profile 数据集
├── qap_task1.jsonl                   # accepted_answer 来源（r_sol 用）
├── qap_task2.jsonl
└── qap_task3.jsonl
```

---

## 环境配置

### Step 1 — Python 依赖

```bash
# Tier 2 评估（最小依赖）
pip install openai python-dotenv

# Stage 3 执行（需要 CrewAI）
pip install crewai crewai-tools openai python-dotenv
```

### Step 2 — Docker 镜像（Stage 3 代码沙盒）

```bash
cd tier2_eval_package
docker build -f Dockerfile.mas-runtime -t mas-runtime .

# 验证
docker run --rm mas-runtime python3 -c "print('ok')"
```

### Step 3 — API Keys

```bash
export OPENAI_API_KEY=sk-...          # 必填，Teacher/Student agent + Tier 2 评估 judge
export OPENAI_BASE_URL=https://...    # 可选，自定义兼容端点（如 bianxie.ai）
export FIRECRAWL_API_KEY=...          # 可选，FirecrawlSearchTool；缺失则该工具跳过
export SERPER_API_KEY=...             # 可选，SerperDevTool；缺失则该工具跳过
```

> **注意**：缺少 `FIRECRAWL_API_KEY` 会导致依赖 `FirecrawlSearchTool` 的 plan step 执行失败（`succeeded=False`），这是当前 baselines 执行成功率偏低的主要原因。

---

## 执行说明

### 模式 A：仅 Tier 2 评估（使用包内预置 100 条 runs）

```bash
cd tier2_eval_package

# 全量评估（100 条）
python3 tier2_evaluator_v2.py --stage3 --out results.json

# 小批量测试
python3 tier2_evaluator_v2.py --stage3 --limit 10 --out pilot.json
```

### 模式 B：完整流水线（重新执行 plans + 评估）

**Step 1：Stage 3 执行**

```bash
cd tier2_eval_package

# 串行执行 100 条（默认随机采样）
python3 batch_eval.py --n 100 --runs-dir runs/

# 并行执行（推荐，4 进程）
python3 batch_eval_parallel.py --n 100 --runs-dir runs/ --workers 4

# 指定模型
python3 batch_eval.py --n 100 --model gpt-4o --student-model gpt-4o-mini --runs-dir runs/
```

执行完成后 `runs/` 和 `batch_summary.json` 自动更新。

**Step 2：Tier 2 评估**

```bash
python3 tier2_evaluator_v2.py --stage3 --out results.json
```

---

## 输出说明

| 文件 | 内容 |
|------|------|
| `results.json` | 每条 run 的四指标评分 + 汇总 aggregate |
| `timeout_log.jsonl` | API timeout / Docker env-fail 事件记录 |

## 评估指标

| 指标 | 含义 |
|------|------|
| EVR | Execution Validity Rate：cov / loop / flow / exec 四项 AND |
| PAS | Personalization Alignment Score：per-utterance PRR 均值 |
| PQS | Pedagogical Quality Score：(NDAR + SPR + IAR) / 3 |
| r_sol | Post-Dialog Solve Rate：学生最终对话是否展示正确理解 |
