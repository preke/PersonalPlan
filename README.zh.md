# MAPLE：多智能体个性化教学规划数据集与训练流水线

## 📌 项目概览

**MAPLE** 是一套面向**多智能体个性化教学规划**场景的数据集与完整训练/评估流水线。仓库内容覆盖：

1. **数据集构造**：以 StackOverflow duplicate-question 簇 + 提问者画像为来源，由 Claude Sonnet 4.6 生成多智能体教学规划方案，经"结构静态检查 + 执行有效性校验"两阶段验证后入选；
2. **数据划分**：在 `question_id` 粒度按 profile 数分层做 80/10/10 划分，避免同问题不同 profile 跨 split 泄漏；
3. **训练**：实现论文 §4 的 hierarchical SFT（PAD / SDP / Joint Alignment）以及 §4.2 的 GRPO（结构+个性化+教学+硬约束 4 类 reward）；
4. **执行与评估**：基于 CrewAI 的 Teacher（GPT-4o）+ Student（GPT-4o-mini）Socratic 执行层，并配 Feasibility / Personalization / Satisfaction 三维度评估；
5. **Baseline 矩阵**：5 层 baseline（闭源 LLM 直跑、多智能体框架 MAS、教育领域 MAS 等）。

数据集最新统计（v15 release）：**3,043 条** plan、**1,730 个** unique canonical questions，其中 **971 个** question 拥有 ≥ 2 个 learner profile，天然形成跨 profile 的个性化研究轴。详细分析见 [personalized_planning_dataset_report.html](./personalized_planning_dataset_report.html)。

---

## 🗂️ 问题分类

数据集按问题意图分为 **7 类**，支持多标签（同一问题可同时归属多类）：

| 类别 | 说明 | 问题数 | 占比 |
|------|------|-------:|-----:|
| `CONCEPTUAL` | 概念理解类：询问原理、机制或设计决策 | 1,029 | 59.34% |
| `API_USAGE` | API / 库的用法类 | 648 | 37.37% |
| `REVIEW` | 代码审查、最佳实践建议类 | 273 | 15.74% |
| `DISCREPANCY` | 实际行为与预期不符的差异分析类 | 227 | 13.09% |
| `ERRORS` | 报错与异常调试类 | 221 | 12.75% |
| `LEARNING` | 学习路径与资源推荐类 | 92 | 5.31% |
| `API_CHANGE` | 因版本更迭引发的 API 变更问题类 | 38 | 2.19% |

> **多意图率：44.75%**，即接近半数问题同时属于两个或以上类别。最常见的共现组合为 `API_USAGE + CONCEPTUAL`（190 题，10.96%）。

---

## 📂 数据结构

每条数据采用 **JSON 格式**，由 `input` 和 `output` 两部分组成。

### Input — 用户需求与画像

| 字段 | 说明 |
|------|------|
| `query` | 任务的自然语言描述（通常为 StackOverflow 问题 title + body 拼接） |
| `learner.about_me` | 用户的背景介绍，包括学习经历与技术栈 |
| `learner.top_tags` | 用户熟悉的技能标签列表（如 `python`、`react` 等） |

### Output — 多智能体规划方案

#### (a) `agents` — 智能体定义

| 字段 | 说明 |
|------|------|
| `agent_role` | 角色名称（按"语言 + 职能"自由命名，如 `cpp_code_validator` / `csharp_docs_retriever`） |
| `goal` | 该智能体的任务目标 |
| `backstory` | 背景说明与能力介绍（CrewAI 原生字段名） |
| `tools` | 可调用的工具列表，从下述 8 工具池中选取 |

**工具池（8 个）**：`CodeInterpreterTool`、`CodeDocsSearchTool`、`FirecrawlSearchTool`、`FileWriterTool`、`ArxivPaperTool`、`RagTool`、`DirectoryReadTool`、`FileReadTool`。其中前 5 个在 v15 release 中被实际选用，后 3 个虽然在工具池声明但模型从未调用 —— 反映 StackOverflow 编程问题的两大主导需求是"跑代码"和"查文档"。

#### (b) `subtasks` — 任务分解

| 字段 | 说明 |
|------|------|
| `id` | 阶段编号（如 `"S2"`） |
| `name` | 阶段名称 |
| `subtask_objective` | 该阶段的总体目标 |
| `steps[].id` | 步骤编号（如 `"S1-2"`） |
| `steps[].agent` | 负责执行的智能体 |
| `steps[].objective` | 步骤目标 |
| `steps[].instruction` | 执行指令 |
| `steps[].depends_on` | 前置依赖步骤（数组） |
| `steps[].requires_human_input` | 是否需要 learner 输入 |
| `steps[].expected_output` | 预期输出描述 |

#### (c) `execution_order` — 执行顺序

满足所有依赖约束后的全局步骤序列，支持条件循环（`loop`）结构。

---

## 📑 数据样例

详细样例请参见 [`plan_examples/`](./plan_examples/)（8 条覆盖不同意图与结构的代表性 plan）。完整结构示例：

```json
{
  "input": {
    "query": "Least Astonishment and the Mutable Default Argument ...",
    "learner": {
      "about_me": "Frontend developer at a startup, 3 years of JavaScript/TypeScript experience. Recently started learning Python for backend work with FastAPI.",
      "top_tags": ["javascript", "typescript", "react", "node.js", "fastapi"]
    }
  },
  "output": {
    "agents": [
      {
        "agent_role": "python_behavior_tutor",
        "goal": "Guide the learner to understand Python's mutable default argument behavior ...",
        "backstory": "Handles learner-facing diagnosis, explanation, ...",
        "tools": []
      }
    ],
    "subtasks": [
      {
        "id": "S1",
        "name": "Diagnose the learner's mental model ...",
        "subtask_objective": "...",
        "steps": [
          {
            "id": "S1-1",
            "agent": "python_behavior_tutor",
            "objective": "Surface the learner's current assumption ...",
            "instruction": "Ask the learner to compare JavaScript and Python behavior ...",
            "tool": null,
            "requires_human_input": true,
            "expected_output": "learner prediction ...",
            "depends_on": []
          }
        ]
      }
    ],
    "execution_order": [
      "S1-1", "S1-2", "S2-1", "S2-2",
      { "loop": { "steps": ["S2-3", "S2-4", "S2-5"],
                  "condition": "S2-4.explanation_accurate == false",
                  "max_iterations": 3 } },
      "S2-6", "..."
    ]
  }
}
```

---

## 🔄 数据构造流程

完整流程文档见 [`personalized_planning_dataset_report.html`](./personalized_planning_dataset_report.html)（v15 release）；代码实现见 [`the_construction_of_MAPLE_datasets/`](./the_construction_of_MAPLE_datasets/)；总览图 [`workflow_of_dataset_construction.png`](./workflow_of_dataset_construction.png)。

整个 pipeline 由 4 个阶段组成，构成"生成 → 验证 → 不合格回退重写"的迭代闭环。

```
                StackOverflow 原始抓取（带 original_questions 链接）
                                │
                                ▼
  ① 数据收集 & 聚类             Crawler 按 duplicate-original 关系把每个 canonical
                                question 与它的所有 duplicates 聚成一个 cluster；
                                cluster 作为基本单位（而非单条 question），
                                因为同一问题的不同 duplicate 由不同背景的用户提问，
                                天然形成 (question, profile) 多对配对。
                                每条样本保留三个字段：question / profile / answer。
                                │
                                ▼
  ② 质量过滤（LLM filter）       Question 质量：accepted answer 是否真正解决问题；
                                              是否需要 stepwise reasoning；
                                              是否多解可辩；
                                              是否可拆为 concept → mistake → demo → practice → transfer。
                                Profile 质量：self_description 是否含具体信号
                                              （职位/技术栈/教育背景/兴趣）。
                                → 1,733 个 question groups 通过，3,062 个 (question, profile) 进入下一阶段
                                │
                                ▼
  ③ Plan 生成                  Generator：Claude Sonnet 4.6
                                输入：question + accepted answer + learner profile + 8 工具池描述
                                输出：单一 JSON 对象 { agents, subtasks, execution_order }
                                Prompt 文件：plan_generation_prompt.txt（数据生成用）
                                            plan_generation_prompt_inference.txt（推理用，不依赖 answer）
                                │
                                ▼
  ④ 两阶段验证                  Stage A — 静态结构检查
                                  Deterministic Python checker，~30 条断言：
                                  JSON 合法 / agent 引用存在 / tool 在池内 /
                                  depends_on 可解且无环 / execution_order 全覆盖 /
                                  agent 实际使用工具与声明一致 / loop 终止条件良好。
                                  失败按 severe/major/minor 分级，映射到 Excellent/Good/Acceptable/Failing。
                                Stage B — 执行有效性校验
                                  实际跑一遍 plan：每 step 派发给对应 agent，
                                  调用声明的工具，收集输出与中间状态；
                                  执行轨迹交给 LLM judge，judge 看 (question, profile, answer, plan, trace)，
                                  评 5 个维度：是否推进解题 / tool 调用合理 / 步骤逻辑连贯 /
                                  内容与 answer 对齐 / 个性化是否真正反映 profile。

                                Stage A ≥ Good 且 Stage B 通过 → 入选；
                                否则回退到 ③ 重新生成，再次进入 Stage A/B。
                                → 3,043 条 plan 通过双重验证，构成 v15 release
```

对应评估 prompt：
- `Plan_Evaluation_Criteria.txt` — 静态结构 + 风格审查
- `Execution_Evaluation_Prompt.txt` — 动态执行评分
- `stage3_execution/{feasibility,personalization,satisfaction}_score_prompt.txt` — 下游三维度评估

---

## 🧪 数据划分（80/10/10）

`build_splits.py` 在 **`question_id`** 粒度按"同 qid 的 profile 数量"分层做 80/10/10 划分（seed=42），保证同 question 不同 profile 的 row 不跨 split。详见 [DATA_SPLIT_REPORT_2026-05-15.html](./DATA_SPLIT_REPORT_2026-05-15.html)。

| Split | qids | rows |
|------|----:|----:|
| train | 1,384 | 2,433 |
| dev   | ~173 | ~305 |
| test  | ~173 | ~305 |

切分清单写到 [`splits/maple_split_v1.json`](./splits/maple_split_v1.json)，并同步生成 SFT (PAD/SDP) 与 GRPO 三套 `{train,dev,test}.jsonl` row 文件。

---

## 🎯 训练流水线

### SFT — 三阶段微调（[`SFT/`](./SFT/)）

实现论文 §4 的 hierarchical SFT + Joint Alignment，使用 HuggingFace AutoTrain（`llm-sft` + `llm-dpo`）+ LoRA 作为后端。

```
  Stage 1 (PAD SFT)            query ⊕ profile          → (T*, A*) 个性化 agents/subtask 元数据
  Stage 2 (SDP SFT)            query ⊕ profile ⊕ gold   → (S*, O*) steps + execution_order
  Stage 3a (On-policy SDP)     PAD-sampled scaffold     → (S*, O*)
  Stage 3b (PAD DPO)           SDP-perplexity preference → DPO on PAD
```

PAD / SDP 是同一 base 上的两个独立 LoRA adapter；推理时串行调用（PAD → 生成 scaffold → SDP → 生成 steps）。详见 [SFT/README.md](./SFT/README.md)。

### GRPO — 强化优化（[`GRPO/`](./GRPO/)）

实现论文 §4.2 的可验证 + 反事实奖励，使用 **TRL `GRPOTrainer` + LoRA + vLLM**。v1 落地 4 个 reward 组件中的核心子集：

| Reward | 实现位置 | v1 默认 |
|---|---|:--:|
| `R_hard` — schema / cycle / tool gate | `rewards.reward_hard_gate` | ✅ |
| `R_struct` — DAG + DC + ATR + 结构 fingerprint Jaccard | `rewards.reward_structural` | ✅ |
| `R_pers` — counterfactual（离线缓存） | `rewards.reward_personalization` | ⬜ |
| `R_ped,hard` — 子任务名共现先修图 | `rewards.reward_pedagogy_hard` | ⬜ |
| `R_ped,soft` — LLM judge ensemble | `rewards.reward_pedagogy_soft` | ⬜ |

升级 v2/v3/v4 仅需切换 YAML toggle 与提供配套数据，训练脚本不变。详见 [GRPO/README.md](./GRPO/README.md) 与 [GRPO/reward_design_writeup.html](./GRPO/reward_design_writeup.html)。

---

## ⚙️ 执行与评估（[`stage3_execution/`](./stage3_execution/)）

基于 **CrewAI** 的 Teacher（GPT-4o）+ Student（GPT-4o-mini）Socratic 教学对话引擎：

| 文件 | 用途 |
|---|---|
| `run_single_plan.py` | 单条 plan 执行入口 |
| `batch_eval.py` | 批量执行 + 断点续跑，结果汇总到 `batch_summary.json` |
| `rerun_targeted.py` | 针对失败 qid 列表的补跑 |
| `plan_mapper_fixed/` | 编译器（静态检查） + 运行时（CrewAI agent 绑定 + 多语言代码执行） + 评估器 |
| `{feasibility,personalization,satisfaction}_score_prompt.txt` | 三维度自动评估 prompt |

**运行结果目录**：

| 目录 | 数据集 | 条数 | 说明 |
|---|---|---:|---|
| `runs/` | latest（旧版） | 100 | 初始随机抽样 |
| `runs_fixedlang/` | latest | 17 | 新增 Swift/TypeScript/Ruby 支持后补跑 |
| `runs_v3/` | v3 | 50 | v3 数据集（profile 个性化加强版）全量执行 |

支持的代码执行语言：Python、C、C++、Java、JavaScript、TypeScript、Swift、Ruby、R。详见 [stage3_execution/README.md](./stage3_execution/README.md) 与 [stage3_execution/100条数据抽样验证.md](./stage3_execution/100条数据抽样验证.md)。

---

## 📊 数据分析（[`analysis/`](./analysis/)）

`analyze.py` 生成分布图（[`figures/`](./analysis/figures/)）+ `stats.json` 数字汇总。报告见 [analysis/REPORT.md](./analysis/REPORT.md)，覆盖：

- 输入侧：query 长度、profile 文本长度、同问题多 profile 分布
- 输出侧：agent / subtask / step 数分布、human-in-the-loop 比例、loop 控制流统计
- 工具侧：工具池实际使用分布、role 家族 UpSet 图
- 个性化侧：skills × plan 内容 Jaccard、skill 命中率、按 intent 拆分的个性化强度
- 复杂度 × intent 热图、plan 质量分布

---

## 🧱 Baseline 矩阵（5 层）

- 实验方案：[EXPERIMENT_PLAN_2026-05-15.html](./EXPERIMENT_PLAN_2026-05-15.html)
- Baseline 设计原则与 schema 翻译方法论：[baseline_design_v1.html](./baseline_design_v1.html)
- Baseline 实施细节（代码级文档）：[BASELINE_DETAILS_FULL.html](./BASELINE_DETAILS_FULL.html) —— 涵盖共用基础设施（data loader / LLM client / prompt sections / schema validator / JSON repair / schema translator / runner / native logger）、T1 闭源 LLM 裸调用的完整实现、关键 bug 与方法忠实度问题汇总，以及真实输出样例

| 层 | 类别 | 代表 baseline |
|---|---|---|
| T1 | 闭源 LLM 直跑 | GPT-4o、Claude Sonnet、Gemini |
| T2 | 开源 LLM 直跑 | 暂留 |
| T3 | 单 agent + 工具链 | — |
| T4 | 通用多智能体框架 MAS | AutoGen、CAMEL |
| T5 | 教育领域 MAS | 保留论文 topology + 后处理翻译为 MAPLE schema |

---

## 📁 项目结构

```
multi_agent_datasets/
├── README.md                                    ← 本文件
│
├── 📦 数据集（顶层）
│   ├── multi_agent_dataset_filtered_qap.jsonl   ← 主数据集（3,043 条）
│   ├── queries_selected_latest.jsonl            ← 筛选后的 canonical query 清单（1,967 条）
│   ├── plan_examples/                           ← 8 条代表性 plan 样例
│   └── datasets_old/                            ← 早期按类别拆分的归档版本
│
├── 🛠️ Prompt 与评估准则
│   ├── prompt_select_query.txt                  ← Query 筛选 prompt
│   ├── plan_generation_prompt.txt               ← 数据生成 prompt（含 accepted answer 锚）
│   ├── plan_generation_prompt_inference.txt     ← 推理 prompt（仅靠 query 推断 destination）
│   ├── Plan_Evaluation_Criteria.txt             ← 静态评估（结构 + 风格审查）
│   └── Execution_Evaluation_Prompt.txt          ← 动态执行评估
│
├── 🔄 数据构造
│   ├── the_construction_of_MAPLE_datasets/      ← 完整构造代码（task_1 / task_2 / task_3）
│   ├── generated_plans/                         ← Sonnet 在 5 条输入上的复现实验（含 compare.py）
│   └── workflow_of_dataset_construction.png     ← 流程总览图
│
├── 🧪 数据划分
│   ├── build_splits.py                          ← 80/10/10 划分器（qid 粒度分层）
│   ├── build_split_report.py                    ← 划分报告生成器
│   ├── splits/
│   │   ├── maple_split_v1.json                  ← 划分清单（qid lists）
│   │   └── split_stats.json
│   └── DATA_SPLIT_REPORT_2026-05-15.html
│
├── 🎯 训练
│   ├── SFT/                                     ← 三阶段微调（PAD / SDP / Joint Alignment）
│   │   ├── README.md
│   │   ├── prompts.py / build_sft_data.py / joint_alignment_data.py
│   │   ├── train_{pad,sdp,joint}.sh
│   │   ├── configs/{pad,sdp,joint_phase_a,joint_phase_b}_autotrain.yml
│   │   └── multi_agent_dataset_filtered_qap_v3.jsonl  (50 条 smoke set)
│   ├── GRPO/                                    ← TRL GRPOTrainer + 4 类 reward
│   │   ├── README.md / GRPO_conversation.html / reward_design_writeup.html
│   │   ├── plan_utils.py / rewards.py
│   │   ├── build_grpo_prompts.py / build_counterfactual_cache.py
│   │   ├── grpo_train.py / train_grpo.sh
│   │   └── configs/grpo.yaml
│   └── finetuning_code/old_version/             ← 早期 stage1/stage2 微调脚本（已归档）
│
├── ⚙️ 执行与评估
│   └── stage3_execution/
│       ├── README.md / 100条数据抽样验证.md
│       ├── run_single_plan.py / batch_eval.py / rerun_targeted.py
│       ├── plan_mapper_fixed/                   ← compiler / runtime / evaluator
│       ├── Dockerfile.mas-runtime / entrypoint.sh
│       ├── {feasibility,personalization,satisfaction}_score_prompt.txt
│       └── runs/ runs_fixedlang/ runs_v3/        ← 执行结果与可读摘要
│
├── 📊 数据分析与实验设计
│   ├── analysis/                                ← analyze.py + REPORT.md + figures/ + stats.json
│   ├── personalized_planning_dataset_report.html ← 数据集统计可视化报告
│   ├── EXPERIMENT_PLAN_2026-05-15.html          ← 实验方案 v2（4 个 RQ + 5 层 baseline + 8 项指标）
│   ├── baseline_design_v1.html                  ← Baseline 设计 + schema 翻译方法论
│   └── BASELINE_DETAILS_FULL.html               ← Baseline 实施细节（代码级，共用基础设施 + T1 完整实现）
│
├── 🗒️ 工作记录与进展
│   ├── notes/2026-05-08-prompt-rewrite-and-validation.md
│   ├── 5_14/GP/                                 ← 数据生成 / Baseline / Prompt 修改进展
│   ├── 5_14/SHC/                                ← GRPO 训练 + Stage 3 Execution Pipeline
│   ├── 5_14/ben/                                ← MAPLE 训练首轮迭代（LoRA 选型）
│   └── 5_14/lyx/                                ← Feasibility / Personalization / Satisfaction 评估
│
├── 流程图/                                       ← 数据构造与训练流程图
├── Q&A.pdf                                      ← 项目常见问题解答
└── workflow_of_dataset_construction.png
```

---

## 📎 关键文档索引

| 文档 | 主题 |
|---|---|
| [the_construction_of_MAPLE_datasets/README.md](./the_construction_of_MAPLE_datasets/) | 数据构造逐步说明（task_1 QAP → task_2 分类 → task_3 筛选） |
| [analysis/REPORT.md](./analysis/REPORT.md) | 数据集完整统计分析（21 张图） |
| [SFT/README.md](./SFT/README.md) | SFT 三阶段实现与论文 §4 公式对应 |
| [GRPO/README.md](./GRPO/README.md) | GRPO reward 设计、v1→v4 升级路径、论文 §4.2 公式对应 |
| [stage3_execution/README.md](./stage3_execution/README.md) | CrewAI 执行层与评估流程 |
| [EXPERIMENT_PLAN_2026-05-15.html](./EXPERIMENT_PLAN_2026-05-15.html) | 实验方案 v2（4 RQ + baseline 矩阵 + 6 周时间表） |
| [baseline_design_v1.html](./baseline_design_v1.html) | 5 层 baseline 设计与公平比较方法论 |
| [BASELINE_DETAILS_FULL.html](./BASELINE_DETAILS_FULL.html) | Baseline 实施细节（代码级，含共用基础设施 + T1 完整实现 + 关键 bug 汇总） |
| [DATA_SPLIT_REPORT_2026-05-15.html](./DATA_SPLIT_REPORT_2026-05-15.html) | Train/Dev/Test 划分分布报告 |
| [personalized_planning_dataset_report.html](./personalized_planning_dataset_report.html) | 数据集统计可视化报告 |
| [notes/2026-05-08-prompt-rewrite-and-validation.md](./notes/2026-05-08-prompt-rewrite-and-validation.md) | Plan 生成 prompt 重写与 Sonnet 复现验证 |
| [5_14/GP/PROGRESS_REPORT_2026-05-14.md](./5_14/GP/PROGRESS_REPORT_2026-05-14.md) | 数据生成 / Baseline / Prompt 三条线进展 |
| [Q&A.pdf](./Q&A.pdf) | 项目常见问题解答 |
