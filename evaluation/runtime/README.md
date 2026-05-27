# Stage 3 Execution

MAS Stage 3 执行层：Teacher（GPT-4o）+ Student（GPT-4o-mini）Socratic 教学对话，基于 CrewAI 框架。

---

## 代码

| 文件 | 用途 |
|------|------|
| `run_single_plan.py` | 单条 plan 执行入口，被 batch 脚本以 subprocess 调用 |
| `batch_eval.py` | 批量执行：随机抽样 N 条，支持断点续跑，结果写入 `batch_summary.json` |
| `rerun_targeted.py` | 针对指定 qid 列表重跑（用于修复后补跑） |
| `plan_mapper_fixed/` | 执行引擎：`compiler.py`（静态检查）、`runtime.py`（CrewAI agent 绑定 + 多语言代码执行）、`evaluator.py`（四项质量检查，独立 LLM 调用） |

**`runtime.py` 支持语言**：Python、C、C++、Java、JavaScript、TypeScript、Swift、Ruby、R；C# 因本机未装 .NET 返回提示。

---

## 数据集

| 文件 | 说明 |
|------|------|
| `multi_agent_dataset_filtered_qap_v3.jsonl` | v3 数据集，50 条，agent goal 和 subtask 按学习者 profile 个性化（相比旧版有实质改进） |

旧版数据集（`multi_agent_dataset_filtered_qap_latest.jsonl`）在仓库根目录。

---

## 运行结果

| 目录 | 数据集 | 条数 | 执行时间 | 说明 |
|------|--------|------|----------|------|
| `runs/` | latest（旧版） | 100 | 2026-05-09 ~ 05-11 | 初始批量验证，100 条随机抽样 |
| `runs_fixedlang/` | latest（旧版） | 17 | 2026-05-11 | 针对代码执行失败的 17 条补跑（新增 Swift/TypeScript/Ruby 支持、修复编译报错措辞后重跑） |
| `runs_v3/` | v3 | 50 | 2026-05-12 | v3 数据集全量执行，含 2 条超时重跑和 6 条 C#/Ruby 问题重跑 |

每个 run 子目录包含：`events.log`（执行事件流）、`execution_log.json`、`step_outputs.json`、`result_readable.md`（可读摘要）。

---

## 质量报告

`100条数据抽样验证.md` — 针对 `runs/` 的详细审计报告，包含执行样例、代码执行失败排查记录、prompt/数据层问题说明。
