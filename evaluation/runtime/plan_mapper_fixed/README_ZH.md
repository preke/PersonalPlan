# CrewAI Plan Mapper（中文版）

将结构化的 Plan JSON 映射为可执行的多智能体工作流。

本项目支持两种执行引擎：

- `flow`（默认）：基于 CrewAI Flow（`@start/@listen/@router`）
- `runtime`：内置编排器（提供 compile/conformance 报告）

支持能力：线性步骤、循环块（loop）、可选交互步骤（`requires_human_input`）、工具调用步骤。

## 一、你需要准备什么

- Python 3.10+
- 虚拟环境（推荐）
- 可用的模型 API 凭证（live 模式）
- 合法的 plan JSON 文件

## 二、环境配置

```bash
cd /home/gp/crewai-plan-mapper
source /home/gp/myvenv/bin/activate
pip install -r requirements.txt
```

### 1) live 模式最少环境变量

- `OPENAI_API_KEY`

如果你使用 OpenAI 兼容网关：

- `OPENAI_API_BASE`（或 `OPENAI_BASE_URL`）

### 2) 工具相关常用环境变量

- `GITHUB_TOKEN`（GithubSearchTool）
- `FIRECRAWL_API_KEY`（FirecrawlSearchTool）
- `SERPER_API_KEY`（SerperDevTool）
- `OPENAI_EMBEDDINGS_API_KEY`
- `OPENAI_EMBEDDINGS_BASE`
- `OPENAI_EMBEDDINGS_MODEL`

### 3) 从 .env 加载（示例）

```bash
set -a
. /home/gp/myvenv/.env
set +a
export OPENAI_BASE_URL="$OPENAI_API_BASE"
```

## 三、Plan 文件格式要求

必须包含：

- `input.query`
- `input.learner.{self_description, skills}`
- `output.agents[]`
- `output.subtasks[].steps[]`
- `output.execution_order[]`

步骤支持字段：

- `id`, `objective`, `instruction`, `expected_output`, `depends_on`
- `tool`（字符串或 `null`）
- `requires_human_input`（可选，默认 `false`）

loop 示例：

```json
{
  "loop": {
    "steps": ["Sx-1", "Sx-2"],
    "condition": "Sx-2.some_flag == false",
    "max_iterations": 2
  }
}
```

## 四、如何运行

### 1) 最简运行（默认 flow）

只传 plan 即可：

```bash
PYTHONPATH=src python -m plan_mapper.cli --plan plans/your_plan.json
```

### 2) live 模式 + 指定模型

```bash
PYTHONPATH=src python -m plan_mapper.cli \
  --plan plans/your_plan.json \
  --mode live \
  --model openai/deepseek-ai/DeepSeek-V3
```

### 3) 使用 runtime 引擎

```bash
PYTHONPATH=src python -m plan_mapper.cli \
  --engine runtime \
  --plan plans/your_plan.json \
  --mode live \
  --runs-dir runs_live
```

### 4) 生成 Flow 脚本（可选）

```bash
PYTHONPATH=src python -m plan_mapper.cli \
  --plan plans/your_plan.json \
  --emit-flow generated/my_flow.py
```

生成后立即执行：

```bash
PYTHONPATH=src python -m plan_mapper.cli \
  --plan plans/your_plan.json \
  --emit-flow generated/my_flow.py \
  --run-generated
```

## 五、CLI 参数说明

- `--plan`（必填）：plan JSON 路径
- `--engine`：`flow`（默认）或 `runtime`
- `--mode`：`smoke` 或 `live`
- `--model`：模型 ID（flow 默认 `openai/gpt-4o-mini`）
- `--runs-dir`：结果输出目录
- `--emit-flow`：输出生成的 Flow 脚本
- `--run-generated`：生成后立即执行脚本
- `--student-model`：给 student agent 单独指定模型
- `--max-rounds`：交互步骤最大轮次
- `--interactive-mode`：`auto`（默认）、`simulated_student`、`teacher_only`

## 六、交互逻辑（requires_human_input）

Flow 引擎下：

- `interactive-mode=auto` 时：
  - 若 plan 中存在 `requires_human_input=true` 步骤，则自动 `simulated_student`
  - 否则自动 `teacher_only`

注意：loop 与互动是两回事。loop 可以在无 student 模拟的情况下照常运行。

## 七、输出文件说明

### 1) Flow 引擎输出

默认写入 `runs_generated/run-<timestamp>/`（可通过 `--runs-dir` 修改）：

- `final_result.md`：人类可读的最终结果（按步骤）
- `final_result.json`：结构化结果
- `interaction_log.md`：仅保留交互步骤的 Teacher/Student 记录

### 2) Runtime 引擎输出

写入 `<runs-dir>/run-<id>/`：

- `compile_report.json`
- `conformance_report.json`
- `execution_report.json`
- `events.log`
- `step_outputs.json`
- `results_full.json`
- `result_readable.md`

## 八、如何从终端快速判断是否出错

成功信号：

- 看到步骤推进日志（如 `STEP ... PASS`）
- Flow/Crew 执行完成
- 最后出现 `Artifacts: ...`

失败信号：

- `Traceback`
- `AuthenticationError`
- `insufficient balance`
- `Compile validation failed`

## 九、适用的数据与任务类型

适用于“结构化步骤驱动”的任务：

- 有明确步骤拆分
- 有 `depends_on` 依赖
- 需要 loop 迭代
- 需要按步骤调用工具

典型场景：

- 教学/辅导型多智能体流程
- Prompt 或流程迭代优化
- 小型实验与验证流程

## 十、工具支持清单（当前代码）

- `FirecrawlSearchTool`
- `RagTool`
- `CodeInterpreterTool`
- `DirectoryReadTool`
- `FileReadTool`
- `FileWriterTool`
- `GithubSearchTool`
- `CodeDocsSearchTool`
- `ArxivPaperTool`
- `SerperDevTool`
- `ScrapeWebsiteTool`

> 提示：如遇工具报错，优先检查环境变量、模型可用性和 docs_url 是否有效。
