# Stage 3 执行层配置说明

Stage 3 是 MAPLE 评估流水线的**plan 执行层**，基于 CrewAI 框架，驱动 Teacher（GPT-4o）+ Student（GPT-4o-mini）Socratic 对话来运行多智能体教学规划。

---

## 1. 环境依赖

### Python 依赖

```bash
pip install crewai crewai-tools openai python-dotenv
```

> **注意**：`crewai_tools` 中部分工具（`FirecrawlSearchTool`、`ArxivPaperTool` 等）需要额外 API key，见第 3 节。若 import 失败，runtime 会将所有工具降级为 `None`，导致依赖这些工具的 plan 执行失败。

---

## 2. Docker 镜像（代码沙盒）

`CodeInterpreterTool` 步骤在 Docker 容器中执行学生代码，支持 Python / Java / Node / Go / Rust / Ruby / R / PHP / Kotlin / C# / SQL。

```bash
# 在 stage3_execution/ 目录下构建
docker build -f Dockerfile.mas-runtime -t mas-runtime .
```

构建完成后验证：

```bash
docker run --rm mas-runtime python3 -c "print('ok')"
```

> 若 Docker 不可用，runtime 会自动降级为**本地执行**（无沙盒隔离），并打印警告。

---

## 3. 环境变量（API Keys）

在项目根目录或 `~/.env` 中配置，或直接 `export`：

| 变量 | 必填 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | **必填** | Teacher/Student agent 调用 GPT-4o / GPT-4o-mini |
| `OPENAI_BASE_URL` | 可选 | 自定义 OpenAI 兼容端点（如 `https://api.bianxie.ai/v1`） |
| `FIRECRAWL_API_KEY` | 可选 | `FirecrawlSearchTool` 网页抓取；缺失时该工具自动跳过 |
| `SERPER_API_KEY` | 可选 | `SerperDevTool` 搜索；缺失时该工具自动跳过 |
| `GITHUB_TOKEN` | 可选 | `GithubSearchTool` 代码搜索；缺失时降级为未鉴权请求 |

最小配置（只需 OpenAI）：

```bash
export OPENAI_API_KEY=sk-...
```

> **缺少 FIRECRAWL_API_KEY 的影响**：依赖 `FirecrawlSearchTool` 的 plan step 会被跳过，导致该 run 的 `succeeded=False`。这是目前 baselines 执行成功率低（7-19%）的主要原因之一。

---

## 4. 运行批量执行

```bash
cd stage3_execution

# 全量批量执行（读取 batch_summary.json 中的 plan 列表）
python3 batch_eval.py --out runs/ --workers 4

# 单条执行（调试用）
python3 run_single_plan.py --qid 28958192 --profile 3

# 并行执行（多进程）
python3 batch_eval_parallel.py --out runs/ --workers 8
```

输出目录结构：

```
runs/
└── run-<qid>-p<profile_idx>-<hash>/
    ├── execution_log.json    # 完整对话记录（每轮 Teacher/Student 交互）
    └── step_outputs.json     # 每个 subtask step 的输出 + meta
```

---

## 5. 执行失败的常见原因

| 现象 | 根因 | 处理方式 |
|------|------|---------|
| `succeeded=False`，工具报 `None` | `crewai_tools` import 失败，所有工具变 None | `pip install crewai-tools` 并检查版本兼容性 |
| `FirecrawlSearchTool skipped` | 缺少 `FIRECRAWL_API_KEY` | 配置 key 或接受该工具跳过 |
| `CodeInterpreterTool` 执行超时 | Docker 容器启动慢 / 代码死循环 | 确认 Docker daemon 运行；调整 `--timeout` 参数 |
| `docker: command not found` | Docker 未安装或 daemon 未启动 | 安装 Docker 并 `docker start` |
| 401 / AuthenticationError | `OPENAI_API_KEY` 过期或余额不足 | 更新 key |

---

## 6. 已知工具可用性

根据 v15 release 数据集分析（见 `README.md` 工具池说明）：

- **实际被 plan 选用**：`CodeInterpreterTool`、`CodeDocsSearchTool`、`FirecrawlSearchTool`、`FileWriterTool`、`ArxivPaperTool`
- **声明但从未被选用**：`DirectoryReadTool`、`FileReadTool`、`RagTool`

其中 `CodeDocsSearchTool`、`FirecrawlSearchTool` 是导致新 LLM 生成 plan 执行失败率高的核心工具，需要确保 `crewai_tools` 版本正确安装。
