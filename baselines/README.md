# `baselines/` — Personalized Task Decomposition Baselines

本目录实现 8 个 baseline，给系统 `(query, learner_profile)` 对，输出针对该 learner 的多智能体教学 plan（§9 JSON）。

| # | Baseline | 类型 | Backbone | 上游来源 |
|---|---|---|---|---|
| L1 | `t1/` (--backend gpt-5) | 闭源 LLM 裸跑 | GPT-5 | 无（bare LLM call） |
| L2 | `t1/` (--backend claude-opus-4-6) | 闭源 LLM 裸跑 | Claude Opus 4.6 | 无 |
| L3 | `t1/` (--backend qwen3-max) | 闭源 LLM 裸跑 | Qwen3-Max | 无 |
| F1 | `autogen_qwen3/` | 领域通用 MAS | qwen3-32b | `autogen-agentchat` (PyPI) |
| F2 | `autoagents/` | 领域通用 MAS | qwen3-32b | clone `external/AutoAgents/` |
| M1 | `aipom/` | 专用 plan-generation MAS（论文方法 port） | qwen3-32b | 上游 prompt 内联，无 runtime 依赖 |
| M2 | `aflow/` | 专用 plan-generation MAS（论文方法 port） | qwen3-32b | 上游 prompt 内联，无 runtime 依赖 |
| M3 | `aop/` | 专用 plan-generation MAS | qwen3-32b | clone `external/Agent-Oriented-Planning/` |

L1-L3 三行共用同一份 `t1/plan.py` + `t1/run.py`，只通过 `--backend` 切。

> **2026-05-19 lineup 变更**：原 M1 `eduplanner/` 和 M2 `genmentor/` 从主表比较中移除——它们的 native 输出是 lesson markdown / skill-gap+session-outline，**不是多智能体编排**；强行翻译成 §9 会系统性歪曲它们的教学意图，比较不公平。代码目录保留在 `eduplanner/` / `genmentor/` 下作 reference，但不再进 main results。论文 Related Work 第 5 段已 document 这一定位（见 `../IPM_plan_education/2-Related_Work.tex`）。新加入的 M1 AIPOM ([EMNLP'25 Demo](https://arxiv.org/abs/2509.24826)) 和 M2 AFlow ([ICLR'25](https://arxiv.org/abs/2410.10762)) native 输出就是 agent-task DAG / agentic workflow，跟 §9 同形态。

---

## 1. 环境配置

### 1.1 主 env（跑 L-tier / F1 AutoGen / M1 AIPOM / M2 AFlow / M3 AOP）

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install openai>=1.0 python-dotenv tqdm
pip install autogen-agentchat>=0.4 autogen-ext[openai]>=0.4    # F1 必需
```

### 1.2 子 venv（跑 F2 AutoAgents）— Python 3.10 严格

```bash
python3.10 -m venv .venvs/autoagents_310
# Windows: .venvs\autoagents_310\Scripts\activate
# Linux/Mac: source .venvs/autoagents_310/bin/activate

pip install -r external/AutoAgents/requirements.txt
# 关键 pinned: litellm==0.7.5 + pydantic==1.10.7
```

> venv 位置可改：默认 `<repo>/.venvs/autoagents_310/`，自动探测 `python.exe` / `Scripts/python.exe` / `bin/python` 三种 layout。也可通过环境变量 `AUTOAGENTS_VENV_PYTHON=<absolute-path-to-python>` 指向任意位置的 venv（如 conda 环境）。

> ⚠️ 已废弃：以前 1.3 是 GenMentor 的 Python 3.10 子 venv，2026-05-19 GenMentor 从 lineup 移除后不再需要。`eduplanner/` / `genmentor/` 代码目录虽然保留但**不在 active 比较里**，跑它们也只需要主 env（EduPlanner 本来就是 inlined prompt，不需子 venv；GenMentor 想跑就还按旧的子 venv 流程，但 main results 不再消费其输出）。

M1 AIPOM 和 M2 AFlow 跟 EduPlanner 同模式——上游 prompt 已内联到 `aipom/meta_prompt.py` / `aflow/meta_prompt.py`，runtime 不依赖任何外部 clone，主 env 就够。

---

## 2. API key

在仓库根目录（`baselines/` 的父目录）建 `.env`：

```env
DASHSCOPE_API_KEY=sk-xxxx              # 阿里云灵积，跑 qwen3-32b / qwen3-max
OPENAI_PROXY_API_KEY=sk-xxxx           # bianxie.ai 代理，跑 GPT-5 / Claude
OPENAI_PROXY_BASE_URL=https://api.bianxie.ai/v1
```

- `DASHSCOPE_API_KEY`：L3 + 全部 F-tier/M-tier 都用
- `OPENAI_PROXY_API_KEY` + `OPENAI_PROXY_BASE_URL`：L1 (GPT-5) + L2 (Claude) 用；不一定走 bianxie.ai，任何 OpenAI-兼容代理都行（直接用 OpenAI 官方 key 也可，把 base_url 改成 `https://api.openai.com/v1`）

---

## 3. 克隆上游论文仓库

只有 2 个 active baseline 必须 clone 上游：

```bash
mkdir -p external && cd external

# F2 AutoAgents
git clone https://github.com/Link-AGI/AutoAgents.git

# M3 AOP
git clone https://github.com/lalaliat/Agent-Oriented-Planning.git
```

L-tier / F1 AutoGen / M1 AIPOM / M2 AFlow 无需 clone 上游。AIPOM 和 AFlow 的 prompt 已分别内联到 `aipom/meta_prompt.py` 和 `aflow/meta_prompt.py`。

可选 clone（runtime 不依赖，只在想对照论文方法源出处时拉）：

```bash
# AIPOM — EMNLP 2025 Demo
git clone https://github.com/megagonlabs/aipom.git
# AFlow — MetaGPT 仓库内 examples/aflow
git clone https://github.com/geekan/MetaGPT.git
# EduPlanner (deprecated lineup，仅 reference)
git clone https://github.com/Zc0812/Edu_Planner.git
# GenMentor (deprecated lineup，仅 reference)
git clone https://github.com/GeminiLight/gen-mentor.git
```

> 如果想把 external/ 放到非默认位置，active baseline 支持环境变量 override：
>
> - `AUTOAGENTS_REPO=<path>` （F2）
> - `AOP_REPO=<path>` （M3）

---

## 4. 必需的根目录文件

仓库根目录还要放：

| 文件 | 用途 |
|---|---|
| `prompt_for_inference.txt` | 推理 prompt 单一源，`compose_t4()` 从这里抽 PREAMBLE+§5+§9+§12 |
| `multi_agent_dataset_filtered_qap_v15_goodplus.jsonl` | 输入数据集，3043 条 (query, learner) 对 |

---

## 5. 跑代码

所有命令从仓库根目录（`baselines/` 父目录）执行。

### 5.1 Smoke test（强烈建议先做，单条样本）

```bash
# L* ×3
python -m baselines.t1.run --backend qwen3-max       --limit 1
python -m baselines.t1.run --backend gpt-5           --limit 1
python -m baselines.t1.run --backend claude-opus-4-6 --limit 1

# F* ×2
python -m baselines.autogen_qwen3.run --limit 1
python -m baselines.autoagents.run    --limit 1

# M* ×3
python -m baselines.aipom.run --limit 1
python -m baselines.aflow.run --limit 1
python -m baselines.aop.run   --limit 1

# Deprecated (kept on disk but NOT in main results)
# python -m baselines.eduplanner.run --limit 1
# python -m baselines.genmentor.run  --limit 1
```

成功标志：`evaluation_results/baselines/<name>/plans.jsonl` 出现 1 行，`failures.jsonl` 为空。

### 5.2 全量跑

去掉 `--limit` 即可。所有 baseline 自带断点续跑（`progress.json` 记录），中断后重跑同命令会跳过已完成的。

### 5.3 自定义输出目录

```bash
python -m baselines.aop.run --limit 5 \
    --output-dir evaluation_results/baselines/_my_test/aop
```

### 5.4 单条耗时参考

| Baseline | 单条耗时 |
|---|---|
| L1 / L2 / L3 | ~30s |
| F1 AutoGen | ~30s |
| F2 AutoAgents | ~11 min |
| M1 AIPOM | ~30s（待 smoke 实测）|
| M2 AFlow | ~30s（待 smoke 实测）|
| M3 AOP | ~40s |

---

## 6. 输出结构

```
evaluation_results/baselines/<name>/
├── plans.jsonl          每行 {"question_id", "profile_index", "generated_plan": {...§9 JSON...}}
├── failures.jsonl       每行 {"key", "error", "traceback"}
├── progress.json        断点续跑状态
└── native_outputs.jsonl F2 / M-tier 才有；记录上游 paper method 的 native 中间产物
```

---

## 7. 子目录说明

```
baselines/
├── common/              共用基础设施
│   ├── prompt_sections.py    切 §1-§12；compose_t4() = PREAMBLE+§5+§9+§12
│   ├── task_description.py   T4_TASK_DESCRIPTION + build_t4_system_message()
│   ├── data_loader.py        流式读输入数据集
│   ├── llm_client.py         统一 LLM 接口（5 个 backend）
│   ├── json_repair.py        LLM 输出 JSON 修复
│   ├── schema_validator.py   §9 strict 校验
│   ├── progress.py           断点续跑
│   ├── native_logger.py      sidecar：写 paper method 中间产物
│   └── runner.py             通用 batch 骨架
├── t1/                  L1/L2/L3 共用
├── autogen_qwen3/       F1（AssistantAgent 单 agent 调用）
├── autoagents/          F2（subprocess 跑上游 Manager + Observer）
├── aipom/               M1（agent-aware DAG planner，AIPOM 论文方法 port）
├── aflow/               M2（operator-composition planner，AFlow 论文方法 port）
├── aop/                 M3（meta-agent + LLM-as-judge replan loop）
├── eduplanner/          [deprecated] 2026-05-19 起从主表移除——native 输出是 lesson markdown
└── genmentor/           [deprecated] 2026-05-19 起从主表移除——native 输出是 skill-gap+session outline
```

---

## 8. 常见问题

**`ModuleNotFoundError: baselines`**
从仓库根目录跑命令（不要 `cd baselines/` 进去），用 `python -m baselines.xxx.run` 形式。

**`KeyError: 'DASHSCOPE_API_KEY'`**
`.env` 没放在 `baselines/` 父目录，或路径不对；检查 `baselines/common/llm_client.py` 的 `load_dotenv()` 指向。

**AutoAgents 子进程 `401 Invalid token`**
`.env` 里的 `DASHSCOPE_API_KEY` 失效或没读到；激活 `.venvs/autoagents_310` 后手动确认环境变量。

**GenMentor `ImportError: modules.personalized_resource_delivery...`** *(deprecated lineup — 2026-05-19 起不再 active)*
`external/gen-mentor/` 没 clone 到位，或目录名不对（必须小写连字符 `gen-mentor`）。

**AOP `os.chdir: /mnt/liao/planner not found`**
`baselines/aop/aop_patch.py` 的 `AOP_REPO` 路径不指向你 clone 的 `external/Agent-Oriented-Planning/`。

**Claude smoke 报 model 不存在**
不同代理上 Claude 的 model id 不一样。改 `baselines/common/llm_client.py` 里 `self.model = "claude-opus-4-6"` 为你代理实际支持的 id。
