# SFT — 三阶段微调 MAPLE 多智能体规划器

本目录实现论文 §4 的 hierarchical SFT + Joint Alignment 流程，使用
HuggingFace AutoTrain 作为训练后端。

## 三阶段总览

```
                              ┌─────────────────┐
  Stage 1 (PAD SFT)           │  PAD adapter    │  → personalization
  query + profile  ─────────► │   (T*, A*)      │    on (T, A)
                              └─────────────────┘
                              ┌─────────────────┐
  Stage 2 (SDP SFT)           │  SDP adapter    │  → pedagogy +
  query + profile + gold(T,A) │   (S*, O*)      │    dependency on (S, O)
  ─────────────────────────►  └─────────────────┘
                              ┌─────────────────┐
  Stage 3 (Joint Alignment)   │  PAD ↔ SDP      │  → close exposure-bias
  alternating Phase A & B     │  refined        │    gap, k ∈ {1, 2}
                              └─────────────────┘
```

| 阶段 | 输入 | 输出 (loss target) | AutoTrain task |
|---|---|---|---|
| **1 — PAD SFT** | `query ⊕ profile` | `(T*, A*)` — agents + subtask 元数据 | `llm-sft` |
| **2 — SDP SFT** | `query ⊕ profile ⊕ gold(T*, A*)` | `(S*, O*)` — steps + execution_order | `llm-sft` |
| **3a — On-policy SDP** | `query ⊕ profile ⊕ PAD-sampled(T', A')` | `(S*, O*)` | `llm-sft` |
| **3b — PAD DPO** | `query ⊕ profile` | DPO pairs `(chosen, rejected)`，由 **scaffold-vs-gold R_struct (Jaccard on agents+subtask names)** 排序 | `llm-dpo` |

> **CLI 命名与 paper §3.2 反**：CLI `phase-a` = paper Phase B (修 SDP)、CLI `phase-b` = paper Phase A (修 PAD)。这是历史命名，没改是为了少动 config。下文行文按 CLI 顺序。

每个 adapter 都是 LoRA on frozen backbone；PAD 和 SDP 是 **同一 base 上的两个独立 LoRA**。

## 文件列表

```
SFT/
├── README.md                                ← 本文件
├── multi_agent_dataset_filtered_qap_v3.jsonl ← 输入（50 条；正式训练时换成完整 3,056 条数据集）
├── prompts.py                               ← PAD / SDP 的 system + user 模板
├── build_sft_data.py                        ← Stage 1 / 2 数据构造
├── joint_alignment_data.py                  ← Stage 3 Phase A / B 数据生成（需 GPU 跑 PAD/SDP）
├── train_pad.sh                             ← Stage 1 启动脚本
├── train_sdp.sh                             ← Stage 2 启动脚本
├── train_joint.sh                           ← Stage 3 启动脚本（含 Phase A↔B 外循环）
├── configs/
│   ├── pad_autotrain.yml                    ← Stage 1 AutoTrain 配置
│   ├── sdp_autotrain.yml                    ← Stage 2 AutoTrain 配置
│   ├── joint_phase_a_autotrain.yml          ← Stage 3 Phase A (SFT) 配置
│   └── joint_phase_b_autotrain.yml          ← Stage 3 Phase B (DPO) 配置
└── data/                                    ← build_sft_data.py 输出
    ├── pad/{train,valid}.jsonl
    ├── sdp/{train,valid}.jsonl
    └── joint/iterK/phase_{a,b}/...          ← train_joint.sh 输出
```

## 环境

```bash
pip install autotrain-advanced
# 联合微调需要额外的：
pip install "torch>=2.1" transformers peft accelerate
```

## 完整训练流程

### 0. 构造 Stage 1 / 2 数据

```bash
cd SFT/
python build_sft_data.py
# 默认读 multi_agent_dataset_filtered_qap_v3.jsonl
# 想用全量数据集 (3056 条) 时：
# python build_sft_data.py --input ../multi_agent_dataset_filtered_qap_latest.jsonl
```

会输出 `data/pad/{train,valid}.jsonl` 和 `data/sdp/{train,valid}.jsonl`。每行是 AutoTrain 兼容的 `messages` 字段（chat-format，loss 只在 assistant token 上计算）。

### 1. Stage 1 — 微调 PAD

```bash
./train_pad.sh
# 或：autotrain --config configs/pad_autotrain.yml
```

输出：`./maple-pad-sft/` 目录，包含 LoRA adapter。

### 2. Stage 2 — 微调 SDP（与 Stage 1 独立，可并行）

```bash
./train_sdp.sh
```

输出：`./maple-sdp-sft/`。

> 训练数据里 SDP 看到的是 **gold scaffold (T*, A*)**，不是 PAD 输出 —— 这是论文 §4 的设计（避免双向耦合，等到 Stage 3 才闭环）。

### 3. Stage 3 — Joint Alignment

```bash
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export PAD_ADAPTER="./maple-pad-sft"
export SDP_ADAPTER="./maple-sdp-sft"
export SRC="multi_agent_dataset_filtered_qap_v3.jsonl"
export K=1     # outer 迭代数，论文用 1~2

./train_joint.sh
```

`train_joint.sh` 内部循环 `K` 次，每次：

- **Phase A** (CLI `phase-a` = paper Phase B，修 SDP): 用 PAD 采样 scaffold → SDP SFT；
- **Phase B** (CLI `phase-b` = paper Phase A，修 PAD): 用 PAD 采 2 个 scaffold，按 **scaffold-vs-gold Jaccard** (agents + subtask names) 排 chosen/rejected → PAD DPO。**不**加载 SDP，比旧的 SDP-perplexity 打分快 ~2×。

每轮结束更新 `PAD_ADAPTER` / `SDP_ADAPTER` 指向新 checkpoint 后继续。

最终 adapter：
- `./maple-pad-dpo-iter${K}/`（PAD 终态，用于推理）
- `./maple-sdp-onpolicy-iter${K}/`（SDP 终态）

## 推理时的串行调用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json
from prompts import PAD_SYSTEM, PAD_USER_TEMPLATE, SDP_SYSTEM, SDP_USER_TEMPLATE

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
# 两个 adapter 挂在同一 base 上
base = PeftModel.from_pretrained(base, "./maple-pad-dpo-iter1", adapter_name="pad")
base.load_adapter("./maple-sdp-onpolicy-iter1", adapter_name="sdp")

def plan(query, learner_desc, learner_skills):
    # PAD pass
    base.set_adapter("pad")
    msgs = [{"role":"system","content":PAD_SYSTEM},
            {"role":"user","content":PAD_USER_TEMPLATE.format(
                query=query, self_description=learner_desc,
                skills=json.dumps(learner_skills))}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = base.generate(**tok(prompt, return_tensors="pt").to(base.device),
                        max_new_tokens=1500, do_sample=False)
    scaffold = json.loads(tok.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip())

    # SDP pass
    base.set_adapter("sdp")
    msgs2 = [{"role":"system","content":SDP_SYSTEM},
             {"role":"user","content":SDP_USER_TEMPLATE.format(
                 query=query, self_description=learner_desc,
                 skills=json.dumps(learner_skills),
                 agents=json.dumps(scaffold["agents"], indent=2),
                 subtasks=json.dumps(scaffold["subtasks"], indent=2))}]
    prompt2 = tok.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
    out2 = base.generate(**tok(prompt2, return_tensors="pt").to(base.device),
                         max_new_tokens=3500, do_sample=False)
    steps_part = json.loads(tok.decode(out2[0], skip_special_tokens=True).split("assistant")[-1].strip())

    # 合并成完整 plan
    return {"agents": scaffold["agents"],
            "subtasks": [
                {**sub_meta, "steps": next(s["steps"] for s in steps_part["subtasks"] if s["id"] == sub_meta["id"])}
                for sub_meta in scaffold["subtasks"]
            ],
            "execution_order": steps_part["execution_order"]}
```

## 注意点与限制

1. **v3 只有 50 条数据**。是 smoke test 规模，loss 能下降但不会有真实泛化。正式训练把 `--input` 改成 `../multi_agent_dataset_filtered_qap_latest.jsonl`（3,056 条）。
2. **AutoTrain CLI 的 YAML schema 在不同版本下可能略有变化**。如果 `peft_model`, `column_mapping` 字段不匹配你的 autotrain 版本，请按它的 docs 调整字段名。最常变动的是 `peft_model` 这一项（用于 resume from existing adapter）。
3. **Stage 3 Phase A/B 必须在 Stage 1+2 完成后才能跑**，因为它依赖已训练的 adapter 做 inference。
4. **DPO 的 reference policy** 在 Phase B 中是当前 PAD adapter（`peft_model` 指向的那个 checkpoint），与论文 §4 公式 (139) 中 $\pi^{\text{ref}}_{\text{PAD}}$ 一致。
5. **GRPO（论文 §4 第二大块）不在此目录内**，留给 `SFT/GRPO/`（已建空目录）后续实现。
6. **量化**：所有配置默认开 `int4` (QLoRA)。如果你的 GPU 内存足够 (≥ 24GB 跑 7B model bf16 + LoRA)，把 `quantization: none` 关掉精度会更好。
7. **base model**：所有 YAML 默认 `Qwen/Qwen2.5-7B-Instruct`。改成 Llama / Mistral 等的话同步改 PAD 和 SDP 两份 config，并确认 tokenizer 的 chat template 不为空。

## 与论文 §4 公式的对应

| 论文 | 实现 |
|---|---|
| $x_{\text{pad}} = I_q \oplus I_p$ | `build_sft_data.py::build_pad_messages` |
| $y_{\text{pad}}$ = serialization of $(\mathcal{T}^\star, \mathcal{A}^\star)$ | `build_sft_data.py::pad_target` |
| $\mathcal{L}_{\text{PAD}}$ | AutoTrain `llm-sft` + LoRA on PAD adapter |
| $x_{\text{sdp}} = I_q \oplus I_p \oplus \mathcal{A}^\star \oplus \mathcal{T}^\star$ | `build_sft_data.py::build_sdp_messages` |
| $y_{\text{sdp}}$ = serialization of $(\mathcal{S}^\star, \mathcal{O}^\star)$ | `build_sft_data.py::sdp_target` |
| $\mathcal{L}_{\text{SDP}}$ | AutoTrain `llm-sft` + LoRA on SDP adapter |
| Phase A: $\mathcal{L}^{\text{ja}}_{\text{SDP}}$ on on-policy $(\mathcal{A}', \mathcal{T}') \sim \pi_{\text{PAD}}$ | `joint_alignment_data.py phase-a` → AutoTrain `llm-sft` 用 `configs/joint_phase_a_autotrain.yml` |
| Phase B: scaffold-vs-gold R_struct preference → DPO on PAD | `joint_alignment_data.py phase-b` → AutoTrain `llm-dpo` 用 `configs/joint_phase_b_autotrain.yml` |
| $\Delta^{w,l}_{\text{PAD}}$, $\beta$ | DPO 内部，`dpo_beta: 0.1` 在配置里 |

## 改动数据 / prompt 的影响

- 改 `prompts.py` → 同步重跑 `build_sft_data.py` → 所有阶段都要重训
- 改 `build_sft_data.py` 的 `pad_target` / `sdp_target` 切分边界 → 同上
- 改 base model → 只动 `configs/*.yml` 里的 `base_model` 字段；prompt 和数据不动
