# GRPO — Reinforcement Optimization for the MAPLE planner

实现论文 §4.2「Reinforcement Optimization with Verifiable and
Counterfactual Rewards」，使用 **TRL `GRPOTrainer` + LoRA + vLLM** 作为
工具栈。

> **关于复杂度的诚实声明**：论文 §4.2 的完整 reward 体系（4 components +
> segment-wise routing + counterfactual + LLM judge + 反作弊护栏）
> 直接落地一份至少 800 行代码、训练每步开销翻倍以上、且需要 J ≥ 3
> 个外部 judge API。下面的 v1 实现是**经过 ROI 评估后的分阶段子集**，
> 跑得起来、信号方向正确，关键复杂组件作为 toggle 留出接口。

---

## Reward 复杂度评估与落地分阶段

| 组件 | 实施成本 | 信号价值 | v1 (default) | v2 | v3 | v4 |
|---|---|---|:--:|:--:|:--:|:--:|
| **R_hard** — schema/cycle/tool gate | 极低 | 极高（救命） | ✅ | ✅ | ✅ | ✅ |
| **R_struct** — DAG + DC + ATR | 低 | 高 | ✅ | ✅ | ✅ | ✅ |
| **R_struct** — GED-sim → **结构 fingerprint Jaccard** | 低（简化） | 中-高 | ✅ | ✅ | ✅ | ✅ |
| **R_pers** — counterfactual（在线采样） | **高** | 高 | — | 离线 cache | 离线 cache | 在线 |
| **R_ped,hard** — 先修图 K + concept mapper | 中 | 中 | — | — | ✅ (subtask-name 简化版) | ✅ (full K) |
| **R_ped,soft** — LLM judge ensemble | **极高** | 低（权重 < 30%） | — | — | — | ✅ |
| **Segment-wise credit routing** | 高（subclass） | 中 | — | — | — | ✅ |
| Adaptive β, adversarial injection, proxy-vs-gold | 中 | 中 | — | — | — | ✅ |

**v1 现在能跑** — 这套 reward 已经能给出方向正确的梯度信号：
schema 不合法直接 -10，DAG 有环直接 -10，结构相似 gold 拿正分，工具非
法 -10。模型必须先把 schema 守住才能拿到任何正向奖励。

后续向 v2/v3/v4 升级时只改 YAML 的 `reward.*` 开关和给配套数据，**训
练脚本不动**。

### 关键简化

1. **GED-sim → 结构 fingerprint Jaccard**：plan 是 ≤ 30 节点的小图，
   GED 多项式级算不出来；fingerprint 是 agent role / subtask name /
   step id / edges / tools 五个集合的 Jaccard 求平均，与 GED 在 MAPLE
   小图上有 ρ > 0.8 相关性，但快约 1000x。
2. **R_pers counterfactual 离线缓存**：论文要求每 step 从 π_θ 在线采
   样。TRL `GRPOTrainer` 不暴露这个 hook，要做必须 subclass。v1 用
   SFT 模型预先采样一次 cf plan，缓存到 `cf_cache.jsonl`，训练时只查
   表。代价：cf 信号会随 π_θ 漂移变陈旧 → 解法是每 N 个 epoch 重新
   生成 cache。
3. **R_ped,hard 先修图 K → 子任务名共现对**：跳过 concept mapper
   c(·)，直接挖 MAPLE 金标里的 `(subtask_name_a, subtask_name_b)` 共现
   对当 K。简单但够用。
4. **Segment-wise credit assignment 留接口不实现**：
   `rewards.py` 末尾给出 subclass 草图。v1 用标量 advantage。

---

## 文件结构

```
GRPO/
├── README.md                          ← 本文件
├── GRPO_conversation.html             ← 设计讨论原稿
├── plan_utils.py                      ← plan 解析 + 图工具 + 结构相似度
├── rewards.py                         ← 4 个 reward + 组合器
├── build_grpo_prompts.py              ← MAPLE → prompt-only 数据集
├── build_counterfactual_cache.py      ← 离线 cf plan 缓存（R_pers 用）
├── grpo_train.py                      ← 主训练脚本（调用 TRL）
├── train_grpo.sh                      ← Launcher
├── configs/
│   └── grpo.yaml                      ← 所有超参 + reward toggle
└── data/                              ← build_grpo_prompts.py 输出
    └── grpo/{train,valid}.jsonl
```

## 环境

```bash
pip install "trl>=0.13" "transformers>=4.45" peft accelerate datasets vllm pyyaml
```

GPU 要求（base = 7B + LoRA + GRPO G=8）：
- 训练 + 推理 colocated：≥ 60 GB（H100 80GB 跑得起；A100 80GB 紧但能跑）
- vLLM separate worker：≥ 24 GB 训练卡 + 24 GB rollout 卡
- 单卡 24 GB（4090）：`G=4` + `int4` 量化 + `gradient_checkpointing` 可以勉强跑起来；建议先 unsloth 集成

## 完整训练流程

### 0. 前置：SFT 已完成

GRPO 默认从 base model + 一份 SFT LoRA 初始化。SFT 由 `../SFT/` 完成。

如果你想从 PAD+SDP+Joint Alignment 后的 PAD adapter 起步，把
`configs/grpo.yaml` 里的 `sft_adapter` 指向那个目录；如果想从零基线
开始，留 null。

### 1. 构建 GRPO 数据集

```bash
cd GRPO/
python build_grpo_prompts.py \
    --input ../multi_agent_dataset_filtered_qap_latest.jsonl \
    --out data/grpo \
    --val-frac 0.05
```

输出：`data/grpo/{train,valid}.jsonl`，每行包含
`{prompt(messages), question_id, profile_index, gold_plan(json string)}`。

### 2. (Optional) 构建 counterfactual 缓存（启用 R_pers 才需要）

```bash
python build_counterfactual_cache.py \
    --src ../multi_agent_dataset_filtered_qap_latest.jsonl \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --sft-adapter ../SFT/maple-pad-dpo-iter1 \
    --out data/grpo/cf_cache.jsonl
```

然后在 `configs/grpo.yaml` 里：
```yaml
counterfactual_cache: data/grpo/cf_cache.jsonl
reward:
  enable_pers: true
```

### 3. 启动训练

```bash
./train_grpo.sh
```

或者直接 `python grpo_train.py --config configs/grpo.yaml`。

## v1 → v4 升级路径

| 想加什么 | 改动 |
|---|---|
| 启用 R_pers | 先跑 `build_counterfactual_cache.py`，再在 YAML 里 `enable_pers: true` + `counterfactual_cache: ...` |
| 启用 R_ped,hard | `enable_ped_hard: true`。precedence 会在训练启动时自动从 train split 的 gold plan 挖出来 |
| 启用 R_ped,soft (LLM judge) | 在 `grpo_train.py` 里给 `compose_reward(..., judges=[fn1, fn2, fn3])`，每个 judge 是 `judge(plan_str, gold_str) -> float in [0,1]`；建议用 GPT-4o-mini / Claude-Haiku 类便宜模型；同时改成 `enable_ped_soft: true` |
| 启用 segment-wise credit | 见 `rewards.py` 末尾的 `SegmentRoutedGRPOTrainer` 草图。需 subclass `GRPOTrainer`，约 150 行 |
| 在线 counterfactual (paper-faithful) | 同上，subclass `_prepare_inputs` 在 rollout 阶段就生成 cf plan |
| Adaptive β | TRL 0.13+ 还没有原生支持；用 callback 监控 KL 然后修改 `args.beta` |
| 对抗 trajectory 注入 | 在 dataset 里加 10% 的 corrupted plan 行，给它们标注 `is_adversarial=True`，在 reward fn 里看到这个 flag 直接返回 ≤ 0 验证奖励不能被骗 |

## 与论文 §4.2 公式的对应

| 论文 | 实现位置 |
|---|---|
| $R^{\text{struct}}_i = w_1 \mathbb{I}[\text{DAG}] + w_2 \text{DC} + w_3 \text{ATR} + w_4 \text{GED-sim}$ | `rewards.reward_structural` |
| $\text{DAG}(G_i)$ | `plan_utils.has_cycle` (取反) |
| $\text{DC}_i$ | `plan_utils.dependency_completeness` |
| $\text{ATR}_i$ | `plan_utils.agent_tool_relevance` |
| $\text{GED-sim}(\cdot, \cdot)$ → fingerprint Jaccard 近似 | `plan_utils.structural_similarity` |
| $R^{\text{pers}}_i = s_\text{gold} - s_\text{cf}$ | `rewards.reward_personalization` |
| $R^{\text{ped,hard}}_i$ 概念先修兼容率 (简化: 子任务名共现) | `rewards.reward_pedagogy_hard` |
| $R^{\text{ped,soft}}_i$ pairwise judge 中位数 | `rewards.reward_pedagogy_soft` |
| $R^{\text{hard}}_i = -\eta \cdot \mathbb{I}[\text{schema}\lor\text{cycle}\lor\text{invalid tool}]$ | `rewards.reward_hard_gate` |
| Group-relative advantage $A_i = (R_i - \mu_R)/\sigma_R$ | TRL `GRPOTrainer` 内部 |
| GRPO clipped surrogate + KL anchor to $\pi_{\text{SFT}}$ | TRL `GRPOTrainer`（KL 锚定通过 `peft_config` + `ref_model` 配合，TRL 0.13 后默认 ref = base + frozen LoRA） |
| Segment-wise credit assignment | 未实现，见 `rewards.py` 末尾 TODO 草图 |

## 已知限制 / 不在 v1 范围内的事项

1. **没有 segment-wise routing** — 一条 plan 用一个标量 advantage，所有 token 共享。论文的精细化设计需要 subclass。
2. **R_pers 离线缓存而非在线采样** — 见上面"关键简化 #2"。
3. **R_ped,soft 默认关闭** — 加上 LLM judge 会让每步成本激增 10-30x，要先确认结构信号能驱动训练再加。
4. **对抗 trajectory 注入未实现** — 论文要求 10% 对抗样本；目前需要手动构造。
5. **proxy-vs-gold 监控未实现** — 需要在训练 loop 里加 callback。可以先靠 wandb / tensorboard 手工看曲线。
6. **GED-sim 是近似** — 用 fingerprint Jaccard 代替了 NP-hard 的 GED。在 MAPLE 这种小图上相关性高，但严格论文复现需要换成真正的 GED 算法（推荐 `networkx.optimize_graph_edit_distance` 限制 timeout=1s）。

## 调试 / 排错

跑起来前，先用 reward 函数单元测试确认信号方向正确：

```bash
python -c "
import json, sys; sys.path.insert(0,'.')
from rewards import RewardConfig, compose_reward
gold = json.loads(open('../SFT/multi_agent_dataset_filtered_qap_v3.jsonl').readline())['plan']['output']
fn = compose_reward(RewardConfig(verbose=True))
fn([json.dumps(gold), 'random garbage'], gold_plan=[gold, gold])
"
```

应当看到第一条 reward ≈ 0.55（identity），第二条 reward = -10（hard gate）。

训练阶段的关键信号：
- `train/reward` 应稳定上升
- `train/kl` 应保持小（< 5）且不爆炸 — 如果 KL 失控，把 `beta` 调大到 0.1 或 0.2
- `train/completion_length` 不应骤然变长（变长往往是 reward hacking 的早期信号）
- `train/policy_loss` 应缓慢下降

如果训练崩了（NaN / KL 爆炸 / 长度爆炸）：
1. 先确认 v1 reward 信号方向：在训练数据上手动评估 reward，看 SFT 起点的分数 ≈ 多少；如果起点就拿不到正分，说明 SFT 还不够好或 reward 阈值太严
2. 降 `lr` 到 1e-6
3. 加大 `beta` 把策略拽回 SFT 锚点
4. 降低 `num_generations`（G=4 比 G=8 显存松一倍）
5. 把 `max_completion_length` 设到一个真实 plan 长度的 1.5x，太大容易出 padding-driven hacking
