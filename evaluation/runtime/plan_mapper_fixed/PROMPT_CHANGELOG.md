# Prompt Changelog · `runtime.py`

记录 `stage3_execution/plan_mapper_fixed/runtime.py` 内 prompt 字符串的所有改动，方便之后回溯、对比 baseline、判断指标变化是 prompt 引起的还是别处。

每条改动都记录：日期 · 文件:行号 · before / after diff · 动机 · 预期影响 · 回滚方式。

---

## 2026-05-22 — 鼓励 teacher agent 在代码题里真编译 / 真跑

**动机**：老师提到「代码编译部分可以再鼓励一下」。MAPLE 里 Java/C++/Kotlin 等编译型语言占比不低，旧 prompt 只有一句 `Use this tool when needed.`，"when needed" 给了 teacher 跳过编译的口子；TOOL_ARG_HINTS[CodeInterpreterTool] 里那句 `ALWAYS call this tool with real code` 是 tool-input 层面的提示，不够强势。希望 teacher 在编译型语言上**真去 compile**（拿到真 compile error），并禁止用 "Expected output:" 这种 speculative 话术。

**影响范围**：仅 `step.tool == "CodeInterpreterTool"` 的 step；其他 tool 的 prompt 维持不变。

### 改动 1 · 新增模块常量 `CODE_TOOL_NUDGE`

位置：紧跟 `TOOL_ARG_HINTS` 定义之后（约 L635）。

```python
# 新增
CODE_TOOL_NUDGE = (
    "Code-execution guidance (this tool compiles and runs code in a sandbox; "
    "treat its output as the source of truth):\n"
    "- For compiled languages (java, c, c++, kotlin, swift, typescript), you MUST "
    "compile via this tool before claiming the code is correct; quote real compile "
    "errors verbatim if they appear.\n"
    "- Quote the real stdout/stderr returned by the tool. Do not write phrases like "
    "\"Expected output:\", \"you would get:\", \"this would produce\", or "
    "\"the result would be\" — those signal you did not actually execute.\n"
    "- Pedagogical ordering: only run/compile AFTER the learner has predicted or "
    "attempted. The tool belongs in validation / feedback / consolidation, not in "
    "the first probe.\n"
    "- Before sending your message, self-check: did I actually run or compile this "
    "code, or am I describing what it would do?\n"
)
```

### 改动 2 · `_run_interactive_step` 内 teacher_prompt（约 L1017）

#### Before
```python
teacher_prompt = (
    f"Objective: {step.objective}\n"
    f"Instruction: {step.instruction}\n"
    f"Expected Output: {step.expected_output}\n"
    f"Context:\n{context_text}\n"
    + (f"Required Tool: {step.tool}. Use this tool when needed.\n" if step.tool else "")
    + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step.tool, '')}\n" if step.tool else "")
    + "Provide your instructional message to the student."
)
```

#### After
```python
teacher_prompt = (
    f"Objective: {step.objective}\n"
    f"Instruction: {step.instruction}\n"
    f"Expected Output: {step.expected_output}\n"
    f"Context:\n{context_text}\n"
    + ((f"Required Tool: {step.tool}. You MUST invoke this tool to compile/run the code in the sandbox; do not describe expected output.\n"
        if step.tool == "CodeInterpreterTool"
        else f"Required Tool: {step.tool}. Use this tool when needed.\n")
       if step.tool else "")
    + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step.tool, '')}\n" if step.tool else "")
    + (CODE_TOOL_NUDGE if step.tool == "CodeInterpreterTool" else "")
    + "Provide your instructional message to the student."
)
```

#### 改了什么
1. 把 `Required Tool: ... Use this tool when needed.` 替换为更强的 `You MUST invoke this tool to compile/run the code in the sandbox; do not describe expected output.` —— 仅当 step.tool 是 CodeInterpreterTool 时生效；其他 tool 保留原措辞。
2. 在 `Tool Input Contract` 行之后追加 `CODE_TOOL_NUDGE`，同样只在 CodeInterpreterTool 时拼。

### 改动 3 · `_run_step` 内 non-interactive prompt（约 L1195）

跟改动 2 完全一样的两个修改，只是 prompt 末尾不一样（`"Return a concise execution response."` 保留）。

### 预期影响

| 指标 | 预期方向 | 解释 |
|---|---|---|
| `n_runs_with_invocation` | ↑ 或 ≈ | gold baseline 已 97/99；理论上限就在这里 |
| `real_exec_rate` | ↑ | 旧 baseline 98.6% (204/207)；想把那 3 个 speculative case 修掉 |
| EVR | ≈ | 直接不变；EVR.exec 子项已经在扫 speculative，行为改善对它正向 |
| NDAR | 须警惕 ↓ | 如果 teacher 把 tool 调用前移到 first probe，就泄漏答案。`CODE_TOOL_NUDGE` 第 3 条加了 "Pedagogical ordering" 来防 |
| SPR | ≈ | scaffolding 顺序不直接受影响 |
| PAS | ≈ | profile 适配跟 tool 调用无关 |
| r_sol | 须警惕 ↓ | 同 NDAR，怕 perfect_from_start。Pedagogical ordering 是关键防线 |

**实测前，这些都是猜测**。要在改动后跑一遍 stage3 + Tier 2 才能验证。

### 与旧 baseline 的对比口径

旧 baseline（`Evaluation/tool_call_report.json`，99 个 gold run）是 **此次改动前** 的 gpt-4o-mini 行为。

```
plan contains CodeInterpreterTool step:  97/99  (98.0%)
runtime actually invoked the tool:       97/99  (98.0%)
real_exec rate:                          204/207 (98.6%)
invocation_rate:                         207/170 (121.8%)
```

改动后重新跑同一批 99 个 plan，按相同口径出新数字，对比即可。

### 回滚方式

如果发现新 prompt 导致 NDAR / r_sol 显著下降（或别的非预期回归）：

1. 删除 `CODE_TOOL_NUDGE` 常量定义
2. 在改动 2 / 3 的两处把多行三元表达式改回 `(f"Required Tool: {step.tool}. Use this tool when needed.\n" if step.tool else "")`
3. 去掉两处的 `+ (CODE_TOOL_NUDGE if step.tool == "CodeInterpreterTool" else "")`

Git diff 可以用 `git log -p stage3_execution/plan_mapper_fixed/runtime.py` 翻到本次 commit 直接 revert。

### 未触动 / 故意不改的部分

- `TOOL_ARG_HINTS[CodeInterpreterTool]` 维持原状（已经有 "ALWAYS call this tool with real code" 一句，跟新 nudge 互不冲突）
- `Objective / Instruction / Expected Output / Context` 4 行不动
- 学生 agent backstory (L916)、`_run_step` 中非 tool 分支的 prompt 都不动
- plan 生成 prompt（不在这个文件里）不动 —— 那条线需要重训才生效，本次不动

---
