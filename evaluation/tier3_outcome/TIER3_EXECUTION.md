# Tier 3 執行說明

## 目前結論

新版 Tier 3 應該分成兩條線：

1. **Sati. / JCC：主實驗指標**
   - 使用 profile-conditioned pairwise LLM judge。
   - 比較 `candidate plan` vs `gold plan`。
   - 不告知 judge 哪個是 gold。
   - AB / BA 各跑一次，消除 position bias。
   - GPT-5、Claude Opus 4.6、Gemini 3 Pro 都跑，最後用 M-judge product 聚合。

2. **Atomic scoring：診斷與 appendix**
   - 使用 one plan x one atomic criterion。
   - LLM 只輸出 labels，不輸出數字。
   - 用來看 candidate 到底在哪些 dimension 輸或贏。

之前的 `tier3_atomic_judge.py` 沒有錯，但它不等於你現在定義的 Sati.。Sati. 要用 `tier3_pairwise_eval.py`。

```text
candidate = GPT-5 generated plan
gold = multi_agent_dataset_filtered_qap.jsonl 裡面的 output
```

## Sati. 定義

Sati. 是 learner preference metric：

```text
Sati. = candidate 被偏好的比例 + 0.5 * Tie 比例
```

每個 judge 對每個 item 做 pairwise choice：

```json
{"choice":"A","justification":"<= 25 words"}
```

`choice` 只能是：

```text
A / B / Tie
```

Judge prompt 會用四個 criteria，而且按優先級判斷。這四個 criteria 不是 atomic scores，而是由 `logic.md` 的研究內容壓縮成四個 **judgment frameworks**：

1. Skill Match：看 mastery、prerequisite、ZPD fit。嚴重 prerequisite gap 會壓過後面的小優勢。
2. Engagement & Learnability：看 learner 是否容易進入、理解、持續跟隨，以及 cognitive load 是否可控。
3. Structural Appropriateness：看 step dependency、actionability、completion condition、observable checkpoint。
4. Personal Relevance：看 profile 是否真的改變 examples、pace、tools、constraints、learning route，而不是表面提到 profile。

注意：這裡不是四個 dimension 分開算分，而是在同一個 pairwise prompt 裡按優先級做整體偏好判斷。LLM 只輸出 `A` / `B` / `Tie`，不輸出 dimension score。

## AB / BA 規則

每個 item 每個 judge 都跑兩次：

| Order | Plan A | Plan B |
|---|---|---|
| AB | candidate | gold |
| BA | gold | candidate |

然後把 A/B choice 轉回 underlying plan：

```text
AB 選 A = candidate
AB 選 B = gold
BA 選 A = gold
BA 選 B = candidate
```

如果 AB 和 BA 指向同一個 underlying plan，就算該 plan win。  
如果 AB / BA 不一致，就計為 Tie。

## M-judge Product

每個 judge 先各自算 Sati.：

```text
Sati_j = candidate_win_rate_j + 0.5 * tie_rate_j
```

三個 judge 的總分不用平均，而用 product：

```text
Sati_M = Sati_GPT5 * Sati_Claude * Sati_Gemini
```

這符合你說的 Aligning-Pedagogy M-judge product 邏輯：只要其中一個 judge 明顯不喜歡 candidate，整體分數會掉很多。

## JCC 定義

JCC 是 Judge Cross-Consistency，用來當 Sati. 的可信度證明。

目前 script 會輸出兩個值：

1. `three_way_agreement_rate`
   - 三個 judge 在同一個 item 上的 stable label 是否完全一致。
   - stable label 是 AB / BA collapse 後的 `candidate` / `gold` / `tie`。

2. `krippendorff_alpha_nominal`
   - nominal Krippendorff alpha。
   - label set 是 `candidate` / `gold` / `tie`。

判讀規則：

```text
alpha < 0.5 -> Sati. 整體不可信，需要重做 rubric
```

```
cd <path-to-repo>/evaluation/tier3_outcome
$env:A8_API_KEY = "<your-a8-api-key>"
```

## Smoke Test：先生成 5 筆 candidate

Workspace 一開始只有 gold dataset，所以要先生成 GPT-5 candidate file。

```powershell
$env:A8_API_KEY = "<your-a8-api-key>"
$PY = "<path-to-your-python>"

& $PY .\tier3_generate_candidates.py `
  --source-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-file .\gpt5_candidate_plans.jsonl `
  --limit-items 5 `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --model gpt-5
```

## Smoke Test：產生 Sati. prompts

不呼叫 API，只先產生 prompts 檢查格式：

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_pairwise_eval.py `
  --candidate-file .\gpt5_candidate_plans.jsonl `
  --gold-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-dir .\tier3_runs\sati_smoke5 `
  --candidate-key output `
  --gold-key output `
  --limit-items 5 `
  --write-prompts
```

5 筆 item 會產生：

```text
5 items x 2 orders = 10 prompts
```

如果跑 3 個 judge models，就是：

```text
10 prompts x 3 judges = 30 API calls
```

## Smoke Test：跑 Sati. + JCC

```powershell
$PY = "<path-to-your-python>"
& $PY -m pip install openai

& $PY .\tier3_pairwise_eval.py `
  --candidate-file .\gpt5_candidate_plans.jsonl `
  --gold-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-dir .\tier3_runs\sati_smoke5 `
  --candidate-key output `
  --gold-key output `
  --limit-items 5 `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --judge-models gpt-5 claude-opus-4-6 gemini-3-pro-preview `
  --run-openai
```

輸出：

```text
tier3_runs/sati_smoke5/sati_pairwise_prompts.jsonl
tier3_runs/sati_smoke5/sati_pairwise_results.jsonl
tier3_runs/sati_smoke5/sati_summary.json
tier3_runs/sati_smoke5/sati_summary.md
```

## Atomic 診斷流程

如果你還想知道 candidate 在哪個 dimension 輸，可以再跑 atomic scoring。這不是 Sati. 主指標。

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_atomic_judge.py `
  --candidate-file .\gpt5_candidate_plans.jsonl `
  --ground-truth-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-dir .\tier3_runs\atomic_smoke5 `
  --candidate-key output `
  --ground-truth-key output `
  --limit-items 5 `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --judge-models gpt-5 claude-opus-4-6 gemini-3-pro-preview `
  --run-openai
```

然後聚合 atomic labels：

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_atomic_aggregate.py `
  --results-file .\tier3_runs\atomic_smoke5\atomic_judge_results.jsonl `
  --out-dir .\tier3_runs\atomic_smoke5 `
  --alpha 0.5
```

## 正式跑完整資料

確認 smoke test 的 JSON 格式、成本、latency 和 JCC 都可以接受後：

1. 重新生成完整 candidate file，不加 `--limit-items 5`。
2. 跑完整 Sati. / JCC。
3. 如果需要診斷，再跑 atomic scoring。

正式 Sati. command：

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_pairwise_eval.py `
  --candidate-file .\gpt5_candidate_plans_full.jsonl `
  --gold-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-dir .\tier3_runs\sati_full `
  --candidate-key output `
  --gold-key output `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --judge-models gpt-5 claude-opus-4-6 gemini-3-pro-preview `
  --run-openai
```

## 現在已修正的違規點

- 舊 pairwise script 把四個 criteria 拆成四個 dimension prompts，已改成單一 Sati. pairwise prompt。
- 舊 script 使用 `gpt5` / `baseline` 命名，容易把 candidate 和 gold 講反，已改成 `candidate` / `gold`。
- 舊 script 使用 OpenAI Responses API，已改成 A8-compatible `chat.completions.create`。
- 舊 script 只支援單一 judge model，已改成支援 `--judge-models`。
- 舊 script 沒有 JCC，已新增 3-way agreement rate 和 nominal Krippendorff alpha。

## Update: evaluate through interaction

最新版本的 Tier 3 Sati. / JCC 仍然使用同一套 pairwise protocol：

```text
candidate plan vs gold plan
AB / BA reversal
candidate / gold / tie collapse
Sati. = candidate_win_rate + 0.5 * tie_rate
JCC = three-judge consistency over collapsed labels
```

唯一改動是：judge 不再只看 static plan，而是優先看 interaction evidence。這個設計是從 Tier 2 evaluator 借來的思想：更像人類評估時，不只看教案寫得好不好，而是看 learner 實際經歷了什麼互動，例如 teacher utterance、student response、follow-up adaptation、feedback、validation、consolidation。

如果沒有 interaction transcript，script 仍然可以跑；prompt 會明確告訴 judge：目前沒有 observed transcript，只能根據 plan 裡明確寫出的互動機會判斷，不要獎勵空泛承諾。

### With interaction logs

如果 candidate 和 gold 都有 interaction / execution log JSONL，可以這樣跑：

Interaction file 可以是兩種格式：

```text
JSONL: 每一行對應一個 item 的 interaction record
JSON array: 單一 item 的 execution_log，例如 Tier 2 runs/run-.../execution_log.json
```

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_pairwise_eval.py `
  --candidate-file .\gpt5_candidate_plans.jsonl `
  --gold-file .\multi_agent_dataset_filtered_qap.jsonl `
  --candidate-interaction-file .\candidate_interactions.jsonl `
  --gold-interaction-file .\gold_interactions.jsonl `
  --candidate-interaction-key execution_log `
  --gold-interaction-key execution_log `
  --out-dir .\tier3_runs\sati_interaction_smoke5 `
  --candidate-key output `
  --gold-key output `
  --limit-items 5 `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --judge-models gpt-5 claude-opus-4-6 gemini-3-pro-preview `
  --run-openai
```

### Without interaction logs

如果暫時沒有 transcript，可以照舊跑：

```powershell
$PY = "<path-to-your-python>"

& $PY .\tier3_pairwise_eval.py `
  --candidate-file .\gpt5_candidate_plans.jsonl `
  --gold-file .\multi_agent_dataset_filtered_qap.jsonl `
  --out-dir .\tier3_runs\sati_smoke5 `
  --candidate-key output `
  --gold-key output `
  --limit-items 5 `
  --base-url https://api.a8.hk/v1 `
  --api-key-env A8_API_KEY `
  --judge-models gpt-5 claude-opus-4-6 gemini-3-pro-preview `
  --run-openai
```

這時 prompt 會在 Plan A / Plan B 後面加入：

```json
{
  "status": "not_provided",
  "instruction": "No observed transcript was provided. Judge only explicit interaction opportunities in the plan."
}
```

所以 Sati. 和 JCC 的計算方式不變，但 judge 的 evidence hierarchy 更接近 human evaluation：有真實互動就看真實互動；沒有真實互動才退回 plan 裡明確設計出的互動。
