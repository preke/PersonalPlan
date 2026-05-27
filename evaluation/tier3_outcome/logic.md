# Tier 3 学习者偏好评估逻辑

## 0. 当前定义

Tier 3 现在有两个层次：

1. **Sati. — Profile-Conditioned Satisfaction**
   - 主指标。
   - 使用 LLM pairwise judge 比较 `candidate plan` 和 `gold plan`。
   - Judge 不知道哪个是 gold。

2. **JCC — Judge Cross-Consistency**
   - 可信度指标。
   - 检查 GPT-5、Claude Opus 4.6、Gemini 3 Pro 三个 judge 是否一致。

Atomic scoring 可以保留作为 appendix 或 error analysis，但不再是 Sati. 的主计算方式。

```text
candidate = GPT-5 generated plan
gold = multi_agent_dataset_filtered_qap.jsonl output
```

## 1. Sati. 定义

Sati. 衡量 learner profile 条件下，judge 是否更偏好 candidate plan。

```text
Sati. = candidate 被偏好的比例 + 0.5 × Tie 比例
```

其中：

- candidate win = AB / BA collapse 后都指向 candidate。
- gold win = AB / BA collapse 后都指向 gold。
- Tie = judge 明确选 Tie，或 AB / BA 结果不一致。

## 2. Pairwise Judge Prompt

每个 prompt 比较同一 learner、同一 query 下的两个 plans。

Judge 只看到：

```text
Plan A
Plan B
learner profile
query
```

Judge 不会被告知：

```text
which plan is candidate
which plan is gold
```

输出格式：

```json
{"choice":"A","justification":"<= 25 words"}
```

`choice` 只能是：

```text
A / B / Tie
```

LLM 不输出分数，不做 Sati. 计算。

## 3. 四个 Criteria 如何压缩成 Sati. Prompt Framework

Sati. prompt 使用四个 criteria，但它们不是四个独立分数，也不是 atomic labels。  
现在的做法是：把每个 dimension 的 research logic 压缩成一个 **judgment framework**，让 LLM 知道该看什么证据，再做 A / B / Tie。

也就是说：

```text
logic.md 的研究依据 -> pairwise prompt 的判断框架
atomic labels -> 不进入 Sati. prompt
numeric formulas -> 不进入 Sati. prompt
```

这样可以保留研究依据，同时避免 prompt 变成一张复杂评分表。

为了让 judge 更一致，Sati. prompt 现在采用 **first decisive criterion** 规则：LLM 先在四个 criteria 内部比较 Plan A / Plan B，再按照 priority order 找第一个有实质差异的 criterion。

```text
Skill Match -> Engagement & Learnability -> Structural Appropriateness -> Personal Relevance
```

如果 Skill Match 已经有明显差距，就由 Skill Match 决定；如果 Skill Match 接近，再看 Engagement；如此往下。若没有任何 criterion 有明确优势，或两个 plan 的优势互相抵消，才选 Tie。

目前 prompt 不加入固定 calibration examples，避免把 judge 推向某一種教學風格。它只定義每個 criterion 什麼算有效 evidence、什麼不算有效 evidence，然後讓 judge 根據 priority order 做 pairwise choice。

這次 prompt refinement 的重點，是把 judge 的評估視角從「哪個答案更直接、更完整」拉回「哪個是更好的 learner-specific tutoring plan」。因為 candidate plan 常常會用比較直接、比較工程化、選項更多的方式回答問題，LLM judge 容易把這些表面特徵誤判成更好的 Skill Match。新的 prompt 因此明確規定：如果兩個 plan 都有 plausible 的解題路徑，不能只因為某個 plan 更直接、選項更多、JSON 更大、看起來更像完整解答就讓它贏；應該比較哪個 plan 更能讓 learner 理解原因、驗證結果、修正錯誤心智模型，並在未來相似問題中遷移使用。

這不是直接告訴 judge 偏好 gold，而是把 evaluation target 定義清楚：Tier 3 的 Sati. 衡量的是 learner profile 條件下的教學滿意度，而不是 technical answer ranking。只有當其中一個 plan 明顯錯誤、不安全、或沒有回答 query 時，technical correctness 才優先決定勝負；如果兩者都可用，勝負應該由教學路徑品質決定。

在描述「什麼算更好」時，prompt 會把 gold plan 常見的教育強項納入各 dimension，而不是另外新增 dimension：

- Skill Match 不只看答案是否貼近 query，也看 plan 是否先診斷或 elicite learner 的 current mental model，避免把錯誤起點當作已知。
- Skill Match 不只看答案是否貼近 query，也看 plan 是否先診斷或 elicite learner 的 current mental model，避免把錯誤起點當作已知。若 plan 只是列出很多 advanced options，但沒有幫 learner 根據能力與限制選擇，不能自動算更好。
- Engagement & Learnability 不只看是否清楚，也看是否讓 learner 透過 prediction、small attempt、reflection 參與學習。要求 learner 先預測、解釋或嘗試，不應被視為拖慢流程；如果這些步驟用於診斷和 feedback，它們反而是 learnability 的正面證據。
- Structural Appropriateness 不只看步驟是否可執行，也看是否有 attempt -> check -> feedback -> revise 的 learning loop。官方文件檢索、code execution、test、compiler/runtime check 等工具如果能讓解釋更可靠，也屬於結構上的有效 evidence。
- Personal Relevance 不只看是否提到 profile，也看 learner background 是否真的改變 analogy、explanation、pace、feedback，並支持 future transfer。把 learner 的工具或背景名稱塞進 generic checklist 不算真正個性化。

因此，新的 decision rule 先做 fatal failure check：如果一個 plan 明顯錯誤、不安全或沒有回答問題，選另一個；否則把兩個 plan 當成 tutoring plans 比較。更好的 plan 應該能建立「目前理解 -> 診斷/預測 -> 解釋 -> 嘗試 -> 驗證 -> feedback -> 遷移」的 learner-specific path，而不是只把 final answer 或多個 possible fixes 交給 learner。

# 四準則的原子判斷設計（v2.0）

四個準則對應四種不同的失敗模式，因此採用四種不同的聚合策略——這個對應本身是本工作的方法學貢獻。

**全域映射規則（除非另行指定）**：A = 1.0 / B = 0.5 / C = 0.0。LLM 僅輸出標籤 + ≤30 字理由，數值由 deterministic 程式碼映射，理由見階段 3。

---

## 準則一：Skill Match（起點與技能的匹配）

**理論依據**：Vygotsky 的 ZPD 模型在教學設計上的三個獨立檢查點：

- 1a 處理 ZPD 的下界——當前能力線在哪裡
- 1c 處理 ZPD 的上界——能達到的能力線在哪裡
- 1b 處理 下界到上界之間的橋——中間的依賴鏈有沒有斷

任何兩個都無法蘊涵第三個。Plan 1（R coalesce）在 1a 上完美（假設了 SQL coalesce、for-loop，這些 learner 都會），1c 上也好（Reduce 折疊對熟悉 loop 的人是「跨得過去的下一格」），但1b 的某些版本可能在「monoid 結合性」這個隱藏假設上斷掉——learner 知道 loop、也能跟到 Reduce，但 plan 沒處理「為什麼 Reduce 對 coalesce 這個運算合法」這個 prereq。三個維度真的是獨立的。

### 原子判斷

1a — Skill Floor Alignment（能力下界對齊）：plan 預設的 learner 已具備能力，是否確實落在 profile 顯示的能力之內？換句話說，plan 有沒有把 learner 不會的東西當作他「應該已經會」？

1b — Prerequisite Coverage（前置條件覆蓋）：plan 在新概念引入前，是否把該概念依賴的所有 prereq 都已經建立或明確處理？這不是評「learner 會不會」，而是評「plan 內部的知識依賴鏈有沒有斷裂」。

1c — ZPD Positioning（近側發展區定位）：plan 的目標難度，是否落在 learner「需要協助才能達到、但有協助就達得到」的那一格？太低（learner 已經會）和太高（即使有協助也達不到）都失敗。

D 級的合法性來自 desirable difficulties (Bjork & Bjork, 2011)；A 比 E 高，因為「太簡單」只是浪費，「太難」會導致 learner 退出 (Sweller et al., 2019)。

---

## 準則二：Engagement & Learnability（可跟隨性）

**理論依據**：直接對應 ARCS（Keller, 1987）與 CLT（Sweller et al., 2019）兩個框架的交集：

2a 對應 ARCS 中的 Attention + Relevance——learner 願不願意開始讀、讀下去
2b 對應一個常被混進 clarity 與 motivation 但實際上獨立的維度——內容是否可解碼
2c 對應 CLT 中的 extraneous + intrinsic load 控制——讀的時候大腦會不會爆
三者獨立的證據在語料中很清楚。Plan 2（HttpHandler）的 2a 還可以（hook 訴諸了 learner 的 ASP.NET MVC 實際工作場景），但 2b 在「private internal data structure」這種措辭上塌陷——learner 看到「internal」會以為自己被要求做不該做的事。動機沒問題，是清晰度的問題。Plan 3（C++ magic_enum）相反：2b 還行（術語都有定義），但 2c 在「PRETTY_FUNCTION + 模板特化 + constexpr substring」三疊一句時塌陷——清晰度沒問題，是負荷的問題。

### 原子判斷

2a — Attention & Motivation Hooks（注意力與動機鉤子）：plan 是否在開頭與關鍵節點，給了 learner「為什麼這值得學」的具體理由？hook 是否與 learner 的真實情境有實質連結，而不只是套用「想像你是…」的泛泛開場？

2b — Clarity of Explanation（解釋清晰度）：plan 中使用的術語、符號、概念引用，是否在當下的 step 對「這個 learner」是可解讀的？沒解釋的關鍵術語、未定義的縮寫、跳躍的記號，都是 clarity 失敗。

2c — Cognitive Load Management（認知負荷管理）：每個 step 的資訊密度是否在合理範圍？多概念是否被適當 chunking？exemplar 是否簡化到 minimal working example？

---

## 準則三：Structural Appropriateness（結構可執行性）

**理論依據**：這個三分法對應 4C/ID（van Merriënboer & Kirschner, 2018）對「學習任務作為可執行流程」的三個獨立要求：

3a 是單步合法性——這一步本身能不能做
3b 是步間合法性——前一步到下一步能不能銜接
3c 是過程合法性——learner 在過程中能不能自我矯正
三者獨立。Plan 6（C# tail reader）的 3a 大致過關（每個 step 有動詞），3b 也過關（順序合理），但 3c 在「verify they can reconstruct」這種訴諸內部狀態的 checkpoint 上塌陷——單步和序列都對，是 checkpoint 失效。一個可能的反例是：plan 可以每個 step 都有極佳 checkpoint（3c 滿分）、單步都 actionable（3a 滿分），但 step 5 的 prereq 在 step 7 才被引入（3b 塌陷）。

### 原子判斷（三項）

3a — Step Actionability（步驟可執行性）：每個 step 是否符合 Mager 三要件——明確的動詞、明確的對象、可觀察的完成條件？learner 讀完該 step 是否知道「現在要做什麼」「做完了長什麼樣」？

3b — Sequence & Dependency Integrity（序列與依賴完整性）：step 之間的順序是否反映知識依賴的拓樸？有沒有後 step 引用了前 step 尚未建立的概念（dependency inversion）？

3c — Checkpoint & Feedback Loop（查核點與回饋循環）：plan 是否在關鍵節點設置了可讓 learner 自我檢驗的 checkpoint？checkpoint 的判準是否客觀（learner 自己知道自己過了沒），還是訴諸不可觀察的內部狀態（「verify they understand」）？

---

## 準則四：Personal Relevance（個性化真實性）

**理論依據**：Aptitude-Treatment Interaction (Cronbach & Snow, 1977)、反事實公平性文獻對「敏感屬性是否真的被使用」的二值判定 (Kusner et al., 2017)。本準則的核心命題：真正的個性化是 plan 與 profile 之間有可觀察的 interaction，而非 plan 提到了 profile。


4a — Profile Reference Density（profile 引用密度與精確度）：plan 引用了 learner profile 中多少屬性？引用是否精確（不杜撰 profile 中沒有的屬性，不誤解既有屬性）？

4b — Counterfactual Sensitivity（反事實敏感度）：如果把 learner profile 換成「同類但不同細節」的另一個 learner，plan 的實質結構（step 順序、舉例選擇、難度梯度、節奏）會不會改變？這是判別 sprinkling 與真個性化的核心測試。

4c — Constraint Compliance（約束遵守）：profile 中明確的禁區、偏好、約束（「不要用數學記號」「請用 OOP 風格」「避免某語言」）是否被遵守？

---



## 5. AB / BA Position Bias 控制

每个 item 每个 judge 跑两次。

| Order | Plan A | Plan B |
|---|---|---|
| AB | candidate | gold |
| BA | gold | candidate |

然后把 A/B choice 转回 underlying plan：

```text
AB chooses A -> candidate
AB chooses B -> gold
BA chooses A -> gold
BA chooses B -> candidate
```

稳定胜负规则：

```text
if AB_label == BA_label:
    stable_label = AB_label
else:
    stable_label = Tie
```

因此：

- AB 选 A，BA 选 B -> 都指向 candidate，candidate win。
- AB 选 B，BA 选 A -> 都指向 gold，gold win。
- AB 选 A，BA 选 A -> 不一致，Tie。
- AB 选 B，BA 选 B -> 不一致，Tie。

## 6. M-judge Product

三个 judge 分别计算 Sati.：

```text
Sati_GPT5
Sati_Claude
Sati_Gemini
```

最终不用平均，而用乘积：

```text
Sati_M = Sati_GPT5 × Sati_Claude × Sati_Gemini
```

原因：

- product 比 average 更敏感。
- 如果一个 judge 明显不偏好 candidate，整体指标会大幅下降。
- 这符合 Aligning-Pedagogy 的 M-judge product 思路。

## 7. JCC — Judge Cross-Consistency

JCC 用来报告 Sati. 的可信度。

每个 judge 对每个 item 先得到 stable label：

```text
candidate / gold / tie
```

然后比较三个 judge 的一致性。

报告两个值：

1. `three_way_agreement_rate`
   - 三个 judge stable label 完全一致的比例。

2. `krippendorff_alpha_nominal`
   - nominal Krippendorff alpha。
   - categories = candidate / gold / tie。

判断规则：

```text
alpha < 0.5 -> Sati. 整体不可信，需要重做 rubric
```

来源对应：

- KELE graduate-student ICC 0.68-0.83。
- Personality-Sim personality categorisation consistency。

## 9. 当前代码对应

主实验：

```text
tier3_pairwise_eval.py
```

负责：

- AB / BA prompt generation。
- A8 OpenAI-compatible API calls。
- GPT-5 / Claude / Gemini 多 judge。
- Sati. per judge。
- M-judge product Sati.。
- JCC three-way agreement。
- Krippendorff alpha nominal。

Candidate generation：

```text
tier3_generate_candidates.py
```

负责：

- 从 `multi_agent_dataset_filtered_qap.jsonl` 的 input 生成 GPT-5 candidate plans。

Atomic diagnostic：

```text
tier3_atomic_judge.py
tier3_atomic_aggregate.py
```

负责：

- 产生 atomic labels。
- 聚合 dimension diagnostic scores。

## 10. 最新 Smoke Test 與 Prompt Refinement 摘要

最新 5-item smoke test 顯示，refined prompt 已經比上一版更穩定，也更符合 Tier 3 的評估目的。這一版結果不是要證明最終實驗已完成，而是用來確認 Sati. pipeline、AB / BA collapse、多 judge 聚合與 JCC 可信度檢查是否正常運作。

最新結果：

```text
GPT-5: candidate win = 0.40, gold win = 0.60, tie = 0.00, Sati. = 0.40
Claude Opus 4.6: candidate win = 0.20, gold win = 0.80, tie = 0.00, Sati. = 0.20
Gemini 3 Pro Preview: candidate win = 0.20, gold win = 0.60, tie = 0.20, Sati. = 0.30
M-judge product Sati. = 0.024
JCC three-way agreement rate = 0.80
Krippendorff alpha = 0.611111
trust flag = ok
```

這代表目前 prompt refinement 有兩個效果：

1. Gold win 明顯增加。這不是因為 prompt 直接告訴 judge 要選 gold，而是因為 prompt 把評估視角從「哪個答案更直接、更完整」校準回「哪個 plan 更像 learner-specific tutoring plan」。
2. JCC 從 unreliable 變成 ok。三個 judge 對大部分 item 的 stable label 更一致，表示 rubric 的可理解性提高。

最新 prompt 的核心邏輯是：

```text
如果其中一個 plan 明顯錯誤、不安全、或沒有回答 query：
    technical correctness 優先，選另一個 plan。
否則：
    把兩個 plan 都當成 tutoring plan 比較。
    比較哪個 plan 更能建立 learner-specific path：
    current understanding -> diagnosis / prediction -> explanation -> attempt -> validation -> feedback -> transfer
```

這個修改特別針對 smoke test 中觀察到的問題：candidate plan 常常比較直接、選項更多、工程化程度更高，因此 LLM judge 容易把 directness、breadth、polished JSON 誤判為 Skill Match。新版 prompt 明確說明：

- 直接答案不自動等於更好。
- 更多 alternatives 不自動等於更好。
- 更大的 JSON 或更多 agents 不自動等於更好。
- 如果 learner prediction、small attempt、reflection 是用來診斷、feedback 或鞏固理解，它們不是拖慢流程，而是 learnability evidence。
- 如果文件檢索、code execution、compiler/runtime check 能提高解釋可靠性，它們是 Structural Appropriateness 的正面 evidence。
- 真正的 Personal Relevance 不是提到 profile，而是 profile 真的改變 analogy、pace、tool use、feedback、practice type 或 transfer route。

因此，這一版 Sati. prompt 更適合用來比較 GPT-5 candidate plan 和 multi-filtered gold plan：candidate 若只是更直接、更完整，不能因此贏；gold 若能更好地診斷 learner、建立 feedback loop、使用 profile 形成教學路徑，judge 應該能把這些視為有效優勢。

## 11. 最新更新：Evaluate Through Interaction

目前 Tier 3 的核心指標仍然是 Sati. 和 JCC，計算方式不變：

```text
Sati. = candidate 被偏好的比例 + 0.5 * Tie 比例
JCC = 三個 judge 在 collapsed stable labels 上的一致性
```

這次只改 evaluation evidence，不改四個 dimensions，也不改 AB / BA、M-judge product、Krippendorff alpha 的計算。

新的想法來自 Tier 2 evaluator：真正像人類評估教學品質時，不應只看 plan 寫得是否完整，而應看 learner 在互動中實際經歷了什麼。Tier 2 會從 `execution_log` 的 `actual_interaction` 裡讀 teacher utterance、student response、agent output，再評估 PAS / PQS / r_sol。Tier 3 現在把這個思想移到 Sati. pairwise judge：

```text
static plan quality
    ↓
interaction-grounded tutoring quality
```

也就是說，judge prompt 現在會同時看到：

```text
Plan A
Plan A interaction evidence
Plan B
Plan B interaction evidence
```

如果有 observed interaction transcript，judge 要優先看互動證據，因為這最接近 learner 實際體驗。重點 evidence 包括：

- teacher 是否根據 learner response 調整下一步；
- learner 是否有機會暴露 confusion 或 current mental model；
- plan 是否讓 learner 做 prediction、small attempt、reflection；
- teacher 是否提供 targeted feedback，而不是直接給 final answer；
- 是否有 validation / code execution / compiler check / test result；
- 最後是否有 consolidation 或 transfer，讓 learner 能把概念用到相似問題。

如果沒有 interaction transcript，script 仍可運行，但 prompt 會明確標記：

```json
{
  "status": "not_provided",
  "instruction": "No observed transcript was provided. Judge only explicit interaction opportunities in the plan."
}
```

這樣做的好處是：Sati. / JCC 的數學計算完全保持一致，但 judge 的判斷更像 human evaluator。它不只是看 plan 聲稱自己會個性化、會互動，而是要求 judge 找到可觀察的互動證據；如果沒有 transcript，就只能看 plan 裡明確寫出的互動設計，不能獎勵空泛承諾。

### 11.1 Tier 2 evaluator 到 Tier 3 Sati. 的具體映射

更仔細看 Tier 2 evaluator，它不是簡單地問「互動多不多」，而是把 interaction 拆成幾種可觀察 evidence：

1. **Instruction Fidelity**
   - 看 plan instruction 和實際 teacher output 是否一致。
   - 同時看 teacher 是否過早教了後面步驟的內容。
   - 對 Tier 3 的意義：這對應到 **Structural Appropriateness**。一個 plan 如果設計了 prediction / feedback / validation，但實際 teacher output 直接跳到答案，這不是好結構。

2. **Workflow Completeness**
   - 看 execution_order 裡的 steps 是否真的有 log。
   - 看 loop 是否正常退出，退出理由是否和內容一致。
   - 看 depends_on 的 upstream output 是否真的被 downstream 使用。
   - 看 tool output 是真實執行結果，還是 speculative language。
   - 對 Tier 3 的意義：這主要對應 **Structural Appropriateness**，也影響 **Skill Match**。因為如果流程斷裂，teacher 就無法根據 learner 的真實狀態調整難度。

3. **Interaction Quality**
   - 對每個 student_response 分類：substantive / shallow / non_participatory。
   - 判斷整體 pattern 是否 reasonable 或 degenerate。
   - 特別重要的是：student 一開始就完美不一定是好事，因為那表示沒有 observable learning gap；真正有教學價值的是 learner early gap -> guided improvement。
   - 對 Tier 3 的意義：這對應 **Engagement & Learnability** 和 **Skill Match**。好的互動不是 learner 說 “thanks”，而是 learner 做 prediction、解釋、寫 code、提出具體疑問，並且 teacher 用這些回應調整下一步。

4. **Content Correctness and Guidance Effectiveness**
   - 先抽出 accepted answer 的 core solution。
   - 檢查 teacher 是否 substantive 地教到 core solution。
   - 檢查 learner response trajectory：early 是否有 gap，middle 是否有 improvement，late 是否能對齊 accepted answer。
   - 重要的是 guidance trace：later improvement 能不能追溯到某個 preceding teacher output。
   - 對 Tier 3 的意義：這是 **Skill Match + Structural Appropriateness + Engagement** 的交叉核心。Tier 3 judge 應該問：這個 plan / interaction 是否真的讓 learner 從原本不懂走到能理解，而不是只把答案貼出來。

5. **PAS / PRR**
   - Tier 2 對每個 teacher utterance 判斷是否反映 learner profile。
   - 對 Tier 3 的意義：這直接對應 **Personal Relevance**。真正個性化不是 plan header 提到 profile，而是 teacher 在實際 utterance 和 feedback 裡使用 learner 的背景、工具、語言、經驗、限制。

6. **NDAR**
   - Tier 2 檢查 teacher utterance 是否直接 reveal accepted answer。
   - 對 Tier 3 的意義：這不是說不能給答案，而是要看答案出現的時機。若 teacher 一開始就 full reveal，learner 沒有 prediction / attempt / feedback 的機會，這會削弱 **Engagement & Learnability**。若 learner 已經嘗試過、或安全/正確性需要直接修正，答案 reveal 可以是合理的。

7. **SPR**
   - Tier 2 檢查 plan 是否有 intro / guide / consolidation phases。
   - 對 Tier 3 的意義：這對應 **Structural Appropriateness**。好的 plan 不只是有很多 steps，而是有：activate prior knowledge -> guided attempt -> consolidate / transfer。

8. **IAR**
   - Tier 2 用 teacher question count / statement count 估計互動性。
   - 對 Tier 3 的意義：這只能作為 weak signal。問題多不一定好，但如果完全沒有問題、沒有 learner attempt，就比較不像 human tutoring。

9. **r_sol**
   - Tier 2 不只看最後 code，還看 learner 在最後 subtask 是否用自己的話展示理解。
   - 對 Tier 3 的意義：這對應 **final understanding / transfer**。Sati. judge 應該偏好能讓 learner 最後解釋原理、邊界與遷移條件的 plan，而不是只得到一段可用 code。

因此，Tier 3 的 interaction-grounded Sati. 不應該把 Tier 2 指標逐個重新計分。正確做法是把 Tier 2 的 evidence model 壓縮進 pairwise judge：

```text
early gap
-> responsive teaching
-> learner action
-> validation / feedback
-> later improvement
-> consolidation / transfer
```

這條 trace 才是 human-like evaluation 的核心。JCC 仍然只計算三個 judge 在 final collapsed labels 上是否一致；Sati. 仍然只計算 candidate 被偏好比例。但每個 judge 做 A / B / Tie 時，必須使用 interaction trace，而不是只看 plan 寫得漂不漂亮。
