#!/usr/bin/env python
"""
Tier 1 §11.2 — LLM-judge metrics (main flow: Pers + Ped).

Implements:
    Pers.   Personalization  = ((mean(SkillMatch, GoalOrient, BgAdapt)) - 1) / 4
    Ped.    Pedagogy         = mean( (PRR-1)/4, NDAR, SPR, (IAR-1)/4 )
                               PRR / IAR via judge (5/3/1 anchored)
                               NDAR via judge on first subtask vs accepted_answer
                                  ({"none":1, "partial":0.5, "full":0})
                               SPR  rule-based: fraction of 6 instructional
                                    phases keyword-detected in steps
                               (NDAR is skipped → /3 fallback when no
                                accepted_answer is available.)

Shared schema / I/O imported from tier1_rule.

Standalone CLI:
    # Probe mode — verify prompts/parsers without calling the API
    .venvs/tier1_eval/Scripts/python.exe tier1_judge.py --probe

    # Real evaluation (needs OPENAI_PROXY_API_KEY in .env)
    .venvs/tier1_eval/Scripts/python.exe tier1_judge.py \\
        --input evaluation_results/baselines/aop/plans.jsonl \\
        --qap   filtered_qap.jsonl \\
        --output evaluation_results/baselines/aop/tier1_judge.csv \\
        --judge gemini-3.1-pro-preview --workers 4 --limit 3

NOTE: Feas.* (DepOrder / TaskDecomp / AgentTool) was retired from the main
flow (2026-05-19 audit): rule TBV/AR/DC cover the same signal at 100% and
OOD experiments showed Feas saturates at ceiling on 5/9 plans.  The
FEAS_RUBRICS dict, _feas_extra_for(), and feas() function are kept
importable for ablation studies but eval_judge() / --gold / JUDGE_COLS /
--probe no longer touch them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tier1_rule import (
    build_gold_map,
    build_qap_lookup,
    get_input,
    get_output,
    load_jsonl,
    outer_pidx,
    outer_qid,
    unwrap,
)


# ============================================================================
# LLM client (lazy)
# ============================================================================

DEFAULT_JUDGE = os.environ.get("TIER1_JUDGE_MODEL", "gemini-3.1-pro-preview")
_OPENAI_CLIENT = None


def _make_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    from openai import OpenAI
    api_key = (os.environ.get("OPENAI_PROXY_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    base_url = os.environ.get("OPENAI_PROXY_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "LLM judges need OPENAI_PROXY_API_KEY (bianxie.ai proxy) or "
            "OPENAI_API_KEY in .env.  Use --probe for offline checks.")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _llm_call(system: str, user: str, judge: str,
              max_tokens: int | None = None) -> str:
    """Single LLM call with NO max_tokens cap.

    Architecture: rubric / calibration / instructions go in the SYSTEM
    message (stable across plans being scored); query + profile + target
    plan go in the USER message (changes per call).  Splitting them
    prevents the judge from confusing WORKED EXAMPLES (calibration plan
    fragments in system) with the TARGET PLAN (real data in user).

    Reasoning models (gpt-5*, o1*, deepseek-r1) emit a hidden reasoning
    chain before the visible JSON answer — a tight max_tokens budget
    truncates the visible content to empty and silently falls back to the
    midpoint anchor, destroying discrimination.  Letting the provider use
    its full per-model output budget keeps both the reasoning and the
    JSON intact.
    """
    client = _make_openai_client()
    kwargs: dict[str, Any] = {
        "model": judge,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    rsp = client.chat.completions.create(**kwargs)
    return (rsp.choices[0].message.content or "").strip()


# ============================================================================
# Response parsing (anchored 5/3/1 + categorical labels)
# ============================================================================

def _strip_thinking(raw: str) -> str:
    """Strip DeepSeek-R1 / o1-style <think>...</think> reasoning blocks
    before JSON extraction (some proxies leave them in, others strip them)."""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def parse_anchor(raw: str, scale_min: int = 1, scale_max: int = 5) -> int:
    """Extract integer score from JSON; clamp to [scale_min, scale_max].

    Full 1-5 Likert (v6 onwards): any of 1, 2, 3, 4, 5 is valid.
    Anchor descriptions exist only for 1 / 3 / 5; scores 2 and 4 are valid
    intermediate values when judge's evidence falls between two anchors.

    Function name kept as `parse_anchor` for backwards compatibility with
    callers, but no longer snaps to a discrete anchor set.
    """
    raw = _strip_thinking(raw)
    m = re.search(r'"score"\s*:\s*(\d+)', raw)
    if not m:
        m = re.search(r"\b([1-5])\b", raw)
    if not m:
        # No parseable integer → return midpoint as conservative default
        return (scale_min + scale_max) // 2
    v = int(m.group(1))
    # Clamp out-of-range values to the scale's endpoints
    return max(scale_min, min(scale_max, v))


def parse_label(raw: str, labels: tuple[str, ...]) -> str:
    """Find which label name appears in the response (JSON or freeform).

    Prefers the canonical JSON keys (reveal / choice / score); falls back to
    raw-text substring match.  Default → last label (most conservative for
    NDAR: 'full' = worst possible)."""
    low = _strip_thinking(raw).lower()
    m = re.search(r'"(?:reveal|choice|score)"\s*:\s*"([a-z]+)"', low)
    if m and m.group(1) in labels:
        return m.group(1)
    for lab in labels:
        if lab in low:
            return lab
    return labels[-1]


def llm_judge_anchor(system: str, user: str,
                     judge: str = DEFAULT_JUDGE) -> int:
    return parse_anchor(_llm_call(system, user, judge))


def llm_judge_label(system: str, user: str,
                    judge: str = DEFAULT_JUDGE,
                    labels: tuple[str, ...] = ("none", "partial", "full")
                    ) -> str:
    return parse_label(_llm_call(system, user, judge), labels)


# ============================================================================
# Prompt template v4 (§11.8 five-section structure + Aligning-Pedagogy
#   DO/DON'T lists + worked examples + reasoning-first JSON)
# ============================================================================
#
# Theoretical grounding for the v4 design (cited in the prompt's SYSTEM line):
#
#   - 5/3/1 anchored Likert + per-sub-dimension independent calls
#       Peng et al. (2025) "KELE", EMNLP Findings 2025 — multi-turn evaluation
#       uses 5-point Likert with GPT-4o as judge, requiring "scores AND
#       detailed analysis to enhance interpretability"; ICC 0.68-0.83.
#       Liu et al. (2024) "Personality-Sim" — multi-aspect independent
#       categorization avoids halo effects.
#
#   - DO / DON'T positive-vs-negative bullet lists + worked OK/REJECT
#     examples + reasoning-first JSON output
#       Dinucu-Jianu et al. (2025) "Aligning Pedagogy", EMNLP 2025, Figures
#       14 & 15 — judge prompts use this exact structure (positive bullets,
#       "It is not acceptable to..." negative bullets, 1 OK + 1 REJECT
#       worked examples with annotated JSON answers, reasoning before
#       decision). This is the empirically validated EMNLP 2025 design.
#
#   - Five-section template (SYSTEM / DIMENSION / CONTEXT / TARGET / OUTPUT)
#       Our internal design doc EVALUATION_DESIGN_2026-05-15.html §11.8;
#       per-anchor wording grounded in Plan_Evaluation_Criteria.txt sections
#       cited inline for each dimension.
#
# Key v4 deviations from v2 (now backed by published prior art):
#   - JSON output puts `reasoning` BEFORE `score` (Aligning-Pedagogy Fig 14);
#     v2 used `evidence` array (no published precedent).
#   - Each prompt embeds 2 worked examples (1 score-5, 1 score-1) showing
#     the JSON answer (Aligning-Pedagogy Fig 14 Example 1 + Example 3).
#   - RUBRIC and EXAMPLES appear BEFORE the TARGET PLAN, matching
#     Aligning-Pedagogy SYSTEM-first ordering rather than doc §11.8's
#     CONTEXT-then-RUBRIC ordering (empirically v1/v2/v3 of pure doc-§11.8
#     ordering all hit qwen3-max ceiling on close-pair data).
# Split into SYSTEM (rubric / calibration / instructions — stable across all
# plans being scored) and USER (the actual query / profile / target plan —
# changes per call).  Keeping them as separate OpenAI messages prevents the
# judge from confusing WORKED EXAMPLES (calibration-only plan fragments in
# the system prompt) with the TARGET PLAN (the real data to score in the
# user message).
ANCHOR_SYSTEM = """You are an expert evaluator of multi-agent teaching plans for personalized
programming education. You must apply the rubric strictly and produce a
JSON object only — no markdown, no commentary outside JSON.

Scoring protocol (5-point anchored Likert, every level has its own anchor):
- Output an integer from 1 to 5.
- Each of the FIVE integers has a SPECIFIC anchor description in the
  RUBRIC section below — read them all before deciding.
- Pick the integer whose anchor your evidence MOST CLOSELY matches.
- If the Score-5 conditions are not directly observable in the plan, you
  MUST choose 4 or lower — never approximate upward.

Output protocol (reasoning-first):
- Write your reasoning BEFORE the score in the JSON object, so the
  reasoning genuinely informs the final decision.
- Reason about concrete step_ids / subtask_ids from the plan, not
  abstract impressions.
- Do not use the literal word "score" inside your reasoning text.

DIMENSION: {dim_name}
{definition}

WHAT THIS DIMENSION REWARDS:
{positive_bullets}

WHAT THIS DIMENSION PENALIZES:
{negative_bullets}

RUBRIC (full 5-point anchored Likert):
  Score 5 (Excellent):   {anchor_5}
  Score 4 (Good):        {anchor_4}
  Score 3 (Adequate):    {anchor_3}
  Score 2 (Weak):        {anchor_2}
  Score 1 (Poor):        {anchor_1}

WORKED EXAMPLES:
  Example A — earns Score 5:
{example_5}

  Example B — earns Score 1:
{example_1}

INSTRUCTIONS:
1. Scan the target plan for evidence relevant to this dimension. Note up
   to 3 concrete step_ids or subtask_ids.
2. Read ALL FIVE anchor descriptions in the RUBRIC carefully.
3. Choose the integer 1-5 whose anchor description your evidence MOST
   CLOSELY matches. If the Score-5 conditions are not directly observable,
   you MUST choose 4 or lower — never approximate upward.
4. Write reasoning first (citing step_ids), then the score.

OUTPUT FORMAT (strict JSON, single line, no markdown fences):
  {{"reasoning": "<3-5 sentences citing concrete step_ids; <= 80 words>", "score": <integer 1 to 5>}}
"""

ANCHOR_USER = """CONTEXT (this is the data to score — distinct from the
worked examples in the system prompt):

  Query:           {query}
  Learner profile: {profile}
{extra}
TARGET PLAN (the JSON you must score now):
{plan}

Now write the JSON evaluation for the TARGET PLAN above. Make sure the JSON is valid:
"""


def _fmt_plan(plan: dict) -> str:
    return json.dumps(get_output(plan), ensure_ascii=False)


def _fmt_profile(plan: dict) -> str:
    return json.dumps(get_input(plan).get("learner", {}), ensure_ascii=False)


def _fmt_query(plan: dict) -> str:
    return get_input(plan).get("query") or ""


def _fmt_answer(plan: dict) -> str:
    return get_input(plan).get("accepted_answer") or ""


# ============================================================================
# Pers. — 3 sub-dims (5/3/1 anchored, doc §11.2 + Plan_Evaluation_Criteria.txt)
# ============================================================================
#
# Each sub-dim is a dict with 8 fields consumed by ANCHOR_SYSTEM.format():
#   definition         — 1-sentence semantic definition + citation
#   positive_bullets   — joined bullet list of DO signals
#   negative_bullets   — joined bullet list of DON'T signals
#   anchor_5/3/1       — Likert anchor descriptions (quantified)
#   example_5          — worked plan fragment + JSON answer that earns 5
#   example_1          — worked plan fragment + JSON answer that earns 1
#
# Worked examples follow Aligning-Pedagogy Fig 14's "Example 1 OK + Example 3
# REJECT" pattern — they teach the judge what the JSON output should look
# like, calibrated against extreme score points.
PERS_RUBRICS: dict[str, dict[str, str]] = {
    "SkillMatch": {
        "definition": (
            "Does the plan's starting difficulty and step granularity match "
            "the learner's declared skill level and prior knowledge "
            "(top_tags, knowledge_level, work background)? A good plan "
            "starts where the learner's understanding is likely to end, "
            "neither over-explaining what they already know nor skipping "
            "prerequisites they lack."),
        "positive_bullets": (
            "  - At least 2 learner-facing steps (requires_human_input=true) "
            "use a tag from learner.top_tags as a CONCRETE BRIDGE (e.g., "
            "\"just like Python's naive datetime that you already use, ...\")\n"
            "  - The plan SKIPS basics the learner has clearly mastered\n"
            "  - Difficulty progression is calibrated to the learner level "
            "(beginner → small jumps; senior → fewer hand-holding steps)"),
        "negative_bullets": (
            "  - Plan explains entry-level concepts to an obviously senior "
            "learner (e.g., \"a HashMap stores key-value pairs\" to a "
            "concurrency expert)\n"
            "  - OR jumps over foundational concepts a beginner clearly "
            "needs (gaps in prerequisites)\n"
            "  - Plan reads as a template — replacing the learner with a "
            "different-background developer would NOT require rewriting "
            "any instruction"),
        "anchor_5": (
            "DEEP starting-point adaptation. ≥2 learner-facing step "
            "instructions explicitly leverage a learner.top_tags entry as a "
            "bridge. Difficulty curve matches the inferred level."),
        "anchor_4": (
            "GOOD. 1 learner-facing step explicitly bridges from a "
            "top_tags entry, AND the rest of the plan's difficulty curve "
            "broadly matches the inferred level (no obvious over- or "
            "under-explanation). Falls short of Score 5 because only 1 "
            "(not ≥2) step uses the concrete bridge."),
        "anchor_3": (
            "SHALLOW. Plan mentions the learner's background in 1 step "
            "(\"since you know X, we'll skip basics\") but the subsequent "
            "content is identical to a generic beginner/expert version."),
        "anchor_2": (
            "WEAK. Only an agent backstory or 1 generic phrase references "
            "the learner's background; no step instruction concretely "
            "uses top_tags. The plan would need at most cosmetic rewriting "
            "for a different-background learner."),
        "anchor_1": (
            "NONE. All step instructions are completely generic. No "
            "leverage of top_tags or knowledge_level anywhere; the plan "
            "could be served to any learner unchanged."),
        "example_5": (
            "    Query: \"How do I implement a thread-safe LRU cache in Java?\"\n"
            "    Learner: top_tags=[\"java\",\"spring\",\"concurrency\"], senior\n"
            "    Plan step S1-1 instruction: \"Build on the lock-striping pattern "
            "you already use with ConcurrentHashMap: wrap a LinkedHashMap "
            "(accessOrder=true) with the same striped-lock approach.\"\n"
            "    → {\"reasoning\": \"S1-1 cites learner's known CHM as a bridge "
            "and skips Map basics, matching senior level.\", \"score\": 5}"),
        "example_1": (
            "    Query: same as above. Learner: same (senior Java/concurrency)\n"
            "    Plan step S1-1 instruction: \"First, let's learn what a Map is "
            "in Java. A Map stores key-value pairs. Try printing one.\"\n"
            "    → {\"reasoning\": \"S1-1 explains Map basics to a senior "
            "concurrency expert; instruction is identical to any beginner plan; "
            "top_tags ignored.\", \"score\": 1}"),
    },
    "GoalOrientation": {
        "definition": (
            "Does the plan's main instructional path lead the learner to "
            "the core solution implied by the query, and does the final "
            "subtask define a verifiable end-state? The plan should not "
            "drift to a competing solution or leave the final outcome "
            "vague."),
        "positive_bullets": (
            "  - ≥70% of steps lie on the path toward the query's core "
            "method/concept\n"
            "  - The final subtask sets a CONCRETE END-STATE that matches "
            "what the query asks (e.g., \"learner produces a working X\")\n"
            "  - No subtask teaches a competing solution that occupies "
            "more steps than the correct one"),
        "negative_bullets": (
            "  - Main subtask focus drifts to a DIFFERENT solution than "
            "what the query implies (e.g., query asks about math.isclose "
            "but plan teaches hand-written epsilon comparison in 4 "
            "subtasks)\n"
            "  - Final subtask has a vague endpoint (\"learner improves "
            "understanding\") with no verifiable artifact\n"
            "  - ≥2 detour subtasks not strictly needed for the query"),
        "anchor_5": (
            "All main subtasks (≥70% of steps) advance the query's core "
            "solution; the final subtask sets a verifiable end-state "
            "matching the query."),
        "anchor_4": (
            "≥70% of steps advance the core solution AND the final "
            "subtask sets a verifiable end-state, BUT 1 subtask is "
            "mildly redundant (e.g., over-explains a covered concept) "
            "or 1 minor detour exists. Direction unambiguously correct."),
        "anchor_3": (
            "Main direction correct but 1-2 subtasks introduce alternative "
            "approaches or detours not strictly required."),
        "anchor_2": (
            "Plan partially drifts. Only 40-60% of steps advance the "
            "query's core solution; the final subtask vaguely connects "
            "to the query but the end-state is not fully verifiable "
            "(no concrete artifact produced)."),
        "anchor_1": (
            "Plan's main focus teaches a DIFFERENT solution than the "
            "query implies, OR the final subtask has no verifiable "
            "endpoint."),
        "example_5": (
            "    Query: \"How do I match a date in Python regex?\"\n"
            "    Plan: S1 probes regex basics, S2 explains \\d{4}-\\d{2}-\\d{2} "
            "pattern, S3 has learner write the pattern, S4 verifies with "
            "re.match. Final subtask: \"learner has a working "
            "YYYY-MM-DD matcher.\"\n"
            "    → {\"reasoning\": \"S2-S4 all advance the regex pattern; "
            "final subtask sets a verifiable end-state (working matcher).\", "
            "\"score\": 5}"),
        "example_1": (
            "    Query: same as above (regex date match)\n"
            "    Plan: S1-S3 teach datetime.strptime parsing; S4 mentions "
            "regex in one line. Final subtask: \"learner understands date "
            "handling.\"\n"
            "    → {\"reasoning\": \"3 of 4 subtasks teach strptime (a "
            "different solution); regex relegated to one mention; endpoint "
            "vague.\", \"score\": 1}"),
    },
    "BackgroundAdaptation": {
        "definition": (
            "Are the plan's examples, analogies, and agent backstories "
            "drawn from the learner's stated background (about_me, "
            "work experience), or are they generic textbook framings? "
            "A good plan would read meaningfully differently if the "
            "learner.about_me were replaced with a different background."),
        "positive_bullets": (
            "  - ≥2 step instructions contain a concrete example/analogy "
            "drawn from learner.about_me (e.g., learner = Django dev → "
            "example uses @cache_page as bridge to Spring @Cacheable)\n"
            "  - At least one agent backstory explicitly cites the "
            "learner's expertise or work context"),
        "negative_bullets": (
            "  - All examples are generic textbook scenarios (e.g., "
            "\"imagine a school address book\")\n"
            "  - Agent backstories are template (\"You are a helpful "
            "tutor\") with no reference to the learner\n"
            "  - Nothing in the plan would change if learner.about_me "
            "were replaced with a completely different background"),
        "anchor_5": (
            "≥2 step instructions use concrete examples or analogies "
            "drawn from learner.about_me, AND at least one agent "
            "backstory cites learner-specific context."),
        "anchor_4": (
            "1 step instruction uses a concrete learner.about_me-drawn "
            "example AND 1 agent backstory cites learner context (i.e., "
            "Score-5's two requirements are each half-met). The plan "
            "feels personalized but not as comprehensively as Score 5."),
        "anchor_3": (
            "1 step example OR 1 agent backstory uses learner-specific "
            "framing; the rest of the plan is generic."),
        "anchor_2": (
            "A single agent backstory glances at the learner's "
            "background (e.g., \"helps backend developers\") but NO step "
            "instruction uses a learner-specific example. The plan would "
            "need partial rewriting for a different learner."),
        "anchor_1": (
            "All examples and agent backstories are generic; the plan "
            "reads as a template."),
        "example_5": (
            "    Learner: about_me=\"Django backend dev for e-commerce\"\n"
            "    Plan step S2-1 instruction: \"Recall how Django's @cache_page "
            "decorator wraps a view function — Spring's @Cacheable shares "
            "the same Aspect-Oriented model; we'll adapt your existing "
            "intuition.\"\n"
            "    Agent backstory: \"You bridge the learner's Django caching "
            "experience to Spring's annotation-based AOP.\"\n"
            "    → {\"reasoning\": \"S2-1 uses Django @cache_page as concrete "
            "bridge; agent backstory cites learner's Django context.\", "
            "\"score\": 5}"),
        "example_1": (
            "    Learner: same Django dev as above\n"
            "    Plan step S2-1 instruction: \"A cache is a software design "
            "pattern that stores frequently accessed data in memory.\"\n"
            "    Agent backstory: \"You are a knowledgeable programming "
            "tutor.\"\n"
            "    → {\"reasoning\": \"S2-1 is generic textbook framing; agent "
            "backstory is a template; nothing references Django.\", "
            "\"score\": 1}"),
    },
}


def pers(plan: dict, judge: str = DEFAULT_JUDGE) -> dict[str, float]:
    q, prof, pl = _fmt_query(plan), _fmt_profile(plan), _fmt_plan(plan)
    raw: dict[str, int] = {}
    user_msg = ANCHOR_USER.format(query=q, profile=prof, extra="", plan=pl)
    for dim, rb in PERS_RUBRICS.items():
        system_msg = ANCHOR_SYSTEM.format(dim_name=dim, **rb)
        raw[dim] = llm_judge_anchor(system_msg, user_msg, judge)
    mean = sum(raw.values()) / len(raw)
    return {
        "Pers": (mean - 1) / 4,
        "Pers_SkillMatch": (raw["SkillMatch"] - 1) / 4,
        "Pers_GoalOrient": (raw["GoalOrientation"] - 1) / 4,
        "Pers_BgAdapt": (raw["BackgroundAdaptation"] - 1) / 4,
    }


# ============================================================================
# Feas. — 3 sub-dims (DepOrdering is reference-grounded per doc §11.2;
#                     reference-grounding pattern from Dinucu-Jianu et al.,
#                     Aligning-Pedagogy, EMNLP 2025)
# ============================================================================
FEAS_RUBRICS: dict[str, dict[str, str]] = {
    "DepOrdering": {
        "definition": (
            "Do depends_on, execution_order, and loop conditions form a "
            "logically consistent DAG that an executor could realize? "
            "A good plan has no cycles, all depends_on references point "
            "to existing steps whose outputs are actually consumed, "
            "execution_order respects topology, and any loop condition "
            "references a step inside the same loop so the condition "
            "can update each iteration."),
        "positive_bullets": (
            "  - Every step listed in depends_on actually has its output "
            "consumed in the dependent step's instruction or tool input\n"
            "  - execution_order is topologically consistent: if B "
            "depends_on A, A appears before B\n"
            "  - Loop conditions reference a step INSIDE the same loop "
            "(so the condition can be updated each iteration)\n"
            "  - No cycles in the dependency graph"),
        "negative_bullets": (
            "  - depends_on references a nonexistent step_id\n"
            "  - Cycle in the dep graph (A → B → A)\n"
            "  - Loop condition references a step OUTSIDE the loop — "
            "condition can never change, causing infinite loop or wrong-"
            "direction termination\n"
            "  - execution_order omits some steps or places dependents "
            "before their prerequisites"),
        "anchor_5": (
            "All four positive conditions hold: every depends_on edge is "
            "used and well-formed; execution_order is topologically "
            "consistent; loop conditions are correctly scoped; no cycles."),
        "anchor_4": (
            "All structural conditions hold (no cycles, all depends_on "
            "exist, execution_order topological, loop scope correct) BUT "
            "1 trivial annotation issue exists, e.g., 1 redundant "
            "depends_on edge whose target IS reached via another path. "
            "Plan executable with no real defect."),
        "anchor_3": (
            "Exactly 1 minor defect: 1 unnecessary depends_on entry, OR "
            "1 minor execution_order inconsistency that does not break "
            "runtime, OR 1 dependency crossing a loop boundary "
            "inappropriately. Plan still executable."),
        "anchor_2": (
            "2 minor defects coexist (e.g., 1 unnecessary dep AND 1 "
            "minor execution_order inconsistency), OR 1 defect borders on "
            "fatal (e.g., execution_order omits 1 short helper step that "
            "doesn't break runtime). Borderline executable."),
        "anchor_1": (
            "At least one fatal defect: dependency cycle; OR depends_on "
            "references a nonexistent step; OR loop condition references "
            "a step OUTSIDE its loop; OR execution_order omits required "
            "steps."),
        "example_5": (
            "    Plan with steps S1, S2, S3.\n"
            "    S2.depends_on=[\"S1\"] and S2 instruction starts: \"Using S1's "
            "output...\"\n"
            "    S3.depends_on=[\"S2\"] and S3 instruction starts: \"Building on "
            "S2's verification result...\"\n"
            "    execution_order=[\"S1\",\"S2\",\"S3\"].\n"
            "    → {\"reasoning\": \"All depends_on edges used; "
            "execution_order matches topology; no loop, so loop-scope check "
            "vacuous.\", \"score\": 5}"),
        "example_1": (
            "    Plan: S2.depends_on=[\"S5\"] but S5 does not exist in the "
            "plan; AND execution_order=[\"S2\",\"S1\"] places S2 before its "
            "actual prerequisite S1.\n"
            "    → {\"reasoning\": \"S2.depends_on references nonexistent "
            "S5; execution_order also violates topology (S2 before S1). "
            "Two fatal defects.\", \"score\": 1}"),
    },
    "TaskDecomp": {
        "definition": (
            "Are subtask objectives distinct (non-overlapping), and is "
            "step granularity such that each step is one agent's coherent "
            "action? A good plan avoids two subtasks that teach the same "
            "result state, avoids steps that bundle work needing different "
            "agent capabilities, and avoids steps too vague to execute."),
        "positive_bullets": (
            "  - Every subtask has a DISTINCT objective; no two could be "
            "merged without losing meaning\n"
            "  - Each step is one agent's continuous action (no agent "
            "switching mid-step; no bundling >2 unrelated concepts)\n"
            "  - All key concepts from the query are covered without "
            "redundant repetition"),
        "negative_bullets": (
            "  - Two subtask objectives describe essentially the same "
            "result state (duplicate teaching)\n"
            "  - A step bundles work requiring different agent "
            "capabilities (e.g., \"retrieve docs AND explain to learner\")\n"
            "  - A step instruction is too vague to execute (\"handle the "
            "backend\")\n"
            "  - A key concept from the query is missing entirely"),
        "anchor_5": (
            "Every subtask distinct; step granularity right for its agent; "
            "all key concepts covered without redundancy."),
        "anchor_4": (
            "All subtasks distinct + key concepts covered, but 1 step "
            "is slightly broader than ideal (bundles 2 mildly related "
            "concepts) or 1 minor concept could be split out. Does not "
            "materially affect executability."),
        "anchor_3": (
            "Exactly 1 issue: 1 subtask could be split or merged; OR 1 "
            "step too coarse; OR 2 adjacent same-agent steps that should "
            "be merged. Other parts fine."),
        "anchor_2": (
            "2 issues coexist (e.g., 1 redundant subtask AND 1 vague "
            "step), OR a key concept from the query is covered only "
            "superficially in 1 step. Decomposition is workable but not "
            "clean."),
        "anchor_1": (
            "Subtask objectives substantially overlap (duplicate "
            "teaching); OR multiple steps too vague to execute; OR a key "
            "concept from the query is missing."),
        "example_5": (
            "    Query: \"How do I write a Python decorator?\"\n"
            "    Subtasks: S1 \"learner predicts what @ syntax does\"; S2 "
            "\"learner reads decorator definition and identifies the wrapper "
            "function\"; S3 \"learner writes a logging decorator\"; S4 "
            "\"learner debugs a stateful counter decorator\".\n"
            "    → {\"reasoning\": \"4 distinct objectives (probe / read / "
            "write / debug); each step is one tutor action; covers core "
            "decorator concepts without overlap.\", \"score\": 5}"),
        "example_1": (
            "    Query: same Python decorator. Subtasks: S1 \"learn what a "
            "decorator is\"; S2 \"understand decorators in Python\"; S3 "
            "\"handle the decorator backend\".\n"
            "    → {\"reasoning\": \"S1 and S2 objectives duplicate; S3 "
            "instruction (\\\"handle the backend\\\") is too vague to "
            "execute.\", \"score\": 1}"),
    },
    "AgentToolApprop": {
        "definition": (
            "Are the declared agents distinct in capability, is every "
            "step.tool in the assigned agent's declared tools list, and "
            "does each step's instruction match the agent's capability "
            "domain? A good plan has no two interchangeable agents, no "
            "step using a tool outside its agent's declared tools, no "
            "step asking an agent to act outside its capability, and no "
            "tool use that adds no value over no-tool."),
        "positive_bullets": (
            "  - Every agent has at least one distinguishing capability "
            "(different tools OR distinct domain) — no two agents are "
            "interchangeable\n"
            "  - Every step.tool is declared in that step.agent's tools "
            "list (or step.tool is null)\n"
            "  - Each step's instruction matches the assigned agent's "
            "description (e.g., a docs_retriever agent only does retrieval, "
            "not concept explanation)\n"
            "  - Every tool use measurably improves the step output over "
            "no-tool"),
        "negative_bullets": (
            "  - Two agents have identical tools AND similar work (one "
            "could be removed)\n"
            "  - A step uses a tool NOT in the assigned agent's declared "
            "tools list (tool-binding error)\n"
            "  - An agent is asked to do work outside its declared "
            "capability (e.g., a code_validator asked to explain a "
            "concept)\n"
            "  - A tool is declared on a step whose instruction needs no "
            "such tool"),
        "anchor_5": (
            "All four positive conditions hold across the plan."),
        "anchor_4": (
            "All agents are distinct AND tools are properly declared, "
            "but 1 minor capability stretch (e.g., an agent doing a "
            "small bit just outside its primary capability) OR 1 marginal "
            "tool use that adds limited value. No tool-binding errors."),
        "anchor_3": (
            "Exactly 1 issue: 1 redundant agent, OR 1 step uses an "
            "undeclared tool, OR 1 capability mismatch, OR 1 superfluous "
            "tool use. Other parts fine."),
        "anchor_2": (
            "2 issues coexist but not at \"≥2 same-type errors\" severity "
            "(e.g., 1 redundant agent AND 1 minor capability mismatch, OR "
            "1 tool-binding error AND 1 borderline capability call). "
            "Plan still mostly functional but cleanly imperfect."),
        "anchor_1": (
            "Multiple agents redundant; OR ≥2 tool-binding errors; OR "
            "≥2 capability mismatches at the step level."),
        "example_5": (
            "    Agents: ConceptInstructor(tools=[]), DocsRetriever("
            "tools=[\"CodeDocsSearchTool\"]), CodeRunner(tools=["
            "\"CodeInterpreterTool\"]).\n"
            "    Steps: S1 (ConceptInstructor, tool=null) explains pattern; "
            "S2 (DocsRetriever, tool=CodeDocsSearchTool) fetches API docs; "
            "S3 (CodeRunner, tool=CodeInterpreterTool) executes learner "
            "code.\n"
            "    → {\"reasoning\": \"3 distinct roles; every step.tool is "
            "declared on its agent; instructions match capability.\", "
            "\"score\": 5}"),
        "example_1": (
            "    Agents: TutorA(tools=[\"CodeInterpreterTool\"]) and TutorB("
            "tools=[\"CodeInterpreterTool\"]) — same tools, same work.\n"
            "    Step S2 (TutorA, tool=\"FirecrawlSearchTool\") — tool not "
            "in TutorA.tools.\n"
            "    Step S3 (TutorA, instruction=\"explain the concept to "
            "learner\") — but TutorA description is \"executes student "
            "code\".\n"
            "    → {\"reasoning\": \"TutorA and TutorB are redundant; S2 "
            "uses undeclared tool; S3 instruction is outside TutorA's "
            "declared capability. Three issues.\", \"score\": 1}"),
    },
}


def _feas_extra_for(dim: str, gold: Optional[dict]) -> str:
    """Per doc §11.2 reference code, only DepOrdering gets the gold
    execution_order as REFERENCE; TaskDecomp / AgentToolApprop receive no
    reference.

    Reference-grounded judging design follows Dinucu-Jianu et al.
    (Aligning-Pedagogy, EMNLP 2025): the reference is provided as ONE
    example of a sensible execution_order so the judge can calibrate, NOT
    as a gold standard the candidate must literally match. We explicitly
    instruct the judge that different structure is fine if internally
    consistent — this neutralizes the length-complexity penalty observed
    in early runs where long candidate plans got scored down for simply
    being structurally different from a short reference.
    """
    if gold is None or dim != "DepOrdering":
        return ""
    eo = get_output(gold).get("execution_order", [])
    return ("REFERENCE (one example of a sensible execution_order for THIS "
            "query — your job is NOT to check whether the candidate matches "
            "this reference. Only check the candidate's own internal "
            "consistency. Different structure is FINE if internally valid.):\n"
            "  " + json.dumps(eo, ensure_ascii=False) + "\n")


def feas(plan: dict, gold: Optional[dict] = None,
         judge: str = DEFAULT_JUDGE) -> dict[str, float]:
    q, prof, pl = _fmt_query(plan), _fmt_profile(plan), _fmt_plan(plan)
    raw: dict[str, int] = {}
    for dim, rb in FEAS_RUBRICS.items():
        extra = _feas_extra_for(dim, gold)
        user_msg = ANCHOR_USER.format(query=q, profile=prof, extra=extra,
                                      plan=pl)
        system_msg = ANCHOR_SYSTEM.format(dim_name=dim, **rb)
        raw[dim] = llm_judge_anchor(system_msg, user_msg, judge)
    mean = sum(raw.values()) / len(raw)
    return {
        "Feas": (mean - 1) / 4,
        "Feas_DepOrder": (raw["DepOrdering"] - 1) / 4,
        "Feas_TaskDecomp": (raw["TaskDecomp"] - 1) / 4,
        "Feas_AgentTool": (raw["AgentToolApprop"] - 1) / 4,
    }


# ============================================================================
# Ped. — PRR + NDAR + SPR + IAR
# (doc §11.2 — acronyms ported from Peng et al., KELE, EMNLP Findings 2025
#  but semantically redefined for plan-level evaluation; original KELE PRR/
#  SPR/IAR target single-turn dialogue with binary yes/no judgments)
# ============================================================================
#
# PRR — Pedagogical Rule Reasonableness (subtask sequence)
PED_PRR_RUBRIC = {
    "definition": (
        "Does the subtask sequence follow progressive teaching (diagnose → "
        "explain → practice → feedback → consolidate), with no later step "
        "depending on knowledge that was not taught earlier? A good plan "
        "introduces every key concept BEFORE it is used downstream and "
        "places attempts before feedback steps."),
    "positive_bullets": (
        "  - Plan starts with a probe / diagnose step before any explain "
        "step (e.g., \"ask learner to predict the output\" before showing "
        "code)\n"
        "  - Explain steps precede apply / practice steps for the same "
        "concept\n"
        "  - Attempt steps precede feedback steps\n"
        "  - Each key concept has a comprehension check before being used "
        "downstream\n"
        "  - No subtask depends on a prerequisite concept that was not "
        "introduced earlier in the plan"),
    "negative_bullets": (
        "  - First subtask hands out the complete solution before learner "
        "has any chance to think\n"
        "  - Apply step appears before the explanation that introduces the "
        "concept it uses\n"
        "  - Feedback step appears before any attempt step (nothing to "
        "give feedback on)\n"
        "  - A later subtask uses terminology / APIs never introduced in "
        "earlier subtasks (prerequisite gap)"),
    "anchor_5": (
        "Subtask sequence strictly follows the progressive order; every "
        "key concept has a comprehension check before downstream use; no "
        "prerequisite gaps."),
    "anchor_4": (
        "Sequence is mostly progressive (5 of 6 expected phases present "
        "in correct order); every concept has a comprehension check "
        "before downstream use; 1 minor sequence quirk (e.g., consolidate "
        "appears slightly early) that does not break the learning flow."),
    "anchor_3": (
        "Mostly progressive but exactly 1 subtask out of order (e.g., 1 "
        "apply step before its demonstration); OR 1 key concept lacks a "
        "comprehension check before being used downstream."),
    "anchor_2": (
        "2 sequence issues coexist (e.g., 1 reversed-order step AND 1 "
        "missing comprehension check), OR sequence is approximately right "
        "but contains 1 clear prerequisite gap that the learner could "
        "still potentially overcome. Borderline pedagogically sound."),
    "anchor_1": (
        "Sequence is chaotic: plan starts directly with apply without "
        "explain; OR S1 hands out the complete solution; OR feedback step "
        "appears before any attempt step; OR multiple prerequisite gaps."),
    "example_5": (
        "    Plan subtasks for \"binary search\":\n"
        "    S1 (probe) \"ask learner how to find a number in sorted array\"\n"
        "    S2 (explain) \"introduce the halving idea\"\n"
        "    S3 (apply)   \"learner writes the loop\"\n"
        "    S4 (validate) \"run tests on learner's code\"\n"
        "    S5 (feedback) \"identify off-by-one errors\"\n"
        "    S6 (consolidate) \"summarize O(log n) takeaway\"\n"
        "    → {\"reasoning\": \"S1-S6 follow probe→explain→apply→validate→\"\n"
        "       \"feedback→consolidate; every concept introduced before use.\","
        " \"score\": 5}"),
    "example_1": (
        "    Plan subtasks for same binary-search query:\n"
        "    S1 (apply) \"here is the binary search code; copy it\"\n"
        "    S2 (probe) \"ask if learner understood\"\n"
        "    → {\"reasoning\": \"S1 hands out the solution before any probe "
        "or explain; S2 (probe) comes AFTER apply — reversed order.\", "
        "\"score\": 1}"),
}

# IAR — Instructional Adaptation Rate
#
# Theoretical grounding (cite in paper §X.Y, NOT in the prompt text per the
# project's citation-free prompt convention):
#   * Kalyuga, Ayres, Chandler, Sweller (2003) Expertise Reversal Effect —
#     instructional methods optimal for novices become ineffective or
#     counter-productive for experts (and vice versa); a single fixed style
#     cannot serve all learner levels.
#   * Sweller (1988, 2010) Cognitive Load Theory — total load = intrinsic
#     + extraneous + germane.  For experts, redundant scaffolding becomes
#     extraneous load; for novices, missing scaffolding inflates intrinsic
#     load.  Method calibration = managing the balance.
#   * Sweller & Cooper (1985); Renkl (2014) Worked Example Effect —
#     novices learn faster from studied worked examples than from pure
#     problem-solving; reverses for experts (Kalyuga 2007).
#   * Collins, Brown, Newman (1989) Cognitive Apprenticeship — instructional
#     continuum modeling → coaching → scaffolding → articulation →
#     reflection → exploration; different levels sit at different points.
#   * Sentance, Waite, Kallia (2019) PRIMM — Predict / Run / Investigate /
#     Modify / Make, programming-novice-specific scaffolding pattern.
#
# Renamed from "Inductive Aptness Rate" (2026-05-19 audit):
#   - The KELE acronym IAR originally means "Instruction Adherence Rate"
#     (model adheres to consultant agent's instruction in KELE's MAS) —
#     completely unrelated to this metric; we keep the acronym for
#     citation continuity but redefine the expansion.
#   - The previous "Inductive vs. Deductive" binary is too coarse to
#     evaluate adaptation comprehensively.  This redesign uses three
#     observation dimensions (level inference, method calibration,
#     within-plan consistency) directly grounded in the Expertise Reversal
#     Effect.
#
# Three observation dimensions (combined into a single anchor 1-5 score):
#   D1. LEVEL INFERENCE — does the plan correctly infer learner's level
#       w.r.t. the QUERY DOMAIN, using multiple signals (knowledge_level
#       field, top_tags overlap with query, about_me seniority words,
#       query intrinsic complexity, cross-domain signal, sub-skill
#       specificity)?  Overall career seniority alone is not sufficient.
#   D2. METHOD CALIBRATION — given the inferred level, do the teaching
#       moves match what the literature predicts for that level?
#       * target-domain novice → ≥2 worked examples + step-by-step +
#         explicit rules + early probe-then-explain
#       * target-domain expert → problem-first framing + minimal
#         scaffolding + efficient rule statements + direct application
#       * cross-domain transfer → ≥2 steps anchor to source-domain
#         analogies before introducing target-domain rules
#       * sub-skill beginner (expert in language X but novice on this
#         specific library / algorithm) → novice treatment on this
#         sub-skill, even though top_tags cover the language
#   D3. WITHIN-PLAN CONSISTENCY — style coherent across ≥3 subtasks; no
#       arbitrary flips between expert-style and novice-style moves
#       without pedagogical reason.
PED_IAR_RUBRIC = {
    "definition": (
        "Does the plan's instructional method — framing, analogies, "
        "questioning strategy, scaffolding density — adapt to the "
        "learner's expertise level WITH RESPECT TO THE QUERY DOMAIN?  "
        "Replacement test: if you swapped the learner with someone of "
        "opposite expertise on this query's topic, would the framing / "
        "analogies / questioning need to be rewritten?  If not, the "
        "method is not adapted.  Level is inferred from multiple "
        "signals (knowledge_level, top_tags overlap with the query "
        "topic, about_me seniority, cross-domain signal, sub-skill "
        "specificity) — not just overall career seniority."),
    "positive_bullets": (
        "  - TARGET-DOMAIN NOVICE (incl. sub-skill beginner — language "
        "familiar but the specific library / algorithm / API in the "
        "query is unknown) → ≥2 steps use worked examples / "
        "step-by-step / probe-then-explain BEFORE abstract rules\n"
        "  - TARGET-DOMAIN EXPERT → ≥2 steps state the rule / API / "
        "contract FIRST with sparse scaffolding; no re-explanation of "
        "concepts within the learner's stated expertise\n"
        "  - CROSS-DOMAIN TRANSFER (learner's expertise is in a "
        "different domain than the query) → ≥2 steps use source-domain "
        "analogies to bridge to the target domain"),
    "negative_bullets": (
        "  - Plan reads as a template — framing / analogies / "
        "questioning would not need rewriting for any other learner "
        "(replacement test passes = failure)\n"
        "  - Expert over-scaffolded on concepts within their stated "
        "expertise; OR novice / sub-skill beginner given dense "
        "rule-first lectures with no worked example, analogy, or probe\n"
        "  - Style flips arbitrarily between expert-mode and novice-mode "
        "across subtasks with no pedagogical reason"),
    "anchor_5": (
        "Style consistently matches the learner's level w.r.t. the "
        "query domain across ≥3 subtasks (novice / sub-skill beginner: "
        "worked-example + probe-then-explain; expert: problem-first + "
        "minimal scaffolding; cross-domain: source-domain analogies)."),
    "anchor_4": (
        "Style matches the inferred level across ≥3 subtasks (correct "
        "dominant pattern), but 1 step is over- or under-scaffolded for "
        "the learner without reversing the overall calibration."),
    "anchor_3": (
        "Style mostly matches but exactly 1 subtask reverses it (e.g., "
        "one expert-paced step in an otherwise novice plan, OR one "
        "novice-style scaffolded step in an otherwise expert plan)."),
    "anchor_2": (
        "2 subtasks reverse the style, OR the style switches arbitrarily "
        "across 30-50% of subtasks with no pedagogical reason. "
        "Alignment to learner level is partial."),
    "anchor_1": (
        "Style is wrong for the level: target-domain expert is walked "
        "through novice-style examples of concepts in their expertise; "
        "OR novice / sub-skill beginner is given a dense rule-first "
        "lecture from S1 with no worked example, analogy, or probe; OR "
        "the plan ignores level entirely."),
    "example_5": (
        "    Learner: \"Python background; familiar with C but not "
        "modern C++\" (top_tags=[\"python\",\"c\",\"syntax\"]). Query: "
        "convert enum to string in modern C++.\n"
        "    S1 \"In Python, Color.RED.name returns 'RED' — predict "
        "whether C++ enums carry their names at runtime.\"\n"
        "    S2 \"Walk through how magic_enum uses __PRETTY_FUNCTION__ "
        "to recover names that C enums discard.\"\n"
        "    → {\"reasoning\": \"Sub-skill beginner in modern C++ "
        "correctly inferred; Python-analogy probe (S1) + scaffolded "
        "explain (S2) match novice treatment on this sub-skill.\", "
        "\"score\": 5}"),
    "example_1": (
        "    Learner: senior Java dev, 8 yr Spring Boot "
        "(top_tags=[\"java\",\"spring\",\"jvm\"]). Query: how Java "
        "class syntax works.\n"
        "    S1 \"Imagine a class as a cooking recipe — collect "
        "ingredients (fields), follow steps (methods).\"\n"
        "    S2 \"What is a class in Java? It is a blueprint for "
        "objects...\"\n"
        "    → {\"reasoning\": \"Target-domain expert walked through "
        "novice cooking-analogy + Java class basics within stated "
        "expertise — same-domain over-explanation.\", \"score\": 1}"),
}

# NDAR — No-Direct-Answer Rate (first subtask)
# Uses a labeled-output prompt (none / partial / full) rather than Likert
# because doc §11.2's NDAR rubric is itself categorical (5/Hit, 3/Partial,
# 1/Miss → none/partial/full mapping). KELE original NDAR is also a binary
# "Whether the teacher's reply provides overly obvious answers" (Peng et
# al., EMNLP Findings 2025). The 3-label form lets us preserve the
# "partial" middle ground that doc §11.2 rubric explicitly defines.
NDAR_SYSTEM = """You are an expert evaluator of multi-agent teaching plans for personalized
programming education. You must apply the rubric strictly and produce a
JSON object only — no markdown, no commentary outside JSON.

Your task: inspect ONLY the FIRST subtask of the plan and label how much
of the accepted answer's core solution it reveals. The teaching principle
behind this check: revealing the solution in the first subtask
short-circuits the learning process, leaving the learner nothing to
discover. Output protocol: reasoning BEFORE the reveal label, so the
reasoning genuinely informs the final decision.

DIMENSION: NDAR (No-Direct-Answer Rate, first subtask only)
This dimension protects against the "answer-before-explanation"
anti-pattern: a plan should not hand the learner the canonical answer
in its very first subtask, no matter how that answer is dressed up
(direct code, step-by-step algorithm, named library call, etc.).

WHAT THIS DIMENSION REWARDS (a plan demonstrates this when):
  - First subtask probes / scaffolds / asks questions WITHOUT presenting
    the accepted answer's code or solution text
  - Learner is given room to attempt, predict, or reason before any
    answer reveal
  - Any reference to the accepted-answer concept is framed as a goal to
    reach, not a fact to copy

WHAT THIS DIMENSION PENALIZES (a plan fails this when):
  - First subtask's step instruction or expected_output contains verbatim
    (or near-verbatim) accepted-answer code / solution
  - First subtask gives away the complete algorithm step by step
  - Learner has nothing material left to derive after S1

LABEL DEFINITIONS:
  "none"    : First subtask only probes / scaffolds. No accepted-answer
              code or solution text appears, even partially. Learner still
              has to figure things out.
  "partial" : First subtask reveals one or two key sub-ideas of the
              accepted answer (e.g., names the right library / API) but
              stops short of the complete solution. Learner can guess
              most of the rest.
  "full"    : First subtask's step instruction or expected_output verbatim
              or near-verbatim states the accepted answer's core code or
              final solution. Learner has nothing material left to derive.

WORKED EXAMPLES (FOR CALIBRATION ONLY — these are hypothetical fragments
shown to anchor what each label looks like; you MUST NOT label the
example subtasks, only the FIRST SUBTASK that will appear in the next
user message):

  Example A — earns "none":
    Accepted answer: use math.isclose(a, b)
    First subtask S1 instruction: Ask the learner to compare 0.1+0.2 == 0.3 in the REPL and explain what they observe.
    -> {{"reasoning": "S1 prompts learner to discover the float comparison problem; never names math.isclose.", "reveal": "none"}}

  Example B — earns "full":
    Accepted answer: same (use math.isclose).
    First subtask S1 instruction: Tell the learner: write math.isclose(a, b) to safely compare two floats; this is the canonical solution.
    -> {{"reasoning": "S1 verbatim states the accepted answer's core API call; nothing left to derive.", "reveal": "full"}}

INSTRUCTIONS (apply these to the FIRST SUBTASK in the next user message):
1. For every step in the first subtask, extract the instruction and
   expected_output text most relevant to whether it reveals the answer.
   Note specific step_ids.
2. Compare against the accepted answer. Note any verbatim or near-verbatim
   overlap (code snippets, API names, full algorithms).
3. Pick the label whose rubric description your evidence BEST matches.
   If no evidence supports \"full\", you MUST pick \"partial\" or \"none\".
4. Write reasoning first (citing step_ids), then the reveal label.

OUTPUT FORMAT (strict JSON, single line, no markdown fences):
  {{"reasoning": "<3-5 sentences citing concrete step_ids; <= 80 words>", "reveal": "none" | "partial" | "full"}}
"""

NDAR_USER = """REFERENCE — accepted answer (the eventual learning target;
the first subtask must NOT just hand it to the learner):
{accepted}

FIRST SUBTASK (the JSON you must label now — distinct from the worked
examples in the system prompt):
{first_sub}

Now write the JSON evaluation for the FIRST SUBTASK above. Make sure the JSON is valid:
"""

# Doc §11.2 PHASE_KEYWORDS (canonical 4 per phase, do NOT extend).
PHASE_KEYWORDS: dict[str, list[str]] = {
    "probe":                ["ask", "elicit", "predict", "what do you think"],
    "retrieve_demonstrate": ["explain", "show", "demonstrate", "example"],
    "apply":                ["try", "implement", "write", "exercise"],
    "validate":             ["check", "verify", "test", "run"],
    "feedback":             ["correct", "debug", "fix", "review"],
    "consolidate":          ["summarize", "recap", "takeaway", "now you know"],
}


def spr(plan: dict) -> float:
    """SPR ∈ [0,1]: fraction of 6 phases keyword-hit in step instructions.

    Doc §11.2 reference code scans `step["instruction"]` only; we widen to
    objective + subtask_objective + metadata.phase to absorb baselines that
    don't put phase signals in `instruction`.
    """
    blobs: list[str] = []
    for st in get_output(plan).get("subtasks", []) or []:
        blobs.append((st.get("name") or "").lower())
        blobs.append((st.get("subtask_objective") or "").lower())
        for step in st.get("steps", []) or []:
            blobs.append((step.get("instruction") or "").lower())
            blobs.append((step.get("objective") or "").lower())
            ph = (step.get("metadata") or {}).get("phase")
            if ph:
                blobs.append(str(ph).lower())
    text = " ".join(blobs)
    if not text.strip():
        return 0.0
    hit = sum(1 for kws in PHASE_KEYWORDS.values()
              if any(kw in text for kw in kws))
    return hit / len(PHASE_KEYWORDS)


def ped(plan: dict, judge: str = DEFAULT_JUDGE) -> dict[str, float]:
    q, prof, pl = _fmt_query(plan), _fmt_profile(plan), _fmt_plan(plan)
    accepted = _fmt_answer(plan)

    user_msg = ANCHOR_USER.format(query=q, profile=prof, extra="", plan=pl)
    prr = llm_judge_anchor(
        ANCHOR_SYSTEM.format(
            dim_name="PRR (Pedagogical Rule Reasonableness)",
            **PED_PRR_RUBRIC),
        user_msg, judge)
    iar = llm_judge_anchor(
        ANCHOR_SYSTEM.format(
            dim_name="IAR (Instructional Adaptation Rate)",
            **PED_IAR_RUBRIC),
        user_msg, judge)
    spr_val = spr(plan)

    if accepted:
        subs = get_output(plan).get("subtasks") or []
        first_sub = subs[0] if subs else {}
        rev = llm_judge_label(
            NDAR_SYSTEM,
            NDAR_USER.format(
                accepted=accepted,
                first_sub=json.dumps(first_sub, ensure_ascii=False),
            ),
            judge, ("none", "partial", "full"))
        ndar = {"none": 1.0, "partial": 0.5, "full": 0.0}[rev]
    else:
        ndar = float("nan")

    if not math.isnan(ndar):
        ped_val = ((prr - 1) / 4 + ndar + spr_val + (iar - 1) / 4) / 4
    else:
        ped_val = ((prr - 1) / 4 + spr_val + (iar - 1) / 4) / 3
    return {
        "Ped": ped_val,
        "Ped_PRR": (prr - 1) / 4,
        "Ped_NDAR": ndar,
        "Ped_SPR": spr_val,
        "Ped_IAR": (iar - 1) / 4,
    }


# ============================================================================
# Per-plan driver
# ============================================================================

JUDGE_COLS = [
    "Pers", "Pers_SkillMatch", "Pers_GoalOrient", "Pers_BgAdapt",
    "Ped", "Ped_PRR", "Ped_NDAR", "Ped_SPR", "Ped_IAR",
]


def eval_judge(plan: dict, gold: Optional[dict] = None,
               judge: str = DEFAULT_JUDGE) -> dict[str, float]:
    # Feas removed from main flow (2026-05-19); call feas(plan, gold, judge)
    # directly for ablation runs.  `gold` is accepted but currently unused.
    del gold
    row: dict[str, float] = {}
    row.update(pers(plan, judge))
    row.update(ped(plan, judge))
    return row


def inject_accepted_answer(plan: dict, qid: Optional[str],
                           pidx: Optional[int],
                           qap_lookup: Optional[dict]) -> dict:
    """Mutate plan.input.accepted_answer in-place when joinable from QAP."""
    if qap_lookup and qid is not None and pidx is not None:
        ans = qap_lookup.get((qid, pidx))
        if ans:
            plan.setdefault("input", {})["accepted_answer"] = ans
    return plan


# ============================================================================
# --probe mode: offline verification (no API call)
# ============================================================================

_SAMPLE_PLAN = {
    "input": {
        "query": "How do I implement a thread-safe LRU cache in Java?",
        "learner": {
            "about_me": "Senior Java backend dev, 6 yr Spring Boot.",
            "top_tags": ["java", "spring", "concurrency"],
        },
        "accepted_answer":
            ("Use LinkedHashMap with accessOrder=true wrapped in "
             "Collections.synchronizedMap, or use Caffeine's Cache."),
    },
    "output": {
        "agents": [
            {"agent_role": "ConceptInstructor",
             "goal": "Refresh thread-safety primitives.",
             "backstory": "Java concurrency expert.",
             "tools": ["RagTool"]},
            {"agent_role": "CodeDemonstrator",
             "goal": "Build a working LRU.",
             "backstory": "Implements demo code in sandbox.",
             "tools": ["CodeInterpreterTool"]},
        ],
        "subtasks": [
            {"id": "S1", "name": "Concept refresh",
             "subtask_objective": "Recall LinkedHashMap accessOrder + monitor lock.",
             "steps": [
                 {"id": "S1-1", "agent": "ConceptInstructor",
                  "objective": "Ask learner to predict invariant.",
                  "instruction": "Ask: what invariant must hold for LRU?",
                  "tool": None, "requires_human_input": True,
                  "expected_output": "Learner verbalizes the invariant.",
                  "depends_on": []},
             ]},
            {"id": "S2", "name": "Implementation",
             "subtask_objective": "Implement and test the cache.",
             "steps": [
                 {"id": "S2-1", "agent": "CodeDemonstrator",
                  "objective": "Demonstrate skeleton.",
                  "instruction": "Show synchronized LinkedHashMap example.",
                  "tool": "CodeInterpreterTool",
                  "requires_human_input": False,
                  "expected_output": "Runnable code printing cache state.",
                  "depends_on": ["S1-1"]},
                 {"id": "S2-2", "agent": "CodeDemonstrator",
                  "objective": "Apply: learner writes their own.",
                  "instruction": "Try implementing; we will run + verify.",
                  "tool": "CodeInterpreterTool",
                  "requires_human_input": True,
                  "expected_output": "Learner's submission compiles and runs.",
                  "depends_on": ["S2-1"]},
             ]},
        ],
        "execution_order": ["S1-1", "S2-1", "S2-2"],
    },
}


def _probe() -> int:
    """Build all prompts on a sample plan + exercise both parsers + spr.
    Returns 0 on success, 1 on any check failure."""
    failures: list[str] = []
    print("=== tier1_judge --probe (offline verification, no API call) ===\n")

    # 1) Sub-dim prompt assembly (Pers 3 + Ped PRR/IAR 2 + NDAR 1 = 6)
    # Each anchor prompt now produces TWO templates (SYSTEM + USER) since
    # the LLM call is split into role=system / role=user messages to
    # cleanly separate rubric/worked-examples from the target plan.
    # Feas (3 sub-dims) is retired from main flow; tested separately when
    # explicitly invoked for ablation.
    n_built = 0
    user_msg = ANCHOR_USER.format(
        query=_fmt_query(_SAMPLE_PLAN),
        profile=_fmt_profile(_SAMPLE_PLAN),
        extra="", plan=_fmt_plan(_SAMPLE_PLAN))
    for dim, rb in PERS_RUBRICS.items():
        ANCHOR_SYSTEM.format(dim_name=dim, **rb)
        n_built += 1
    for dim, rb in (("PRR", PED_PRR_RUBRIC), ("IAR", PED_IAR_RUBRIC)):
        ANCHOR_SYSTEM.format(dim_name=dim, **rb)
        n_built += 1
    _ = user_msg  # exercised but not asserted on
    NDAR_USER.format(
        accepted=_fmt_answer(_SAMPLE_PLAN),
        first_sub=json.dumps(
            get_output(_SAMPLE_PLAN)["subtasks"][0],
            ensure_ascii=False))
    # NDAR_SYSTEM is a constant template (no placeholders) — exercise it
    # by length check.
    assert "NDAR" in NDAR_SYSTEM
    n_built += 1
    print(f"  [1/4] Built {n_built}/6 system+user prompt pairs — OK")

    # 2) parse_anchor on canonical / edge / no-score responses.
    # v6: full 1-5 Likert — accept any integer in [1, 5], clamp out-of-range.
    cases_anchor = [
        ('{"score": 5, "justification": "good"}', 5),
        ('{"score": 4, "justification": "near top"}', 4),  # v6: 4 is valid
        ('{"score": 3}', 3),
        ('{"score": 2, "justification": "near bottom"}', 2),  # v6: 2 is valid
        ('{"score": 1, "justification": "bad"}', 1),
        ('{"score": 7}', 5),                                # out-of-range → clamp to 5
        ('I think 3 fits best', 3),                         # bare integer match
        ('No JSON, no integer here', 3),                    # midpoint fallback
    ]
    for raw, expected in cases_anchor:
        got = parse_anchor(raw)
        if got != expected:
            failures.append(f"parse_anchor({raw!r}) -> {got}, want {expected}")
    print(f"  [2/4] parse_anchor: {len(cases_anchor)} cases, "
          f"{len(cases_anchor)-len([f for f in failures if 'parse_anchor' in f])} pass")

    # 3) parse_label on the 3 NDAR outcomes
    cases_label = [
        ('{"reveal": "none", "evidence": []}', "none"),
        ('{"reveal": "partial"}', "partial"),
        ('{"reveal": "full"}', "full"),
        ('the answer is partially leaked', "partial"),
    ]
    for raw, expected in cases_label:
        got = parse_label(raw, ("none", "partial", "full"))
        if got != expected:
            failures.append(f"parse_label({raw!r}) -> {got}, want {expected}")
    print(f"  [3/4] parse_label: {len(cases_label)} cases, "
          f"{len(cases_label)-len([f for f in failures if 'parse_label' in f])} pass")

    # 4) SPR rule-based: sample plan should cover probe / demo / apply / validate
    s_val = spr(_SAMPLE_PLAN)
    expected_min = 3 / 6  # at least 3 phases covered by the sample plan
    if s_val < expected_min:
        failures.append(f"spr(_SAMPLE_PLAN) = {s_val:.4f} "
                        f"< expected_min {expected_min}")
    print(f"  [4/4] SPR(sample plan) = {s_val:.4f}  "
          f"(>= {expected_min:.4f} expected: {'OK' if s_val >= expected_min else 'FAIL'})")

    if failures:
        print("\n--- FAILURES ---")
        for f in failures:
            print("  ", f)
        return 1
    print("\nAll offline checks passed.")
    return 0


# ============================================================================
# CLI
# ============================================================================

def _summarize(df: pd.DataFrame) -> None:
    headline = [c for c in ("Pers", "Ped") if c in df.columns]
    if not headline:
        return
    print("\n=== Mean per-metric (Tier 1 §11.2 LLM judges) ===")
    print(df[headline].apply(pd.to_numeric, errors="coerce").mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tier 1 §11.2 LLM-judge main flow: Pers + Ped "
                    "(Feas retired — call feas() directly for ablation).")
    p.add_argument("--probe", action="store_true",
                   help="Offline check: build prompts + run parsers (no API).")
    p.add_argument("--input", help="Plans JSONL.")
    p.add_argument("--gold", default=None,
                   help="Gold JSONL — accepted for backward compatibility "
                        "but currently unused (Feas retired from main flow).")
    p.add_argument("--qap", default=None,
                   help="filtered_qap.jsonl — supplies accepted_answer for "
                        "Ped.NDAR via (question_id, profile_index) join.")
    p.add_argument("--output", help="Output CSV path.")
    p.add_argument("--judge", default=DEFAULT_JUDGE,
                   help=f"LLM judge model id (default {DEFAULT_JUDGE}).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="If --output already exists, skip rows whose "
                        "(question_id, profile_index) are present with "
                        "empty _error; only new/failed rows are recomputed. "
                        "Final CSV contains existing successful rows + "
                        "newly computed rows.")
    args = p.parse_args()

    if args.probe:
        sys.exit(_probe())

    if not args.input or not args.output:
        p.error("--input and --output are required (or use --probe).")

    rows_in = load_jsonl(args.input)
    if args.limit:
        rows_in = rows_in[: args.limit]
    gold_map = build_gold_map(args.gold) if args.gold else {}
    qap_lookup = build_qap_lookup(args.qap) if args.qap else None

    # ---- Resume: load already-successful rows from --output --------------
    # Key = (str(question_id), int(profile_index)); a row is considered
    # "done" only if _error is empty / NaN. Rows with errors are recomputed.
    #
    # Why _norm_key: pandas.read_csv infers all-numeric question_id columns
    # as float64, so "28703241" round-trips as 28703241.0. Without
    # normalization, CSV key ("28703241.0", 1) ≠ JSONL key ("28703241", 1)
    # and resume mis-matches every row.
    def _norm_key(qid_raw, pidx_raw):
        if qid_raw is None or (isinstance(qid_raw, float) and pd.isna(qid_raw)):
            return None
        s = str(qid_raw).strip()
        if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
            s = s[:-2]
        try:
            if isinstance(pidx_raw, float) and pd.isna(pidx_raw):
                return None
            p = int(float(pidx_raw)) if pidx_raw is not None else 0
        except (TypeError, ValueError):
            return None
        return (s, p)

    done_rows: dict[tuple[str, int], dict] = {}
    if args.resume and Path(args.output).exists():
        try:
            prev = pd.read_csv(args.output, encoding="utf-8-sig",
                               dtype={"question_id": str})
            for _, r in prev.iterrows():
                err = r.get("_error", "")
                if pd.isna(err) or not str(err).strip():
                    key = _norm_key(r.get("question_id"), r.get("profile_index"))
                    if key is None:
                        continue
                    row_dict = {k: (None if pd.isna(v) else v)
                                for k, v in r.to_dict().items()}
                    # Normalize the stored qid so downstream writer doesn't
                    # re-introduce ".0" via dtype-mixing.
                    row_dict["question_id"] = key[0]
                    row_dict["profile_index"] = key[1]
                    done_rows[key] = row_dict
            print(f"[resume] loaded {len(done_rows)} successful prior rows "
                  f"from {args.output}")
        except Exception as e:
            print(f"[resume] failed to load existing CSV ({type(e).__name__}: "
                  f"{e}); running from scratch")
            done_rows = {}

    rows_to_run: list[dict] = []
    for row in rows_in:
        key = _norm_key(outer_qid(row), outer_pidx(row))
        if key is None or key not in done_rows:
            rows_to_run.append(row)
    if args.resume:
        print(f"[resume] {len(rows_to_run)} rows to compute, "
              f"{len(done_rows)} carried over (of {len(rows_in)} total)")

    def _job(row: dict) -> dict[str, Any]:
        qid, pidx = outer_qid(row), outer_pidx(row)
        plan = unwrap(row)
        inject_accepted_answer(plan, qid, pidx, qap_lookup)
        gold = gold_map.get((qid, pidx))
        try:
            metrics = eval_judge(plan, gold, judge=args.judge)
            err = ""
        except Exception as e:
            metrics = {k: float("nan") for k in JUDGE_COLS}
            err = f"{type(e).__name__}: {e}"
        return {"question_id": qid, "profile_index": pidx,
                **metrics, "_error": err}

    new_rows: list[dict] = []
    if args.workers <= 1:
        for row in tqdm(rows_to_run, desc="judge"):
            new_rows.append(_job(row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_job, r) for r in rows_to_run]
            for f in tqdm(as_completed(futs), total=len(futs), desc="judge"):
                new_rows.append(f.result())

    out_rows: list[dict] = list(done_rows.values()) + new_rows
    df = pd.DataFrame(out_rows)
    # Force question_id to be written as a clean string (no "28703241.0"),
    # so subsequent --resume runs can re-read the key without dtype drift.
    if not df.empty and "question_id" in df.columns:
        def _strip_dot_zero(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            s = str(v).strip()
            if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
                s = s[:-2]
            return s
        df["question_id"] = df["question_id"].apply(_strip_dot_zero)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out_rows)} rows -> {args.output} "
          f"({len(new_rows)} newly computed, {len(done_rows)} carried over)")
    _summarize(df)


if __name__ == "__main__":
    main()
