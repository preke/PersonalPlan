"""Reward functions for GRPO training of the MAPLE planner.

Paper §4.2 specifies four components:

  R_struct  (verifiable; DAG + DC + ATR + GED-sim to gold)
  R_pers    (counterfactual; sim(plan, gold) - sim(plan, plan_cf))
  R_ped     (hybrid; λ * prereq_compat + (1-λ) * LLM_judge)
  R_hard    (gate; -η on schema/cycle/invalid-tool)

Practical staging (see README for rationale):

  v1 ─ R_struct (Jaccard fingerprint proxy of GED-sim) + R_hard
       This is the runnable default and what `compose_reward` returns
       when constructed with the default flags.

  v2 ─ + R_pers using a *frozen* counterfactual cache (pre-sampled with
       the SFT model). Adds personalization signal at almost no extra
       cost during training. Toggle with `enable_pers=True`.

  v3 ─ + R_ped,hard via mined subtask-precedence pairs.
       Toggle with `enable_ped_hard=True`.

  v4 ─ + R_ped,soft via LLM judge ensemble.
       Toggle with `enable_ped_soft=True`. Requires API access or
       a local judge model — expect significant per-step latency.

All component functions return a SCALAR per trajectory. Aggregation is
in `compose_reward` (simple weighted sum, paper default
w_s:w_p:w_e = 0.55:0.25:0.20). Segment-wise routing is documented in
the README but NOT applied in the scalar reward — it requires
subclassing TRL's GRPOTrainer, which we stub at the end of this file.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Optional

from plan_utils import (
    agent_tool_relevance,
    dependency_completeness,
    has_cycle,
    invalid_tool_calls,
    mine_subtask_precedence,
    parse_plan,
    plan_components,
    prerequisite_compatibility,
    schema_valid,
    structural_similarity,
)


# ======================================================================
# R_struct — verifiable structural reward
# ======================================================================

def reward_structural(plan_text: str, gold: dict | None) -> float:
    """R_struct = w1·DAG + w2·DC + w3·ATR + w4·StructuralSim(plan, gold)
    Weights taken from paper §4.2 (rounded for first-cut)."""
    plan = parse_plan(plan_text)
    if plan is None:
        return 0.0
    dag = 0.0 if has_cycle(plan) else 1.0
    dc = dependency_completeness(plan)
    atr = agent_tool_relevance(plan)
    sim = structural_similarity(plan, gold) if gold else 0.0
    # Paper does not fix exact (w1, w2, w3, w4); these are reasonable defaults
    # and the overall component weight w_s in compose_reward dominates the magnitudes.
    return 0.25 * dag + 0.25 * dc + 0.20 * atr + 0.30 * sim


# ======================================================================
# R_hard — disqualifying gate
# ======================================================================

def reward_hard_gate(plan_text: str, eta: float = 10.0) -> float:
    """R_hard = -eta if (invalid_schema | has_cycle | any_invalid_tool_call).
    NOT z-scored; magnitude is intentionally large to dominate other terms."""
    plan = parse_plan(plan_text)
    if not schema_valid(plan):
        return -eta
    if has_cycle(plan):
        return -eta
    if invalid_tool_calls(plan) > 0:
        return -eta
    return 0.0


# ======================================================================
# R_pers — counterfactual personalization (v2)
# ======================================================================

def reward_personalization(
    plan_text: str,
    gold: dict | None,
    cf_plan: dict | None,
) -> float:
    """R_pers = sim(plan, gold) - sim(plan, plan_cf).

    Paper §4.2: cf_plan is sampled from the *current* policy with a
    randomly-swapped profile. To stay compatible with TRL's GRPOTrainer
    (which does not expose the model to reward_funcs), our v1
    implementation passes in a counterfactual sampled from the FROZEN
    SFT model (see `build_counterfactual_cache.py`).

    A v2 implementation should override GRPOTrainer._prepare_inputs to
    sample cf_plans from π_θ at every step. We leave that as a TODO.
    """
    if cf_plan is None:
        return 0.0
    plan = parse_plan(plan_text)
    s_gold = structural_similarity(plan, gold)
    s_cf = structural_similarity(plan, cf_plan)
    return s_gold - s_cf


# ======================================================================
# R_pers_lite — drop-in personalization reward, no counterfactual rollout
# ======================================================================
#
# v2 design (after empirical testing showed v1 mis-scored gold plans):
#
#   reward = fraction of step.instructions that contain at least one
#            "personalization sentence" — a sentence that addresses the
#            learner directly (you/your/the learner) AND references a
#            profile-specific signal (skill, domain word from about_me,
#            or a bridging phrase like "background", "project work",
#            "analogy") in the SAME sentence.
#
# Why a per-step co-occurrence signal?
#
#   - Gold plans in MAPLE personalize STRUCTURALLY: bridges, analogies,
#     2nd-person addressing inside step.instruction text. They do NOT
#     stuff skill keywords into agent backstories.
#   - Empirical check on MAPLE: gold instructions contain "you" / "the
#     learner" + profile-context phrases roughly 60-90% of the time;
#     query-only generic plans contain neither.
#   - Anti-hacking: requires CO-OCCURRENCE within a sentence and
#     DISTRIBUTION across multiple steps. Both keyword-only stuffing
#     and pure-second-person boilerplate fail.

import re

# Allow-list of "personalization-eligible" lexical anchors. These are
# words that, when co-occurring with a 2nd-person reference, signal
# learner-grounded teaching rather than generic instruction.
_BRIDGE_ANCHORS = {
    "background", "experience", "skill", "skills", "expertise",
    "project", "projects", "domain", "stack", "framework", "language",
    "analogy", "analogous", "analogies", "bridge",
    "intuition", "familiar", "familiarity", "owns", "already",
    "compared", "unlike", "similar", "like", "draws", "drawing",
    "your", "the learner", "learner's", "the user", "you've",
}

_PERSON_REF_RE = re.compile(
    r"\b(?:you|your|you'?re|you'?ve|yourself|the learner|learner'?s|the user)\b",
    re.IGNORECASE,
)

# Sentence splitter — keep simple; instructions are short.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

_DOMAIN_WORD_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+#\-]{2,})\b")
_STOPWORDS_LITE = {
    "the", "and", "for", "with", "from", "your", "you", "this", "that", "have",
    "are", "was", "were", "will", "would", "should", "could", "but", "not",
    "into", "what", "when", "where", "which", "why", "how", "who", "use", "using",
    "they", "them", "their", "all", "any", "some", "more", "most", "many", "much",
    "such", "than", "then", "very", "also", "just", "only", "etc",
}


def _extract_profile_signals(profile: dict | None) -> tuple[set[str], set[str]]:
    """Return (skill_terms, domain_words) — lowercased, deduped, stop-filtered."""
    if not isinstance(profile, dict):
        return set(), set()
    raw_skills = profile.get("skills") or profile.get("top_tags") or []
    skills = {str(s).lower().strip() for s in raw_skills if str(s).strip()}
    desc = profile.get("self_description") or profile.get("about_me") or ""
    domain_words = set()
    for tok in _DOMAIN_WORD_RE.findall(desc):
        tl = tok.lower()
        if len(tl) >= 4 and tl not in _STOPWORDS_LITE:
            domain_words.add(tl)
    domain_words -= skills
    return skills, domain_words


def _is_personalization_sentence(
    sentence: str,
    profile_signals: set[str],
    require_anchor: bool = True,
) -> bool:
    """A sentence counts as personalization iff:
      (a) it contains a 2nd-person / learner reference, AND
      (b) it contains at least one profile signal OR a bridge anchor word.
    """
    if not _PERSON_REF_RE.search(sentence):
        return False
    s_low = sentence.lower()
    # Profile signal hit?
    if any(sig in s_low for sig in profile_signals):
        return True
    if require_anchor and any(a in s_low for a in _BRIDGE_ANCHORS):
        return True
    return False


def reward_personalization_lite(
    plan_text: str,
    profile: dict | None,
    *,
    cap: float = 0.8,
    min_step_chars: int = 150,
) -> float:
    """Counterfactual-free personalization reward (v2).

    Score = fraction of step.instructions that ALL of:
      (1) are at least `min_step_chars` characters long (boilerplate filter)
      (2) contain a 2nd-person / learner reference
      (3) co-locate that reference with a profile signal or bridge anchor
          in the SAME sentence
    capped at `cap`.

    Returns a scalar in [0, cap].

    Important caveat: this scorer can be gamed by *long* templated text
    that paraphrases bridge phrases without real teaching content. The
    intended defense is COMPOSITION with R_struct: in
    `compose_reward` the total reward is
        w_struct * sim(plan, profile-specific gold) + w_pers * R_pers_lite + ...
    R_struct collapses on such templated attacks because their
    step.instructions diverge from gold structurally. Use R_pers_lite as
    a complementary signal, not a standalone discriminator.
    """
    skills, domain_words = _extract_profile_signals(profile)
    profile_signals = skills | domain_words

    plan = parse_plan(plan_text)
    if plan is None:
        return 0.0
    p = plan_components(plan)
    total_steps = len(p["steps_by_id"])
    if total_steps == 0:
        return 0.0

    n_covered = 0
    for sid, st in p["steps_by_id"].items():
        instr = (st.get("instruction", "") or "").strip()
        # (1) length filter — rejects "You are a X developer." style boilerplate
        if len(instr) < min_step_chars:
            continue
        # (2) fast-fail on 2nd-person reference
        if not _PERSON_REF_RE.search(instr):
            continue
        # (3) sentence-level co-occurrence check
        sents = _SENT_SPLIT_RE.split(instr) if instr else []
        if not sents:
            sents = [instr]
        for s in sents:
            if _is_personalization_sentence(s, profile_signals):
                n_covered += 1
                break

    fraction = n_covered / total_steps
    return min(fraction, 1.0) * cap


# ======================================================================
# R_ped — pedagogy
# ======================================================================
#
# Two cheap, rule-based signals that together capture what the paper's
# R_ped supervises (teaching-order correctness + canonical phase
# integration). No LLM judge needed at v1.
#
#   reward_pedagogy_hard       — subtask-name precedence (simplification
#                                  of paper's prerequisite graph K)
#   reward_pedagogy_phases     — canonical phase coverage + ordering
#
# Both are bounded in [0, 1]; `reward_pedagogy_lite` linearly combines
# them (50/50 default). The paper's R_ped,soft (LLM judge) can still
# be plugged in via `reward_pedagogy_soft` when needed.

def reward_pedagogy_hard(plan_text: str, precedence_pairs: set[tuple[str, str]]) -> float:
    """R_ped,hard from mined subtask-precedence pairs (a simplified
    stand-in for the paper's prerequisite graph K)."""
    plan = parse_plan(plan_text)
    return prerequisite_compatibility(plan, precedence_pairs)


# ----------------------------------------------------------------------
# Phase coverage — purely rule-based stand-in for R_ped,soft
# ----------------------------------------------------------------------

# Canonical teaching phases. Order matters: list defines expected
# left-to-right progression in a well-structured plan.
_PED_PHASE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("probe", [
        "ask the learner", "have the learner predict", "ask them to predict",
        "what do you think", "your intuition", "prior knowledge",
        "before introducing", "before showing", "surface", "activate",
        "have them guess", "have them describe",
    ]),
    ("retrieve_demonstrate", [
        "look up", "retrieve", "documentation", "official",
        "demonstrate", "produce concrete", "show concrete",
        "execute the following", "run the following",
    ]),
    ("apply", [
        "ask the learner to write", "ask the learner to revise",
        "ask the learner to rewrite", "have the learner write",
        "have the learner implement", "submit",
        "write a", "implement",
    ]),
    ("validate", [
        "test case", "test cases", "validate", "verify",
        "compile", "check that", "set ", "expected_output",
        "outcome=", "implementation_correct",
    ]),
    ("feedback", [
        "explain what went wrong", "targeted feedback", "what to fix",
        "point to the specific", "tell the learner the precise",
        "give targeted feedback", "specific fix",
    ]),
    ("consolidate", [
        "summarize", "consolidate", "wrap up", "wrap-up",
        "decision rule", "transfer", "general rule",
        "in their own words", "decision framework",
        "one-line rule", "their own decision",
    ]),
]
_PHASE_CANONICAL_ORDER = [name for name, _ in _PED_PHASE_KEYWORDS]


def _identify_phase(text: str) -> str | None:
    """Map a step's instruction+objective text to one canonical phase
    (or None if no phase keyword fires). First match wins — keyword
    lists are roughly ordered by specificity."""
    tl = text.lower()
    for phase, keywords in _PED_PHASE_KEYWORDS:
        for kw in keywords:
            if kw in tl:
                return phase
    return None


def reward_pedagogy_phases(plan_text: str) -> float:
    """Phase coverage and ordering score in [0, 1].

    Five binary checks averaged:
      1. probe phase appears in first 50% of steps
      2. validate phase appears anywhere
      3. feedback phase appears anywhere (typically inside a loop)
      4. consolidate phase appears in last 40% of steps
      5. detected phases appear in canonical order (probe ≺ retrieve ≺
         apply ≺ validate ≺ feedback ≺ consolidate)
    """
    plan = parse_plan(plan_text)
    if plan is None:
        return 0.0
    p = plan_components(plan)
    if not p["steps_by_id"]:
        return 0.0

    # Sort steps by id (S1-1 < S1-2 < S2-1 etc. with simple lexicographic sort
    # on the (subtask_idx, step_idx) tuple).
    def _step_sort_key(sid: str) -> tuple[int, int]:
        # id format: "S<I>-<J>"; fall back to lexicographic if non-standard.
        try:
            i, j = sid[1:].split("-", 1)
            return int(i), int(j)
        except Exception:
            return (10**9, 10**9)
    sorted_ids = sorted(p["steps_by_id"].keys(), key=_step_sort_key)
    n = len(sorted_ids)

    # Map position → phase
    phase_first_pos: dict[str, int] = {}
    for pos, sid in enumerate(sorted_ids):
        st = p["steps_by_id"][sid]
        text = (st.get("instruction") or "") + " " + (st.get("objective") or "")
        phase = _identify_phase(text)
        if phase and phase not in phase_first_pos:
            phase_first_pos[phase] = pos

    half_first = max(int(n * 0.5), 1)
    threshold_late = int(n * 0.6)  # "last 40%" means pos >= n*0.6

    # 5 checks
    has_probe_early = (
        "probe" in phase_first_pos and phase_first_pos["probe"] < half_first
    )
    has_validate = "validate" in phase_first_pos
    has_feedback = "feedback" in phase_first_pos
    has_consolidate_late = (
        "consolidate" in phase_first_pos
        and phase_first_pos["consolidate"] >= threshold_late
    )
    ordered_phases = [
        phase_first_pos[p] for p in _PHASE_CANONICAL_ORDER if p in phase_first_pos
    ]
    is_ordered = all(
        ordered_phases[i] <= ordered_phases[i + 1]
        for i in range(len(ordered_phases) - 1)
    ) if len(ordered_phases) >= 2 else False

    checks = [
        has_probe_early,
        has_validate,
        has_feedback,
        has_consolidate_late,
        is_ordered,
    ]
    return sum(1 for c in checks if c) / len(checks)


def reward_pedagogy_lite(
    plan_text: str,
    precedence_pairs: set[tuple[str, str]] | None = None,
    *,
    w_precedence: float = 0.5,
    w_phases: float = 0.5,
) -> float:
    """Combined cheap pedagogy reward.

    R_ped_lite = w_precedence · subtask_precedence_compat
               + w_phases     · phase_coverage

    `precedence_pairs` is the set mined by plan_utils.mine_subtask_precedence;
    pass None to set the precedence term to 0 (phases-only mode).
    """
    if precedence_pairs:
        prec = reward_pedagogy_hard(plan_text, precedence_pairs)
    else:
        prec = 0.0
    phases = reward_pedagogy_phases(plan_text)
    return w_precedence * prec + w_phases * phases


def reward_pedagogy_soft(
    plan_text: str,
    gold: dict | None,
    judges: list[Callable[[str, str], float]] | None,
) -> float:
    """R_ped,soft via pairwise LLM judge ensemble.

    `judges` is a list of callables: judge(plan_str, gold_str) -> [0,1]
    where 1.0 means plan ≻ gold pedagogically. We apply order
    randomization and take the median across J judges.

    First-cut keeps `judges=None` so this returns 0 — turn on only after
    structural rewards are training stably.
    """
    if not judges:
        return 0.0
    if not gold:
        return 0.0
    gold_str = json.dumps(gold, ensure_ascii=False)
    scores = []
    for j in judges:
        if random.random() < 0.5:
            s = j(plan_text, gold_str)
        else:
            s = 1.0 - j(gold_str, plan_text)
        scores.append(s)
    return sorted(scores)[len(scores) // 2]  # median


# ======================================================================
# Composer — produces a TRL-compatible reward_func
# ======================================================================

@dataclass
class RewardConfig:
    # Component toggles. v1 = struct + hard only.
    enable_struct: bool = True
    enable_hard: bool = True

    # Personalization mode:
    #   "off"            → no R_pers (default)
    #   "lite"           → reward_personalization_lite (no extra rollout; recommended starting point)
    #   "counterfactual" → reward_personalization (requires cf_cache from build_counterfactual_cache.py)
    pers_mode: str = "off"

    # Pedagogy mode:
    #   "off"  → no R_ped (default)
    #   "lite" → reward_pedagogy_lite (precedence + phase coverage, both rule-based, zero LLM cost)
    #   "full" → lite + LLM judge ensemble (reward_pedagogy_soft) blended via lambda_ped
    ped_mode: str = "off"

    # Component weights (paper default: 0.55 / 0.25 / 0.20).
    w_struct: float = 0.55
    w_pers: float = 0.25
    w_ped: float = 0.20

    # Inside ped_mode="lite": split between precedence vs phase coverage.
    w_ped_precedence: float = 0.5
    w_ped_phases: float = 0.5

    # Inside ped_mode="full": λ blends lite (hard) vs soft. Paper: λ ∈ [0.7, 0.9].
    lambda_ped: float = 0.8

    # R_hard magnitude (paper: η; large enough to dominate normalized soft scores).
    eta: float = 10.0

    # Diagnostic toggle: log each component to stdout.
    verbose: bool = False


def compose_reward(
    cfg: RewardConfig,
    *,
    precedence_pairs: set[tuple[str, str]] | None = None,
    cf_cache: dict[str, dict] | None = None,
    judges: list[Callable[[str, str], float]] | None = None,
) -> Callable:
    """Return a callable suitable for TRL's `reward_funcs=[...]`.

    The returned function expects to be called by TRL's GRPOTrainer
    with at minimum:
        completions : list[str]         # model outputs
        prompts     : list[str]         # original prompts (for joining
                                          back to gold via the dataset)
        question_id : list[str]         # if dataset provides it,
                                          used to look up cf_cache and gold
        gold_plan   : list[dict]        # passed through from dataset
        learner_profile : list[dict]    # used by pers_mode="lite"

    See `grpo_train.py` for how the dataset is constructed to expose
    these fields.
    """

    if cfg.pers_mode not in ("off", "lite", "counterfactual"):
        raise ValueError(
            f"pers_mode must be one of 'off' | 'lite' | 'counterfactual', "
            f"got {cfg.pers_mode!r}"
        )
    if cfg.ped_mode not in ("off", "lite", "full"):
        raise ValueError(
            f"ped_mode must be one of 'off' | 'lite' | 'full', got {cfg.ped_mode!r}"
        )

    def reward_fn(completions, **kwargs):
        prompts = kwargs.get("prompts", [None] * len(completions))
        question_ids = kwargs.get("question_id", [None] * len(completions))
        gold_plans = kwargs.get("gold_plan", [None] * len(completions))
        learner_profiles = kwargs.get("learner_profile", [None] * len(completions))

        scores: list[float] = []
        for i, comp in enumerate(completions):
            gold = gold_plans[i] if i < len(gold_plans) else None
            qid = question_ids[i] if i < len(question_ids) else None
            profile = (
                learner_profiles[i] if i < len(learner_profiles) else None
            )

            # Components (each may be deactivated by config).
            r_struct = reward_structural(comp, gold) if cfg.enable_struct else 0.0
            r_hard = reward_hard_gate(comp, eta=cfg.eta) if cfg.enable_hard else 0.0

            # R_pers — three dispatch modes
            r_pers = 0.0
            if cfg.pers_mode == "lite":
                r_pers = reward_personalization_lite(comp, profile)
            elif cfg.pers_mode == "counterfactual":
                if cf_cache is not None and qid is not None:
                    cf_plan = cf_cache.get(qid)
                    r_pers = reward_personalization(comp, gold, cf_plan)

            # R_ped — three dispatch modes
            r_ped_lite = 0.0
            r_ped_soft = 0.0
            if cfg.ped_mode == "lite":
                r_ped_lite = reward_pedagogy_lite(
                    comp,
                    precedence_pairs=precedence_pairs,
                    w_precedence=cfg.w_ped_precedence,
                    w_phases=cfg.w_ped_phases,
                )
                r_ped = r_ped_lite
            elif cfg.ped_mode == "full":
                r_ped_lite = reward_pedagogy_lite(
                    comp,
                    precedence_pairs=precedence_pairs,
                    w_precedence=cfg.w_ped_precedence,
                    w_phases=cfg.w_ped_phases,
                )
                if judges:
                    r_ped_soft = reward_pedagogy_soft(comp, gold, judges)
                r_ped = cfg.lambda_ped * r_ped_lite + (1 - cfg.lambda_ped) * r_ped_soft
            else:
                r_ped = 0.0

            total = (
                cfg.w_struct * r_struct
                + cfg.w_pers * r_pers
                + cfg.w_ped * r_ped
                + r_hard
            )
            if cfg.verbose:
                print(
                    f"  [{i:02d}] struct={r_struct:.3f} pers={r_pers:.3f} "
                    f"ped_lite={r_ped_lite:.3f} ped_s={r_ped_soft:.3f} "
                    f"hard={r_hard:+.1f} → R={total:+.3f}"
                )
            scores.append(total)
        return scores

    return reward_fn


# ======================================================================
# TODO: segment-wise credit assignment (paper §4.2 Level 2)
# ======================================================================
#
# The scalar `reward_fn` above feeds TRL's standard advantage path:
# A_i for every token in trajectory i. The paper instead routes:
#
#     agent tokens   ← z(R_pers)
#     subtask tokens ← z(R_ped)
#     step/dep tokens← z(R_struct)
#     hard gate      ← uniformly added
#
# To implement this you would subclass `trl.GRPOTrainer` and override
# `_get_per_token_logps` / `compute_loss` to:
#   1. tag each generated token with its plan segment (agent/subtask/step)
#      based on a streaming JSON parser tracking the current key path,
#   2. assemble a per-token advantage vector from the four component scores,
#   3. apply the GRPO clipped surrogate on this per-token advantage.
#
# We DO NOT implement this in v1. The scalar reward above is the
# correct first target — once it trains stably, swap in the segment-wise
# version. A minimal sketch:
#
#     class SegmentRoutedGRPOTrainer(GRPOTrainer):
#         def _compute_advantages(self, rewards_per_segment, segments):
#             ...
#
# Sketch only — full implementation is ~150 lines and requires careful
# alignment between tokenizer offsets and JSON spans.
