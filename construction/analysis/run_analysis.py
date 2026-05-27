"""
MAPLE_Construction / multi_agent_dataset_filtered_qap.jsonl 全面分析

参考:
  - analysis/analyze.py
  - MAPLE_Construction/analysis_paper/analyze_paper.py
  - EXPERIMENT_PLAN_2026-05-15.html (RQ1-RQ4, Ped/PVS/PNG/Skill-Match dimensions)
  - EVALUATION_DESIGN_2026-05-15.html

新增维度 (相对旧版):
  P1. 教学法 (pedagogy patterns) 分类: 10 类策略 (predict-tell, scaffolding, retrieval, etc.)
  P2. 教学骨架 (scaffolding skeleton) 三段式覆盖: Activate -> Apply/Validate -> Consolidate
  P3. 学习者 profile 画像: top_tags 共现 / 主题 / 长度分桶
  P4. 个性化信号在文本里的渗透: skill-mention / second-person / bridge anchors
  P5. profile-pair plan divergence (1-Jaccard agent_role / tool / subtask token)
  P6. Plan complexity vs profile_index (是否随 profile_index 改变?)
  P7. Intent x complexity / Intent x pedagogy 交叉

输出:
  - claude_analysis/figures/*.png
  - claude_analysis/stats.json
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MAPLE_DIR = HERE.parent
ROOT = MAPLE_DIR.parent
DATA_PATH = MAPLE_DIR / "multi_agent_dataset_filtered_qap.jsonl"
LABELS_PATH = ROOT / "the_construction_of_MAPLE_datasets/task_3/classified_results.jsonl"
SPLIT_PATH = ROOT / "splits/maple_split_v1.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
STATS_OUT = HERE / "stats.json"

# ----------------------------------------------------------------------
# Publication style (colorblind-safe, paper context)
# ----------------------------------------------------------------------
sns.set_theme(style="ticks", context="paper", font_scale=1.05)
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
PALETTE_MAIN = sns.color_palette("crest", 8)
PALETTE_SET = sns.color_palette("Set2", 8)
INTENT_COLORS = dict(zip(
    ["API_USAGE", "CONCEPTUAL", "DISCREPANCY", "ERRORS",
     "REVIEW", "API_CHANGE", "LEARNING", "UNKNOWN"],
    sns.color_palette("Set2", 8),
))
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 160,
    "savefig.facecolor": "white",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
})


def save_fig(name, dpi=200):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_jsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_learner(lrn):
    """统一两种 schema."""
    if not lrn:
        return "", []
    desc = lrn.get("about_me") or lrn.get("self_description") or ""
    skills = lrn.get("top_tags") or lrn.get("skills") or []
    return desc.strip(), list(skills)


def agent_desc_text(agent):
    return agent.get("backstory") or agent.get("description") or ""


def normalize_token(s):
    return re.findall(r"[A-Za-z][A-Za-z0-9_+#\.-]+", s.lower())


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


# ----------------------------------------------------------------------
# Role-family taxonomy & pedagogy patterns
# ----------------------------------------------------------------------
ROLE_FAMILIES = [
    ("tutor", ["tutor", "instructor", "teacher", "coach", "explainer",
               "mentor", "guide", "facilitator"]),
    ("retriever", ["retriever", "docs", "documentation", "searcher",
                   "finder", "lookup", "fetcher"]),
    ("validator", ["validator", "checker", "tester", "verifier",
                   "executor", "runner", "evaluator", "assessor"]),
    ("debugger", ["debugger", "diagnostician", "analyzer", "diagnoser",
                  "inspector", "troubleshooter"]),
    ("planner", ["planner", "designer", "architect", "strategist",
                 "coordinator", "orchestrator"]),
    ("reviewer", ["reviewer", "critic", "auditor", "qa", "quality"]),
    ("generator", ["generator", "writer", "composer", "author",
                   "producer", "creator"]),
    ("translator", ["translator", "interpreter", "converter", "transformer"]),
]


def classify_role(role):
    rl = (role or "").lower()
    for fam, kws in ROLE_FAMILIES:
        if any(k in rl for k in kws):
            return fam
    return "other"


# ============================================================
# Pedagogy pattern detectors (regex over instruction text)
# 设计参考 EXPERIMENT_PLAN: PRR / NDAR / SPR / IAR + Skill-Match / Background-Adaptation
# ============================================================
PEDAGOGY_PATTERNS = {
    "prediction_elicitation":      re.compile(r"\b(predict|prediction|guess|hypothesi[sz]e|what do you think|before showing|first ask)\b", re.I),
    "explanation_walkthrough":     re.compile(r"\b(explain|walk.{0,15}through|describe to|teach|step.by.step|illustrate)\b", re.I),
    "worked_example":              re.compile(r"\b(example|worked example|sample code|demonstration|show .{0,15}how)\b", re.I),
    "guided_practice":             re.compile(r"\b(apply|try|implement|practice|hands.on|exercise|write the code)\b", re.I),
    "scaffolding_iterative":       re.compile(r"\b(loop|iterate|until|retry|iteration|refine)\b", re.I),
    "validation_feedback":         re.compile(r"\b(validate|verify|check|test|confirm|assess|evaluate|compile|run)\b", re.I),
    "reflection_metacognition":    re.compile(r"\b(reflect|generaliz[ez]|extract.{0,10}rule|takeaway|summari[sz]e|articulate)\b", re.I),
    "retrieval_practice":          re.compile(r"\b(recall|retrieve|remember|look up|search|find.{0,10}document|fetch)\b", re.I),
    "analogy_priorknowledge":      re.compile(r"\b(analog[yi]|like.{0,10}(php|html|javascript|python|c#|c\+\+|java)|in your |from your (background|portfolio)|prior knowledge)\b", re.I),
    "diagnostic_assessment":       re.compile(r"\b(diagnos|identify.{0,10}error|misconception|gap|misunderstanding|root cause)\b", re.I),
}

# Scaffolding skeleton: Activate -> Apply/Validate -> Consolidate
SKELETON_ACTIVATE  = re.compile(r"\b(activate|surface|reveal|elicit|baseline|prior knowledge|mental model|misconception|prediction|gap)\b", re.I)
SKELETON_APPLY     = re.compile(r"\b(apply|implement|fix|write|build|construct|debug|validate|verify|test|run|compile|loop)\b", re.I)
SKELETON_CONSOLID  = re.compile(r"\b(consolidate|generali[sz]e|reflect|takeaway|extract.{0,10}rule|summari[sz]e|decision rule|articulate)\b", re.I)

# Personalization signal in plan text
SECOND_PERSON = re.compile(r"\b(you|your|the learner)\b", re.I)
BRIDGE_ANCHORS = [
    "your portfolio", "your background", "your project", "as a beginner",
    "as a senior", "your experience", "in your", "from your", "the learner's",
    "given your", "you mentioned", "since you", "you already",
]

INTENT_ORDER = ["API_USAGE", "CONCEPTUAL", "DISCREPANCY", "ERRORS",
                "REVIEW", "API_CHANGE", "LEARNING"]


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
records = load_jsonl(DATA_PATH)
N = len(records)

intent_labels = {}
if LABELS_PATH.exists():
    for o in load_jsonl(LABELS_PATH):
        intent_labels[str(o["question_id"])] = o.get("labels", [])

split_def = None
if SPLIT_PATH.exists():
    split_def = json.load(open(SPLIT_PATH))

STATS = {"meta": {"dataset_path": str(DATA_PATH), "n_records": N}}


def primary_intent(qid):
    labs = intent_labels.get(str(qid), [])
    return labs[0] if labs else "UNKNOWN"


# ======================================================================
# Section A — OVERVIEW
# ======================================================================
print("=" * 70 + "\nA. OVERVIEW\n" + "=" * 70)

qids = Counter(str(r["question_id"]) for r in records)
profile_idx_dist = Counter(r["profile_index"] for r in records)
unique_qids = len(qids)
nprof_per_qid = Counter(qids.values())


def profile_key(p):
    desc, sk = extract_learner(p)
    return (desc, tuple(sorted(sk)))


unique_profiles = len(set(profile_key(r["plan"]["input"]["learner"]) for r in records))

learner_schema_counter = Counter()
for r in records:
    lrn = r["plan"]["input"].get("learner", {})
    learner_schema_counter[tuple(sorted(lrn.keys()))] += 1

query_lens = np.array([len(r["plan"]["input"]["query"]) for r in records])
desc_lens, skills_counts = [], []
for r in records:
    desc, sk = extract_learner(r["plan"]["input"]["learner"])
    desc_lens.append(len(desc))
    skills_counts.append(len(sk))
desc_lens = np.array(desc_lens)
skills_counts = np.array(skills_counts)

# agent / step textual stats
agent_goal_lens, agent_backstory_lens = [], []
step_instr_lens, step_obj_lens, step_exp_lens = [], [], []
sub_obj_lens = []
for r in records:
    out = r["plan"]["output"]
    for a in out["agents"]:
        agent_goal_lens.append(len(a.get("goal", "")))
        agent_backstory_lens.append(len(agent_desc_text(a)))
    for s in out["subtasks"]:
        sub_obj_lens.append(len(s.get("subtask_objective", "")))
        for st in s.get("steps", []):
            step_instr_lens.append(len(st.get("instruction", "")))
            step_obj_lens.append(len(st.get("objective", "")))
            step_exp_lens.append(len(st.get("expected_output", "")))

STATS["A_overview"] = {
    "total_records": N,
    "unique_questions": unique_qids,
    "unique_profiles_by_text": unique_profiles,
    "rows_per_qid_mean": N / unique_qids,
    "learner_schema_distribution": {str(k): v for k, v in learner_schema_counter.items()},
    "profile_index_distribution": dict(profile_idx_dist),
    "nprof_per_qid_distribution": dict(nprof_per_qid),
    "max_profiles_per_question": max(qids.values()),
    "questions_with_multi_profile": sum(1 for v in qids.values() if v > 1),
    "query_len_chars": {"mean": float(query_lens.mean()), "median": float(np.median(query_lens)),
                         "p25": float(np.percentile(query_lens, 25)),
                         "p75": float(np.percentile(query_lens, 75)),
                         "min": int(query_lens.min()), "max": int(query_lens.max())},
    "desc_len_chars": {"mean": float(desc_lens.mean()), "median": float(np.median(desc_lens)),
                        "min": int(desc_lens.min()), "max": int(desc_lens.max())},
    "skills_per_profile": {"mean": float(skills_counts.mean()), "median": float(np.median(skills_counts)),
                            "min": int(skills_counts.min()), "max": int(skills_counts.max())},
    "agent_goal_len_chars": {"mean": safe_mean(agent_goal_lens), "median": float(np.median(agent_goal_lens))},
    "agent_backstory_len_chars": {"mean": safe_mean(agent_backstory_lens), "median": float(np.median(agent_backstory_lens))},
    "subtask_objective_len_chars": {"mean": safe_mean(sub_obj_lens), "median": float(np.median(sub_obj_lens))},
    "step_instruction_len_chars": {"mean": safe_mean(step_instr_lens), "median": float(np.median(step_instr_lens))},
    "step_objective_len_chars": {"mean": safe_mean(step_obj_lens), "median": float(np.median(step_obj_lens))},
    "step_expected_output_len_chars": {"mean": safe_mean(step_exp_lens), "median": float(np.median(step_exp_lens))},
}
for k, v in STATS["A_overview"].items():
    print(f"  {k}: {v}")

# --- A1: profile_index dist
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = sorted(profile_idx_dist.keys())
bars = ax.bar(ks, [profile_idx_dist[k] for k in ks],
              color=OKABE_ITO[4], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 25, f"{int(h):,}",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("profile_index")
ax.set_ylabel("Record count")
ax.set_title("A1 · Records by profile_index")
sns.despine()
save_fig("A1_profile_index.png")

# --- A2: nprof_per_qid
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = sorted(nprof_per_qid.keys())
bars = ax.bar(ks, [nprof_per_qid[k] for k in ks],
              color=OKABE_ITO[1], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
            f"{nprof_per_qid[k]}\n({nprof_per_qid[k]/unique_qids:.1%})",
            ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Number of profile variants for one question")
ax.set_ylabel("Question count")
ax.set_title(f"A2 · Profile multiplicity per question (n_q={unique_qids})")
sns.despine()
save_fig("A2_profiles_per_question.png")

# --- A3: query / about_me / skill length grid
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, data, title, color in zip(
    axes,
    [query_lens, desc_lens, skills_counts],
    ["Query length (chars)", "Learner about_me length (chars)", "Top-tags per learner"],
    [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[3]],
):
    bins = 40 if title.startswith("Top") is False else max(int(data.max())+1, 2)
    if title.startswith("Top"):
        bins = range(0, int(data.max())+2)
    sns.histplot(data, bins=bins, color=color, edgecolor="white", ax=ax)
    mu, med = float(np.mean(data)), float(np.median(data))
    ax.axvline(mu, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean={mu:.1f}\nMedian={med:.0f}")
    ax.set_title(title)
    ax.set_ylabel("Record count")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)
fig.suptitle("A3 · Input-side length distributions", y=1.05, fontsize=12, fontweight="semibold")
save_fig("A3_input_lengths.png")

# --- A4: agent text lengths
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, data, title, color in zip(
    axes,
    [agent_goal_lens, agent_backstory_lens, step_instr_lens],
    ["Agent goal (chars)", "Agent backstory (chars)", "Step instruction (chars)"],
    [OKABE_ITO[5], OKABE_ITO[6], OKABE_ITO[0]],
):
    cap = np.percentile(data, 99)
    bins = np.linspace(0, cap, 35)
    sns.histplot(np.clip(data, 0, cap), bins=bins, color=color, edgecolor="white", ax=ax)
    mu, med = float(np.mean(data)), float(np.median(data))
    ax.axvline(mu, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean={mu:.0f}  Median={med:.0f}")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)
fig.suptitle("A4 · Output-side length distributions  (clipped at p99)",
             y=1.05, fontsize=12, fontweight="semibold")
save_fig("A4_output_lengths.png")


# ======================================================================
# Section B — Plan complexity (agents · subtasks · steps · DAG · loops)
# ======================================================================
print("\n" + "=" * 70 + "\nB. PLAN COMPLEXITY\n" + "=" * 70)

n_agents, n_subtasks, n_steps, steps_per_subtask = [], [], [], []
n_tools_total_plan, n_tools_unique_plan = [], []
loops_per_plan, loop_step_counts, loop_max_iter = [], [], []
human_input_per_plan, human_input_ratio = [], []
agent_step_counts = []  # used to measure agent activity per plan

for r in records:
    out = r["plan"]["output"]
    agents = out["agents"]
    n_agents.append(len(agents))

    subs = out["subtasks"]
    n_subtasks.append(len(subs))
    total_steps, hi = 0, 0
    for s in subs:
        steps = s.get("steps", [])
        steps_per_subtask.append(len(steps))
        total_steps += len(steps)
        for st in steps:
            if st.get("requires_human_input"):
                hi += 1
    n_steps.append(total_steps)
    human_input_per_plan.append(hi)
    human_input_ratio.append(hi / total_steps if total_steps else 0.0)

    # tools per plan
    pool = []
    for a in agents:
        for t in a.get("tools", []) or []:
            pool.append(t)
    n_tools_total_plan.append(len(pool))
    n_tools_unique_plan.append(len(set(pool)))

    # loops in execution_order
    nloops = 0
    for item in out["execution_order"]:
        if isinstance(item, dict) and "loop" in item:
            nloops += 1
            ld = item["loop"]
            if "steps" in ld:
                loop_step_counts.append(len(ld["steps"]))
            elif "step" in ld:
                loop_step_counts.append(1 if isinstance(ld["step"], str) else len(ld["step"]))
            loop_max_iter.append(ld.get("max_iterations"))
    loops_per_plan.append(nloops)

    # agent activity per plan: how many steps each agent owns
    act = Counter()
    for s in subs:
        for st in s.get("steps", []):
            act[st.get("agent")] += 1
    agent_step_counts.extend(act.values())


def dag_metrics(subtasks):
    nodes = []
    parents = defaultdict(list)
    for s in subtasks:
        for st in s.get("steps", []):
            sid = st.get("id")
            if not sid:
                continue
            nodes.append(sid)
            for d in (st.get("depends_on") or []):
                parents[sid].append(d)
    nodes_set = set(nodes)
    layer = {}

    def depth(n, seen):
        if n in layer:
            return layer[n]
        if n in seen:
            return 0
        seen.add(n)
        ps = [p for p in parents.get(n, []) if p in nodes_set]
        layer[n] = 0 if not ps else 1 + max(depth(p, seen) for p in ps)
        return layer[n]

    for n in nodes:
        depth(n, set())
    if not layer:
        return (0, 0, 0, 0, 0.0)
    longest = max(layer.values()) + 1
    max_width = max(Counter(layer.values()).values())
    par_ratio = 1 - longest / len(nodes) if len(nodes) else 0
    return (len(nodes), sum(len(v) for v in parents.values()), longest, max_width, par_ratio)


dag_stats = [dag_metrics(r["plan"]["output"]["subtasks"]) for r in records]
n_edges = [x[1] for x in dag_stats]
longest_path = [x[2] for x in dag_stats]
max_widths = [x[3] for x in dag_stats]
par_ratio = [x[4] for x in dag_stats]

STATS["B_complexity"] = {
    "agents_per_plan":   {"mean": safe_mean(n_agents),   "median": float(np.median(n_agents)),
                          "min": min(n_agents), "max": max(n_agents),
                          "distribution": dict(Counter(n_agents))},
    "subtasks_per_plan": {"mean": safe_mean(n_subtasks), "median": float(np.median(n_subtasks)),
                          "min": min(n_subtasks), "max": max(n_subtasks),
                          "distribution": dict(Counter(n_subtasks))},
    "steps_per_plan":    {"mean": safe_mean(n_steps),    "median": float(np.median(n_steps)),
                          "min": min(n_steps), "max": max(n_steps)},
    "steps_per_subtask": {"mean": safe_mean(steps_per_subtask),
                          "median": float(np.median(steps_per_subtask))},
    "tools_total_per_plan_mean":  safe_mean(n_tools_total_plan),
    "tools_unique_per_plan_mean": safe_mean(n_tools_unique_plan),
    "plans_with_loop":  sum(1 for x in loops_per_plan if x > 0),
    "plans_with_multiple_loops": sum(1 for x in loops_per_plan if x > 1),
    "loops_total": sum(loops_per_plan),
    "loop_step_count_mean": safe_mean(loop_step_counts),
    "loop_max_iter_distribution": dict(Counter(loop_max_iter)),
    "human_input_steps_per_plan_mean": safe_mean(human_input_per_plan),
    "human_input_step_ratio_overall": sum(human_input_per_plan) / sum(n_steps),
    "dag_longest_path": {"mean": safe_mean(longest_path), "median": float(np.median(longest_path)),
                         "min": min(longest_path), "max": max(longest_path)},
    "dag_max_layer_width": {"mean": safe_mean(max_widths)},
    "dag_parallelizable_ratio_mean": safe_mean(par_ratio),
    "agent_step_load_mean": safe_mean(agent_step_counts),
}
for k, v in STATS["B_complexity"].items():
    print(f"  {k}: {v}")

# --- B1: agents / subtasks / steps four-panel
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, data, title, color in zip(
    axes,
    [n_agents, n_subtasks, n_steps, steps_per_subtask],
    ["Agents per plan", "Subtasks per plan", "Steps per plan", "Steps per subtask"],
    [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3]],
):
    bins = range(min(data), max(data)+2)
    sns.histplot(data, bins=bins, color=color, edgecolor="white", discrete=True, ax=ax)
    mu = float(np.mean(data))
    ax.axvline(mu, color="red", linestyle="--", linewidth=1.2, label=f"Mean={mu:.2f}")
    ax.set_xlabel(title.split(" per ")[0])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)
fig.suptitle("B1 · Plan structural counts", y=1.05, fontsize=12, fontweight="semibold")
save_fig("B1_plan_counts.png")

# --- B2: loops / max_iter / human-input ratio
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
# loops per plan
c = Counter(loops_per_plan)
ks = sorted(c.keys())
axes[0].bar(ks, [c[k] for k in ks], color=OKABE_ITO[4], edgecolor="white")
for k in ks:
    axes[0].text(k, c[k], str(c[k]), ha="center", va="bottom", fontsize=8)
axes[0].set_title("Loops per plan")
axes[0].set_xlabel("# loops")
axes[0].set_ylabel("Plan count")
sns.despine(ax=axes[0])
# loop max iter
c = Counter([x for x in loop_max_iter if x is not None])
ks = sorted(c.keys())
axes[1].bar([str(k) for k in ks], [c[k] for k in ks], color=OKABE_ITO[5], edgecolor="white")
for i, k in enumerate(ks):
    axes[1].text(i, c[k], str(c[k]), ha="center", va="bottom", fontsize=8)
axes[1].set_title("Loop max_iterations distribution")
axes[1].set_xlabel("max_iterations")
axes[1].set_ylabel("Loop count")
sns.despine(ax=axes[1])
# human input ratio
sns.histplot(human_input_ratio, bins=20, color=OKABE_ITO[6], edgecolor="white", ax=axes[2])
axes[2].axvline(np.mean(human_input_ratio), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(human_input_ratio):.1%}")
axes[2].set_title("Human-input step ratio per plan")
axes[2].set_xlabel("Ratio of steps needing human input")
axes[2].set_ylabel("Plan count")
axes[2].legend()
sns.despine(ax=axes[2])
fig.suptitle("B2 · Iterative scaffolding & human-in-loop signals",
             y=1.05, fontsize=12, fontweight="semibold")
save_fig("B2_loops_humanin.png")

# --- B3: DAG metrics
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(longest_path, bins=range(min(longest_path), max(longest_path)+2),
             color=OKABE_ITO[0], edgecolor="white", ax=axes[0])
axes[0].axvline(np.mean(longest_path), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(longest_path):.2f}")
axes[0].set_title("Critical-path depth (DAG longest path)")
axes[0].set_xlabel("Depth in steps")
axes[0].set_ylabel("Plan count")
axes[0].legend()
sns.despine(ax=axes[0])
sns.histplot(par_ratio, bins=20, color=OKABE_ITO[2], edgecolor="white", ax=axes[1])
axes[1].axvline(np.mean(par_ratio), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(par_ratio):.2f}")
axes[1].set_title("Parallelizable ratio (1 - depth / nodes)")
axes[1].set_xlabel("Ratio")
axes[1].set_ylabel("Plan count")
axes[1].legend()
sns.despine(ax=axes[1])
fig.suptitle("B3 · DAG topology", y=1.05, fontsize=12, fontweight="semibold")
save_fig("B3_dag_metrics.png")


# ======================================================================
# Section C — Agent design (roles, families, tools)
# ======================================================================
print("\n" + "=" * 70 + "\nC. AGENT DESIGN & TOOL USAGE\n" + "=" * 70)

role_counter, family_counter = Counter(), Counter()
tools_per_agent, zero_tool = [], 0
tool_counter = Counter()
plan_family_set = []
plan_tool_set = []
roles_per_plan = []
unique_roles_per_plan = []

for r in records:
    fams_in_plan = set()
    tools_in_plan = set()
    roles = []
    for a in r["plan"]["output"]["agents"]:
        role = a["agent_role"]
        roles.append(role)
        role_counter[role] += 1
        fam = classify_role(role)
        family_counter[fam] += 1
        fams_in_plan.add(fam)
        tools = a.get("tools", []) or []
        tools_per_agent.append(len(tools))
        if not tools:
            zero_tool += 1
        for t in tools:
            tool_counter[t] += 1
            tools_in_plan.add(t)
    roles_per_plan.append(len(roles))
    unique_roles_per_plan.append(len(set(roles)))
    plan_family_set.append(fams_in_plan)
    plan_tool_set.append(tools_in_plan)

STATS["C_agents"] = {
    "total_agent_instances": sum(roles_per_plan),
    "unique_agent_roles_global": len(role_counter),
    "top_roles": role_counter.most_common(25),
    "roles_per_plan_mean": safe_mean(roles_per_plan),
    "unique_roles_per_plan_mean": safe_mean(unique_roles_per_plan),
    "duplicate_role_plans": sum(1 for t, u in zip(roles_per_plan, unique_roles_per_plan) if t != u),
    "tools_per_agent_mean": safe_mean(tools_per_agent),
    "zero_tool_agents_pct": zero_tool / max(len(tools_per_agent), 1),
    "tools_global_unique": len(tool_counter),
    "tools_global_counts": dict(tool_counter.most_common()),
    "role_family_distribution": dict(family_counter),
}
print(f"  unique_agent_roles_global: {len(role_counter)}")
print(f"  top10 roles: {role_counter.most_common(10)}")
print(f"  tools_global_unique: {len(tool_counter)}  -> {dict(tool_counter)}")
print(f"  role family distribution: {dict(family_counter)}")

# --- C1: tools per agent (discrete bar)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
c = Counter(tools_per_agent)
ks = sorted(c.keys())
bars = ax.bar(ks, [c[k] for k in ks], color=OKABE_ITO[0], edgecolor="white")
for k in ks:
    ax.text(k, c[k], f"{c[k]:,}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Tools per agent")
ax.set_ylabel("Agent count")
ax.set_title("C1 · Tools attached per agent")
sns.despine()
save_fig("C1_tools_per_agent.png")

# --- C2: tool distribution
fig, ax = plt.subplots(figsize=(8, 3.8))
items = tool_counter.most_common()
names = [x[0] for x in items]
vals = [x[1] for x in items]
bars = ax.barh(names[::-1], vals[::-1], color=sns.color_palette("crest", len(names)),
               edgecolor="white")
for i, v in enumerate(vals[::-1]):
    ax.text(v + max(vals)*0.005, i, f" {v:,}", va="center", fontsize=9)
ax.set_title(f"C2 · Tool inventory (n={len(names)} distinct tools)")
ax.set_xlabel("Occurrences across all agents")
sns.despine()
save_fig("C2_tool_inventory.png")

# --- C3: role family distribution (donut)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
fam_items = sorted(family_counter.items(), key=lambda x: -x[1])
labels = [f"{k}\n({v:,})" for k, v in fam_items]
ax.pie([v for _, v in fam_items], labels=labels,
       autopct="%1.1f%%", startangle=90,
       wedgeprops=dict(width=0.45, edgecolor="white"),
       colors=sns.color_palette("Set2", len(fam_items)),
       textprops={"fontsize": 9})
ax.set_title("C3 · Agent role-family distribution")
save_fig("C3_role_families.png")

# --- C4: UpSet of role-family combos per plan
fam_order = [f for f, _ in fam_items]
combo_counter = Counter()
for s in plan_family_set:
    combo_counter[tuple(sorted(s, key=lambda x: fam_order.index(x)))] += 1
top_combos = combo_counter.most_common(15)

fig = plt.figure(figsize=(12, 6))
gs = plt.GridSpec(2, 1, height_ratios=[2, 1.3], hspace=0.08)
ax_bar = fig.add_subplot(gs[0])
ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)
x = np.arange(len(top_combos))
counts = [c for _, c in top_combos]
combos = [c for c, _ in top_combos]
colors = sns.color_palette("crest", len(top_combos))
ax_bar.bar(x, counts, color=colors, edgecolor="white", width=0.6)
for i, cnt in enumerate(counts):
    ax_bar.text(i, cnt, f"{cnt}\n({cnt/N*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)
ax_bar.set_ylabel("Plans")
ax_bar.set_title("C4 · Top-15 role-family compositions per plan  (UpSet)")
ax_bar.tick_params(axis="x", labelbottom=False)
sns.despine(ax=ax_bar)
for i, combo in enumerate(combos):
    active = [fam_order.index(f) for f in combo]
    for j, fam in enumerate(fam_order):
        if fam in combo:
            ax_dot.scatter(i, j, color="black", s=80, zorder=3)
        else:
            ax_dot.scatter(i, j, facecolors="none", edgecolors="lightgray", s=40, zorder=2)
    if len(active) > 1:
        ax_dot.plot([i, i], [min(active), max(active)], color="black", linewidth=1.5)
ax_dot.set_yticks(range(len(fam_order)))
ax_dot.set_yticklabels(fam_order)
ax_dot.set_xticks([])
ax_dot.invert_yaxis()
sns.despine(ax=ax_dot, left=False, bottom=True)
save_fig("C4_role_family_upset.png")

# --- C5: top-30 raw roles
fig, ax = plt.subplots(figsize=(9, 7))
top_roles = role_counter.most_common(30)
names = [x[0] for x in top_roles][::-1]
vals = [x[1] for x in top_roles][::-1]
ax.barh(names, vals, color=sns.color_palette("crest_r", len(names)),
        edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + max(vals)*0.005, i, f" {v}", va="center", fontsize=8)
ax.set_title(f"C5 · Top-30 most frequent agent_roles "
             f"(global unique = {len(role_counter):,})")
ax.set_xlabel("Occurrences")
sns.despine()
save_fig("C5_top30_roles.png")


# ======================================================================
# Section D — Learner profile portrait (top_tags landscape)
# ======================================================================
print("\n" + "=" * 70 + "\nD. LEARNER PROFILE PORTRAIT\n" + "=" * 70)

tag_counter = Counter()
tag_pair_counter = Counter()
per_record_tags = []
desc_len_by_skill_count = defaultdict(list)
for r in records:
    desc, sk = extract_learner(r["plan"]["input"]["learner"])
    sk_l = [t.strip().lower() for t in sk if t]
    per_record_tags.append(sk_l)
    for t in sk_l:
        tag_counter[t] += 1
    for a, b in combinations(sorted(set(sk_l)), 2):
        tag_pair_counter[(a, b)] += 1
    desc_len_by_skill_count[len(sk_l)].append(len(desc))

STATS["D_learner"] = {
    "unique_tags": len(tag_counter),
    "top_30_tags": tag_counter.most_common(30),
    "top_15_pairs": tag_pair_counter.most_common(15),
    "tags_per_profile_mean": safe_mean([len(t) for t in per_record_tags]),
}
print(f"  unique tags: {len(tag_counter)}")
print(f"  top10 tags: {tag_counter.most_common(10)}")
print(f"  top5 co-occurring pairs: {tag_pair_counter.most_common(5)}")

# D1: top tags
fig, ax = plt.subplots(figsize=(9, 7))
top = tag_counter.most_common(30)
names = [x[0] for x in top][::-1]
vals = [x[1] for x in top][::-1]
ax.barh(names, vals, color=sns.color_palette("crest_r", len(names)),
        edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + max(vals)*0.005, i, f" {v}", va="center", fontsize=8)
ax.set_title(f"D1 · Top-30 learner tags (global unique = {len(tag_counter):,})")
ax.set_xlabel("Records mentioning this tag")
sns.despine()
save_fig("D1_top_tags.png")

# D2: tag co-occurrence heatmap (top 15)
top15 = [t for t, _ in tag_counter.most_common(15)]
mat = np.zeros((15, 15))
for i, a in enumerate(top15):
    for j, b in enumerate(top15):
        if i == j:
            mat[i, j] = tag_counter[a]
        else:
            key = tuple(sorted([a, b]))
            mat[i, j] = tag_pair_counter.get(key, 0)
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(mat, annot=True, fmt=".0f", cmap="crest",
            xticklabels=top15, yticklabels=top15, ax=ax,
            cbar_kws={"label": "co-occurrence count (diag = total)"})
ax.set_title("D2 · Learner top-tags co-occurrence (top-15)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
save_fig("D2_tag_cooccurrence.png")


# ======================================================================
# Section E — Pedagogy patterns (10 strategies) & three-act skeleton
# ======================================================================
print("\n" + "=" * 70 + "\nE. PEDAGOGY PATTERNS\n" + "=" * 70)


def plan_full_text(r):
    out = r["plan"]["output"]
    parts = []
    for a in out["agents"]:
        parts.append(a.get("goal", ""))
        parts.append(agent_desc_text(a))
    for s in out["subtasks"]:
        parts.append(s.get("name", ""))
        parts.append(s.get("subtask_objective", ""))
        for st in s.get("steps", []):
            parts.append(st.get("objective", ""))
            parts.append(st.get("instruction", ""))
            parts.append(st.get("expected_output", ""))
    return "\n".join(parts)


pat_present_plan = {k: 0 for k in PEDAGOGY_PATTERNS}
pat_step_count = {k: 0 for k in PEDAGOGY_PATTERNS}
pat_per_plan = []  # list of (set of patterns) for diversity

skeleton_per_plan = []  # 3-tuple of (act_present, app_present, con_present)

for r in records:
    full = plan_full_text(r)
    present = set()
    for k, pat in PEDAGOGY_PATTERNS.items():
        if pat.search(full):
            present.add(k)
            pat_present_plan[k] += 1
    pat_per_plan.append(present)

    # count step-level hits
    for s in r["plan"]["output"]["subtasks"]:
        for st in s.get("steps", []):
            t = (st.get("instruction", "") + " " + st.get("objective", "")).lower()
            for k, pat in PEDAGOGY_PATTERNS.items():
                if pat.search(t):
                    pat_step_count[k] += 1

    sub_text = " ".join(s.get("name", "") + " " + s.get("subtask_objective", "")
                        for s in r["plan"]["output"]["subtasks"])
    skeleton_per_plan.append((
        bool(SKELETON_ACTIVATE.search(sub_text)),
        bool(SKELETON_APPLY.search(sub_text)),
        bool(SKELETON_CONSOLID.search(sub_text)),
    ))

pat_per_plan_count = [len(s) for s in pat_per_plan]

STATS["E_pedagogy"] = {
    "pattern_plan_coverage_pct": {k: v / N for k, v in pat_present_plan.items()},
    "pattern_step_count": pat_step_count,
    "patterns_per_plan_mean": safe_mean(pat_per_plan_count),
    "patterns_per_plan_median": float(np.median(pat_per_plan_count)),
    "skeleton_full_triple_pct":
        sum(1 for a, b, c in skeleton_per_plan if a and b and c) / N,
    "skeleton_breakdown": {
        "activate":    sum(1 for a, b, c in skeleton_per_plan if a) / N,
        "apply":       sum(1 for a, b, c in skeleton_per_plan if b) / N,
        "consolidate": sum(1 for a, b, c in skeleton_per_plan if c) / N,
    },
}
for k, v in STATS["E_pedagogy"].items():
    print(f"  {k}: {v}")

# --- E1: pedagogy pattern coverage (plan-level)
fig, ax = plt.subplots(figsize=(9, 4.5))
items = sorted(pat_present_plan.items(), key=lambda x: -x[1])
names = [x[0] for x in items]
vals = [x[1] for x in items]
pct = [v / N for v in vals]
bars = ax.barh(names[::-1], pct[::-1],
               color=sns.color_palette("crest_r", len(items)),
               edgecolor="white")
for i, (v, p) in enumerate(zip(vals[::-1], pct[::-1])):
    ax.text(p + 0.005, i, f" {v:,}  ({p:.1%})", va="center", fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_xlabel("Fraction of plans containing the pattern (plan-level)")
ax.set_title("E1 · Pedagogy strategy coverage across plans")
sns.despine()
save_fig("E1_pedagogy_coverage.png")

# --- E2: patterns per plan (diversity)
fig, ax = plt.subplots(figsize=(7, 3.8))
c = Counter(pat_per_plan_count)
ks = sorted(c.keys())
bars = ax.bar(ks, [c[k] for k in ks], color=OKABE_ITO[4], edgecolor="white")
for k in ks:
    ax.text(k, c[k], f"{c[k]}", ha="center", va="bottom", fontsize=8)
ax.axvline(np.mean(pat_per_plan_count), color="red", linestyle="--", linewidth=1.2,
           label=f"Mean={np.mean(pat_per_plan_count):.2f}")
ax.set_xlabel("Number of distinct pedagogy patterns in a plan")
ax.set_ylabel("Plan count")
ax.set_title("E2 · Pedagogy diversity per plan")
ax.legend()
sns.despine()
save_fig("E2_pedagogy_diversity.png")

# --- E3: scaffolding skeleton (Activate-Apply-Consolidate)
sk_cov = STATS["E_pedagogy"]["skeleton_breakdown"]
sk_full = STATS["E_pedagogy"]["skeleton_full_triple_pct"]
fig, ax = plt.subplots(figsize=(6, 3.8))
names = ["Activate", "Apply", "Consolidate", "Full triple"]
vals = [sk_cov["activate"], sk_cov["apply"], sk_cov["consolidate"], sk_full]
bars = ax.bar(names, vals, color=[OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[5]],
              edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v + 0.01, f"{v:.1%}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Fraction of plans")
ax.set_title("E3 · Three-act scaffolding skeleton coverage")
sns.despine()
save_fig("E3_scaffolding_skeleton.png")

# --- E4: pedagogy pattern co-occurrence heatmap (binary jaccard between patterns over plans)
pats = list(PEDAGOGY_PATTERNS)
M = np.zeros((len(pats), len(pats)))
for i, a in enumerate(pats):
    for j, b in enumerate(pats):
        if i == j:
            M[i, j] = pat_present_plan[a]
        else:
            both = sum(1 for s in pat_per_plan if a in s and b in s)
            M[i, j] = both
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(M, annot=True, fmt=".0f", cmap="crest",
            xticklabels=pats, yticklabels=pats, ax=ax,
            cbar_kws={"label": "Plan count (diag = single coverage)"})
ax.set_title("E4 · Pedagogy pattern co-occurrence across plans")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
save_fig("E4_pedagogy_cooccurrence.png")


# ======================================================================
# Section F — Personalization
# ======================================================================
print("\n" + "=" * 70 + "\nF. PERSONALIZATION SIGNAL\n" + "=" * 70)

# F.1 — per-record textual personalization signals
sec_pers_pct_records = []   # fraction of step instructions with 2nd-person pronoun
bridge_anchor_count = []    # # of anchor phrases per plan
skill_in_plan_ratio = []
desc_token_in_plan_ratio = []
plans_with_at_least_one_skill_mentioned = 0

for r in records:
    desc, sk = extract_learner(r["plan"]["input"]["learner"])
    full = plan_full_text(r).lower()
    # second-person at step level
    sp_hits, total_steps = 0, 0
    for s in r["plan"]["output"]["subtasks"]:
        for st in s.get("steps", []):
            total_steps += 1
            if SECOND_PERSON.search(st.get("instruction", "")):
                sp_hits += 1
    sec_pers_pct_records.append(sp_hits / total_steps if total_steps else 0)
    # bridge anchors
    bridge_anchor_count.append(sum(full.count(a) for a in BRIDGE_ANCHORS))
    # skill hit ratio
    if sk:
        hits = sum(1 for s in sk if str(s).lower() in full)
        skill_in_plan_ratio.append(hits / len(sk))
        if hits >= 1:
            plans_with_at_least_one_skill_mentioned += 1
    # desc token hit
    toks = [t for t in normalize_token(desc) if len(t) >= 4]
    if toks:
        desc_token_in_plan_ratio.append(
            sum(1 for t in toks if t in full) / len(toks))

# F.2 — same-question cross-profile divergence
qid_to_records = defaultdict(list)
for r in records:
    qid_to_records[str(r["question_id"])].append(r)


def plan_features(r):
    out = r["plan"]["output"]
    roles = [a["agent_role"] for a in out["agents"]]
    families = [classify_role(a["agent_role"]) for a in out["agents"]]
    tools = set()
    for a in out["agents"]:
        for t in a.get("tools", []) or []:
            tools.add(t)
    subtask_tokens = set()
    for s in out["subtasks"]:
        subtask_tokens.update(normalize_token(s.get("name", "")))
        subtask_tokens.update(normalize_token(s.get("subtask_objective", "")))
    full = plan_full_text(r).lower()
    pats_present = {k for k, pat in PEDAGOGY_PATTERNS.items() if pat.search(full)}
    n_step = sum(len(s.get("steps", [])) for s in out["subtasks"])
    return {"roles": roles, "families": families, "tools": tools,
            "subtask_tokens": subtask_tokens,
            "pedagogy": pats_present, "n_step": n_step}


role_jac, fam_jac, tool_jac, subtask_jac, ped_jac = [], [], [], [], []
step_diff = []
intent_pair_diverg = defaultdict(list)  # intent -> [1 - role_jaccard]
pair_count = 0
for qid, lst in qid_to_records.items():
    if len(lst) < 2:
        continue
    feats = [plan_features(r) for r in lst]
    pi = primary_intent(qid)
    for i, j in combinations(range(len(feats)), 2):
        a, b = feats[i], feats[j]
        jr = jaccard(a["roles"], b["roles"])
        jf = jaccard(a["families"], b["families"])
        jt = jaccard(a["tools"], b["tools"])
        js = jaccard(a["subtask_tokens"], b["subtask_tokens"])
        jp = jaccard(a["pedagogy"], b["pedagogy"])
        role_jac.append(jr)
        fam_jac.append(jf)
        tool_jac.append(jt)
        subtask_jac.append(js)
        ped_jac.append(jp)
        step_diff.append(abs(a["n_step"] - b["n_step"]))
        intent_pair_diverg[pi].append(1 - jr)
        pair_count += 1

STATS["F_personalization"] = {
    "second_person_ratio_per_plan_mean": safe_mean(sec_pers_pct_records),
    "bridge_anchor_count_per_plan_mean": safe_mean(bridge_anchor_count),
    "plans_with_at_least_one_skill_mentioned": plans_with_at_least_one_skill_mentioned,
    "plans_with_at_least_one_skill_pct": plans_with_at_least_one_skill_mentioned / N,
    "skill_in_plan_ratio_mean": safe_mean(skill_in_plan_ratio),
    "skill_in_plan_ratio_median": float(np.median(skill_in_plan_ratio)) if skill_in_plan_ratio else 0,
    "desc_token_in_plan_ratio_mean": safe_mean(desc_token_in_plan_ratio),
    "cross_profile_pairs": pair_count,
    "cross_profile_jaccard_means": {
        "agent_role":   safe_mean(role_jac),
        "role_family":  safe_mean(fam_jac),
        "tools":        safe_mean(tool_jac),
        "subtask_tokens": safe_mean(subtask_jac),
        "pedagogy":     safe_mean(ped_jac),
    },
    "cross_profile_step_diff_mean": safe_mean(step_diff),
}
for k, v in STATS["F_personalization"].items():
    print(f"  {k}: {v}")

# --- F1: skill hit ratio + 2nd-person ratio + bridge anchors
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(skill_in_plan_ratio, bins=20, color=OKABE_ITO[0], edgecolor="white", ax=axes[0])
axes[0].axvline(np.mean(skill_in_plan_ratio), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(skill_in_plan_ratio):.2%}")
axes[0].set_title("F1a · Fraction of learner top_tags\nappearing in plan text")
axes[0].set_xlabel("Hit ratio")
axes[0].set_ylabel("Plans")
axes[0].legend()
sns.despine(ax=axes[0])

sns.histplot(sec_pers_pct_records, bins=20, color=OKABE_ITO[1], edgecolor="white", ax=axes[1])
axes[1].axvline(np.mean(sec_pers_pct_records), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(sec_pers_pct_records):.1%}")
axes[1].set_title("F1b · Steps containing 2nd-person\npronoun (you/your/the learner)")
axes[1].set_xlabel("Per-plan ratio")
axes[1].set_ylabel("Plans")
axes[1].legend()
sns.despine(ax=axes[1])

bins = range(0, max(max(bridge_anchor_count) + 1, 2))
sns.histplot(bridge_anchor_count, bins=bins, color=OKABE_ITO[2], edgecolor="white",
             discrete=True, ax=axes[2])
axes[2].axvline(np.mean(bridge_anchor_count), color="red", linestyle="--", linewidth=1.2,
                label=f"Mean={np.mean(bridge_anchor_count):.2f}")
axes[2].set_title("F1c · Bridge-anchor phrases per plan\n('your portfolio', 'as a beginner', ...)")
axes[2].set_xlabel("Anchor occurrences in plan text")
axes[2].set_ylabel("Plans")
axes[2].legend()
sns.despine(ax=axes[2])
fig.suptitle("F1 · Personalization signal in plan text",
             y=1.05, fontsize=12, fontweight="semibold")
save_fig("F1_personalization_signal.png")

# --- F2: cross-profile Jaccard boxplot
fig, ax = plt.subplots(figsize=(9, 4.5))
df = pd.DataFrame({
    "Agent roles": role_jac,
    "Role families": fam_jac,
    "Tools": tool_jac,
    "Subtask tokens": subtask_jac,
    "Pedagogy patterns": ped_jac,
})
sns.boxplot(data=df, ax=ax, palette="Set2", fliersize=2, linewidth=1.2)
ax.set_ylabel("Jaccard similarity (same query, different profile)")
ax.set_title(f"F2 · Cross-profile plan divergence  (n={pair_count:,} pairs)\n"
             "Lower = more profile-specific personalization")
sns.despine()
save_fig("F2_cross_profile_jaccard.png")


# ======================================================================
# Section G — Intent x complexity x pedagogy
# ======================================================================
print("\n" + "=" * 70 + "\nG. INTENT × COMPLEXITY × PEDAGOGY\n" + "=" * 70)

rows = []
for i, r in enumerate(records):
    pi = primary_intent(r["question_id"])
    plan_pats = pat_per_plan[i]
    rows.append({
        "qid": str(r["question_id"]),
        "primary_intent": pi,
        "n_agents": n_agents[i],
        "n_subtasks": n_subtasks[i],
        "n_steps": n_steps[i],
        "n_loops": loops_per_plan[i],
        "n_human_input": human_input_per_plan[i],
        "longest_path": longest_path[i],
        "n_tools_total": n_tools_total_plan[i],
        "n_ped_patterns": len(plan_pats),
        "has_loop": 1 if loops_per_plan[i] > 0 else 0,
    })
df_plans = pd.DataFrame(rows)

intent_summary = df_plans.groupby("primary_intent").agg(
    n=("qid", "size"),
    agents=("n_agents", "mean"),
    subtasks=("n_subtasks", "mean"),
    steps=("n_steps", "mean"),
    longest_path=("longest_path", "mean"),
    loops_mean=("n_loops", "mean"),
    loop_pct=("has_loop", "mean"),
    human_input_mean=("n_human_input", "mean"),
    tools=("n_tools_total", "mean"),
    ped_patterns=("n_ped_patterns", "mean"),
).round(2)
ord_idx = ([i for i in INTENT_ORDER if i in intent_summary.index] +
           [i for i in intent_summary.index if i not in INTENT_ORDER])
intent_summary = intent_summary.reindex(ord_idx)
print(intent_summary.to_string())
STATS["G_intent_summary"] = intent_summary.reset_index().to_dict("records")

# --- G1: heatmap of normalized means
cols = ["agents", "subtasks", "steps", "longest_path", "loops_mean",
        "loop_pct", "human_input_mean", "tools", "ped_patterns"]
norm = intent_summary[cols].copy()
for c in cols:
    norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(norm, annot=intent_summary[cols], fmt=".2f",
            cmap="crest", ax=ax, cbar_kws={"label": "min-max normalized"})
ax.set_title("G1 · Plan complexity & pedagogy by primary query intent\n"
             "(cells = raw means · colors = min-max normalized per column)")
ax.set_xlabel("")
plt.xticks(rotation=20, ha="right")
save_fig("G1_intent_complexity_heatmap.png")

# --- G2: pedagogy pattern coverage by intent
pat_by_intent = defaultdict(lambda: Counter())
intent_counts = Counter()
for i, r in enumerate(records):
    pi = primary_intent(r["question_id"])
    intent_counts[pi] += 1
    for p in pat_per_plan[i]:
        pat_by_intent[pi][p] += 1

pats = list(PEDAGOGY_PATTERNS)
intents_present = [i for i in INTENT_ORDER if i in intent_counts]
M = np.zeros((len(pats), len(intents_present)))
for r, p in enumerate(pats):
    for c, intent in enumerate(intents_present):
        M[r, c] = pat_by_intent[intent].get(p, 0) / max(intent_counts[intent], 1)
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.heatmap(M, annot=True, fmt=".2f", cmap="crest",
            xticklabels=intents_present, yticklabels=pats, ax=ax,
            cbar_kws={"label": "Fraction of plans of that intent"})
ax.set_title("G2 · Pedagogy pattern usage by primary query intent")
plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)
save_fig("G2_pedagogy_by_intent.png")

# --- G3: divergence (1 - role Jaccard) by intent
div_rows = []
for intent in INTENT_ORDER + ["UNKNOWN"]:
    vals = intent_pair_diverg.get(intent, [])
    if vals:
        div_rows.append({"intent": intent, "mean_diff": np.mean(vals),
                         "n_pairs": len(vals)})
div_df = pd.DataFrame(div_rows)
if not div_df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=div_df, x="intent", y="mean_diff", hue="intent",
                palette="crest", legend=False, ax=ax)
    for i, row in div_df.reset_index(drop=True).iterrows():
        ax.text(i, row["mean_diff"], f"{row['mean_diff']:.2f}\n(n={row['n_pairs']})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("G3 · Cross-profile personalization divergence by intent")
    ax.set_ylabel("Mean (1 − Jaccard of agent_role)")
    plt.xticks(rotation=20, ha="right")
    sns.despine()
    save_fig("G3_divergence_by_intent.png")


# ======================================================================
# Section H — Complexity vs profile_index (does plan adapt with profile?)
# ======================================================================
print("\n" + "=" * 70 + "\nH. COMPLEXITY VS PROFILE_INDEX\n" + "=" * 70)

df_plans["profile_index"] = [r["profile_index"] for r in records]
prof_summary = df_plans.groupby("profile_index").agg(
    n=("qid", "size"),
    agents=("n_agents", "mean"),
    subtasks=("n_subtasks", "mean"),
    steps=("n_steps", "mean"),
    loops_pct=("has_loop", "mean"),
    human_input=("n_human_input", "mean"),
    ped=("n_ped_patterns", "mean"),
).round(3)
print(prof_summary.to_string())
STATS["H_by_profile_index"] = prof_summary.reset_index().to_dict("records")

fig, ax = plt.subplots(figsize=(9, 4.5))
cols = ["agents", "subtasks", "steps", "loops_pct", "human_input", "ped"]
norm = prof_summary[cols].copy()
for c in cols:
    norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
sns.heatmap(norm, annot=prof_summary[cols], fmt=".2f", cmap="crest",
            ax=ax, cbar_kws={"label": "min-max normalized"})
ax.set_title("H1 · Plan signal by profile_index  (cells = raw means)")
ax.set_xlabel("")
ax.set_ylabel("profile_index")
save_fig("H1_profile_index_heatmap.png")


# ======================================================================
# Section I — Schema & referential validity
# ======================================================================
print("\n" + "=" * 70 + "\nI. SCHEMA / REFERENTIAL VALIDITY\n" + "=" * 70)


def _has_cycle(nodes, parents):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    stack = []
    for start in list(nodes):
        if color[start] != WHITE:
            continue
        stack.append((start, iter(parents.get(start, []))))
        color[start] = GRAY
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is None:
                color[u] = BLACK
                stack.pop()
                continue
            if v not in color:
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE:
                color[v] = GRAY
                stack.append((v, iter(parents.get(v, []))))
    return False


q_unknown_agent = q_unknown_depends = q_unknown_exec = q_missing_exec = 0
q_cycle = q_bad = 0
loop_singular = loop_plural = 0
for r in records:
    out = r["plan"]["output"]
    role_names = {a["agent_role"] for a in out["agents"]}
    step_ids = set()
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            step_ids.add(st.get("id"))
    bad = False
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            if st.get("agent") not in role_names:
                q_unknown_agent += 1; bad = True; break
        if bad:
            break
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            for d in (st.get("depends_on") or []):
                if d not in step_ids:
                    q_unknown_depends += 1; bad = True; break
            else:
                continue
            break
        else:
            continue
        break
    flat = []
    for x in out["execution_order"]:
        if isinstance(x, str):
            flat.append(x)
        elif isinstance(x, dict) and "loop" in x:
            ld = x["loop"]
            if "steps" in ld:
                loop_plural += 1
                flat.extend(ld["steps"])
            elif "step" in ld:
                loop_singular += 1
                flat.extend([ld["step"]] if isinstance(ld["step"], str) else ld["step"])
    eo_set = set(flat)
    if eo_set - step_ids:
        q_unknown_exec += 1; bad = True
    if step_ids - eo_set:
        q_missing_exec += 1; bad = True
    parents = defaultdict(list)
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            for d in (st.get("depends_on") or []):
                parents[st["id"]].append(d)
    if _has_cycle(step_ids, parents):
        q_cycle += 1; bad = True
    if bad:
        q_bad += 1

STATS["I_quality"] = {
    "plans_with_unknown_agent_ref": q_unknown_agent,
    "plans_with_unknown_depends_on": q_unknown_depends,
    "plans_with_unknown_step_in_exec": q_unknown_exec,
    "plans_with_missing_step_in_exec": q_missing_exec,
    "plans_with_cycle": q_cycle,
    "plans_with_any_issue": q_bad,
    "valid_plans": N - q_bad,
    "valid_rate": (N - q_bad) / N,
    "loop_schema_plural_steps": loop_plural,
    "loop_schema_singular_step": loop_singular,
}
for k, v in STATS["I_quality"].items():
    print(f"  {k}: {v}")

fig, ax = plt.subplots(figsize=(8, 3.8))
labels = ["unknown_agent_ref", "unknown_depends_on", "unknown_step_in_exec",
          "missing_step_in_exec", "cycle"]
vals = [q_unknown_agent, q_unknown_depends, q_unknown_exec, q_missing_exec, q_cycle]
bars = ax.bar(labels, vals,
              color=[OKABE_ITO[5]]*5, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, str(v),
            ha="center", va="bottom", fontsize=9)
ax.set_title(f"I1 · Schema / referential issues "
             f"(valid plans = {N - q_bad}/{N} = {(N-q_bad)/N:.1%})")
ax.set_ylabel("Plan count")
plt.xticks(rotation=18, ha="right")
sns.despine()
save_fig("I1_quality.png")


# ======================================================================
# Save stats.json
# ======================================================================
def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _conv(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_conv(x) for x in o]
    if isinstance(o, tuple):
        return [_conv(x) for x in o]
    return o


with open(STATS_OUT, "w", encoding="utf-8") as f:
    json.dump(_conv(STATS), f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print(f"Saved: {STATS_OUT}")
print(f"Figures: {FIG_DIR}")
print("=" * 70)
