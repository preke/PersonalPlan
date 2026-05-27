"""
Paper-grade analysis of MAP-PPL final dataset (3043 records, 1730 unique qids).

Sections:
  A) Overview
  B) Train / Dev / Test split balance              (NEW)
  C) Plan complexity (agents, subtasks, steps, DAG)
  D) Agent design (roles, role families, tools)
  E) Personalization / one-to-many                  (EXPANDED)
  F) Intent x complexity
  G) Lexical diversity                              (NEW)
  H) Tool / Role co-occurrence + Zipf               (NEW)
  I) Schema / referential validity

Outputs:
  - figures/*.png        publication-quality figures
  - stats.json           machine-readable numbers
"""

import json
import re
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_PATH = HERE.parent / "multi_agent_dataset_filtered_qap.jsonl"
LABELS_PATH = ROOT / "the_construction_of_MAPLE_datasets/task_3/classified_results.jsonl"
SPLIT_PATH = ROOT / "splits/maple_split_v1.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
STATS_OUT = HERE / "stats.json"

# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
# EMNLP/ACL-style paper figures: serif, compact font sizes (the figures
# are typically scaled to ~3 in. width in a 0.48\linewidth two-column
# layout, so smaller base sizes keep tick/label text legible without
# dominating the panel).
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
PALETTE_MAIN = ["#2E5077", "#4F8A8B", "#E07A5F", "#C7A33E",
                "#7B61FF", "#5A6E80", "#9C7A5F", "#6C8E68"]
PALETTE_SET = ["#4F8A8B", "#E07A5F", "#2E5077", "#C7A33E",
               "#7B61FF", "#9C7A5F", "#6C8E68", "#5A6E80"]
SPLIT_COLORS = {"train": "#2E5077", "dev": "#4F8A8B", "test": "#E07A5F"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif",
                   "Liberation Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#3A3A3A",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#3A3A3A",
    "ytick.color": "#3A3A3A",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "figure.dpi": 160,
    "savefig.facecolor": "white",
    "grid.color": "#E2E2E2",
    "grid.linewidth": 0.4,
})


def save_fig(name, dpi=220):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def add_value_labels(ax, bars, fmt="{:.0f}", offset=2, fs=9, color="#222222"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, color=color)


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


def agent_desc_text(agent):
    return agent.get("backstory") or agent.get("description") or ""


def extract_learner(lrn):
    if not lrn:
        return "", []
    desc = lrn.get("about_me") or lrn.get("self_description") or ""
    skills = lrn.get("top_tags") or lrn.get("skills") or []
    return desc.strip(), list(skills)


def normalize_token(s):
    return re.findall(r"[A-Za-z][A-Za-z0-9_+#\.-]+", s.lower())


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


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
records = load_jsonl(DATA_PATH)
N = len(records)

intent_labels = {}
for o in load_jsonl(LABELS_PATH):
    intent_labels[str(o["question_id"])] = o.get("labels", [])

split_def = json.load(open(SPLIT_PATH))
train_q = set(map(str, split_def["train_qids"]))
dev_q = set(map(str, split_def["dev_qids"]))
test_q = set(map(str, split_def["test_qids"]))


def split_of(qid):
    qid = str(qid)
    if qid in train_q:
        return "train"
    if qid in dev_q:
        return "dev"
    if qid in test_q:
        return "test"
    return "unassigned"


INTENT_ORDER = ["API_USAGE", "CONCEPTUAL", "DISCREPANCY", "ERRORS",
                "REVIEW", "API_CHANGE", "LEARNING"]
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
    rl = role.lower()
    for fam, kws in ROLE_FAMILIES:
        if any(k in rl for k in kws):
            return fam
    return "other"


def primary_intent(qid):
    labs = intent_labels.get(str(qid), [])
    return labs[0] if labs else "UNKNOWN"


STATS = {}

# ======================================================================
# A. OVERVIEW
# ======================================================================
qids = Counter(str(r["question_id"]) for r in records)
profile_idx_dist = Counter(r["profile_index"] for r in records)
unique_qids = len(qids)


def profile_key(p):
    desc, skills = extract_learner(p)
    return (desc, tuple(sorted(skills)))


unique_profiles = len(set(profile_key(r["plan"]["input"]["learner"]) for r in records))

learner_schema_counter = Counter()
for r in records:
    lrn = r["plan"]["input"].get("learner", {})
    learner_schema_counter[tuple(sorted(lrn.keys()))] += 1


def is_empty_profile(p):
    desc, skills = extract_learner(p)
    return not desc and not skills


empty_profile_count = sum(1 for r in records if is_empty_profile(r["plan"]["input"]["learner"]))
nonempty_profile_count = N - empty_profile_count

query_lens = [len(r["plan"]["input"]["query"]) for r in records]
desc_lens = []
skills_counts = []
for r in records:
    desc, skills = extract_learner(r["plan"]["input"]["learner"])
    desc_lens.append(len(desc))
    skills_counts.append(len(skills))

agent_desc_lens = []
goal_lens = []
for r in records:
    for a in r["plan"]["output"]["agents"]:
        agent_desc_lens.append(len(agent_desc_text(a)))
        goal_lens.append(len(a.get("goal", "")))

instr_lens = []
for r in records:
    for s in r["plan"]["output"]["subtasks"]:
        for st in s["steps"]:
            instr_lens.append(len(st.get("instruction", "")))

nprof_per_qid = Counter(qids.values())

STATS["A_overview"] = {
    "total_records": N,
    "unique_questions": unique_qids,
    "unique_profiles_by_text": unique_profiles,
    "rows_per_qid_mean": N / unique_qids,
    "empty_profile_records": empty_profile_count,
    "nonempty_profile_records": nonempty_profile_count,
    "nonempty_profile_pct": nonempty_profile_count / N,
    "learner_schema_distribution": {str(k): v for k, v in learner_schema_counter.items()},
    "questions_with_multi_profile": sum(1 for v in qids.values() if v > 1),
    "max_profiles_per_question": max(qids.values()),
    "profile_index_distribution": dict(profile_idx_dist),
    "nprof_per_qid_distribution": dict(nprof_per_qid),
    "query_len_chars": {
        "mean": float(np.mean(query_lens)), "median": float(np.median(query_lens)),
        "min": int(min(query_lens)), "max": int(max(query_lens)),
        "p25": float(np.percentile(query_lens, 25)),
        "p75": float(np.percentile(query_lens, 75)),
    },
    "desc_len_chars_all": {
        "mean": float(np.mean(desc_lens)), "median": float(np.median(desc_lens)),
        "min": int(min(desc_lens)), "max": int(max(desc_lens)),
    },
    "skills_per_profile_all": {
        "mean": float(np.mean(skills_counts)), "median": float(np.median(skills_counts)),
        "min": int(min(skills_counts)), "max": int(max(skills_counts)),
    },
    "agent_backstory_len_chars": {
        "mean": float(np.mean(agent_desc_lens)),
        "median": float(np.median(agent_desc_lens)),
    },
    "agent_goal_len_chars": {
        "mean": float(np.mean(goal_lens)),
        "median": float(np.median(goal_lens)),
    },
    "instruction_len_chars": {
        "mean": float(np.mean(instr_lens)),
        "median": float(np.median(instr_lens)),
    },
}

# --- A1: profile_index dist
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = sorted(profile_idx_dist.keys())
bars = ax.bar(ks, [profile_idx_dist[k] for k in ks],
              color=PALETTE_MAIN[2], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 25, f"{int(h):,}",
            ha="center", va="bottom", fontsize=7.5, color="#222222")
ax.set_xlabel("profile_index"); ax.set_ylabel("Number of records")
ax.set_title("Records per profile_index (across questions)")
sns.despine(ax=ax)
save_fig("A1_profile_index_distribution.png")

# --- A2: Query length (KDE + hist)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.histplot(query_lens, bins=40, color=PALETTE_MAIN[3], edgecolor="white",
             alpha=0.85, ax=ax, kde=True, line_kws={"linewidth": 1.6})
ax.axvline(np.mean(query_lens), color="#c0392b", linestyle="--", lw=1.4,
           label=f"mean={np.mean(query_lens):.0f}")
ax.axvline(np.median(query_lens), color="#d4a017", linestyle=":", lw=1.4,
           label=f"median={np.median(query_lens):.0f}")
ax.set_xlabel("Query length (characters)"); ax.set_ylabel("Number of records")
ax.set_title("Query length distribution")
ax.legend()
sns.despine(ax=ax)
save_fig("A2_query_length.png")

# --- A3: Self-description length
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.histplot(desc_lens, bins=40, color=PALETTE_MAIN[4], edgecolor="white",
             alpha=0.85, ax=ax, kde=True, line_kws={"linewidth": 1.6})
ax.axvline(np.mean(desc_lens), color="#c0392b", linestyle="--", lw=1.4,
           label=f"mean={np.mean(desc_lens):.0f}")
ax.set_xlabel("Self-description length (characters)"); ax.set_ylabel("Number of records")
ax.set_title("Learner `about_me` length")
ax.legend()
sns.despine(ax=ax)
save_fig("A3_selfdesc_length.png")

# --- A4: skills count
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = sorted(set(skills_counts))
counter = Counter(skills_counts)
bars = ax.bar(ks, [counter[k] for k in ks],
              color=PALETTE_MAIN[5], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 15, f"{int(h):,}",
            ha="center", va="bottom", fontsize=7.5, color="#222222")
ax.set_xlabel("Number of skills / top_tags"); ax.set_ylabel("Number of profiles")
ax.set_title("Skills per learner profile")
sns.despine(ax=ax)
save_fig("A4_skills_count.png")

# --- A5: nprof_per_qid bar
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ks = sorted(nprof_per_qid.keys())
bars = ax.bar(ks, [nprof_per_qid[k] for k in ks],
              color=PALETTE_MAIN[1], edgecolor="white", linewidth=1.5)
total_q = sum(nprof_per_qid.values())
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 8,
            f"{int(h):,}\n({h/total_q*100:.1f}%)",
            ha="center", va="bottom", fontsize=7.5, color="#222222")
ax.set_xlabel("# profiles bound to the same question_id")
ax.set_ylabel("Number of questions (qids)")
ax.set_title("Profile-fan-out per question (one-to-many)")
sns.despine(ax=ax)
save_fig("A5_nprof_per_qid.png")

# ======================================================================
# B. TRAIN / DEV / TEST SPLIT BALANCE  (NEW)
# ======================================================================
for r in records:
    r["__split__"] = split_of(r["question_id"])
    r["__primary_intent__"] = primary_intent(r["question_id"])

split_rows = Counter(r["__split__"] for r in records)
split_qids = {s: set() for s in ["train", "dev", "test"]}
for r in records:
    s = r["__split__"]
    if s in split_qids:
        split_qids[s].add(str(r["question_id"]))

# Intent x split table
intent_split_rows = pd.DataFrame(
    0, index=INTENT_ORDER + ["UNKNOWN"], columns=["train", "dev", "test"])
for r in records:
    pi = r["__primary_intent__"] if r["__primary_intent__"] in INTENT_ORDER else "UNKNOWN"
    if r["__split__"] in intent_split_rows.columns:
        intent_split_rows.loc[pi, r["__split__"]] += 1

# Per-split % within column
intent_split_pct = intent_split_rows.div(intent_split_rows.sum(axis=0), axis=1) * 100

# profile_index x split
pidx_split = pd.DataFrame(
    0, index=sorted(set(r["profile_index"] for r in records)),
    columns=["train", "dev", "test"])
for r in records:
    if r["__split__"] in pidx_split.columns:
        pidx_split.loc[r["profile_index"], r["__split__"]] += 1
pidx_split_pct = pidx_split.div(pidx_split.sum(axis=0), axis=1) * 100

# nprof x split
nprof_split = pd.DataFrame(
    0, index=sorted(nprof_per_qid.keys()), columns=["train", "dev", "test"])
for q, c in qids.items():
    s = split_of(q)
    if s in nprof_split.columns:
        nprof_split.loc[c, s] += 1


def _chi2(observed):
    obs = np.array(observed, dtype=float)
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    if total == 0:
        return None, None
    exp = row_sums @ col_sums / total
    mask = exp > 0
    chi2 = ((obs - exp) ** 2 / np.where(mask, exp, 1.0))[mask].sum()
    n = total
    r, c = obs.shape
    cramers_v = math.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else 0
    return float(chi2), float(cramers_v)


chi2_intent, v_intent = _chi2(intent_split_rows.values)
chi2_pidx, v_pidx = _chi2(pidx_split.values)
chi2_nprof, v_nprof = _chi2(nprof_split.values)

STATS["B_splits"] = {
    "rows": {s: int(split_rows[s]) for s in ["train", "dev", "test"]},
    "qids": {s: len(split_qids[s]) for s in ["train", "dev", "test"]},
    "intent_x_split": intent_split_rows.to_dict(),
    "intent_x_split_pct": intent_split_pct.round(2).to_dict(),
    "profile_index_x_split": pidx_split.to_dict(),
    "nprof_x_split": nprof_split.to_dict(),
    "balance_chi2": {
        "intent":         {"chi2": chi2_intent, "cramers_v": v_intent},
        "profile_index":  {"chi2": chi2_pidx,   "cramers_v": v_pidx},
        "nprof_per_qid":  {"chi2": chi2_nprof,  "cramers_v": v_nprof},
    },
}

# --- B1: rows + qids per split, side-by-side
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
splits = ["train", "dev", "test"]
row_counts = [split_rows[s] for s in splits]
qid_counts = [len(split_qids[s]) for s in splits]
colors = [SPLIT_COLORS[s] for s in splits]
b1 = axes[0].bar(splits, row_counts, color=colors, edgecolor="white", linewidth=1.5)
for b, v in zip(b1, row_counts):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 25,
                 f"{v:,}\n({v/N*100:.1f}%)", ha="center", va="bottom", fontsize=8)
axes[0].set_title(f"Row count per split (N={N:,})")
axes[0].set_ylabel("# rows")
sns.despine(ax=axes[0])
b2 = axes[1].bar(splits, qid_counts, color=colors, edgecolor="white", linewidth=1.5)
for b, v in zip(b2, qid_counts):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                 f"{v:,}\n({v/unique_qids*100:.1f}%)", ha="center",
                 va="bottom", fontsize=8)
axes[1].set_title(f"Unique question_id per split (total={unique_qids:,})")
axes[1].set_ylabel("# qids")
sns.despine(ax=axes[1])
save_fig("B1_split_sizes.png")

# --- B2: intent distribution per split (grouped + stacked %)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
intent_plot = intent_split_rows.copy()
intent_plot = intent_plot.loc[(intent_plot.sum(axis=1) > 0)]
intent_plot.plot(kind="bar", ax=axes[0], color=[SPLIT_COLORS[s] for s in splits],
                 edgecolor="white", width=0.78)
axes[0].set_title("Primary intent × split (counts)")
axes[0].set_xlabel("Primary intent")
axes[0].set_ylabel("# rows")
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(title="split", loc="upper right")
sns.despine(ax=axes[0])

# Normalized (within split) — to compare proportions
intent_pct = intent_plot.div(intent_plot.sum(axis=0), axis=1) * 100
intent_pct.T.plot(kind="bar", stacked=True, ax=axes[1],
                  color=sns.color_palette("Set2", len(intent_plot.index)),
                  edgecolor="white", width=0.65)
axes[1].set_title("Primary intent share within each split (%)")
axes[1].set_xlabel("Split")
axes[1].set_ylabel("% of rows")
axes[1].tick_params(axis="x", rotation=0)
axes[1].set_ylim(0, 100)
axes[1].legend(title="intent", bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=7.5)
sns.despine(ax=axes[1])
save_fig("B2_intent_x_split.png")

# --- B3: nprof_per_qid x split + profile_index x split
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
nprof_plot = nprof_split.copy()
nprof_plot.plot(kind="bar", ax=axes[0], color=[SPLIT_COLORS[s] for s in splits],
                edgecolor="white", width=0.78)
axes[0].set_title("# profiles per qid (stratification bucket) × split")
axes[0].set_xlabel("# profiles per qid")
axes[0].set_ylabel("# qids")
axes[0].legend(title="split")
sns.despine(ax=axes[0])

pidx_plot_pct = pidx_split.div(pidx_split.sum(axis=0), axis=1) * 100
pidx_plot_pct.plot(kind="bar", ax=axes[1],
                   color=[SPLIT_COLORS[s] for s in splits],
                   edgecolor="white", width=0.78)
axes[1].set_title("profile_index distribution within each split (%)")
axes[1].set_xlabel("profile_index")
axes[1].set_ylabel("% of split rows")
axes[1].legend(title="split")
sns.despine(ax=axes[1])
save_fig("B3_split_strata.png")

# --- B4: complexity stats per split (boxplots)
agents_per_plan = []
steps_per_plan = []
for r in records:
    agents_per_plan.append(len(r["plan"]["output"]["agents"]))
    steps_per_plan.append(
        sum(len(s.get("steps", [])) for s in r["plan"]["output"]["subtasks"]))

split_df = pd.DataFrame({
    "split": [r["__split__"] for r in records],
    "n_agents": agents_per_plan,
    "n_steps": steps_per_plan,
    "query_len": query_lens,
    "desc_len": desc_lens,
})
split_df = split_df[split_df["split"].isin(splits)].copy()

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, col, title in zip(axes,
                          ["n_agents", "n_steps", "query_len", "desc_len"],
                          ["# agents / plan", "# steps / plan",
                           "Query length (chars)", "Profile description (chars)"]):
    sns.boxplot(data=split_df, x="split", y=col, ax=ax,
                palette=SPLIT_COLORS, width=0.55, fliersize=2)
    ax.set_title(title)
    ax.set_xlabel("")
    sns.despine(ax=ax)
save_fig("B4_complexity_per_split.png")

# ======================================================================
# C. PLAN COMPLEXITY
# ======================================================================
per_plan_agents_total = []
per_plan_agents_unique = []
per_plan_subtasks_total = []
per_plan_subtasks_unique_names = []
per_plan_steps_total = []
per_plan_tools_total = []
per_plan_tools_unique = []

ds_total_agents = 0
ds_unique_agent_roles = set()
ds_total_subtasks = 0
ds_unique_subtask_names = set()
ds_total_steps = 0
ds_unique_tools = set()

n_agents = []
n_subtasks = []
n_steps = []
steps_per_subtask = []
human_input_per_plan = []
human_input_ratio_per_plan = []
loops_per_plan = []
loop_step_counts = []
loop_max_iter = []

for r in records:
    out = r["plan"]["output"]
    agents = out["agents"]
    subs = out["subtasks"]
    a_roles = [a["agent_role"] for a in agents]
    per_plan_agents_total.append(len(agents))
    per_plan_agents_unique.append(len(set(a_roles)))
    n_agents.append(len(agents))
    ds_total_agents += len(agents)
    ds_unique_agent_roles.update(a_roles)

    sub_names = [s.get("name", "") for s in subs]
    per_plan_subtasks_total.append(len(subs))
    per_plan_subtasks_unique_names.append(len(set(sub_names)))
    n_subtasks.append(len(subs))
    ds_total_subtasks += len(subs)
    ds_unique_subtask_names.update(sub_names)

    total_steps = 0
    hi = 0
    for s in subs:
        steps = s.get("steps", [])
        steps_per_subtask.append(len(steps))
        total_steps += len(steps)
        for st in steps:
            if st.get("requires_human_input"):
                hi += 1
    per_plan_steps_total.append(total_steps)
    n_steps.append(total_steps)
    ds_total_steps += total_steps
    human_input_per_plan.append(hi)
    human_input_ratio_per_plan.append(hi / total_steps if total_steps else 0.0)

    tools = []
    for a in agents:
        for t in a.get("tools", []) or []:
            tools.append(t)
            ds_unique_tools.add(t)
    per_plan_tools_total.append(len(tools))
    per_plan_tools_unique.append(len(set(tools)))

    nloops = 0
    for item in out["execution_order"]:
        if isinstance(item, dict) and "loop" in item:
            nloops += 1
            ld = item["loop"]
            if "steps" in ld:
                loop_step_counts.append(len(ld["steps"]))
            elif "step" in ld:
                loop_step_counts.append(
                    1 if isinstance(ld["step"], str) else len(ld["step"]))
            loop_max_iter.append(ld.get("max_iterations"))
    loops_per_plan.append(nloops)

STATS["C_counts_vs_unique"] = {
    "per_plan": {
        "agents_per_plan_mean": float(np.mean(per_plan_agents_total)),
        "agents_per_plan_median": float(np.median(per_plan_agents_total)),
        "unique_agent_roles_per_plan_mean": float(np.mean(per_plan_agents_unique)),
        "duplicate_agent_role_within_plan_count":
            sum(1 for t, u in zip(per_plan_agents_total, per_plan_agents_unique) if t != u),
        "subtasks_per_plan_mean": float(np.mean(per_plan_subtasks_total)),
        "unique_subtask_names_per_plan_mean": float(np.mean(per_plan_subtasks_unique_names)),
        "duplicate_subtask_name_within_plan_count":
            sum(1 for t, u in zip(per_plan_subtasks_total, per_plan_subtasks_unique_names) if t != u),
        "steps_per_plan_mean": float(np.mean(per_plan_steps_total)),
        "tools_per_plan_total_mean": float(np.mean(per_plan_tools_total)),
        "unique_tools_per_plan_mean": float(np.mean(per_plan_tools_unique)),
    },
    "dataset_total": {
        "total_agent_instances": ds_total_agents,
        "unique_agent_roles_global": len(ds_unique_agent_roles),
        "total_subtask_instances": ds_total_subtasks,
        "unique_subtask_names_global": len(ds_unique_subtask_names),
        "total_step_instances": ds_total_steps,
        "unique_tools_global": len(ds_unique_tools),
    },
}

STATS["C_complexity"] = {
    "agents_per_plan": {
        "mean": float(np.mean(n_agents)),
        "median": float(np.median(n_agents)),
        "min": int(min(n_agents)), "max": int(max(n_agents)),
        "distribution": dict(Counter(n_agents)),
    },
    "subtasks_per_plan": {
        "mean": float(np.mean(n_subtasks)),
        "median": float(np.median(n_subtasks)),
        "min": int(min(n_subtasks)), "max": int(max(n_subtasks)),
        "distribution": dict(Counter(n_subtasks)),
    },
    "steps_per_plan": {
        "mean": float(np.mean(n_steps)),
        "median": float(np.median(n_steps)),
        "min": int(min(n_steps)), "max": int(max(n_steps)),
    },
    "steps_per_subtask": {
        "mean": float(np.mean(steps_per_subtask)),
        "median": float(np.median(steps_per_subtask)),
    },
    "human_input_steps_per_plan_mean": float(np.mean(human_input_per_plan)),
    "human_input_step_ratio_overall": sum(human_input_per_plan) / sum(n_steps),
    "plans_with_loop": sum(1 for x in loops_per_plan if x > 0),
    "plans_with_loop_pct": sum(1 for x in loops_per_plan if x > 0) / N,
    "plans_with_multiple_loops": sum(1 for x in loops_per_plan if x > 1),
    "loops_total": int(sum(loops_per_plan)),
    "loop_step_count_mean": float(np.mean(loop_step_counts)) if loop_step_counts else 0,
    "loop_max_iter_distribution": dict(Counter(loop_max_iter)),
}

# C1 agents per plan
fig, ax = plt.subplots(figsize=(6.5, 3.8))
c = Counter(n_agents)
ks = sorted(c.keys())
bars = ax.bar(ks, [c[k] for k in ks],
              color=PALETTE_MAIN[3], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 30,
            f"{int(h):,}\n({h/N*100:.1f}%)", ha="center", va="bottom", fontsize=7.5)
ax.set_xlabel("# agents in plan"); ax.set_ylabel("# plans")
ax.set_title(f"Agents per plan (mean={np.mean(n_agents):.2f})")
sns.despine(ax=ax)
save_fig("C1_agents_per_plan.png")

# C2 subtasks per plan
fig, ax = plt.subplots(figsize=(6.5, 3.8))
c = Counter(n_subtasks)
ks = sorted(c.keys())
bars = ax.bar(ks, [c[k] for k in ks],
              color=PALETTE_MAIN[4], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    ax.text(b.get_x() + b.get_width() / 2, h + 30, f"{int(h):,}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xlabel("# subtasks in plan"); ax.set_ylabel("# plans")
ax.set_title(f"Subtasks per plan (mean={np.mean(n_subtasks):.2f})")
sns.despine(ax=ax)
save_fig("C2_subtasks_per_plan.png")

# C3 steps per plan
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.histplot(n_steps, bins=range(min(n_steps), max(n_steps) + 2),
             color=PALETTE_MAIN[2], edgecolor="white", alpha=0.85,
             kde=True, ax=ax, line_kws={"linewidth": 1.6})
ax.axvline(np.mean(n_steps), color="#c0392b", linestyle="--", lw=1.4,
           label=f"mean={np.mean(n_steps):.2f}")
ax.axvline(np.median(n_steps), color="#d4a017", linestyle=":", lw=1.4,
           label=f"median={np.median(n_steps):.0f}")
ax.set_xlabel("# steps in plan"); ax.set_ylabel("# plans")
ax.set_title("Steps per plan")
ax.legend()
sns.despine(ax=ax)
save_fig("C3_steps_per_plan.png")

# C4 human input ratio
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.histplot(human_input_ratio_per_plan, bins=20,
             color=PALETTE_MAIN[5], edgecolor="white", alpha=0.85, ax=ax)
ax.axvline(np.mean(human_input_ratio_per_plan), color="#c0392b", linestyle="--",
           lw=1.4, label=f"mean={np.mean(human_input_ratio_per_plan):.1%}")
ax.set_xlabel("Fraction of steps with requires_human_input=true")
ax.set_ylabel("# plans")
ax.set_title("Human-in-the-loop intensity (per plan)")
ax.legend()
sns.despine(ax=ax)
save_fig("C4_human_input_ratio.png")

# C5 loops
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
c = Counter(loops_per_plan)
ks = sorted(c.keys())
bars = axes[0].bar(ks, [c[k] for k in ks],
                   color=PALETTE_MAIN[2], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width() / 2, h + 20, f"{int(h):,}",
                 ha="center", va="bottom", fontsize=7.5)
axes[0].set_xlabel("# loops in execution_order"); axes[0].set_ylabel("# plans")
axes[0].set_title(f"Loops per plan ({sum(1 for x in loops_per_plan if x>0):,}/{N:,} have ≥1)")
sns.despine(ax=axes[0])

c2 = Counter([k for k in loop_max_iter if k is not None])
ks2 = sorted(c2.keys())
bars = axes[1].bar([str(k) for k in ks2], [c2[k] for k in ks2],
                   color=PALETTE_MAIN[5], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks2):
    h = b.get_height()
    axes[1].text(b.get_x() + b.get_width() / 2, h + 5, f"{int(h):,}",
                 ha="center", va="bottom", fontsize=7.5)
axes[1].set_xlabel("max_iterations"); axes[1].set_ylabel("# loop blocks")
axes[1].set_title("Loop max_iterations distribution")
sns.despine(ax=axes[1])
save_fig("C5_loops.png")


def dag_metrics(subtasks):
    nodes = set()
    edges = []
    for s in subtasks:
        for st in s.get("steps", []):
            sid = st.get("id")
            if not sid:
                continue
            nodes.add(sid)
            for d in st.get("depends_on", []) or []:
                edges.append((d, sid))
    parents = defaultdict(list)
    for u, v in edges:
        parents[v].append(u)
    layer = {}

    def depth(n, seen):
        if n in layer:
            return layer[n]
        if n in seen:
            return 0
        seen.add(n)
        ps = parents.get(n, [])
        if not ps:
            layer[n] = 0
        else:
            layer[n] = 1 + max(depth(p, seen) for p in ps)
        return layer[n]
    for n in nodes:
        depth(n, set())
    if not layer:
        return (0, 0, 0, 0, 0.0)
    layer_count = Counter(layer.values())
    longest = max(layer.values()) + 1
    max_width = max(layer_count.values())
    parallelizable = 1 - longest / len(nodes) if len(nodes) else 0
    return len(nodes), len(edges), longest, max_width, parallelizable


dag_stats = [dag_metrics(r["plan"]["output"]["subtasks"]) for r in records]
n_nodes_l = [x[0] for x in dag_stats]
n_edges_l = [x[1] for x in dag_stats]
longest_path = [x[2] for x in dag_stats]
max_widths = [x[3] for x in dag_stats]
par_ratio = [x[4] for x in dag_stats]

STATS["C_dag"] = {
    "longest_path_len": {
        "mean": float(np.mean(longest_path)),
        "median": float(np.median(longest_path)),
        "min": int(min(longest_path)), "max": int(max(longest_path)),
    },
    "max_layer_width": {
        "mean": float(np.mean(max_widths)),
        "median": float(np.median(max_widths)),
    },
    "edges_per_plan_mean": float(np.mean(n_edges_l)),
    "edges_per_node_mean": float(np.mean(n_edges_l) / np.mean(n_nodes_l)),
    "parallelizable_ratio_mean": float(np.mean(par_ratio)),
}

# C6: DAG
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
sns.histplot(longest_path, bins=range(min(longest_path), max(longest_path) + 2),
             color=PALETTE_MAIN[3], edgecolor="white", alpha=0.85,
             kde=True, ax=axes[0], line_kws={"linewidth": 1.6})
axes[0].axvline(np.mean(longest_path), color="#c0392b", linestyle="--", lw=1.4,
                label=f"mean={np.mean(longest_path):.2f}")
axes[0].set_xlabel("Critical path length (depth)")
axes[0].set_ylabel("# plans")
axes[0].set_title("DAG critical-path depth")
axes[0].legend()
sns.despine(ax=axes[0])

sns.histplot(par_ratio, bins=15, color=PALETTE_MAIN[5], edgecolor="white",
             alpha=0.85, ax=axes[1], kde=True, line_kws={"linewidth": 1.6})
axes[1].axvline(np.mean(par_ratio), color="#c0392b", linestyle="--", lw=1.4,
                label=f"mean={np.mean(par_ratio):.2f}")
axes[1].set_xlabel("Parallelizable ratio = 1 - depth/|nodes|")
axes[1].set_ylabel("# plans")
axes[1].set_title("Parallelization potential")
axes[1].legend()
sns.despine(ax=axes[1])
save_fig("C6_dag_metrics.png")

# ======================================================================
# D. AGENT DESIGN
# ======================================================================
role_counter = Counter()
tools_per_agent = []
zero_tool = 0
tool_counter = Counter()
roles_per_plan = []
plan_role_set = []
for r in records:
    roles = set()
    for a in r["plan"]["output"]["agents"]:
        role = a["agent_role"]
        role_counter[role] += 1
        roles.add(role)
        tools = a.get("tools", []) or []
        tools_per_agent.append(len(tools))
        if not tools:
            zero_tool += 1
        for t in tools:
            tool_counter[t] += 1
    roles_per_plan.append(len(roles))
    plan_role_set.append(roles)

family_counter = Counter()
plan_family_set = []
for r in records:
    fams = set()
    for a in r["plan"]["output"]["agents"]:
        fam = classify_role(a["agent_role"])
        family_counter[fam] += 1
        fams.add(fam)
    plan_family_set.append(fams)

STATS["D_agents"] = {
    "unique_agent_roles": len(role_counter),
    "top20_roles": role_counter.most_common(20),
    "tools_per_agent_mean": float(np.mean(tools_per_agent)),
    "zero_tool_agents_pct": zero_tool / len(tools_per_agent),
    "total_agents": len(tools_per_agent),
    "unique_tools": len(tool_counter),
    "all_tools": tool_counter.most_common(),
    "role_family_distribution": dict(family_counter),
    "roles_per_plan_mean": float(np.mean(roles_per_plan)),
}

# D1 top roles
fig, ax = plt.subplots(figsize=(8.5, 5.2))
top15 = role_counter.most_common(15)[::-1]
names = [t[0] for t in top15]
vals = [t[1] for t in top15]
bars = ax.barh(names, vals, color=sns.color_palette("crest_r", 15),
               edgecolor="white", linewidth=1.2)
for i, v in enumerate(vals):
    ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=7.5, color="#222222")
ax.set_xlabel("# occurrences"); ax.set_title("Top-15 agent_role names")
sns.despine(ax=ax, left=True)
save_fig("D1_top_roles.png")

# D2 tools per agent + Top tools
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
c = Counter(tools_per_agent)
ks = sorted(c.keys())
bars = axes[0].bar(ks, [c[k] for k in ks],
                   color=PALETTE_MAIN[3], edgecolor="white", linewidth=1.5)
for b, k in zip(bars, ks):
    h = b.get_height()
    axes[0].text(b.get_x() + b.get_width() / 2, h + 50, f"{int(h):,}",
                 ha="center", va="bottom", fontsize=7.5)
axes[0].set_xlabel("# tools per agent"); axes[0].set_ylabel("# agents")
axes[0].set_title("Tools per agent")
sns.despine(ax=axes[0])

top_tools = tool_counter.most_common(10)[::-1]
tnames = [t[0] for t in top_tools]
tvals = [t[1] for t in top_tools]
axes[1].barh(tnames, tvals, color=sns.color_palette("rocket_r", 10),
             edgecolor="white", linewidth=1.2)
for i, v in enumerate(tvals):
    axes[1].text(v + max(tvals) * 0.01, i, f"{v:,}", va="center", fontsize=7.5)
axes[1].set_xlabel("# occurrences"); axes[1].set_title("Top tools (dataset-wide)")
sns.despine(ax=axes[1], left=True)
save_fig("D2_tools.png")

# D3 role family pie
fig, ax = plt.subplots(figsize=(7, 5.5))
# D3 role family pie: small slices use legend instead of inline labels
# to avoid overlap; majority slices keep their inline label.
fam_items = sorted(family_counter.items(), key=lambda x: -x[1])
fam_total = sum(v for _, v in fam_items)
fam_pct = [v / fam_total for _, v in fam_items]
fam_labels = [
    f"{k}\n(n={v:,})" if pct >= 0.05 else ""
    for (k, v), pct in zip(fam_items, fam_pct)
]
wedges, texts, autotexts = ax.pie(
    [v for _, v in fam_items],
    labels=fam_labels,
    autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
    startangle=90,
    colors=PALETTE_SET[:len(fam_items)],
    wedgeprops=dict(edgecolor="white", linewidth=1.2),
    textprops=dict(fontsize=8))
for t in autotexts:
    t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(8)
# add a legend for the small slices that we suppressed inline
small_handles = [
    (w, f"{k} (n={v:,}, {pct*100:.1f}%)")
    for w, (k, v), pct in zip(wedges, fam_items, fam_pct)
    if pct < 0.05
]
if small_handles:
    ax.legend(
        [h for h, _ in small_handles],
        [lab for _, lab in small_handles],
        loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=7, frameon=False, handlelength=1.2)
save_fig("D3_role_families.png")

# D4 role-family combinations per plan (UpSet-style)
fam_order = [f for f, _ in fam_items]
combo_counter = Counter()
for s in plan_family_set:
    combo_counter[tuple(sorted(s, key=fam_order.index))] += 1

top_combos = combo_counter.most_common(10)
fig = plt.figure(figsize=(12, 5.5))
gs = plt.GridSpec(2, 1, height_ratios=[2, 1.3], hspace=0.1)
ax_bar = fig.add_subplot(gs[0])
ax_dot = fig.add_subplot(gs[1], sharex=ax_bar)
x = np.arange(len(top_combos))
counts = [c for _, c in top_combos]
combos = [c for c, _ in top_combos]
colors = sns.color_palette("crest_r", len(top_combos))
bars = ax_bar.bar(x, counts, color=colors, edgecolor="white", linewidth=1.4, width=0.65)
for i, cnt in enumerate(counts):
    ax_bar.text(i, cnt + max(counts) * 0.02, f"{cnt:,}\n({cnt/N*100:.1f}%)",
                ha="center", va="bottom", fontsize=7)
ax_bar.set_ylabel("# plans")
ax_bar.set_title("Top-10 agent role-family combinations per plan")
ax_bar.tick_params(axis="x", labelbottom=False)
sns.despine(ax=ax_bar)
for i, combo in enumerate(combos):
    active = [fam_order.index(f) for f in combo]
    for j, fam in enumerate(fam_order):
        if fam in combo:
            ax_dot.scatter(i, j, color="#2c3e50", s=90, zorder=3)
        else:
            ax_dot.scatter(i, j, facecolors="none",
                           edgecolors="#cccccc", s=42, zorder=2)
    if len(active) > 1:
        ax_dot.plot([i, i], [min(active), max(active)],
                    color="#2c3e50", linewidth=1.7)
ax_dot.set_yticks(range(len(fam_order)))
ax_dot.set_yticklabels(fam_order)
ax_dot.set_xticks([])
ax_dot.invert_yaxis()
ax_dot.grid(False)
sns.despine(ax=ax_dot, left=False, bottom=False)
save_fig("D4_role_family_upset.png")

# ======================================================================
# E. PERSONALIZATION / ONE-TO-MANY  (EXPANDED)
# ======================================================================
qid_to_records = defaultdict(list)
for r in records:
    qid_to_records[str(r["question_id"])].append(r)

multi_qids = [q for q, lst in qid_to_records.items() if len(lst) > 1]
total_pairs = sum(len(lst) * (len(lst) - 1) // 2
                  for lst in qid_to_records.values() if len(lst) > 1)

same_profile_pair_count = 0
for qid, lst in qid_to_records.items():
    if len(lst) < 2:
        continue
    for i, j in combinations(range(len(lst)), 2):
        a = profile_key(lst[i]["plan"]["input"]["learner"])
        b = profile_key(lst[j]["plan"]["input"]["learner"])
        if a == b:
            same_profile_pair_count += 1


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


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
    n_step = sum(len(s.get("steps", [])) for s in out["subtasks"])
    n_loop = sum(1 for x in out["execution_order"]
                 if isinstance(x, dict) and "loop" in x)
    return {"roles": roles, "families": families, "tools": tools,
            "subtask_tokens": subtask_tokens, "n_step": n_step, "n_loop": n_loop}


agent_role_jaccards = []
agent_family_jaccards = []
subtask_name_jaccards = []
tool_jaccards = []
step_count_diffs = []
loop_count_diffs = []
intent_to_pair_personalization = defaultdict(list)
# Also bucket plan-pair similarity by profile-pair similarity
profile_vs_plan_pairs = []  # (skill_jaccard, role_jaccard, family_jaccard)

for qid, lst in qid_to_records.items():
    if len(lst) < 2:
        continue
    feats = [plan_features(r) for r in lst]
    profs = [extract_learner(r["plan"]["input"]["learner"]) for r in lst]
    pi = primary_intent(qid)
    for i, j in combinations(range(len(feats)), 2):
        a, b = feats[i], feats[j]
        jr = jaccard(a["roles"], b["roles"])
        jf = jaccard(a["families"], b["families"])
        jt = jaccard(a["tools"], b["tools"])
        js = jaccard(a["subtask_tokens"], b["subtask_tokens"])
        agent_role_jaccards.append(jr)
        agent_family_jaccards.append(jf)
        tool_jaccards.append(jt)
        subtask_name_jaccards.append(js)
        step_count_diffs.append(abs(a["n_step"] - b["n_step"]))
        loop_count_diffs.append(abs(a["n_loop"] - b["n_loop"]))
        intent_to_pair_personalization[pi].append(1 - jr)
        # profile pair similarity
        sk_a = set(profs[i][1]); sk_b = set(profs[j][1])
        sk_j = (len(sk_a & sk_b) / len(sk_a | sk_b)) if (sk_a or sk_b) else 1.0
        profile_vs_plan_pairs.append((sk_j, jr, jf))

STATS["E_personalization"] = {
    "nonempty_profile_records": nonempty_profile_count,
    "questions_with_multi_profile": len(multi_qids),
    "total_pairs": total_pairs,
    "pair_with_identical_profile_text": same_profile_pair_count,
    "agent_role_jaccard_mean": float(np.mean(agent_role_jaccards)) if agent_role_jaccards else 0,
    "agent_role_jaccard_median": float(np.median(agent_role_jaccards)) if agent_role_jaccards else 0,
    "agent_family_jaccard_mean": float(np.mean(agent_family_jaccards)) if agent_family_jaccards else 0,
    "tool_jaccard_mean": float(np.mean(tool_jaccards)) if tool_jaccards else 0,
    "subtask_name_jaccard_mean": float(np.mean(subtask_name_jaccards)) if subtask_name_jaccards else 0,
    "step_count_diff_mean": float(np.mean(step_count_diffs)) if step_count_diffs else 0,
}

# E1 cross-profile Jaccard violin
fig, ax = plt.subplots(figsize=(8, 4.4))
df_div = pd.DataFrame({
    "Agent\nroles": agent_role_jaccards,
    "Role\nfamilies": agent_family_jaccards,
    "Tools": tool_jaccards,
    "Subtask\ntokens": subtask_name_jaccards,
})
df_long = df_div.melt(var_name="dim", value_name="jac")
sns.violinplot(data=df_long, x="dim", y="jac", ax=ax,
               palette="Set2", inner="box", linewidth=1.2, cut=0)
for i, col in enumerate(df_div.columns):
    m = df_div[col].mean()
    ax.scatter(i, m, marker="D", color="#c0392b", s=40, zorder=5,
               edgecolor="white", linewidth=1.2, label="mean" if i == 0 else None)
    ax.text(i, m + 0.04, f"{m:.2f}", color="#c0392b", fontsize=7.5,
            ha="center", fontweight="bold")
ax.set_ylabel("Jaccard similarity")
ax.set_xlabel("")
ax.set_title(f"Cross-profile plan similarity (n={total_pairs:,} pairs)\n"
             "Lower = more personalization rewriting")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="upper right")
sns.despine(ax=ax)
save_fig("E1_jaccard_violin.png")


# E2 skill lexical leakage
def plan_text(r):
    out = r["plan"]["output"]
    parts = []
    for a in out["agents"]:
        parts.append(a.get("agent_role", ""))
        parts.append(a.get("goal", ""))
        parts.append(agent_desc_text(a))
    for s in out["subtasks"]:
        parts.append(s.get("name", ""))
        parts.append(s.get("subtask_objective", ""))
        for st in s.get("steps", []):
            parts.append(st.get("instruction", ""))
            parts.append(st.get("objective", ""))
            parts.append(st.get("expected_output", ""))
    return " ".join(parts).lower()


skill_hit_ratio = []
skill_at_least_one = 0
desc_token_hit_ratio = []
for r in records:
    desc, sk = extract_learner(r["plan"]["input"]["learner"])
    text = plan_text(r)
    sk_l = [s.lower() for s in sk if s]
    if sk_l:
        hits = sum(1 for s in sk_l if s in text)
        skill_hit_ratio.append(hits / len(sk_l))
        if hits >= 1:
            skill_at_least_one += 1
    desc_tokens = [t for t in normalize_token(desc) if len(t) >= 4]
    if desc_tokens:
        hits = sum(1 for t in desc_tokens if t in text)
        desc_token_hit_ratio.append(hits / len(desc_tokens))

STATS["E_personalization"].update({
    "records_with_skills": len(skill_hit_ratio),
    "skill_lex_hit_ratio_mean":
        float(np.mean(skill_hit_ratio)) if skill_hit_ratio else 0,
    "skill_lex_hit_ratio_median":
        float(np.median(skill_hit_ratio)) if skill_hit_ratio else 0,
    "skill_at_least_one_record_count": skill_at_least_one,
    "skill_at_least_one_pct":
        skill_at_least_one / len(skill_hit_ratio) if skill_hit_ratio else 0,
    "desc_token_hit_ratio_mean":
        float(np.mean(desc_token_hit_ratio)) if desc_token_hit_ratio else 0,
})

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(skill_hit_ratio, bins=20, color=PALETTE_MAIN[4],
             edgecolor="white", alpha=0.85, ax=ax)
ax.axvline(np.mean(skill_hit_ratio), color="#c0392b", linestyle="--", lw=1.4,
           label=f"mean={np.mean(skill_hit_ratio):.1%}")
ax.set_xlabel("Fraction of learner skills appearing verbatim in plan text")
ax.set_ylabel("# records")
ax.set_title(f"Profile→Plan skill lexical leakage\n"
             f"≥1 skill present in {skill_at_least_one/len(skill_hit_ratio):.1%} of plans")
ax.legend()
sns.despine(ax=ax)
save_fig("E2_skill_hit_ratio.png")

# E3 personalization by intent
intent_div_rows = []
for intent in INTENT_ORDER:
    vals = intent_to_pair_personalization.get(intent, [])
    if vals:
        intent_div_rows.append({
            "intent": intent,
            "mean_diff": float(np.mean(vals)),
            "median_diff": float(np.median(vals)),
            "n_pairs": len(vals),
        })
intent_div_df = pd.DataFrame(intent_div_rows)
STATS["E_personalization"]["divergence_by_intent"] = (
    intent_div_df.to_dict("records") if not intent_div_df.empty else [])

if not intent_div_df.empty:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.barplot(data=intent_div_df, x="intent", y="mean_diff",
                palette=PALETTE_MAIN[:len(intent_div_df)], ax=ax)
    ymax = float(intent_div_df["mean_diff"].max())
    ax.set_ylim(0, ymax * 1.28)
    for i, row in intent_div_df.reset_index(drop=True).iterrows():
        ax.text(i, row["mean_diff"] + ymax * 0.03,
                f"{row['mean_diff']:.2f}\n(n={row['n_pairs']:,})",
                ha="center", va="bottom", fontsize=6.5, color="#333333")
    ax.set_ylabel(r"Mean $1-\mathrm{Jaccard}$ (agent roles)")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right")
    sns.despine(ax=ax)
    save_fig("E3_personalization_by_intent.png")

# E4 NEW: profile-similarity vs plan-similarity scatter (effect size sanity check)
if profile_vs_plan_pairs:
    arr = np.array(profile_vs_plan_pairs)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # 2D hex density
    hb = ax.hexbin(arr[:, 0], arr[:, 1], gridsize=22,
                   cmap="crest_r", mincnt=1, edgecolors="white", linewidths=0.2)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("# profile-pair × plan-pair instances")
    # regression line
    if len(arr) >= 2:
        m, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        xs = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 100)
        ax.plot(xs, m * xs + b, color="#c0392b", lw=1.8, linestyle="--",
                label=f"linear fit: y={m:.2f}x+{b:.2f}")
        # correlation
        if np.std(arr[:, 0]) > 0 and np.std(arr[:, 1]) > 0:
            corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            ax.text(0.04, 0.93, f"Pearson r = {corr:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))
            STATS["E_personalization"]["profile_vs_plan_pearson_r"] = corr
        ax.legend(loc="lower right")
    ax.set_xlabel("Profile skill-set Jaccard (same-qid pair)")
    ax.set_ylabel("Plan agent-role Jaccard")
    ax.set_title("Do more-similar profiles yield more-similar plans?")
    sns.despine(ax=ax)
    save_fig("E4_profile_vs_plan_similarity.png")

# E5 NEW: divergence by nprof bucket (one-to-many depth effect)
nprof_bucket_div = defaultdict(list)
for qid, lst in qid_to_records.items():
    if len(lst) < 2:
        continue
    feats = [plan_features(r) for r in lst]
    nprof = len(lst)
    for i, j in combinations(range(len(feats)), 2):
        nprof_bucket_div[nprof].append(
            1 - jaccard(feats[i]["roles"], feats[j]["roles"]))

if nprof_bucket_div:
    rows = []
    for k in sorted(nprof_bucket_div):
        for v in nprof_bucket_div[k]:
            rows.append({"nprof": k, "1-jaccard": v})
    bdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.violinplot(data=bdf, x="nprof", y="1-jaccard", ax=ax,
                   palette="crest_r", inner="quartile", linewidth=1.0, cut=0)
    means = bdf.groupby("nprof")["1-jaccard"].mean()
    counts = bdf.groupby("nprof")["1-jaccard"].size()
    for i, k in enumerate(sorted(nprof_bucket_div)):
        ax.scatter(i, means[k], marker="D", color="#c0392b", s=40, zorder=5,
                   edgecolor="white", linewidth=1.2)
        ax.text(i, 1.04, f"n={counts[k]:,}", ha="center", fontsize=7, color="#555555")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Plan divergence (1 - role Jaccard)")
    ax.set_xlabel("# profiles per question (one-to-many depth)")
    ax.set_title("Personalization signal vs. one-to-many fan-out")
    sns.despine(ax=ax)
    save_fig("E5_divergence_by_nprof.png")

# ======================================================================
# J. PEDAGOGY / REWARD-ALIGNED ATTRIBUTES  (NEW)
# ======================================================================
PHASE_PATTERNS = {
    "probe": re.compile(
        r"\b(probe|ask|predict|prediction|hypothes|surface|diagnose|identify|"
        r"think out loud|articulate|restate|explain what|before revealing)\b",
        re.I,
    ),
    "retrieve_demonstrate": re.compile(
        r"\b(retrieve|look up|official|docs?|documentation|spec|api reference|"
        r"demonstrat|walk through|worked example|example|show|illustrat)\b",
        re.I,
    ),
    "apply": re.compile(
        r"\b(apply|implement|write|revise|fix|refactor|construct|produce|submit|"
        r"code|patch|change|rewrite)\b",
        re.I,
    ),
    "validate": re.compile(
        r"\b(validate|verify|test|run|compile|execute|check|confirm|pass/fail|"
        r"actual vs expected|compiler output)\b",
        re.I,
    ),
    "feedback": re.compile(
        r"\b(feedback|revise|revision|failed|failure|correct only|targeted|"
        r"still wrong|if .* fails|iterate|iteration)\b",
        re.I,
    ),
    "consolidate": re.compile(
        r"\b(consolidate|general rule|one sentence|reflect|transfer|reusable|"
        r"summarize|summary|takeaway|future)\b",
        re.I,
    ),
}

METHOD_PATTERNS = {
    "Socratic probing": PHASE_PATTERNS["probe"],
    "Scaffolding/fading": re.compile(
        r"\b(scaffold|step-by-step|lead them|do not hint|do not show|without writing|"
        r"before revealing|gradual|guided)\b",
        re.I,
    ),
    "Analogy/bridging": re.compile(
        r"\b(analogy|analogies|like in|familiar|bridge|anchor|coming from|"
        r"based on your|your background|your experience|your project)\b",
        re.I,
    ),
    "Docs grounding": PHASE_PATTERNS["retrieve_demonstrate"],
    "Worked example": re.compile(
        r"\b(worked example|example|demonstrat|walk through|show|sample|minimal|"
        r"representative)\b",
        re.I,
    ),
    "Practice/application": PHASE_PATTERNS["apply"],
    "Validation/testing": PHASE_PATTERNS["validate"],
    "Iterative feedback": PHASE_PATTERNS["feedback"],
    "Metacognitive consolidation": PHASE_PATTERNS["consolidate"],
}

PHASE_ORDER = ["probe", "retrieve_demonstrate", "apply", "validate"]
BRIDGE_ANCHORS = [
    "your background", "your experience", "your project", "your portfolio",
    "your github", "your current", "your existing", "your skill",
    "your skills", "your top tags", "your domain", "your role",
    "based on your", "given your", "from your", "coming from",
    "as a beginner", "as an experienced", "you already know",
    "connect it to", "anchor", "analogy", "like in", "because you",
    "familiar with", "your mental model", "your prior", "your stack",
]
SECOND_PERSON = re.compile(r"\b(you|your|learner)\b", re.I)


def score_range(x, lo, hi):
    if x < lo:
        return max(0.0, x / lo)
    if x <= hi:
        return 1.0
    return max(0.0, 1.0 - (x - hi) / hi)


def plan_step_rows(record):
    rows = []
    global_idx = 0
    for sub_idx, sub in enumerate(record["plan"]["output"]["subtasks"]):
        sub_prefix = " ".join([
            sub.get("name", ""),
            sub.get("subtask_objective", ""),
        ])
        for st in sub.get("steps", []):
            text = " ".join([
                sub_prefix,
                st.get("objective", ""),
                st.get("instruction", ""),
                st.get("expected_output", ""),
            ])
            rows.append({
                "global_idx": global_idx,
                "sub_idx": sub_idx,
                "instruction": st.get("instruction", ""),
                "text": text,
            })
            global_idx += 1
    return rows


def first_phase_positions(rows):
    out = {}
    for phase, pat in PHASE_PATTERNS.items():
        hits = [r for r in rows if pat.search(r["text"])]
        if hits:
            out[phase] = {
                "step": min(r["global_idx"] for r in hits),
                "subtask": min(r["sub_idx"] for r in hits),
            }
    return out


def profile_signal_terms(record):
    desc, skills = extract_learner(record["plan"]["input"]["learner"])
    skill_terms = [s.lower() for s in skills if s]
    desc_terms = [t for t in normalize_token(desc) if len(t) >= 4]
    return skill_terms, desc_terms


phase_plan_hits = Counter()
phase_step_hits = Counter()
phase_condition_hits = Counter()
method_plan_hits = Counter()
method_step_hits = Counter()
method_by_intent = defaultdict(Counter)
phase_sequence_counter = Counter()

phasecov_values = []
size_quality_values = []
r_ped_values = []
pers_proxy_values = []
grounded_steps_per_plan = []
plan_text_len_chars = []
plan_text_len_tokens = []
plan_json_len_chars = []
step_instruction_lens = []

for r in records:
    rows = plan_step_rows(r)
    all_text = plan_text(r)
    plan_text_len_chars.append(len(all_text))
    plan_text_len_tokens.append(len(normalize_token(all_text)))
    plan_json_len_chars.append(len(json.dumps(r["plan"], ensure_ascii=False)))
    step_instruction_lens.extend(len(x["instruction"]) for x in rows)

    positions = first_phase_positions(rows)
    for phase in PHASE_PATTERNS:
        if phase in positions:
            phase_plan_hits[phase] += 1
    for row in rows:
        for phase, pat in PHASE_PATTERNS.items():
            if pat.search(row["text"]):
                phase_step_hits[phase] += 1

    n_sub = len(r["plan"]["output"]["subtasks"])
    probe_first_half = (
        "probe" in positions and positions["probe"]["subtask"] < max(1, math.ceil(n_sub / 2))
    )
    validate_present = "validate" in positions
    order_present = all(p in positions for p in PHASE_ORDER)
    ordered_core = (
        order_present
        and positions["probe"]["step"] <= positions["retrieve_demonstrate"]["step"]
        and positions["retrieve_demonstrate"]["step"] <= positions["apply"]["step"]
        and positions["apply"]["step"] <= positions["validate"]["step"]
    )
    phase_condition_hits["probe_first_half"] += int(probe_first_half)
    phase_condition_hits["validate_present"] += int(validate_present)
    phase_condition_hits["ordered_core_4_phase"] += int(ordered_core)
    phasecov = (int(probe_first_half) + int(validate_present) + int(ordered_core)) / 3
    phasecov_values.append(phasecov)

    first_order = sorted(
        [(positions[p]["step"], p) for p in PHASE_ORDER if p in positions],
        key=lambda x: x[0],
    )
    phase_sequence_counter[" > ".join(p for _, p in first_order) or "none"] += 1

    n_step = len(rows)
    size_q = (score_range(n_sub, 3, 5) + score_range(n_step, 8, 13)) / 2
    size_quality_values.append(size_q)
    r_ped_values.append((phasecov + size_q) / 2)

    intent = primary_intent(r["question_id"])
    for method, pat in METHOD_PATTERNS.items():
        if pat.search(all_text):
            method_plan_hits[method] += 1
            method_by_intent[intent][method] += 1
        for row in rows:
            if pat.search(row["text"]):
                method_step_hits[method] += 1

    skills, desc_terms = profile_signal_terms(r)
    grounded = 0
    for row in rows:
        inst = row["instruction"].lower()
        has_second_person = bool(SECOND_PERSON.search(inst))
        has_skill = any(s and s in inst for s in skills)
        has_desc = any(t and t in inst for t in desc_terms)
        has_anchor = any(a in inst for a in BRIDGE_ANCHORS)
        if len(row["instruction"]) >= 220 and has_second_person and (has_skill or has_desc or has_anchor):
            grounded += 1
    grounded_steps_per_plan.append(grounded)
    pers_proxy_values.append(grounded / n_step if n_step else 0.0)

total_steps = sum(n_steps)
phase_stats = {
    p: {
        "plans": int(phase_plan_hits[p]),
        "plan_pct": phase_plan_hits[p] / N,
        "steps": int(phase_step_hits[p]),
        "step_pct": phase_step_hits[p] / total_steps,
    }
    for p in PHASE_PATTERNS
}
method_stats = {
    m: {
        "plans": int(method_plan_hits[m]),
        "plan_pct": method_plan_hits[m] / N,
        "steps": int(method_step_hits[m]),
        "step_pct": method_step_hits[m] / total_steps,
    }
    for m in METHOD_PATTERNS
}

STATS["J_pedagogy"] = {
    "phase_stats": phase_stats,
    "phase_conditions": {
        "probe_first_half_pct": phase_condition_hits["probe_first_half"] / N,
        "validate_present_pct": phase_condition_hits["validate_present"] / N,
        "ordered_core_4_phase_pct": phase_condition_hits["ordered_core_4_phase"] / N,
        "phasecov_v2_mean": float(np.mean(phasecov_values)),
        "phasecov_v2_full_pct": sum(1 for x in phasecov_values if x == 1.0) / N,
    },
    "plan_size_quality_mean": float(np.mean(size_quality_values)),
    "r_ped_proxy_mean": float(np.mean(r_ped_values)),
    "method_stats": method_stats,
    "top_phase_sequences": phase_sequence_counter.most_common(10),
    "personalization_proxy": {
        "grounded_step_ratio_overall": sum(grounded_steps_per_plan) / total_steps,
        "grounded_steps_per_plan_mean": float(np.mean(grounded_steps_per_plan)),
        "plans_with_grounded_step_pct": sum(1 for x in grounded_steps_per_plan if x > 0) / N,
        "r_pers_proxy_mean": float(np.mean(pers_proxy_values)),
        "r_pers_proxy_median": float(np.median(pers_proxy_values)),
    },
    "lengths": {
        "plan_text_chars_mean": float(np.mean(plan_text_len_chars)),
        "plan_text_tokens_mean": float(np.mean(plan_text_len_tokens)),
        "plan_json_chars_mean": float(np.mean(plan_json_len_chars)),
        "step_instruction_chars_mean": float(np.mean(step_instruction_lens)),
        "step_instruction_chars_median": float(np.median(step_instruction_lens)),
    },
}

# J1: phase coverage and reward conditions
# J1 is rendered as a 0.48\linewidth panel in the paper; keep it a
# single chart so the bar labels remain legible after scaling.
fig, ax = plt.subplots(figsize=(4.2, 3.0))
phase_order_plot = ["probe", "retrieve_demonstrate", "apply",
                    "validate", "feedback", "consolidate"]
phase_pcts = [phase_stats[p]["plan_pct"] * 100 for p in phase_order_plot]
bars = ax.barh(phase_order_plot[::-1], phase_pcts[::-1],
               color=PALETTE_MAIN[:len(phase_order_plot)][::-1],
               edgecolor="white", linewidth=0.8)
for b, v in zip(bars, phase_pcts[::-1]):
    ax.text(min(v + 1.5, 99), b.get_y() + b.get_height() / 2,
            f"{v:.1f}%", va="center", fontsize=8, color="#222222")
ax.set_xlim(0, 112)
ax.set_xlabel("% plans containing phase signal")
sns.despine(ax=ax, left=True)
save_fig("J1_phase_coverage.png")

# J2: pedagogical method usage
method_df = pd.DataFrame([
    {"method": m, "plan_pct": v["plan_pct"] * 100, "step_pct": v["step_pct"] * 100}
    for m, v in method_stats.items()
]).sort_values("plan_pct")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
bars = ax.barh(method_df["method"], method_df["plan_pct"],
               color=sns.color_palette("rocket_r", len(method_df)),
               edgecolor="white", linewidth=1.2)
for b, v in zip(bars, method_df["plan_pct"]):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
            va="center", fontsize=7.5)
ax.set_xlim(0, 105)
ax.set_xlabel("% plans containing method signal")
ax.set_title("Pedagogical method taxonomy inferred from plan text")
sns.despine(ax=ax, left=True)
save_fig("J2_pedagogical_methods.png")

# J3: reward-proxy distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
for ax, vals, title, xlabel in [
    (axes[0], pers_proxy_values, "Personalization proxy", "grounded steps / steps"),
    (axes[1], phasecov_values, "PhaseCov_v2 proxy", "condition hits / 3"),
    (axes[2], r_ped_values, "R_ped proxy", "mean(PhaseCov, SizeQuality)"),
]:
    sns.histplot(vals, bins=20, color=PALETTE_MAIN[3], edgecolor="white",
                 alpha=0.85, ax=ax)
    ax.axvline(np.mean(vals), color="#c0392b", linestyle="--", lw=1.4,
               label=f"mean={np.mean(vals):.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    sns.despine(ax=ax)
save_fig("J3_reward_proxy_distributions.png")

# J4: method hit rate by intent
intent_method_rows = []
intent_counts = Counter(primary_intent(r["question_id"]) for r in records)
for intent in INTENT_ORDER:
    if intent_counts[intent] == 0:
        continue
    row = {"intent": intent}
    for method in METHOD_PATTERNS:
        row[method] = method_by_intent[intent][method] / intent_counts[intent] * 100
    intent_method_rows.append(row)
if intent_method_rows:
    im_df = pd.DataFrame(intent_method_rows).set_index("intent")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    sns.heatmap(im_df, annot=True, fmt=".0f", cmap="crest_r", linewidths=0.5,
                linecolor="white", cbar_kws={"label": "% plans"}, ax=ax)
    ax.set_title("Pedagogical method coverage by primary intent (%)")
    ax.set_xlabel("")
    ax.set_ylabel("Primary intent")
    plt.xticks(rotation=35, ha="right")
    save_fig("J4_methods_by_intent.png")

# ======================================================================
# F. INTENT × COMPLEXITY
# ======================================================================
rows = []
for i, r in enumerate(records):
    qid = str(r["question_id"])
    labs = intent_labels.get(qid, [])
    pi = labs[0] if labs else "UNKNOWN"
    rows.append({
        "qid": qid,
        "primary_intent": pi,
        "n_agents": n_agents[i],
        "n_subtasks": n_subtasks[i],
        "n_steps": n_steps[i],
        "n_loops": loops_per_plan[i],
        "n_human_input": human_input_per_plan[i],
        "longest_path": longest_path[i],
        "n_tools": sum(len(a.get("tools", []) or [])
                       for a in r["plan"]["output"]["agents"]),
    })
df = pd.DataFrame(rows)

intent_rows = []
for intent, g in df.groupby("primary_intent"):
    intent_rows.append({
        "primary_intent": intent,
        "n": int(len(g)),
        "agents": float(g["n_agents"].mean()),
        "subtasks": float(g["n_subtasks"].mean()),
        "steps": float(g["n_steps"].mean()),
        "longest_path": float(g["longest_path"].mean()),
        "loops_mean": float(g["n_loops"].mean()),
        "loop_pct": float((g["n_loops"] > 0).mean()),
        "human_input_mean": float(g["n_human_input"].mean()),
        "tools": float(g["n_tools"].mean()),
    })
intent_summary = pd.DataFrame(intent_rows).set_index("primary_intent").round(2)
intent_summary = intent_summary.reindex(
    [i for i in INTENT_ORDER if i in intent_summary.index] +
    [i for i in intent_summary.index if i not in INTENT_ORDER])
STATS["F_intent_summary"] = intent_summary.reset_index().to_dict("records")

# F1 normalized heatmap
cols = ["agents", "subtasks", "steps", "longest_path",
        "loops_mean", "loop_pct", "human_input_mean", "tools"]
norm = intent_summary[cols].copy()
for c in cols:
    norm[c] = (norm[c] - norm[c].min()) / (norm[c].max() - norm[c].min() + 1e-9)
fig, ax = plt.subplots(figsize=(11, 4.6))
sns.heatmap(norm, annot=intent_summary[cols], fmt=".2f", cmap="crest_r",
            ax=ax, cbar_kws={"label": "normalized"}, linewidths=0.6,
            linecolor="white")
ax.set_title("Plan complexity by primary query intent (normalized)")
ax.set_xlabel("")
ax.set_ylabel("Primary intent")
save_fig("F1_intent_complexity_heatmap.png")

# F2 intent distribution donut
intent_dist = Counter()
for r in records:
    qid = str(r["question_id"])
    labs = intent_labels.get(qid, [])
    if labs:
        intent_dist[labs[0]] += 1
    else:
        intent_dist["UNKNOWN"] += 1
fig, ax = plt.subplots(figsize=(7.5, 5.5))
items = sorted(intent_dist.items(), key=lambda x: -x[1])
wedges, _, autotexts = ax.pie(
    [v for _, v in items],
    labels=[f"{k}\n({v:,}, {v/N*100:.1f}%)" for k, v in items],
    startangle=90, colors=sns.color_palette("Set2", len(items)),
    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
    autopct="%1.1f%%", pctdistance=0.78,
    textprops=dict(fontsize=7.5))
for t in autotexts:
    t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(9)
ax.set_title("Primary intent distribution")
save_fig("F2_intent_donut.png")

# ======================================================================
# G. LEXICAL DIVERSITY  (NEW)
# ======================================================================
all_query_tokens = []
all_plan_tokens = []
all_skill_tokens = []
all_role_tokens = []
for r in records:
    all_query_tokens.extend(normalize_token(r["plan"]["input"]["query"]))
    all_plan_tokens.extend(normalize_token(plan_text(r)))
    _, skills = extract_learner(r["plan"]["input"]["learner"])
    all_skill_tokens.extend([s.lower() for s in skills])
    for a in r["plan"]["output"]["agents"]:
        all_role_tokens.append(a["agent_role"].lower())


def distinct_n(tokens, n):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


lex = {
    "query": {
        "tokens": len(all_query_tokens),
        "vocab": len(set(all_query_tokens)),
        "ttr": len(set(all_query_tokens)) / max(1, len(all_query_tokens)),
        "distinct_1": distinct_n(all_query_tokens, 1),
        "distinct_2": distinct_n(all_query_tokens, 2),
        "distinct_3": distinct_n(all_query_tokens, 3),
    },
    "plan_text": {
        "tokens": len(all_plan_tokens),
        "vocab": len(set(all_plan_tokens)),
        "ttr": len(set(all_plan_tokens)) / max(1, len(all_plan_tokens)),
        "distinct_1": distinct_n(all_plan_tokens, 1),
        "distinct_2": distinct_n(all_plan_tokens, 2),
        "distinct_3": distinct_n(all_plan_tokens, 3),
    },
    "skills": {
        "tokens": len(all_skill_tokens),
        "vocab": len(set(all_skill_tokens)),
    },
    "agent_roles": {
        "tokens": len(all_role_tokens),
        "vocab": len(set(all_role_tokens)),
    },
}
STATS["G_lexical"] = lex

# G1: distinct-n bars for query vs plan
fig, ax = plt.subplots(figsize=(7, 3.8))
labels = ["distinct-1", "distinct-2", "distinct-3"]
q_vals = [lex["query"][k.replace("-", "_")] for k in labels]
p_vals = [lex["plan_text"][k.replace("-", "_")] for k in labels]
x = np.arange(len(labels))
w = 0.36
b1 = ax.bar(x - w/2, q_vals, w, label="query", color=PALETTE_MAIN[2],
            edgecolor="white", linewidth=1.2)
b2 = ax.bar(x + w/2, p_vals, w, label="plan text", color=PALETTE_MAIN[5],
            edgecolor="white", linewidth=1.2)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Distinct-n ratio (vocab / total ngrams)")
ax.set_title("Lexical diversity — query vs plan text")
ax.legend()
sns.despine(ax=ax)
save_fig("G1_distinct_n.png")

# G2: top tags wordcount-style (top 30 skills)
top_skills = Counter(all_skill_tokens).most_common(25)[::-1]
fig, ax = plt.subplots(figsize=(8.5, 6))
ax.barh([s for s, _ in top_skills], [c for _, c in top_skills],
        color=sns.color_palette("rocket_r", 25),
        edgecolor="white", linewidth=1.1)
for i, (_, c) in enumerate(top_skills):
    ax.text(c + max(c for _, c in top_skills) * 0.01, i, f"{c:,}",
            va="center", fontsize=7.5)
ax.set_xlabel("# occurrences across learner profiles")
ax.set_title("Top-25 learner skills / top_tags")
sns.despine(ax=ax, left=True)
save_fig("G2_top_skills.png")

# ======================================================================
# H. TOOL & ROLE LONG-TAIL + CO-OCCURRENCE  (NEW)
# ======================================================================
# H1: tool Zipf
fig, ax = plt.subplots(figsize=(7.5, 4.2))
tool_freqs = sorted(tool_counter.values(), reverse=True)
xs = np.arange(1, len(tool_freqs) + 1)
ax.plot(xs, tool_freqs, color=PALETTE_MAIN[2], lw=2)
ax.fill_between(xs, tool_freqs, color=PALETTE_MAIN[2], alpha=0.18)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Tool rank (log)")
ax.set_ylabel("# occurrences (log)")
top10_share = sum(tool_freqs[:10]) / max(1, sum(tool_freqs))
ax.set_title(f"Tool frequency Zipf curve "
             f"(top-10 tools cover {top10_share*100:.1f}% of usage)")
sns.despine(ax=ax)
save_fig("H1_tool_zipf.png")
STATS["H_long_tail"] = {
    "top10_tool_share": top10_share,
    "unique_tools": len(tool_counter),
}

# H2: tool co-occurrence heatmap (top-12 tools)
top_tools_12 = [t for t, _ in tool_counter.most_common(12)]
idx = {t: i for i, t in enumerate(top_tools_12)}
M = np.zeros((len(top_tools_12), len(top_tools_12)), dtype=int)
for r in records:
    plan_tools = set()
    for a in r["plan"]["output"]["agents"]:
        for t in a.get("tools", []) or []:
            if t in idx:
                plan_tools.add(t)
    for a, b in combinations(plan_tools, 2):
        M[idx[a], idx[b]] += 1
        M[idx[b], idx[a]] += 1
    for t in plan_tools:
        M[idx[t], idx[t]] += 1
cm = pd.DataFrame(M, index=top_tools_12, columns=top_tools_12)
fig, ax = plt.subplots(figsize=(8.5, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="crest_r", ax=ax,
            cbar_kws={"label": "# plans containing both"}, linewidths=0.6,
            linecolor="white", square=True,
            annot_kws={"fontsize": 8.5})
ax.set_title("Tool co-occurrence within plans (top-12 tools)")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
save_fig("H2_tool_cooccurrence.png")

# H3: role-family co-occurrence
fam_order_all = [f for f in fam_order]
M2 = np.zeros((len(fam_order_all), len(fam_order_all)), dtype=int)
fidx = {f: i for i, f in enumerate(fam_order_all)}
for plan_fams in plan_family_set:
    for f in plan_fams:
        M2[fidx[f], fidx[f]] += 1
    for a, b in combinations(plan_fams, 2):
        M2[fidx[a], fidx[b]] += 1
        M2[fidx[b], fidx[a]] += 1
cm2 = pd.DataFrame(M2, index=fam_order_all, columns=fam_order_all)
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(cm2, annot=True, fmt="d", cmap="rocket_r", ax=ax,
            cbar_kws={"label": "# plans"}, linewidths=0.6, linecolor="white",
            square=True, annot_kws={"fontsize": 9})
ax.set_title("Role-family co-occurrence within plans")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
save_fig("H3_role_family_cooccurrence.png")

# ======================================================================
# I. QUALITY / SCHEMA
# ======================================================================
q_unknown_agent = 0
q_unknown_depends = 0
q_unknown_exec = 0
q_missing_in_exec = 0
q_extra_in_exec = 0
q_cycle = 0
q_loop_step_unknown = 0
q_bad = 0
loop_schema_singular = 0
loop_schema_plural = 0
missing_depends_on = 0
missing_backstory = 0

for r in records:
    out = r["plan"]["output"]
    role_names = {a["agent_role"] for a in out["agents"]}
    step_ids = set()
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            step_ids.add(st.get("id"))
            if "depends_on" not in st:
                missing_depends_on += 1
    for a in out["agents"]:
        if "backstory" not in a and "description" not in a:
            missing_backstory += 1

    bad = False
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            if st.get("agent") not in role_names:
                q_unknown_agent += 1
                bad = True
                break
        if bad:
            break

    saw_unknown_dep = False
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            for d in (st.get("depends_on") or []):
                if d not in step_ids:
                    saw_unknown_dep = True
                    break
            if saw_unknown_dep:
                break
        if saw_unknown_dep:
            break
    if saw_unknown_dep:
        q_unknown_depends += 1
        bad = True

    flat = []
    for x in out["execution_order"]:
        if isinstance(x, str):
            flat.append(x)
        elif isinstance(x, dict) and "loop" in x:
            loop_def = x["loop"]
            if "steps" in loop_def:
                loop_schema_plural += 1
                sids = loop_def["steps"]
            elif "step" in loop_def:
                loop_schema_singular += 1
                sids = ([loop_def["step"]]
                        if isinstance(loop_def["step"], str)
                        else loop_def["step"])
            else:
                sids = []
            for sid in sids:
                flat.append(sid)
                if sid not in step_ids:
                    q_loop_step_unknown += 1
    eo_set = set(flat)
    unknown = eo_set - step_ids
    missing = step_ids - eo_set
    if unknown:
        q_unknown_exec += 1
        bad = True
    if missing:
        q_missing_in_exec += 1
        bad = True
    parents = defaultdict(list)
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            for d in (st.get("depends_on") or []):
                parents[st["id"]].append(d)
    if _has_cycle(step_ids, parents):
        q_cycle += 1
        bad = True
    if bad:
        q_bad += 1

STATS["I_quality"] = {
    "plans_with_unknown_agent_ref": q_unknown_agent,
    "plans_with_unknown_depends_on": q_unknown_depends,
    "plans_with_unknown_step_in_exec": q_unknown_exec,
    "plans_with_missing_step_in_exec": q_missing_in_exec,
    "plans_with_loop_step_unknown": q_loop_step_unknown,
    "plans_with_cycle": q_cycle,
    "plans_with_any_issue": q_bad,
    "valid_plans": N - q_bad,
    "valid_rate": (N - q_bad) / N,
    "loop_schema_plural_steps": loop_schema_plural,
    "loop_schema_singular_step": loop_schema_singular,
    "steps_missing_depends_on": missing_depends_on,
    "agents_missing_backstory": missing_backstory,
}

# I1
quality = STATS["I_quality"]
fig, ax = plt.subplots(figsize=(9, 4.2))
keys = ["plans_with_unknown_agent_ref", "plans_with_unknown_depends_on",
        "plans_with_unknown_step_in_exec", "plans_with_missing_step_in_exec",
        "plans_with_cycle", "plans_with_loop_step_unknown"]
labels_short = ["unknown\nagent_ref", "unknown\ndepends_on",
                "unknown_step\nin_exec", "missing_step\nin_exec",
                "DAG\ncycle", "loop_step\nunknown"]
vals = [quality[k] for k in keys]
colors_q = ["#c0392b" if v > 0 else "#27ae60" for v in vals]
bars = ax.bar(labels_short, vals, color=colors_q, edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2,
            b.get_height() + max(vals, default=1) * 0.02 + 0.1,
            f"{int(v):,}", ha="center", va="bottom", fontsize=8)
ax.set_title(f"Schema / referential issues — "
             f"{N - q_bad:,}/{N:,} plans fully valid ({(N-q_bad)/N:.1%})")
ax.set_ylabel("# plans")
sns.despine(ax=ax)
save_fig("I1_quality.png")

# ======================================================================
# Save stats
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

print(f"Saved stats to {STATS_OUT}")
print(f"Figures dir:  {FIG_DIR}")
print(f"N records:    {N:,}")
print(f"Splits:       train={split_rows['train']:,}, "
      f"dev={split_rows['dev']:,}, test={split_rows['test']:,}")
print(f"valid_rate={(N-q_bad)/N:.3%}, "
      f"agents/plan={np.mean(n_agents):.2f}, "
      f"steps/plan={np.mean(n_steps):.2f}, "
      f"unique_roles={len(role_counter):,}, "
      f"unique_tools={len(tool_counter):,}")
