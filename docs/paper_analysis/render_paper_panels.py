"""Re-render the three MAP-PPL figures used in the EMNLP main paper.

Goal: on-page font size comparable to 10pt body. The three figures are
embedded at ~0.30--0.33 \\textwidth of a two-column 11pt acl.sty page,
so the saved PNG is rendered at a compact figsize (3.5--4 in wide) with
large source-side fonts (~13--14pt); the resulting LaTeX scale factor
is close to 1, so labels print near body size.

Outputs:
  figures_paper/F2_intent_donut.png
  figures_paper/E1_jaccard_violin.png
  figures_paper/J1_phase_coverage.png
"""

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "multi_agent_dataset_filtered_qap.jsonl"
ROOT = HERE.parent.parent
LABELS_PATH = ROOT / "the_construction_of_MAPLE_datasets/task_3/classified_results.jsonl"
OUT_DIR = HERE / "figures_paper"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Compact, body-size-friendly style.
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 13,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.facecolor": "white",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
})


def save_fig(name, dpi=240):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close()


# ------------------------------------------------------------------
# Load records
# ------------------------------------------------------------------
records = []
with DATA_PATH.open() as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Optional intent labels from task_3 classifier.
intent_map = {}
if LABELS_PATH.exists():
    with LABELS_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("question_id")
            labels = d.get("labels")
            if isinstance(labels, list) and labels:
                primary = labels[0]
            else:
                primary = d.get("intent") or d.get("primary_intent") or d.get("label")
            if qid and primary:
                intent_map[str(qid)] = str(primary)

INTENT_ORDER = [
    "Conceptual",
    "Api_Usage",
    "Discrepancy",
    "Review",
    "Errors",
    "Api_Change",
    "Learning",
]

# ------------------------------------------------------------------
# F2 intent donut
# ------------------------------------------------------------------
def normalize_intent(s):
    if not s:
        return "Other"
    s = str(s).strip()
    canon = {
        "conceptual": "Conceptual",
        "api_usage": "Api_Usage",
        "api usage": "Api_Usage",
        "discrepancy": "Discrepancy",
        "review": "Review",
        "errors": "Errors",
        "api_change": "Api_Change",
        "api change": "Api_Change",
        "learning": "Learning",
    }
    return canon.get(s.lower(), s)


intent_counts = Counter()
if not intent_map:
    raise RuntimeError(f"Intent labels not found at {LABELS_PATH}")
for r in records:
    qid = str(r["question_id"])
    intent_counts[normalize_intent(intent_map.get(qid))] += 1

labels = [k for k in INTENT_ORDER if intent_counts.get(k)]
sizes = [intent_counts[k] for k in labels]
total = sum(sizes)
colors = sns.color_palette("crest_r", n_colors=len(labels))

fig, ax = plt.subplots(figsize=(3.8, 3.4))
wedges, _ = ax.pie(sizes, startangle=90, counterclock=False, colors=colors,
                   wedgeprops=dict(width=0.42, edgecolor="white",
                                   linewidth=1.0))
ax.text(0, 0.06, f"{total:,}", ha="center", va="center",
        fontsize=18, fontweight="bold", color="#333")
ax.text(0, -0.14, "queries", ha="center", va="center", fontsize=12,
        color="#666")

legend_labels = [f"{lab} ({100*sz/total:.1f}\\%)".replace("\\%", "%")
                 for lab, sz in zip(labels, sizes)]
ax.legend(wedges, legend_labels, loc="center left",
          bbox_to_anchor=(1.02, 0.5), fontsize=11, frameon=False)
ax.set_title("Query intent distribution", fontsize=13, pad=8)
save_fig("F2_intent_donut.png")

# ------------------------------------------------------------------
# E1 cross-profile Jaccard violin
# ------------------------------------------------------------------
def agent_desc_text(a):
    parts = [a.get("agent_role", ""), a.get("goal", ""),
             a.get("backstory", "")]
    return " ".join([p for p in parts if p])


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


def role_family(role):
    rl = role.lower()
    for fam, kws in ROLE_FAMILIES:
        if any(k in rl for k in kws):
            return fam
    return "other"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


by_qid = defaultdict(list)
for r in records:
    by_qid[r["question_id"]].append(r)

multi_qids = [qid for qid, lst in by_qid.items() if len(lst) >= 2]

agent_role_J, agent_family_J, tool_J, subtask_name_J = [], [], [], []
for qid in multi_qids:
    lst = by_qid[qid]
    feats = []
    for r in lst:
        out = r["plan"]["output"]
        roles = [a.get("agent_role", "") for a in out["agents"]]
        fams = [role_family(rr) for rr in roles]
        tools = set()
        for a in out["agents"]:
            tools.update(a.get("tools", []))
        sub_tokens = set()
        for s in out["subtasks"]:
            for tok in re.findall(r"[A-Za-z]+", s.get("name", "").lower()):
                if len(tok) >= 3:
                    sub_tokens.add(tok)
        feats.append((roles, fams, tools, sub_tokens))
    for i, j in combinations(range(len(feats)), 2):
        a, b = feats[i], feats[j]
        agent_role_J.append(jaccard(a[0], b[0]))
        agent_family_J.append(jaccard(a[1], b[1]))
        tool_J.append(jaccard(a[2], b[2]))
        subtask_name_J.append(jaccard(a[3], b[3]))

total_pairs = len(agent_role_J)

fig, ax = plt.subplots(figsize=(4.2, 3.4))
df_div = pd.DataFrame({
    "Agent\nroles": agent_role_J,
    "Role\nfamilies": agent_family_J,
    "Tools": tool_J,
    "Subtask\ntokens": subtask_name_J,
})
df_long = df_div.melt(var_name="dim", value_name="jac")
sns.violinplot(data=df_long, x="dim", y="jac", ax=ax,
               palette="Set2", inner="box", linewidth=1.2, cut=0)
for i, col in enumerate(df_div.columns):
    m = df_div[col].mean()
    ax.scatter(i, m, marker="D", color="#c0392b", s=45, zorder=5,
               edgecolor="white", linewidth=1.1)
    ax.text(i, m + 0.05, f"{m:.2f}", color="#c0392b", fontsize=11,
            ha="center", fontweight="bold")
ax.set_ylabel("Jaccard similarity")
ax.set_xlabel("")
ax.set_title(f"Cross-profile plan similarity\n(n={total_pairs:,} pairs)",
             fontsize=12)
ax.set_ylim(-0.05, 1.10)
sns.despine(ax=ax)
save_fig("E1_jaccard_violin.png")

# ------------------------------------------------------------------
# J1 phase coverage
# ------------------------------------------------------------------
PHASE_PATTERNS = {
    "probe": re.compile(
        r"\b(probe|ask|predict|prediction|hypothes|surface|diagnose|identify|"
        r"think out loud|articulate|restate|explain what|before revealing)\b",
        re.I,
    ),
    "retrieve--demo": re.compile(
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


def plan_blob(r):
    out = r["plan"]["output"]
    parts = []
    for s in out["subtasks"]:
        for st in s.get("steps", []):
            parts.append(st.get("instruction", ""))
            parts.append(st.get("objective", ""))
            parts.append(st.get("expected_output", ""))
    return " ".join(parts)


phase_hit = Counter()
n_plans = len(records)
for r in records:
    blob = plan_blob(r)
    for phase, pat in PHASE_PATTERNS.items():
        if pat.search(blob):
            phase_hit[phase] += 1

phases = ["probe", "retrieve--demo", "apply", "validate",
          "feedback", "consolidate"]
rates = [phase_hit[p] / n_plans for p in phases]

fig, ax = plt.subplots(figsize=(4.6, 3.6))
palette = sns.color_palette("crest_r", n_colors=len(phases))
bars = ax.bar(range(len(phases)), rates, color=palette,
              edgecolor="white", linewidth=0.6)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 0.02,
            f"{rate*100:.1f}%",
            ha="center", fontsize=11, fontweight="bold", color="#333")
ax.set_ylim(0, 1.18)
ax.set_ylabel("Plans covering phase")
ax.set_xlabel("")
ax.set_xticks(range(len(phases)))
ax.set_xticklabels(phases, rotation=35, ha="right")
ax.set_title("Pedagogical phase coverage", fontsize=13)
sns.despine(ax=ax)
save_fig("J1_phase_coverage.png")

print("Wrote", sorted(p.name for p in OUT_DIR.glob("*.png")))
