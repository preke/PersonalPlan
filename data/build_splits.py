"""Build Train/Dev/Test splits for MAPLE and materialize them for SFT + GRPO.

Implements §2.2 of EXPERIMENT_PLAN_2026-05-15.html:

  * split key  = question_id  (NEVER row index — same query / different
    profile rows must not straddle splits)
  * stratum    = number of profile variants the qid owns
  * proportion = 80 / 10 / 10
  * algorithm  = random.shuffle within each stratum, seed = 42
  * write      = splits/maple_split_v1.json (qid lists, frozen)

Then re-emits the existing PAD / SDP / GRPO row builders into three
splits each — replacing the previous valid-only files.

Outputs (overwrites the old 2-way splits):

  SFT/data/pad/{train,dev,test}.jsonl
  SFT/data/sdp/{train,dev,test}.jsonl
  GRPO/data/grpo/{train,dev,test}.jsonl
  splits/maple_split_v1.json
  splits/split_stats.json     # consumed by the HTML report
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "multi_agent_dataset_filtered_qap.jsonl"
SPLIT_DIR = ROOT / "splits"
SFT_DIR = ROOT / "SFT"
GRPO_DIR = ROOT / "GRPO"

sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(GRPO_DIR))

from prompts import (  # noqa: E402  SFT/prompts.py
    PAD_SYSTEM,
    PAD_USER_TEMPLATE,
    SDP_SYSTEM,
    SDP_USER_TEMPLATE,
)
from build_sft_data import build_pad_messages, build_sdp_messages  # noqa: E402
from build_grpo_prompts import build_row as build_grpo_row  # noqa: E402

SEED = 42
TEST_FRAC = 0.10
DEV_FRAC = 0.10


# ---------- load --------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- stratified qid split ---------------------------------------------

def stratified_qid_split(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Replicates EXPERIMENT_PLAN §2.2 exactly.

    stratum = number of distinct profile_index values for that qid.
    Within each stratum, shuffle (seed=42), then carve out
    round(n*0.10) test and round(n*0.10) dev; remainder is train.
    """
    qid_to_profs: dict[str, set[int]] = defaultdict(set)
    for d in records:
        qid_to_profs[str(d["question_id"])].add(int(d["profile_index"]))

    strata: dict[int, list[str]] = defaultdict(list)
    for q, profs in qid_to_profs.items():
        strata[len(profs)].append(q)

    rng = random.Random(SEED)
    train_q: list[str] = []
    dev_q: list[str] = []
    test_q: list[str] = []
    for k in sorted(strata.keys()):
        ql = strata[k]
        rng.shuffle(ql)
        n = len(ql)
        n_test = round(n * TEST_FRAC)
        n_dev = round(n * DEV_FRAC)
        test_q += ql[:n_test]
        dev_q += ql[n_test:n_test + n_dev]
        train_q += ql[n_test + n_dev:]

    return {"train_qids": train_q, "dev_qids": dev_q, "test_qids": test_q}


# ---------- split-aware row materialization ----------------------------------

def partition_rows(records: list[dict[str, Any]], split: dict[str, list[str]]):
    qid_to_split = {q: "train" for q in split["train_qids"]}
    qid_to_split.update({q: "dev" for q in split["dev_qids"]})
    qid_to_split.update({q: "test" for q in split["test_qids"]})

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for r in records:
        s = qid_to_split[str(r["question_id"])]
        buckets[s].append(r)
    # determinism for downstream shuffling
    for s in buckets:
        buckets[s].sort(key=lambda d: (str(d["question_id"]), int(d["profile_index"])))
    return buckets


def materialize_sft(buckets: dict[str, list[dict[str, Any]]]) -> None:
    """Write PAD + SDP train/dev/test JSONL (chat-template ready)."""
    sft_data = SFT_DIR / "data"
    # Clean previous 2-way valid.jsonl so nothing stale remains.
    for sub in ("pad", "sdp"):
        for legacy in ("valid.jsonl",):
            p = sft_data / sub / legacy
            if p.exists():
                p.unlink()
    for split, rows in buckets.items():
        pad_rows = [build_pad_messages(r) for r in rows]
        sdp_rows = [build_sdp_messages(r) for r in rows]
        write_jsonl(sft_data / "pad" / f"{split}.jsonl", pad_rows)
        write_jsonl(sft_data / "sdp" / f"{split}.jsonl", sdp_rows)


def materialize_grpo(buckets: dict[str, list[dict[str, Any]]]) -> None:
    """Write prompt-only train/dev/test JSONL."""
    grpo_data = GRPO_DIR / "data" / "grpo"
    for legacy in ("valid.jsonl",):
        p = grpo_data / legacy
        if p.exists():
            p.unlink()
    for split, rows in buckets.items():
        out_rows = [build_grpo_row(r) for r in rows]
        write_jsonl(grpo_data / f"{split}.jsonl", out_rows)


# ---------- diagnostic stats --------------------------------------------------

def collect_stats(records, split, buckets):
    qid_to_profs: dict[str, set[int]] = defaultdict(set)
    qid_to_rowcount: dict[str, int] = defaultdict(int)
    profile_idx_counter: Counter = Counter()
    for d in records:
        q = str(d["question_id"])
        qid_to_profs[q].add(int(d["profile_index"]))
        qid_to_rowcount[q] += 1
        profile_idx_counter[int(d["profile_index"])] += 1

    nprof_counter: Counter = Counter(len(s) for s in qid_to_profs.values())

    def per_split(qids: list[str], rows: list[dict[str, Any]]):
        qids_set = set(qids)
        nprof_dist: Counter = Counter()
        prof_idx_dist: Counter = Counter()
        for q in qids:
            nprof_dist[len(qid_to_profs[q])] += 1
        for r in rows:
            prof_idx_dist[int(r["profile_index"])] += 1
        # how many qids in this split have ≥3 profiles (used for
        # profile-sensitivity subset, see EXPERIMENT_PLAN §2.2)
        n_ge3 = sum(1 for q in qids if len(qid_to_profs[q]) >= 3)
        n_ge2 = sum(1 for q in qids if len(qid_to_profs[q]) >= 2)
        # leakage check
        leaks = [str(r["question_id"]) for r in rows if str(r["question_id"]) not in qids_set]
        return {
            "n_qids": len(qids),
            "n_rows": len(rows),
            "nprof_per_qid": dict(sorted(nprof_dist.items())),
            "profile_index_dist": dict(sorted(prof_idx_dist.items())),
            "n_qids_ge2_profiles": n_ge2,
            "n_qids_ge3_profiles": n_ge3,
            "leakage_rows": len(leaks),
        }

    splits_stats = {
        s: per_split(split[f"{s}_qids"], buckets[s])
        for s in ("train", "dev", "test")
    }

    # cross-split disjointness checks
    tq = set(split["train_qids"])
    dq = set(split["dev_qids"])
    eq = set(split["test_qids"])
    overlap = {
        "train_dev": len(tq & dq),
        "train_test": len(tq & eq),
        "dev_test": len(dq & eq),
    }

    return {
        "global": {
            "n_rows": len(records),
            "n_unique_qids": len(qid_to_profs),
            "nprof_per_qid": dict(sorted(nprof_counter.items())),
            "profile_index_dist": dict(sorted(profile_idx_counter.items())),
        },
        "splits": splits_stats,
        "overlap_check": overlap,
        "seed": SEED,
        "fractions": {"test": TEST_FRAC, "dev": DEV_FRAC, "train": 1 - TEST_FRAC - DEV_FRAC},
        "src_file": str(SRC.relative_to(ROOT)),
    }


# ---------- main --------------------------------------------------------------

def main() -> None:
    records = load_jsonl(SRC)
    split = stratified_qid_split(records)
    buckets = partition_rows(records, split)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_DIR / "maple_split_v1.json", "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    materialize_sft(buckets)
    materialize_grpo(buckets)

    stats = collect_stats(records, split, buckets)
    with open(SPLIT_DIR / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("== Split summary ==")
    for s in ("train", "dev", "test"):
        ss = stats["splits"][s]
        print(f"  {s:>5}: {ss['n_qids']:>5} qids / {ss['n_rows']:>5} rows "
              f"(≥3-profile qids: {ss['n_qids_ge3_profiles']})")
    print(f"  overlap (must be 0,0,0): {stats['overlap_check']}")


if __name__ == "__main__":
    main()
