"""Build PAD/SDP SFT datasets from the frozen MAPLE question_id split.

This script does not create a new split. It reads the already-frozen
``maple_split_v1.json`` from EXPERIMENT_PLAN_2026-05-15 and materializes:

  data/pad/{train,dev,test}.jsonl
  data/sdp/{train,dev,test}.jsonl
  data/raw/{train,dev,test}.jsonl

Each output row is AutoTrain-compatible and has a ``messages`` column:

  PAD input  = query + learner profile
  PAD target = agents + subtasks without steps

  SDP input  = query + learner profile + gold agents/subtasks
  SDP target = steps + execution_order

Important: do not random-split rows here. The experiment plan requires all
rows with the same question_id to stay in the same split to avoid leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prompts import PAD_SYSTEM, PAD_USER_TEMPLATE, SDP_SYSTEM, SDP_USER_TEMPLATE

HERE = Path(__file__).resolve().parent


def extract_learner(lrn: dict[str, Any]) -> tuple[str, list[str]]:
    """Normalize the learner schema variants used by MAPLE."""
    desc = lrn.get("self_description") or lrn.get("about_me") or ""
    skills = lrn.get("skills") or lrn.get("top_tags") or []
    return desc.strip(), list(skills)


def agent_record(a: dict[str, Any]) -> dict[str, Any]:
    """Normalize the agent record. backstory/description are aliases."""
    return {
        "agent_role": a.get("agent_role", ""),
        "goal": a.get("goal", ""),
        "backstory": a.get("backstory") or a.get("description") or "",
        "tools": list(a.get("tools") or []),
    }


def pad_target(plan_output: dict[str, Any]) -> dict[str, Any]:
    """High-level slice: agents + subtask metadata, no steps."""
    return {
        "agents": [agent_record(a) for a in plan_output["agents"]],
        "subtasks": [
            {
                "id": s["id"],
                "name": s["name"],
                "subtask_objective": s["subtask_objective"],
            }
            for s in plan_output["subtasks"]
        ],
    }


def sdp_target(plan_output: dict[str, Any]) -> dict[str, Any]:
    """Lower-level slice: steps grouped by subtask + execution_order."""
    return {
        "subtasks": [
            {"id": s["id"], "steps": list(s.get("steps", []))}
            for s in plan_output["subtasks"]
        ],
        "execution_order": list(plan_output.get("execution_order", [])),
    }


def build_pad_messages(record: dict[str, Any]) -> dict[str, Any]:
    inp = record["plan"]["input"]
    query = inp["query"]
    desc, skills = extract_learner(inp["learner"])
    target = pad_target(record["plan"]["output"])

    user_text = PAD_USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
    )
    assistant_text = json.dumps(target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": PAD_SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "question_id": record["question_id"],
        "profile_index": record["profile_index"],
    }


def build_sdp_messages(record: dict[str, Any]) -> dict[str, Any]:
    inp = record["plan"]["input"]
    query = inp["query"]
    desc, skills = extract_learner(inp["learner"])
    scaffold = pad_target(record["plan"]["output"])
    target = sdp_target(record["plan"]["output"])

    user_text = SDP_USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
        agents=json.dumps(scaffold["agents"], ensure_ascii=False, indent=2),
        subtasks=json.dumps(scaffold["subtasks"], ensure_ascii=False, indent=2),
    )
    assistant_text = json.dumps(target, ensure_ascii=False, indent=2)
    return {
        "messages": [
            {"role": "system", "content": SDP_SYSTEM},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "question_id": record["question_id"],
        "profile_index": record["profile_index"],
    }


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(path: Path) -> dict[str, set[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    required = {"train_qids", "dev_qids", "test_qids"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Split file {path} is missing keys: {sorted(missing)}")
    return {
        "train": {str(qid) for qid in raw["train_qids"]},
        "dev": {str(qid) for qid in raw["dev_qids"]},
        "test": {str(qid) for qid in raw["test_qids"]},
    }


def partition_records(
    records: list[dict[str, Any]],
    split: dict[str, set[str]],
) -> dict[str, list[dict[str, Any]]]:
    qid_to_split = {}
    for split_name, qids in split.items():
        for qid in qids:
            if qid in qid_to_split:
                raise ValueError(f"question_id {qid} appears in multiple splits")
            qid_to_split[qid] = split_name

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    missing_qids: set[str] = set()
    for record in records:
        qid = str(record["question_id"])
        split_name = qid_to_split.get(qid)
        if split_name is None:
            missing_qids.add(qid)
            continue
        buckets[split_name].append(record)

    if missing_qids:
        sample = ", ".join(sorted(missing_qids)[:5])
        raise ValueError(
            f"{len(missing_qids)} question_id values from the source are absent "
            f"from the split file. Examples: {sample}"
        )

    for rows in buckets.values():
        rows.sort(key=lambda r: (str(r["question_id"]), int(r["profile_index"])))
    return buckets


def remove_legacy_valid_files(out_dir: Path) -> None:
    for subdir in ("pad", "sdp"):
        legacy = out_dir / subdir / "valid.jsonl"
        if legacy.exists():
            legacy.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(HERE / "multi_agent_dataset_filtered_qap.jsonl"),
        help="Source MAPLE JSONL. Default: full 2026-05-15 dataset.",
    )
    ap.add_argument(
        "--split-file",
        default=str(HERE / "maple_split_v1.json"),
        help="Frozen question_id split from EXPERIMENT_PLAN_2026-05-15.",
    )
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    split_path = Path(args.split_file).resolve()
    out_dir = Path(args.out_dir).resolve()

    records = load_jsonl(in_path)
    split = load_split(split_path)
    buckets = partition_records(records, split)
    remove_legacy_valid_files(out_dir)

    print(f"Source            : {in_path}")
    print(f"Split file        : {split_path}")
    print(f"Total records     : {len(records)}")
    print(
        "Train / Dev / Test: "
        f"{len(buckets['train'])} / {len(buckets['dev'])} / {len(buckets['test'])}"
    )

    for split_name, split_rows in buckets.items():
        write_jsonl(out_dir / "raw" / f"{split_name}.jsonl", split_rows)
        pad_rows = [build_pad_messages(row) for row in split_rows]
        sdp_rows = [build_sdp_messages(row) for row in split_rows]
        write_jsonl(out_dir / "pad" / f"{split_name}.jsonl", pad_rows)
        write_jsonl(out_dir / "sdp" / f"{split_name}.jsonl", sdp_rows)
        print(
            f"  wrote {len(pad_rows):>4} PAD / {len(sdp_rows):>4} SDP "
            f"rows to {out_dir.name}/{{pad,sdp}}/{split_name}.jsonl"
        )

    print("\nSanity: assistant-text character length per stage (train):")
    pad_lens = [
        len(row["messages"][-1]["content"])
        for row in load_jsonl(out_dir / "pad" / "train.jsonl")
    ]
    sdp_lens = [
        len(row["messages"][-1]["content"])
        for row in load_jsonl(out_dir / "sdp" / "train.jsonl")
    ]
    if pad_lens:
        print(
            f"  PAD assistant chars: min {min(pad_lens)}, "
            f"max {max(pad_lens)}, mean {sum(pad_lens) // len(pad_lens)}"
        )
    if sdp_lens:
        print(
            f"  SDP assistant chars: min {min(sdp_lens)}, "
            f"max {max(sdp_lens)}, mean {sum(sdp_lens) // len(sdp_lens)}"
        )


if __name__ == "__main__":
    main()
