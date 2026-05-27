"""Materialize canonical train/dev/test JSONL files for the Hugging Face dataset release.

Reads:
  --source   multi_agent_dataset_filtered_qap.jsonl  (full 3,043 rows)
  --split    data/maple_split_v1.json                (qid → split mapping)
Writes:
  <out-dir>/train.jsonl
  <out-dir>/dev.jsonl
  <out-dir>/test.jsonl
  <out-dir>/sample.jsonl  (100 rows from train, for quick inspection)

Each output row preserves the original {question_id, profile_index, plan}
structure so that downstream loaders see the same schema as the source.

Usage:
    python scripts/build_canonical_splits.py \\
        --source path/to/multi_agent_dataset_filtered_qap.jsonl \\
        --split  data/maple_split_v1.json \\
        --out-dir /path/to/maple-hf-staging
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_split(path: Path) -> dict[str, set]:
    with path.open() as f:
        raw = json.load(f)
    return {
        "train": {str(q) for q in raw["train_qids"]},
        "dev":   {str(q) for q in raw["dev_qids"]},
        "test":  {str(q) for q in raw["test_qids"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split",  type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    qid2split = load_split(args.split)
    qid_to_split = {}
    for split_name, qids in qid2split.items():
        for q in qids:
            qid_to_split[q] = split_name

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: (args.out_dir / f"{name}.jsonl").open("w") for name in ("train", "dev", "test")}
    counts: Counter[str] = Counter()
    orphan_qids: set[str] = set()

    sample_rows: list[str] = []
    with args.source.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id"))
            split = qid_to_split.get(qid)
            if split is None:
                orphan_qids.add(qid)
                continue
            writers[split].write(line + "\n")
            counts[split] += 1
            if split == "train" and len(sample_rows) < args.sample_size:
                sample_rows.append(line)

    for w in writers.values():
        w.close()

    sample_path = args.out_dir / "sample.jsonl"
    with sample_path.open("w") as f:
        for r in sample_rows:
            f.write(r + "\n")

    print("=== canonical splits written ===")
    for name in ("train", "dev", "test"):
        print(f"  {name:5s}: {counts[name]:5d} rows -> {args.out_dir / (name + '.jsonl')}")
    print(f"  sample : {len(sample_rows):5d} rows -> {sample_path}")
    if orphan_qids:
        print(f"WARNING: {len(orphan_qids)} qid(s) in source but not in split JSON (skipped). "
              f"Example: {next(iter(orphan_qids))}")


if __name__ == "__main__":
    main()
