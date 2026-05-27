"""
Stream (query, learner) pairs from the main dataset.
Renames profile.self_description → learner.about_me,
        profile.skills → learner.top_tags.
Does NOT yield the accepted answer (inference prompt does not need it).
"""
import json
from pathlib import Path
from typing import Iterator

# Project root = baselines/common/data_loader.py → common → baselines → ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DEFAULT = _PROJECT_ROOT / "multi_agent_dataset_filtered_qap_v15_goodplus.jsonl"


def load_main_dataset(jsonl_path: Path = DATASET_DEFAULT,
                      limit: int = None) -> Iterator[dict]:
    """
    Each yielded item:
        {
            "key": "<question_id>__<profile_idx>",
            "question_id": "...",
            "profile_idx": int,
            "query": "...",
            "learner": {"about_me": str, "top_tags": list[str]}
        }
    """
    yielded = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj["question_id"]
            # Two supported layouts:
            # (A) QAP layout (filtered_qap.jsonl): canonical_query + profiles_answers
            # (B) Main dataset layout (multi_agent_dataset_filtered_qap.jsonl):
            #     plan.input.{query, learner{about_me, top_tags}} per row,
            #     with one profile per row indexed by profile_index.
            if "canonical_query" in obj:
                query = obj["canonical_query"]
                for idx, pa in enumerate(obj.get("profiles_answers", [])):
                    profile = pa.get("profile", {})
                    yield {
                        "key": f"{qid}__{idx}",
                        "question_id": qid,
                        "profile_idx": idx,
                        "query": query,
                        "learner": {
                            "about_me": profile.get("self_description", ""),
                            "top_tags": profile.get("skills", []),
                        },
                    }
                    yielded += 1
                    if limit and yielded >= limit:
                        return
            else:
                plan = obj.get("plan", {})
                inp = plan.get("input", {})
                query = inp.get("query", "")
                learner_in = inp.get("learner", {}) or {}
                idx = obj.get("profile_index", 0)
                yield {
                    "key": f"{qid}__{idx}",
                    "question_id": qid,
                    "profile_idx": idx,
                    "query": query,
                    "learner": {
                        "about_me": learner_in.get("about_me", ""),
                        "top_tags": learner_in.get("top_tags", []),
                    },
                }
                yielded += 1
                if limit and yielded >= limit:
                    return
