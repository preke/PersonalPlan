---
language:
  - en
license: cc-by-sa-4.0
pretty_name: "MAPLE: Multi-Agent Personalized Learning Plans"
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
  - question-answering
tags:
  - multi-agent
  - personalized-learning
  - educational-ai
  - planning
  - stack-overflow
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: dev
        path: dev.jsonl
      - split: test
        path: test.jsonl
---

# MAPLE: Multi-Agent Personalized Learning Plans

> Use this file as the dataset card on Hugging Face Hub. The `data_files` block
> above assumes you upload `train.jsonl`, `dev.jsonl`, `test.jsonl` (built by
> `data/build_splits.py`) at the dataset repo root. A 100-row `sample.jsonl`
> is recommended for quick inspection.

## Summary

MAPLE pairs Stack Overflow duplicate-question clusters with synthetic learner
profiles, then asks Claude Sonnet 4.6 to draft a **multi-agent teaching
plan**. Plans pass through a two-stage filter — static structural validation
and CrewAI execution feasibility — before entering the release set.

- **3,043** plans (v15)
- **1,730** unique canonical questions
- **971** questions with ≥ 2 learner profiles → cross-profile personalization axis
- **7** question categories (multi-label): `CONCEPTUAL`, `API_USAGE`, `REVIEW`,
  `DISCREPANCY`, `ERRORS`, `LEARNING`, `API_CHANGE`
- **Multi-intent rate**: 44.75% (questions in ≥ 2 categories)

## Data fields

Each row is a JSON object with `input` and `output`.

### `input`

| field | description |
|---|---|
| `query` | Natural-language task description (SO title + body) |
| `learner.about_me` | Background paragraph (study history, tech stack) |
| `learner.top_tags` | List of familiar skill tags |

### `output`

- `agents[]` — agent definitions (`agent_role`, `goal`, `backstory`, `tools`)
- `subtasks[]` — staged decomposition (id, name, objective, `steps[]`)
- `execution_order` — dependency-respecting step sequence (supports `loop`)

Tool pool (8 tools): `CodeInterpreterTool`, `CodeDocsSearchTool`,
`FirecrawlSearchTool`, `FileWriterTool`, `ArxivPaperTool`, `RagTool`,
`DirectoryReadTool`, `FileReadTool`. (The first five are actually used in v15.)

## Splits

Stratified 80/10/10 at the `question_id` level, bucketed by profile count, to
prevent cross-profile leakage of the same question.

| split | plans | unique question_ids |
|---|---:|---:|
| train | 2,433 | 1,384 |
| dev   |   305 |   173 |
| test  |   305 |   173 |
| **total** | **3,043** | **1,730** |

See `splits/maple_split_v1.json` in the GitHub repo for the canonical
mapping and `splits/split_stats.json` for per-bucket statistics.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("preke/maple")
print(ds)
print(ds["train"][0])
```

## How MAPLE was built

1. **Query selection** — Stack Overflow duplicate-question clusters filtered to
   programming questions answerable by the 8-tool agent pool.
2. **Profile pairing** — synthetic learner profiles (skills + about-me) matched
   to each canonical question; multi-profile questions form the
   personalization axis.
3. **Plan generation** — Claude Sonnet 4.6 with the prompt in
   `prompts/plan_generation_prompt.txt`.
4. **Static filter** — structural validity check (agent definitions, subtask
   schema, dependency DAG).
5. **Execution filter** — CrewAI Teacher (GPT-4o) + Student (GPT-4o-mini)
   Socratic execution must complete without runtime failure.

Full code and prompts are in the GitHub repository.

## License

**CC BY-SA 4.0**, inherited from the Stack Overflow source. Derivative works
must use a compatible ShareAlike license. See `LICENSE-DATA` in the GitHub
repo for details.

## Citation

```bibtex
@article{maple2026,
  title   = {MAPLE: Multi-Agent Personalized Learning Plans},
  author  = {TBD},
  year    = {2026},
  journal = {TBD}
}
```

## Limitations and ethical considerations

- Plans are model-generated and not pedagogically validated by human experts;
  use as a research artifact, not as production curriculum.
- Learner profiles are synthetic and do not represent real users.
- Stack Overflow content may contain outdated APIs; verify before deploying
  any plan in a teaching setting.

## Contact

Open an issue on the GitHub repository for questions.
