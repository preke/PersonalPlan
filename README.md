# MAPLE: Multi-Agent Personalized Learning Plans

> 📄 Paper: TBD &nbsp;|&nbsp; 🤗 Dataset: `preke/maple` &nbsp;|&nbsp; 🇨🇳 [中文版 README](README.zh.md)

MAPLE is a dataset and a full training/evaluation pipeline for **multi-agent
personalized teaching plans**. Each plan turns a real Stack Overflow question
into a multi-agent CrewAI workflow tailored to a specific learner profile,
filtered by a static structural check and an execution-feasibility check.

| stat | value |
|---|---:|
| plans (v15 release) | 3,043 |
| unique canonical questions | 1,730 |
| questions with ≥ 2 learner profiles | 971 |
| multi-intent rate (≥ 2 categories) | 44.75% |

The 971 multi-profile questions form a natural axis for **cross-profile
personalization** research: same question, different learners, different
plans.

## What's in this release

| component | path | what it does |
|---|---|---|
| Dataset construction | [`construction/`](construction/) | Build the dataset from Stack Overflow + learner profiles |
| Hierarchical SFT | [`training/sft/`](training/sft/) | PAD / SDP / Joint Alignment (paper §4) |
| GRPO | [`training/grpo/`](training/grpo/) | 4-reward GRPO (paper §4.2) |
| Plan execution runtime | [`evaluation/runtime/`](evaluation/runtime/) | CrewAI Teacher (GPT-4o) + Student (GPT-4o-mini) |
| Tier 1 (static) | [`evaluation/tier1_static/`](evaluation/tier1_static/) | Structural validity + counterfactual checks |
| Tier 2 (execution) | [`evaluation/tier2_execution/`](evaluation/tier2_execution/) | Feasibility scoring on real runs |
| Tier 3 (outcome) | [`evaluation/tier3_outcome/`](evaluation/tier3_outcome/) | Cross-method pairwise judge (Sati. / JCC) |
| Baselines | [`baselines/`](baselines/) | 8 baseline methods (AutoGen, AFlow, AutoAgents, EduPlanner, GenMentor, AOP, AIPoM, plus shared utils) |
| Prompts | [`prompts/`](prompts/) | All LLM prompts used in generation and scoring |
| Docs | [`docs/`](docs/) | Design HTMLs, paper-analysis code, workflow figure |
| Data examples | [`data/examples/`](data/examples/) | 8 representative plans + a 100-row sample |

## Quick start

```bash
git clone https://github.com/preke/PersonalPlan.git
cd maple

# Python ≥ 3.10 recommended
python -m venv .venv && source .venv/bin/activate
pip install -U huggingface_hub

# Fetch the full dataset (~36 MB) from Hugging Face
bash scripts/download_data.sh

# Or just inspect the 100-row sample shipped in the repo
head -1 data/examples/sample.jsonl | python -m json.tool
```

For per-task dependencies (training, evaluation), see the README in each
subdirectory.

## Reproducing paper results

See [`docs/reproduce.md`](docs/reproduce.md) for the table-by-table and
figure-by-figure command list.

## Dataset

The full MAPLE dataset is hosted on Hugging Face: **`preke/maple`** (link will
be added at release). It is licensed under **CC BY-SA 4.0** (inherited from
Stack Overflow). Its dataset card is in [`data/DATASET_CARD.md`](data/DATASET_CARD.md).

| split | plans | unique question_ids |
|---|---:|---:|
| train | 2,433 | 1,384 |
| dev   |   305 |   173 |
| test  |   305 |   173 |
| **total** | **3,043** | **1,730** |

Canonical split index: [`data/maple_split_v1.json`](data/maple_split_v1.json).
Per-bucket statistics: [`data/split_stats.json`](data/split_stats.json).

## License

- **Code** — MIT, see [`LICENSE`](LICENSE)
- **Dataset** — CC BY-SA 4.0, see [`LICENSE-DATA`](LICENSE-DATA)

Code is free to remix; the dataset and any derivative dataset must remain
ShareAlike under Stack Overflow's terms.

## Citation

```bibtex
@article{maple2026,
  title   = {MAPLE: Multi-Agent Personalized Learning Plans},
  author  = {TBD},
  year    = {2026}
}
```

## Acknowledgements

Question source: Stack Overflow (CC BY-SA 4.0). Plan generation uses Claude
Sonnet 4.6; execution layer uses GPT-4o and GPT-4o-mini via CrewAI.

## Contact

Open an issue, or see the author's email in `CITATION.cff`.
