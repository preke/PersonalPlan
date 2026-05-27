# MAP-PPL Qwen3-8B SFT + Joint Alignment

This repository contains the current MAP-PPL training workflow used for the
Qwen3-8B experiment. The authoritative runbook is:

- `readmore_517_sft.md`

Use that file as the source of truth for server setup, training commands,
checkpoint checks, and recovery steps.

## Current Workflow

- Stage 1a: PAD SFT, 3 epochs
- Stage 1b: SDP SFT, 2 epochs
- Stage 2: Joint Alignment, K=1
  - Phase A: PAD DPO from structural scaffold preferences
  - Phase B: SDP on-policy SFT from PAD-sampled scaffold inputs

The split is frozen by `maple_split_v1.json`, grouped by `question_id`.
Current split counts are:

- Train: 2433 rows
- Dev: 305 rows
- Test: 305 rows

## Important Current Notes

- Do not use AutoTrain as the launcher for this experiment.
- Use `train_sft_lora.py` for PAD/SDP SFT and Phase B SDP on-policy SFT.
- Use `train_dpo_lora.py` for Phase A PAD DPO.
- Use `joint_alignment_data.py` for the two Joint data-generation stages.
- Joint data generation uses `SRC=data/raw/train.jsonl`.
- Joint scaffold generation uses `BATCH_SIZE=32` by default in the runbook.
- Batched generation should not silently truncate prompts.
- `joint_alignment_data.py` handles adapter/base tokenizer-size mismatch by
  resizing base embeddings before loading the LoRA adapter.

## Tool Constraint Fix

The SDP prompt now explicitly provides the eight valid CrewAI tool class names:

- `FirecrawlSearchTool`
- `RagTool`
- `CodeInterpreterTool`
- `DirectoryReadTool`
- `FileReadTool`
- `FileWriterTool`
- `CodeDocsSearchTool`
- `ArxivPaperTool`

This fixes the earlier mismatch where the model learned semantic tool names
while evaluation expected fixed CrewAI class names.

## Core Files

- `build_sft_data.py`: builds PAD/SDP SFT data and raw split files.
- `prompts.py`: prompt templates, including SDP CrewAI tool constraints.
- `train_sft_lora.py`: TRL SFTTrainer LoRA runner.
- `train_dpo_lora.py`: TRL DPOTrainer LoRA runner.
- `joint_alignment_data.py`: Joint data generation for PAD DPO and SDP
  on-policy SFT.
- `checkpoint_metadata.py`: writes checkpoint metadata when metric values are
  provided.
- `configs/`: YAML training configs.
- `training logs/`: local training logs and checkpoint-location notes.

## Checkpoints

Expected server-side adapter outputs:

- `maple-pad-sft/`
- `maple-sdp-sft/`
- `maple-pad-dpo-iter1/`
- `maple-sdp-onpolicy-iter1/`

Stage 2 uses four separate manual commands in `readmore_517_sft.md`, so if one
stage fails, rerun only that stage rather than restarting the full workflow.

## Package Notes

This folder may include HTML reports and historical experiment notes for
documentation. When documentation conflicts, follow `readmore_517_sft.md`.
