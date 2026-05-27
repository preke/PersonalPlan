# MAP-PPL 2026-05-15 SFT + Joint execution notes

Scope: your part includes Stage 1 SFT and Stage 2 Joint Alignment.

- Stage 1a: LoRA-PAD, 3 epochs
- Stage 1b: LoRA-SDP, 2 epochs
- Stage 2: Joint Alignment, K=1
- Frozen split: `maple_split_v1.json`, grouped by `question_id`
- Train/dev/test rows: 2433 / 305 / 305
- Backbone in the experiment plan: `Qwen/Qwen3-8B-Instruct`

## Current Status

Already done:

- `build_sft_data.py` reads `maple_split_v1.json`; it does not row-random split.
- It now writes both SFT data and raw split data:
  - `data/pad/{train,dev,test}.jsonl`
  - `data/sdp/{train,dev,test}.jsonl`
  - `data/raw/{train,dev,test}.jsonl`
- Current validated counts:
  - Train: 2433 rows
  - Dev: 305 rows
  - Test: 305 rows
- `valid.jsonl` is not used.
- PAD/SDP configs match the Stage 1 hyperparameter table.
- Stage 1 checkpoint saving is enabled every 50 steps.
- Stage 1 logging is configured every 1 step.
- Manual Joint commands use `configs/`.
- Manual Joint commands use `SRC=data/raw/train.jsonl`, so Joint uses Train only.
- Manual Joint order follows the experiment plan:
  - Phase A: PAD DPO
  - Phase B: SDP on-policy SFT
- `joint_alignment_data.py` resizes base embeddings before loading adapters when adapter tokenizer length differs.
- Stage 1 SFT was completed with TRL `SFTTrainer`, not AutoTrain:
  - PAD: 459 steps / 3 epochs, final `eval_loss = 0.7246`, final `train_loss = 0.7207`.
  - SDP: 306 steps / 2 epochs, final `eval_loss = 0.5097`, final `train_loss = 0.5307`.
  - Output folders contain LoRA adapter files such as `adapter_config.json` and `adapter_model.safetensors`.
- The recommended manual Joint flow avoids AutoTrain too:
  - Phase A PAD DPO uses `train_dpo_lora.py` with TRL `DPOTrainer`.
  - Phase B SDP on-policy SFT uses `train_sft_lora.py` with TRL `SFTTrainer`.

Still manual:

- Environment cleanup on the server, especially removing packages that pin old `transformers`.
- Download the base model locally because the server cannot reach Hugging Face.
- Confirm LoRA output after every SFT/DPO run.
- Stage 1 epoch-end SV/AR checks are not automated in this folder.
- Stage 2 Gate diagnostics (`pass@1`, `pass@8`, GED-sim, and one LLM-judge sanity check) are not automated in this folder.
- `checkpoint_metadata.py` can write required `metadata.json` files once you provide the metric values.
- `eval_plan_structure.py` can compute SV/AR from a JSONL of generated plans.

## Important Environment Notes

Do not use `autotrain-advanced==0.8.36` as the launcher. It pins `transformers==4.48.0`, which cannot load Qwen3 (`model_type: qwen3`). Stage 1 uses `train_sft_lora.py` with TRL `SFTTrainer`; Stage 2 uses `train_dpo_lora.py` plus `train_sft_lora.py`.

Recommended clean environment:

```bash
conda create -n maple-sft python=3.12 -y
conda activate maple-sft

pip install -U pip
pip uninstall -y autotrain-advanced
pip install "transformers>=4.51.1" trl peft accelerate bitsandbytes datasets pyyaml safetensors tensorboard modelscope
```

Download the base model locally:

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /root/autodl-tmp/Qwen3-8B-Instruct
```

Then run with:

```bash
export BASE_MODEL=/root/autodl-tmp/Qwen3-8B-Instruct
```

For a smoke test only, you may use a smaller local model, but PAD, SDP, and Joint must all use the same `BASE_MODEL`.

## File Checklist

Use for your part:

- `build_sft_data.py`: builds SFT data and raw train/dev/test split from `multi_agent_dataset_filtered_qap.jsonl` + `maple_split_v1.json`.
- `prompts.py`: PAD/SDP prompt templates.
- `configs/pad_autotrain.yml`: Stage 1a PAD SFT config, read by `train_sft_lora.py`.
- `configs/sdp_autotrain.yml`: Stage 1b SDP SFT config, read by `train_sft_lora.py`.
- `train_sft_lora.py`: direct TRL SFTTrainer runner for Stage 1 LoRA.
- `train_dpo_lora.py`: direct TRL DPOTrainer runner for Stage 2 PAD DPO.
- `train_pad.sh`: runs PAD.
- `train_sdp.sh`: runs SDP.
- `joint_alignment_data.py`: generates Joint Phase A/B data.
- `checkpoint_metadata.py`: writes experiment-plan `metadata.json` for a checkpoint.
- `eval_plan_structure.py`: computes basic SV/AR for generated plan JSONL files.
- `configs/joint_phase_b_autotrain.yml`: Stage 2 Phase A PAD DPO template, read by `train_dpo_lora.py`.
- `configs/joint_phase_a_autotrain.yml`: Stage 2 Phase B SDP SFT template, read by `train_sft_lora.py`.
- `train_joint.sh`: runs Joint Alignment K iterations.

Do not use as the main training entrypoint:

- `build_split_report.py`: report generation only.
- `build_grpo_prompts.py`: GRPO helper, not needed for PAD/SDP/Joint.

## Stage 1 Hyperparameters

Both PAD and SDP:

```text
base model: /root/autodl-tmp/Qwen3-8B-Instruct
LoRA r / alpha / dropout: 16 / 32 / 0.05
target modules: all-linear
optimizer: paged_adamw_8bit
learning rate: 1e-4
scheduler: cosine
warmup_ratio: 0.02
weight_decay: 0.01
per-device batch / grad_accum: 2 / 8
effective batch: 16
max seq length: 3072
bf16: enabled
gradient checkpointing: enabled
eval_steps: 50
save_steps: 50
logging_steps: 1
```

Stage-specific:

```text
PAD epochs: 3
SDP epochs: 2
PAD data path: data/pad
SDP data path: data/sdp
validation split: dev
```

## Stage 2 Joint Settings

Phase A, PAD DPO:

```text
input source: data/raw/train.jsonl
data output: data/joint/iter1/phase_a_pad_dpo/train.jsonl
config template: configs/joint_phase_b_autotrain.yml
trainer: TRL DPOTrainer via train_dpo_lora.py
project output: maple-pad-dpo-iter1
DPO beta: 0.1
lr: 5e-5
epochs: 1
```

Important deviation from strict paper text:

```text
The paper/plan says Phase A should choose scaffold pairs by:
PAD samples 2 scaffolds -> frozen SDP generates a plan for each -> rule-based reward R_struct + R_pers + R_ped chooses winner/loser.

Current implementation chooses scaffold pairs by:
PAD samples 2 scaffolds -> compute scaffold-vs-gold structural similarity -> higher structural similarity chooses winner.
```

So Stage 2 keeps the K=1 PAD-DPO -> SDP-on-policy-SFT structure, but Phase A uses scaffold-vs-gold structural Jaccard as a practical preference proxy. This matches the currently uploaded root `joint_alignment_data.py`; it does not load the SDP adapter for Phase A scoring and it does not implement the full generated-plan `R_struct + R_pers + R_ped` reward stack.

Phase B, SDP on-policy SFT:

```text
input source: data/raw/train.jsonl
data output: data/joint/iter1/phase_b_sdp_sft/train.jsonl
config template: configs/joint_phase_a_autotrain.yml
trainer: TRL SFTTrainer via train_sft_lora.py
project output: maple-sdp-onpolicy-iter1
lr: 5e-5
epochs: 1
```

## Upload Files To Server

From local:


```bash
cd "<repo>/training/sft/smallmodel_training"

ssh -p 19003 root@connect.westd.seetacloud.com "mkdir -p ~/SFTV4/configs"

scp -P 19003 build_sft_data.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 prompts.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 train_sft_lora.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 train_dpo_lora.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 train_pad.sh root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 train_sdp.sh root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 train_joint.sh root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 joint_alignment_data.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 checkpoint_metadata.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 eval_plan_structure.py root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 readmore_515_sft.md root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 multi_agent_dataset_filtered_qap.jsonl root@connect.westd.seetacloud.com:~/SFTV4/
scp -P 19003 maple_split_v1.json root@connect.westd.seetacloud.com:~/SFTV4/
scp -r -P 19003 configs root@connect.westd.seetacloud.com:~/SFTV4/
```

## Build Data

```bash
cd ~root/autodl-tmp/SFTV4/
python build_sft_data.py
```

Check:

```bash
wc -l data/raw/train.jsonl data/raw/dev.jsonl data/raw/test.jsonl
wc -l data/pad/train.jsonl data/pad/dev.jsonl data/pad/test.jsonl
wc -l data/sdp/train.jsonl data/sdp/dev.jsonl data/sdp/test.jsonl
```

Expected:

```text
2433 data/raw/train.jsonl
305  data/raw/dev.jsonl
305  data/raw/test.jsonl
2433 data/pad/train.jsonl
305  data/pad/dev.jsonl
305  data/pad/test.jsonl
2433 data/sdp/train.jsonl
305  data/sdp/dev.jsonl
305  data/sdp/test.jsonl
```

## Train Stage 1

```bash
chmod +x train_pad.sh train_sdp.sh train_joint.sh
export BASE_MODEL=/root/autodl-tmp/Qwen3-8B-Instruct

./train_pad.sh 2>&1 | tee pad_sft_515.log
./train_sdp.sh 2>&1 | tee sdp_sft_515.log
```

Expected output folders:

```text
maple-pad-sft/
maple-sdp-sft/
```

Verify LoRA, not full fine-tune:

```bash
ls maple-pad-sft
ls maple-sdp-sft
```

Good signs:

```text
adapter_config.json
adapter_model.safetensors
```

Danger sign:

```text
model.safetensors
```

If you see only a large full `model.safetensors`, stop and re-check that `train_sft_lora.py` used PEFT/LoRA.

## Train Stage 2 Joint

Important: only run this after the Stage 1 output folders are verified:

```bash
ls maple-pad-sft/adapter_config.json maple-pad-sft/adapter_model.safetensors
ls maple-sdp-sft/adapter_config.json maple-sdp-sft/adapter_model.safetensors
```

Set the model and adapters:

```bash
export BASE_MODEL=/root/autodl-tmp/Qwen3-8B-Instruct
export PAD_ADAPTER=./maple-pad-sft
export SDP_ADAPTER=./maple-sdp-sft
export SRC=data/raw/train.jsonl
export K=1
export DEVICE=cuda:0
export MAX_NEW_PAD=1536
export BATCH_SIZE=32
export FLUSH_EVERY=50
```

Preferred recovery-safe run mode:

Run the four Joint steps manually instead of launching one long `train_joint.sh`
job. This creates clear restart points: if a command fails, fix the issue and
rerun only that command or the next incomplete command. Do not rerun completed
training phases unless their output folder is missing or invalid.

Important: the commands below assume the active `joint_alignment_data.py` is the
updated root structural-Jaccard version. For `phase-b`, it supports
`--pad-adapter`, `--batch-size`, and `--flush-every`, but does not support
`--sdp-adapter`. Phase A PAD DPO therefore does not load the SDP adapter during
data generation. Batched generation does not truncate prompts; if batch size 32
causes OOM, lower `BATCH_SIZE` to 16 or 8 and rerun the same stage.

### Stage 2.1: Generate Phase A PAD DPO Data

```bash
python joint_alignment_data.py phase-b \
  --src "$SRC" \
  --base-model "$BASE_MODEL" \
  --device "$DEVICE" \
  --pad-adapter "$PAD_ADAPTER" \
  --pairs-per-query 1 \
  --max-new-pad "$MAX_NEW_PAD" \
  --batch-size "$BATCH_SIZE" \
  --flush-every "$FLUSH_EVERY" \
  --out data/joint/iter1/phase_a_pad_dpo \
  2>&1 | tee joint_phase_a_generate_515.log
```

Check:

```bash
wc -l data/joint/iter1/phase_a_pad_dpo/train.jsonl
head -n 1 data/joint/iter1/phase_a_pad_dpo/train.jsonl
```

Continue only if the row count is reasonable. If this produces a tiny accidental
subset, stop and inspect skipped-row logs before training DPO.

### Stage 2.2: Train Phase A PAD DPO

```bash
cp configs/joint_phase_b_autotrain.yml configs/_phase_a_pad_dpo_iter1.yml

sed -i.bak \
  -e "s|^base_model:.*|base_model: $BASE_MODEL|" \
  -e "s|^project_name:.*|project_name: maple-pad-dpo-iter1|" \
  -e "s|^  path:.*|  path: data/joint/iter1/phase_a_pad_dpo|" \
  -e "s|^  peft_model:.*|  peft_model: $PAD_ADAPTER|" \
  configs/_phase_a_pad_dpo_iter1.yml

rm -f configs/_phase_a_pad_dpo_iter1.yml.bak

python train_dpo_lora.py --config configs/_phase_a_pad_dpo_iter1.yml \
  2>&1 | tee joint_phase_a_pad_dpo_train_515.log
```

Check:

```bash
ls maple-pad-dpo-iter1/adapter_config.json maple-pad-dpo-iter1/adapter_model.safetensors
```

If this check passes, the PAD side has a safe boundary checkpoint:
`maple-pad-dpo-iter1`.

### Stage 2.3: Generate Phase B SDP On-Policy SFT Data

Use the refined PAD adapter from Stage 2.2:

```bash
python joint_alignment_data.py phase-a \
  --src "$SRC" \
  --base-model "$BASE_MODEL" \
  --device "$DEVICE" \
  --pad-adapter ./maple-pad-dpo-iter1 \
  --samples-per-query 1 \
  --max-new-pad "$MAX_NEW_PAD" \
  --batch-size "$BATCH_SIZE" \
  --flush-every "$FLUSH_EVERY" \
  --out data/joint/iter1/phase_b_sdp_sft \
  2>&1 | tee joint_phase_b_generate_515.log
```

Check:

```bash
wc -l data/joint/iter1/phase_b_sdp_sft/train.jsonl
head -n 1 data/joint/iter1/phase_b_sdp_sft/train.jsonl
```

### Stage 2.4: Train Phase B SDP On-Policy SFT

```bash
cp configs/joint_phase_a_autotrain.yml configs/_phase_b_sdp_sft_iter1.yml

sed -i.bak \
  -e "s|^base_model:.*|base_model: $BASE_MODEL|" \
  -e "s|^project_name:.*|project_name: maple-sdp-onpolicy-iter1|" \
  -e "s|^  path:.*|  path: data/joint/iter1/phase_b_sdp_sft|" \
  -e "s|^  peft_model:.*|  peft_model: $SDP_ADAPTER|" \
  configs/_phase_b_sdp_sft_iter1.yml

rm -f configs/_phase_b_sdp_sft_iter1.yml.bak

python train_sft_lora.py --config configs/_phase_b_sdp_sft_iter1.yml \
  2>&1 | tee joint_phase_b_sdp_sft_train_515.log
```

Check:

```bash
ls maple-sdp-onpolicy-iter1/adapter_config.json maple-sdp-onpolicy-iter1/adapter_model.safetensors
```

Expected output folders:

```text
maple-pad-dpo-iter1/
maple-sdp-onpolicy-iter1/
```

Optional one-shot mode:

Only use `./train_joint.sh` after confirming it matches the four commands above
and uses `train_dpo_lora.py` / `train_sft_lora.py`, not AutoTrain. Manual mode is
preferred because it is easier to resume after data-generation or training
errors.

Final LoRA verification:

```bash
ls maple-pad-dpo-iter1
ls maple-sdp-onpolicy-iter1
```

Good signs:

```text
adapter_config.json
adapter_model.safetensors
```

Danger sign:

```text
model.safetensors
```

If you see a full `model.safetensors`, stop and re-check that the training
script loaded PEFT/LoRA instead of full fine-tuning.

## Stage 2 Checkpoint Safety

Stage 2 uses `epochs: 1` and `save_strategy: epoch`, so the two main durable
adapter checkpoints are:

```text
maple-pad-dpo-iter1/
maple-sdp-onpolicy-iter1/
```

That is enough boundary protection for the four-step manual workflow:

- If Stage 2.1 fails, rerun Stage 2.1 only.
- If Stage 2.2 fails before `maple-pad-dpo-iter1` is valid, rerun Stage 2.2.
- If Stage 2.3 fails, keep `maple-pad-dpo-iter1` and rerun Stage 2.3 only.
- If Stage 2.4 fails before `maple-sdp-onpolicy-iter1` is valid, rerun Stage 2.4.

During Stage 2.1 and Stage 2.3 data generation, `joint_alignment_data.py`
prints progress after every generation batch and writes a partial checkpoint to:

```text
data/joint/iter1/phase_a_pad_dpo/train.jsonl.tmp
data/joint/iter1/phase_b_sdp_sft/train.jsonl.tmp
```

Those `.tmp` files are for inspection/recovery if generation crashes. A
successful run writes the final `train.jsonl` and removes the `.tmp` file.

This is not strong mid-epoch resume protection. If the DPO or SDP-on-policy
training process dies halfway through its single epoch, the practical recovery
path is to rerun that one training command. If mid-epoch recovery becomes
necessary, create a copied config with `save_strategy: steps` and a reasonable
`save_steps` before launching the run, and document the deviation.

## Monitor

During Stage 1:

- Every 1 step: training logs.
- Every 50 steps: dev eval.
- Every 50 steps: checkpoint save.
- Stop only if dev NLL rises more than 10% for 2 consecutive evals or NaN/Inf appears.

During Stage 2:

- Watch Phase A DPO data generation: many skipped rows means PAD generation is poor, JSON is unparseable, or `MAX_NEW_PAD` is too low for complete scaffold output.
- `joint_alignment_data.py` uses `max_new_pad=1536` by default; for a faster smoke test use `MAX_NEW_PAD=1024`.
- The current root `joint_alignment_data.py` does not use the SDP log-likelihood scorer; Phase A PAD DPO rows include `delta_struct`.
- Phase A DPO data should contain only `prompt`, `chosen`, `rejected`, plus ids.
- Phase B SFT data should contain `messages`.

Check generated Joint data:

```bash
head -n 1 data/joint/iter1/phase_a_pad_dpo/train.jsonl
head -n 1 data/joint/iter1/phase_b_sdp_sft/train.jsonl
```

## Metadata Per Checkpoint

The experiment plan requires a `metadata.json` beside each saved checkpoint.
This repository now has `checkpoint_metadata.py` for the code-generated parts:

- `data_split_sha`: computed automatically from `maple_split_v1.json`
- `git_sha`: computed automatically if `git` is available
- `hparams`: extracted from the config `params:` block
- `ckpt_path`: recorded from your argument

You still need to provide the observed metrics from logs/evaluators:

- `step`
- `epoch`
- `wall_clock_h`
- `dev_nll`
- `dev_sv`
- `dev_ar`
- `dev_dc_mean`
- `dev_atr_mean`

Example for PAD:

```bash
python checkpoint_metadata.py \
  --stage stage1a_PAD \
  --ckpt-path maple-pad-sft/checkpoint-350 \
  --step 350 \
  --epoch 2.3 \
  --wall-clock-h 15.2 \
  --dev-nll 1.42 \
  --dev-sv 0.91 \
  --dev-ar 0.84 \
  --dev-dc-mean 0.78 \
  --dev-atr-mean 0.66 \
  --config configs/pad_autotrain.yml
```

Example for SDP:

```bash
python checkpoint_metadata.py \
  --stage stage1b_SDP \
  --ckpt-path maple-sdp-sft/checkpoint-300 \
  --step 300 \
  --epoch 2.0 \
  --wall-clock-h 14.0 \
  --dev-nll 1.50 \
  --dev-sv 0.90 \
  --dev-ar 0.83 \
  --dev-dc-mean 0.76 \
  --dev-atr-mean 0.64 \
  --config configs/sdp_autotrain.yml
```

Example for Joint:

```bash
python checkpoint_metadata.py \
  --stage stage2_JA \
  --ckpt-path maple-sdp-onpolicy-iter1 \
  --step 0 \
  --epoch 1.0 \
  --wall-clock-h 7.0 \
  --dev-nll 1.40 \
  --dev-sv 0.91 \
  --dev-ar 0.84 \
  --dev-dc-mean 0.78 \
  --dev-atr-mean 0.66 \
  --config configs/_phase_b_sdp_sft_iter1.yml
```

Do not invent these numbers. If an optional metric is not supplied, the script records it as `null`; rerun the command or edit the metadata after the metric is measured.

## Evaluation Matrix

Code-supported now:

- `Dev NLL every 50 steps`: `train_sft_lora.py` runs eval every 50 steps through TRL. Use the logged eval loss as the dev NLL proxy unless you add a separate target-token-only NLL evaluator.
- `Dev SV / AR`: use `eval_plan_structure.py` after you generate Dev predictions.
- `data_split_sha`: use `checkpoint_metadata.py`.

Still human/manual or separate tooling:

- `Dev DC / ATR`: no implementation in this folder yet. If another teammate has the rule-based metric script, run it and pass the values into `checkpoint_metadata.py`.
- `Dev GED-sim`: no implementation in this folder yet; run only at Stage 2 Gate if/when the GED evaluator is available.
- `LLM judge Pers./Feas./Sati.`: manual/API evaluation only at Stage 2 end, not inside training.
- `pass@1 / pass@8`: no script in this folder yet; run at Stage 2 end if the Gate diagnostic script is available.

SV/AR example, assuming you have generated Dev predictions in `runs/pad_dev_generations.jsonl` with an `output` column:

```bash
python eval_plan_structure.py runs/pad_dev_generations.jsonl --column output --mode pad --out runs/pad_dev_sv_ar.json
```

For a full plan output:

```bash
python eval_plan_structure.py runs/full_dev_generations.jsonl --column output --mode full --out runs/full_dev_sv_ar.json
```

## After Stage 2

Before GRPO handoff / Gate:

- Run rule-based checks on Dev 100 if available.
- Compute pass@1 and pass@8 on Dev 100.
- Save the hard-gate report to `runs/sft_gate_report.json`.
- Run one LLM-judge sanity check only at Stage 2 end, not inside training.
- Do not touch Test yet.
- Do not submit a GRPO job until this gate passes.

## Known Risks

- Server dependency conflict: `autotrain-advanced==0.8.36` pins `transformers==4.48.0`, which cannot load Qwen3. Use the TRL scripts instead.
- Hugging Face may be blocked; use `HF_ENDPOINT` or local model path.
- Always verify adapter output.
- `train_dpo_lora.py` parses the current DPO `prompt` / `chosen` / `rejected` rows directly. If you change `joint_alignment_data.py` output format, update `train_dpo_lora.py` at the same time.
- If Joint DPO produces too few rows, do not train on a tiny accidental subset without recording it.

## SFT And Joint Caveats From The Plan

Code-followed items:

- LoRA loading with adapter embeddings: `joint_alignment_data.py` resizes the base model embeddings if the adapter tokenizer length differs before `PeftModel.from_pretrained(...)`.
- Stage 1b SDP uses gold scaffold: `build_sft_data.py` builds SDP inputs from gold `agents` and `subtasks`, not PAD outputs.
- No LLM judge is called by the SFT or Joint training scripts.
- Gradient checkpointing is enabled in configs via `disable_gradient_checkpointing: false`.
- Stage 2 order in the recommended manual commands follows the plan:
  - Phase A: PAD DPO
  - Phase B: SDP on-policy SFT
- Stage 2 Phase A DPO uses `dpo_beta: 0.1`.
- Stage 2 Phase A and Phase B both use `lr: 5.0e-5`.
- Stage 2 uses Train only through `SRC=data/raw/train.jsonl`.
- DPO `delta_struct` is written for diagnostics; `delta_ll` is not used by the current root structural-Jaccard script.
- DPO `chosen` and `rejected` are written as Python-list strings and parsed by `train_dpo_lora.py`.
- Stage 2 Phase A currently uses scaffold-vs-gold structural Jaccard as the PAD DPO preference proxy. This is a documented deviation from the full generated-plan reward selection in the experiment text.

Manual / not fully implemented items:

- Dev NLL alone is not enough. You still need generated Dev samples and SV/AR checks after each epoch.
- DC/ATR are not implemented in this folder.
- Stage 2 Gate diagnostics are not implemented in this folder: `pass@1`, `pass@8`, GED-sim, and one LLM-judge sanity check must be run with separate tooling.
- The experiment text says Phase A should choose scaffold pairs by generating plans and applying rule-based `R_struct + R_pers + R_ped`. That full reward stack is not implemented here. Current root `joint_alignment_data.py` uses scaffold-vs-gold structural Jaccard as the PAD DPO preference proxy, so report this explicitly.
- The pass@1/pass@8 diagnostic gate is a hard gate before GRPO. This folder does not currently implement `runs/sft_gate_report.json`; use separate tooling and do not touch Test.
