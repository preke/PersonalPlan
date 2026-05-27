"""GRPO training for the MAP-PPL planner — LoRA + TRL + (optionally) vLLM.

Paper §4.2 Reinforcement Optimization with Verifiable and Counterfactual
Rewards. v1 implementation (see README for the staged plan).

Required components in this file:
  1. load base model + SFT LoRA initializer (the GRPO target)
  2. construct π_SFT reference for the KL anchor (frozen copy)
  3. build a HuggingFace Dataset from the prompts produced by
     build_grpo_prompts.py
  4. compose the reward function (rewards.compose_reward)
  5. run trl.GRPOTrainer

Things v1 deliberately does NOT do (see rewards.py + README):
  - segment-wise credit assignment
  - online (in-step) counterfactual sampling — uses precomputed cache
  - LLM judge ensemble
  - adversarial trajectory injection
  - adaptive β
All of those are toggles or subclass extensions to add later.

CLI:
  python grpo_train.py --config configs/grpo.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rewards import RewardConfig, compose_reward                       # noqa: E402
from plan_utils import mine_subtask_precedence                         # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_gold_plan(gold_str: str | dict) -> dict | None:
    if isinstance(gold_str, dict):
        return gold_str
    try:
        return json.loads(gold_str)
    except Exception:
        return None


def parse_cf_cache(path: str | None) -> dict[str, dict]:
    """Counterfactual cache format: jsonl with rows
        {"question_id": ..., "cf_plan": <plan dict or json string>}
    Returns dict keyed by question_id."""
    if not path:
        return {}
    cache: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r["question_id"])
            cf = r["cf_plan"]
            if isinstance(cf, str):
                cf = json.loads(cf)
            cache[qid] = cf
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs" / "grpo.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)

    # --- heavy imports deferred so --help works without GPU stack ---
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    base_model = cfg["base_model"]
    sft_adapter = cfg.get("sft_adapter")               # peft adapter to start from (optional)
    output_dir = cfg["output_dir"]

    # --- tokenizer ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- dataset ------------------------------------------------------
    data_cfg = cfg["data"]
    raw = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_path"],
            "valid": data_cfg["valid_path"],
        },
    )

    # TRL expects "prompt" column. We have it from build_grpo_prompts.py
    # (in chat-messages form). The trainer will apply chat template at
    # rollout time when we pass `processing_class=tokenizer`.

    # Optional: mine subtask precedence pairs from the gold plans for
    # R_ped,hard. This is a one-time pass over the training data.
    rcfg = RewardConfig(**cfg["reward"])
    precedence_pairs: set[tuple[str, str]] = set()
    # Mine subtask precedence pairs whenever R_ped is enabled (lite or full both
    # depend on it).
    if rcfg.ped_mode in ("lite", "full"):
        gold_rows = [json.loads(r["gold_plan"]) for r in raw["train"]]
        precedence_pairs = mine_subtask_precedence(
            [{"plan": {"output": p}} for p in gold_rows]
        )
        print(f"Mined {len(precedence_pairs)} subtask precedence pairs.")

    # Optional: load precomputed counterfactual cache for R_pers
    # (only needed when pers_mode == "counterfactual"; "lite" needs no cache).
    cf_cache: dict[str, dict] = {}
    if rcfg.pers_mode == "counterfactual":
        cf_cache_path = cfg.get("counterfactual_cache")
        cf_cache = parse_cf_cache(cf_cache_path)
        print(f"Loaded {len(cf_cache)} counterfactual rollouts from cache.")
        if not cf_cache:
            print(
                "  WARNING: pers_mode=counterfactual but cache empty. "
                "R_pers will return 0 for all trajectories. "
                "Run build_counterfactual_cache.py first, "
                "or switch pers_mode to 'lite'."
            )

    # The TRL reward_func receives dataset columns as kwargs (per-completion).
    # Our `compose_reward` returns the right callable for that signature.
    reward_fn = compose_reward(
        rcfg,
        precedence_pairs=precedence_pairs or None,
        cf_cache=cf_cache or None,
        judges=None,    # v1: LLM judges off
    )

    # Adapter: TRL reward funcs receive dataset columns as strings. Wrap so
    # rewards.compose_reward gets parsed dicts.
    def reward_wrapper(completions, **kwargs):
        if "gold_plan" in kwargs:
            kwargs["gold_plan"] = [parse_gold_plan(g) for g in kwargs["gold_plan"]]
        if "learner_profile" in kwargs:
            parsed = []
            for p in kwargs["learner_profile"]:
                if isinstance(p, dict):
                    parsed.append(p)
                else:
                    try:
                        parsed.append(json.loads(p))
                    except Exception:
                        parsed.append(None)
            kwargs["learner_profile"] = parsed
        return reward_fn(completions, **kwargs)

    # --- LoRA config --------------------------------------------------
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- GRPOConfig ---------------------------------------------------
    train_cfg = cfg["training"]
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("grad_accum", 8),
        learning_rate=train_cfg.get("lr", 5e-6),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),

        # GRPO-specific
        num_generations=train_cfg.get("num_generations", 8),  # G in paper
        max_prompt_length=train_cfg.get("max_prompt_length", 2048),
        max_completion_length=train_cfg.get("max_completion_length", 3500),
        beta=train_cfg.get("beta", 0.04),                    # KL coefficient
        epsilon=train_cfg.get("epsilon", 0.2),               # PPO clip
        temperature=train_cfg.get("temperature", 0.9),
        top_p=train_cfg.get("top_p", 0.95),

        # vLLM for rollouts (recommended)
        use_vllm=train_cfg.get("use_vllm", True),
        vllm_device=train_cfg.get("vllm_device", "auto"),
        vllm_gpu_memory_utilization=train_cfg.get("vllm_gpu_mem", 0.5),

        # General training knobs
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy="no",         # we monitor proxy-vs-gold separately
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=True,
        report_to=cfg.get("report_to", ["tensorboard"]),
        seed=cfg.get("seed", 42),
    )

    # --- Build trainer ------------------------------------------------
    # If sft_adapter is given, the user is expected to:
    #   1. Load base_model
    #   2. Apply the SFT LoRA adapter via PeftModel.from_pretrained
    #   3. Merge or keep as-is (TRL will train *the LoRA params*)
    # The simplest pattern is to use the merged checkpoint as base_model.
    # If you want to *continue training* the existing LoRA, pass it as
    # `model` (a PeftModel) and set `peft_config=None` (TRL will reuse
    # the already-attached adapter). For the v1 default we initialize a
    # fresh LoRA on top of base_model — change if you've trained an SFT
    # adapter and want to start from it.

    trainer = GRPOTrainer(
        model=base_model,
        reward_funcs=[reward_wrapper],
        args=grpo_config,
        train_dataset=raw["train"],
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(
        f"\nStarting GRPO training\n"
        f"  base model       : {base_model}\n"
        f"  sft adapter init : {sft_adapter or '(none — training fresh LoRA on base)'}\n"
        f"  num_generations  : {grpo_config.num_generations}\n"
        f"  beta (KL coef)   : {grpo_config.beta}\n"
        f"  rewards enabled  : "
        f"struct={rcfg.enable_struct} hard={rcfg.enable_hard} "
        f"pers_mode={rcfg.pers_mode} ped_mode={rcfg.ped_mode}\n"
        f"  train rows       : {len(raw['train'])}\n"
        f"  output dir       : {output_dir}\n"
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    print(f"Done. Final adapter saved to {os.path.join(output_dir, 'final')}")


if __name__ == "__main__":
    main()
