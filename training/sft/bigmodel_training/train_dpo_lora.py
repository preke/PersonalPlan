"""Train MAP-PPL Stage-2 PAD DPO with TRL DPOTrainer.
Uses precompute_ref_log_probs=True with ref_model=None so that TRL
computes reference log-probs from the policy model at init time
(before any training, policy == ref). Only ONE model on GPU at any time.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any
import re

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_think_and_fences(text: str) -> str:
    s = text.strip()

    # Remove a leading <think>...</think> block if present
    s = re.sub(r"^\s*<think>.*?</think>\s*", "", s, flags=re.DOTALL)

    # Remove markdown code fences like ```json ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)

    return s.strip()


def message_payload_to_text(value: Any) -> str:
    if value is None:
        return ""

    # Already plain text
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""

        # Try JSON parse first
        try:
            parsed = json.loads(s)
            return message_payload_to_text(parsed)
        except Exception:
            pass

        # Then Python literal parse
        try:
            parsed = ast.literal_eval(s)
            return message_payload_to_text(parsed)
        except Exception:
            # Not serialized structure; treat as raw completion text
            return _strip_think_and_fences(s)

    # Dict payload
    if isinstance(value, dict):
        if "content" in value:
            return message_payload_to_text(value["content"])
        if "messages" in value:
            return message_payload_to_text(value["messages"])
        return _strip_think_and_fences(json.dumps(value, ensure_ascii=False))

    # List payload (chat messages or blocks)
    if isinstance(value, list):
        if len(value) == 0:
            return ""

        # Chat messages [{"role": "...", "content": "..."}]
        if all(isinstance(x, dict) for x in value):
            assistant_chunks = []
            fallback_chunks = []
            for m in value:
                c = m.get("content", "")
                if isinstance(c, list):
                    # content blocks format
                    c = "".join(
                        str(b.get("text", "")) for b in c if isinstance(b, dict)
                    )
                c = str(c)
                if not c.strip():
                    continue
                fallback_chunks.append(c)
                if m.get("role") == "assistant":
                    assistant_chunks.append(c)

            joined = "\n".join(assistant_chunks if assistant_chunks else fallback_chunks)
            return _strip_think_and_fences(joined)

        # Generic list
        return _strip_think_and_fences("\n".join(str(x) for x in value))

    # Fallback
    return _strip_think_and_fences(str(value))


def build_dataset(data_dir: Path, split_name: str) -> Dataset:
    rows = []
    for row in load_jsonl(data_dir / f"{split_name}.jsonl"):
        rows.append(
            {
                "prompt": row["prompt"],
                "chosen": message_payload_to_text(row["chosen"]),
                "rejected": message_payload_to_text(row["rejected"]),
                "question_id": row.get("question_id"),
                "profile_index": row.get("profile_index"),
            }
        )
    return Dataset.from_list(rows)


def optimizer_name(name: str) -> str:
    if name == "paged_adamw_8bit":
        return "paged_adamw_8bit"
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--optim", default=None)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config).resolve())
    data_cfg = cfg["data"]
    params = cfg["params"]
    base_model = args.base_model or os.environ.get("BASE_MODEL") or cfg["base_model"]
    project_name = args.project or os.environ.get("PROJECT") or cfg["project_name"]
    adapter_path = params.get("peft_model")
    data_dir = Path(data_cfg["path"]).resolve()
    max_length = int(params.get("block_size", params.get("model_max_length", 4096)))

    # --- Build datasets ---
    train_dataset = build_dataset(data_dir, data_cfg.get("train_split", "train"))
    eval_dataset = None
    valid_split = data_cfg.get("valid_split")
    if valid_split and (data_dir / f"{valid_split}.jsonl").exists():
        eval_dataset = build_dataset(data_dir, valid_split)

    # --- Load policy model (base + SFT LoRA) ---
    tokenizer_source = adapter_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # --- LoRA config (only needed if no existing adapter) ---
    peft_config = None
    if not adapter_path:
        peft_config = LoraConfig(
            r=int(params["lora_r"]),
            lora_alpha=int(params["lora_alpha"]),
            lora_dropout=float(params.get("lora_dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=params.get("target_modules", "all-linear"),
        )

    # --- DPO Config ---
    optim = args.optim or params.get("optimizer", "paged_adamw_8bit")
    dpo_args = DPOConfig(
        output_dir=project_name,
        num_train_epochs=float(params["epochs"]),
        per_device_train_batch_size=int(params["batch_size"]),
        per_device_eval_batch_size=int(params["batch_size"]),
        gradient_accumulation_steps=int(params["gradient_accumulation"]),
        learning_rate=float(params["lr"]),
        lr_scheduler_type=params.get("scheduler", "cosine"),
        warmup_ratio=float(params.get("warmup_ratio", 0.0)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        max_grad_norm=float(params.get("max_grad_norm", 1.0)),
        bf16=str(params.get("mixed_precision", "")).lower() == "bf16",
        gradient_checkpointing=not bool(params.get("disable_gradient_checkpointing", False)),
        logging_steps=int(params.get("logging_steps", 5)),
        eval_strategy=params.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=int(params.get("eval_steps", 50)) if eval_dataset else None,
        save_strategy=params.get("save_strategy", "epoch"),
        save_total_limit=int(params.get("save_total_limit", 2)),
        optim=optimizer_name(optim),
        report_to=[cfg.get("log", "tensorboard")],
        seed=int(params.get("seed", 42)),
        beta=float(params.get("dpo_beta", 0.1)),
        max_length=max_length,
        remove_unused_columns=True,
        precompute_ref_log_probs=True,
    )

    # --- Train ---
    # ref_model=None → TRL uses the policy model at init to compute ref logps.
    # At init time policy == SFT checkpoint (correct reference).
    # Only ONE model ever on GPU.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(project_name)
    tokenizer.save_pretrained(project_name)


if __name__ == "__main__":
    main()