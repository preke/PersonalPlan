"""Train MAPLE Stage-1 LoRA adapters with TRL SFTTrainer.

AutoTrain 0.8.36 pins transformers==4.48.0, which cannot load Qwen3
(`model_type: qwen3`). This script keeps the existing YAML configs and data
layout, but bypasses the AutoTrain CLI so the server can use a newer
Transformers + TRL stack.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


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


def split_messages(row: dict[str, Any]) -> dict[str, Any]:
    """Convert system + user + assistant messages into prompt/completion."""
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Each SFT row must end with an assistant message")
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
        "question_id": row.get("question_id"),
        "profile_index": row.get("profile_index"),
    }


def build_dataset(data_dir: Path, split_name: str) -> Dataset:
    rows = load_jsonl(data_dir / f"{split_name}.jsonl")
    return Dataset.from_list([split_messages(row) for row in rows])


def optimizer_name(name: str) -> str:
    if name == "paged_adamw_8bit":
        return "paged_adamw_8bit"
    return name


def load_base_model(base_model: str, adapter_path: str | None = None):
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
    return tokenizer, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Existing PAD/SDP YAML config")
    ap.add_argument("--base-model", default=None, help="Override config base_model")
    ap.add_argument("--project", default=None, help="Override config project_name")
    ap.add_argument("--optim", default=None, help="Override optimizer")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config).resolve())
    data_cfg = cfg["data"]
    params = cfg["params"]

    base_model = args.base_model or os.environ.get("BASE_MODEL") or cfg["base_model"]
    project_name = args.project or os.environ.get("PROJECT") or cfg["project_name"]
    peft_model = params.get("peft_model")
    data_dir = Path(data_cfg["path"]).resolve()

    train_dataset = build_dataset(data_dir, data_cfg.get("train_split", "train"))
    eval_dataset = None
    valid_split = data_cfg.get("valid_split")
    if valid_split and (data_dir / f"{valid_split}.jsonl").exists():
        eval_dataset = build_dataset(data_dir, valid_split)

    tokenizer, model = load_base_model(base_model, peft_model)

    lora_config = None
    if not peft_model:
        lora_config = LoraConfig(
            r=int(params["lora_r"]),
            lora_alpha=int(params["lora_alpha"]),
            lora_dropout=float(params.get("lora_dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=params.get("target_modules", "all-linear"),
        )

    optim = args.optim or params.get("optimizer", "paged_adamw_8bit")
    sft_args = SFTConfig(
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
        logging_steps=int(params.get("logging_steps", 1)),
        eval_strategy=params.get("eval_strategy", "steps") if eval_dataset else "no",
        eval_steps=int(params.get("eval_steps", 50)) if eval_dataset else None,
        save_strategy=params.get("save_strategy", "steps"),
        save_steps=int(params.get("save_steps", 50)),
        save_total_limit=int(params.get("save_total_limit", 3)),
        optim=optimizer_name(optim),
        report_to=[cfg.get("log", "tensorboard")],
        seed=int(params.get("seed", 42)),
        max_length=int(params.get("block_size", params.get("model_max_length", 3072))),
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(project_name)
    tokenizer.save_pretrained(project_name)


if __name__ == "__main__":
    main()
