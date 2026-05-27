"""Generate Stage 3 (Joint Alignment) training data.

Stage 3 needs model rollouts, so it cannot be a pure offline dataset
construction the way Stages 1-2 are. This script uses the already-trained
PAD and SDP adapters to GENERATE the data for two AutoTrain runs:

  Phase A — on-policy SDP SFT
    Input:  base model + frozen PAD adapter
    For each training record, sample K scaffolds from PAD (on-policy).
    Pair each sampled scaffold with the GOLD (S, O) target.
    Output: data/joint/phase_a/{train,valid}.jsonl
            → feed to AutoTrain SFT to update the SDP adapter
              (configs/joint_phase_a_autotrain.yml)

  Phase B — PAD DPO with scaffold-level structural preferences
    Input:  base model + frozen PAD adapter (to sample). SDP is NOT loaded.
    For each training record, sample 2 PAD scaffolds and score each by
    structural similarity to the gold scaffold (Jaccard over agent roles
    and subtask names). Higher similarity → chosen, lower → rejected.
    Output: data/joint/phase_b/{train,valid}.jsonl  (prompt/chosen/rejected)
            → feed to AutoTrain DPO to update the PAD adapter
              (configs/joint_phase_b_autotrain.yml)

Phases A and B alternate (paper §4: K ∈ {1, 2} outer iterations).

Naming note: CLI `phase-a` = paper §3.2 "Phase B" (修 SDP via on-policy SFT);
CLI `phase-b` = paper §3.2 "Phase A" (修 PAD via DPO). The mismatch is a
historical artifact; train_joint.sh runs them in CLI order (a then b),
which is reversed from the paper's stated order. Both orders converge for
K ≥ 2; with K=1 (the default) the difference is small but real.

CLI:

  # Phase A — on-policy SDP data
  python joint_alignment_data.py phase-a \
      --src multi_agent_dataset_filtered_qap_v3.jsonl \
      --base-model /root/autodl-tmp/Qwen-32B-Instruct \
      --pad-adapter ./maple-pad-sft \
      --samples-per-query 1 \
      --out data/joint/phase_a

  # Phase B — DPO pairs for PAD (scaffold-level R_struct, no SDP needed)
  python joint_alignment_data.py phase-b \
      --src multi_agent_dataset_filtered_qap_v3.jsonl \
      --base-model /root/autodl-tmp/Qwen-32B-Instruct \
      --pad-adapter ./maple-pad-sft \
      --pairs-per-query 1 \
      --out data/joint/phase_b

Requires: torch, transformers, peft, accelerate (and a GPU for realistic
throughput).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Heavy imports are deferred so `--help` works without a torch install.

from collections import Counter

from prompts import PAD_SYSTEM, PAD_USER_TEMPLATE, SDP_SYSTEM, SDP_USER_TEMPLATE
from build_sft_data import (
    extract_learner,
    pad_target,
    sdp_target,
    load_jsonl,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent


# ---------- helpers ----------------------------------------------------------

def _try_parse_json(text: str):
    """Try to extract a single JSON object from a model's free-form output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def render_chat(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_base_and_adapter(base_model, adapter_path, device, dtype):
    """Load the base model + one PEFT adapter."""
    import torch  # noqa
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    try:
        tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    target_vocab_size = _adapter_vocab_size(adapter_path) or len(tok)
    current_vocab_size = base.get_input_embeddings().weight.shape[0]
    if current_vocab_size != target_vocab_size:
        print(
            f"[load] resizing base token embeddings "
            f"{current_vocab_size} -> {target_vocab_size} before loading adapter",
            flush=True,
        )
        base.resize_token_embeddings(target_vocab_size)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tok, model


def _adapter_vocab_size(adapter_path):
    """Return adapter-saved embedding vocab size without loading all weights."""
    adapter_dir = Path(adapter_path)
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(("embed_tokens.weight", "lm_head.weight")):
                        return int(f.get_slice(key).get_shape()[0])
        except Exception as exc:
            print(f"[load] warning: could not inspect adapter vocab size: {exc}", flush=True)
    return None


def sample_scaffold(model, tok, query, desc, skills, *, temperature, top_p, max_new):
    """Sample one PAD scaffold (T, A). Returns (parsed_or_None, raw_text)."""
    import torch  # noqa

    user_text = PAD_USER_TEMPLATE.format(
        query=query,
        self_description=desc if desc else "(no self-description provided)",
        skills=json.dumps(skills, ensure_ascii=False),
    )
    messages = [
        {"role": "system", "content": PAD_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    prompt = render_chat(tok, messages)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    return _try_parse_json(gen), gen


def batch_sample_scaffolds(
    model,
    tok,
    prompts_data,
    *,
    temperature,
    top_p,
    max_new,
    batch_size,
    phase_label,
):
    """Batch sample PAD scaffolds without truncating prompts."""
    import torch  # noqa

    if not prompts_data:
        return []

    batch_size = max(1, int(batch_size))
    old_padding_side = tok.padding_side
    tok.padding_side = "left"
    results = []
    total = len(prompts_data)
    try:
        for start in range(0, total, batch_size):
            batch = prompts_data[start : start + batch_size]
            texts = []
            for query, desc, skills in batch:
                user_text = PAD_USER_TEMPLATE.format(
                    query=query,
                    self_description=desc if desc else "(no self-description provided)",
                    skills=json.dumps(skills, ensure_ascii=False),
                )
                messages = [
                    {"role": "system", "content": PAD_SYSTEM},
                    {"role": "user", "content": user_text},
                ]
                texts.append(render_chat(tok, messages))

            inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tok.pad_token_id,
                )

            input_len = inputs.input_ids.shape[1]
            for j in range(len(batch)):
                gen = tok.decode(out[j][input_len:], skip_special_tokens=True)
                results.append((_try_parse_json(gen), gen))

            done = min(start + len(batch), total)
            print(
                f"  [{phase_label}/generate] {done}/{total} scaffolds generated "
                f"(batch_size={batch_size})",
                flush=True,
            )
    finally:
        tok.padding_side = old_padding_side
    return results


def _multiset_jaccard(a, b) -> float:
    """Weighted Jaccard over multisets. Mirrors GRPO/plan_utils.py."""
    ca, cb = Counter(a), Counter(b)
    if not ca and not cb:
        return 1.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


def _jaccard(a, b) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def scaffold_struct_score(scaffold, gold_scaffold) -> float:
    """Scaffold-only structural similarity to gold, in [0, 1].

    Narrowed reward used in paper §3.2 Phase A (DPO on PAD): we want a
    signal that *only* reflects PAD↔SDP coordination quality at the
    scaffold level, leaving R_pers / R_ped for GRPO. We average two
    Jaccard axes:
      - agent_role multiset
      - subtask name set
    Both axes mirror the first two components of structural_similarity
    in GRPO/plan_utils.py, so Phase A and the GRPO R_struct share a
    consistent geometry. Returns 0.0 if `scaffold` could not be parsed.
    """
    if not isinstance(scaffold, dict):
        return 0.0
    p_agents = [a.get("agent_role", "") for a in (scaffold.get("agents") or [])]
    g_agents = [a.get("agent_role", "") for a in (gold_scaffold.get("agents") or [])]
    p_subs = {s.get("name", "") for s in (scaffold.get("subtasks") or [])}
    g_subs = {s.get("name", "") for s in (gold_scaffold.get("subtasks") or [])}
    return 0.5 * _multiset_jaccard(p_agents, g_agents) + 0.5 * _jaccard(p_subs, g_subs)


# ---------- phase A ----------------------------------------------------------

def cmd_phase_a(args):
    import torch  # noqa

    tok, pad_model = load_base_and_adapter(
        args.base_model, args.pad_adapter, args.device, torch.bfloat16
    )
    src_rows = load_jsonl(Path(args.src).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "train.jsonl.tmp"
    out_rows = []
    skipped = 0
    print(
        f"  [phase-a] {len(src_rows)} source rows, "
        f"samples_per_query={args.samples_per_query}, batch_size={args.batch_size}",
        flush=True,
    )

    flush_every = max(1, int(args.flush_every))
    for chunk_start in range(0, len(src_rows), flush_every):
        chunk = src_rows[chunk_start : chunk_start + flush_every]
        jobs = []
        prompts_data = []
        for offset, r in enumerate(chunk, chunk_start + 1):
            inp = r["plan"]["input"]
            query = inp["query"]
            desc, skills = extract_learner(inp["learner"])
            gold_sdp = sdp_target(r["plan"]["output"])
            for k in range(args.samples_per_query):
                jobs.append((offset, r, k, query, desc, skills, gold_sdp))
                prompts_data.append((query, desc, skills))

        results = batch_sample_scaffolds(
            pad_model,
            tok,
            prompts_data,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new=args.max_new_pad,
            batch_size=args.batch_size,
            phase_label="phase-a",
        )

        for job, (scaffold, _) in zip(jobs, results):
            i, r, k, query, desc, skills, gold_sdp = job
            if scaffold is None or "agents" not in scaffold or "subtasks" not in scaffold:
                skipped += 1
                continue
            user_text = SDP_USER_TEMPLATE.format(
                query=query,
                self_description=desc if desc else "(no self-description provided)",
                skills=json.dumps(skills, ensure_ascii=False),
                agents=json.dumps(scaffold["agents"], ensure_ascii=False, indent=2),
                subtasks=json.dumps(scaffold["subtasks"], ensure_ascii=False, indent=2),
            )
            assistant_text = json.dumps(gold_sdp, ensure_ascii=False, indent=2)
            out_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SDP_SYSTEM},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "question_id": r["question_id"],
                    "profile_index": r["profile_index"],
                    "scaffold_sample_id": k,
                }
            )
        done = min(chunk_start + len(chunk), len(src_rows))
        write_jsonl(tmp_path, out_rows)
        print(
            f"  [phase-a] {done}/{len(src_rows)} rows processed; "
            f"{len(out_rows)} ok, {skipped} skipped; partial={tmp_path}",
            flush=True,
        )

    write_jsonl(out_dir / "train.jsonl", out_rows)
    if tmp_path.exists():
        tmp_path.unlink()
    print(
        f"\n[phase-a] wrote {len(out_rows)} rows to {out_dir/'train.jsonl'} "
        f"(skipped {skipped} unparseable scaffolds)"
    )


# ---------- phase B ----------------------------------------------------------

def cmd_phase_b(args):
    """Plan §3.2 Phase A: DPO on PAD using scaffold-level R_struct.

    Preferences come from rule-based scaffold-vs-gold structural similarity;
    SDP is NOT loaded. This narrows the Phase A signal to coordination
    (which scaffolds are structurally close to the gold scaffold) and lets
    GRPO own the wider R_pers / R_ped Pareto. It is also substantially
    faster than the previous SDP-perplexity scoring: one PAD generation
    per scaffold, plus an O(|scaffold|) Jaccard — no SDP forward pass.
    """
    import torch  # noqa

    tok, pad_model = load_base_and_adapter(
        args.base_model, args.pad_adapter, args.device, torch.bfloat16
    )

    src_rows = load_jsonl(Path(args.src).resolve())
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "train.jsonl.tmp"
    out_rows = []
    skipped = 0
    print(
        f"  [phase-b] {len(src_rows)} source rows, "
        f"pairs_per_query={args.pairs_per_query}, batch_size={args.batch_size}",
        flush=True,
    )

    flush_every = max(1, int(args.flush_every))
    for chunk_start in range(0, len(src_rows), flush_every):
        chunk = src_rows[chunk_start : chunk_start + flush_every]
        jobs = []
        prompts_data = []
        for offset, r in enumerate(chunk, chunk_start + 1):
            inp = r["plan"]["input"]
            query = inp["query"]
            desc, skills = extract_learner(inp["learner"])
            gold_scaffold = pad_target(r["plan"]["output"])
            for pair_idx in range(args.pairs_per_query):
                jobs.append((offset, r, pair_idx, query, desc, skills, gold_scaffold))
                prompts_data.extend([(query, desc, skills), (query, desc, skills)])

        results = batch_sample_scaffolds(
            pad_model,
            tok,
            prompts_data,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new=args.max_new_pad,
            batch_size=args.batch_size,
            phase_label="phase-b",
        )

        result_idx = 0
        for job in jobs:
            i, r, pair_idx, query, desc, skills, gold_scaffold = job
            candidates = []
            for _ in range(2):
                scaffold, raw_text = results[result_idx]
                result_idx += 1
                score = scaffold_struct_score(scaffold, gold_scaffold)
                candidates.append((score, raw_text))
            if len(candidates) < 2 or candidates[0][0] == candidates[1][0]:
                skipped += 1
                continue
            candidates.sort(key=lambda x: -x[0])  # descending
            chosen_text, rejected_text = candidates[0][1], candidates[1][1]
            user_text = PAD_USER_TEMPLATE.format(
                query=query,
                self_description=desc if desc else "(no self-description provided)",
                skills=json.dumps(skills, ensure_ascii=False),
            )
            prompt_messages = [
                {"role": "system", "content": PAD_SYSTEM},
                {"role": "user", "content": user_text},
            ]
            prompt_text = render_chat(tok, prompt_messages)
            out_rows.append(
                {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "question_id": r["question_id"],
                    "profile_index": r["profile_index"],
                    "pair_idx": pair_idx,
                    "delta_struct": candidates[0][0] - candidates[1][0],
                }
            )
        done = min(chunk_start + len(chunk), len(src_rows))
        write_jsonl(tmp_path, out_rows)
        print(
            f"  [phase-b] {done}/{len(src_rows)} rows processed; "
            f"{len(out_rows)} pairs, {skipped} skipped; partial={tmp_path}",
            flush=True,
        )

    write_jsonl(out_dir / "train.jsonl", out_rows)
    if tmp_path.exists():
        tmp_path.unlink()
    print(
        f"\n[phase-b] wrote {len(out_rows)} DPO pairs to {out_dir/'train.jsonl'} "
        f"(skipped {skipped} ties/unparseable)"
    )


# ---------- CLI --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common_args = lambda p: (
        p.add_argument("--src", required=True, help="MAPLE JSONL (training records)."),
        p.add_argument("--base-model", required=True),
        p.add_argument("--device", default="auto"),
        p.add_argument("--temperature", type=float, default=0.8),
        p.add_argument("--top-p", type=float, default=0.9),
        p.add_argument("--max-new-pad", type=int, default=1536),
        p.add_argument("--batch-size", type=int, default=32),
        p.add_argument(
            "--flush-every",
            type=int,
            default=50,
            help="Write partial train.jsonl.tmp after this many source rows.",
        ),
        p.add_argument("--out", required=True),
    )

    pa = sub.add_parser("phase-a", help="On-policy SDP training data.")
    common_args(pa)
    pa.add_argument("--pad-adapter", required=True)
    pa.add_argument("--samples-per-query", type=int, default=1)
    pa.set_defaults(func=cmd_phase_a)

    pb = sub.add_parser(
        "phase-b",
        help="PAD DPO pairs scored by scaffold-vs-gold structural Jaccard.",
    )
    common_args(pb)
    pb.add_argument("--pad-adapter", required=True)
    pb.add_argument("--pairs-per-query", type=int, default=1)
    pb.set_defaults(func=cmd_phase_b)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
