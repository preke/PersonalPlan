"""Pre-generate the counterfactual cache used by R_pers (v1).

Paper §4.2 wants the counterfactual plan to be sampled from π_θ (current
policy) at every training step, with a randomly-swapped profile. TRL's
GRPOTrainer doesn't expose hooks to do that cleanly. The v1
work-around: sample one counterfactual plan PER training record using
the *frozen SFT model* and store it in a jsonl cache. R_pers then uses
this fixed reference for the entire training run.

Tradeoff:
  + zero per-step inference cost (cf is cached)
  + works inside TRL without subclassing
  − cf signal is stale w.r.t. current policy → R_pers gradient becomes
    progressively less informative as π_θ drifts from π_SFT.

Mitigation: re-run this script every N training epochs (or every time
you change the SFT init) to refresh the cache.

Usage:
  python build_counterfactual_cache.py \
      --src ../multi_agent_dataset_filtered_qap_latest.jsonl \
      --base-model Qwen/Qwen2.5-7B-Instruct \
      --sft-adapter ../SFT/maple-pad-dpo-iter1  (or whichever full-plan SFT)
      --out data/grpo/cf_cache.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Heavy imports deferred so --help works without torch installed.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="MAP-PPL JSONL (the GRPO source).")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sft-adapter", default=None,
                    help="Optional SFT LoRA adapter to load on top of base_model.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=3500)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    src = [json.loads(l) for l in open(args.src) if l.strip()]
    print(f"Loaded {len(src)} source records")

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    if args.sft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()

    # Pool of profiles to draw counterfactuals from
    def extract_learner(lrn):
        desc = lrn.get("self_description") or lrn.get("about_me") or ""
        skills = lrn.get("skills") or lrn.get("top_tags") or []
        return desc.strip(), list(skills)

    profile_pool = [extract_learner(r["plan"]["input"]["learner"]) for r in src]
    profile_pool = [p for p in profile_pool if p[0] or p[1]]

    # Reuse the GRPO prompt builder for consistency
    from build_grpo_prompts import SYSTEM_PROMPT, USER_TEMPLATE

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, r in enumerate(src, 1):
            qid = str(r["question_id"])
            query = r["plan"]["input"]["query"]
            true_desc, true_skills = extract_learner(r["plan"]["input"]["learner"])

            # Pick a counterfactual profile (different from the true one)
            cf_desc, cf_skills = rng.choice(profile_pool)
            tries = 0
            while (cf_desc, tuple(cf_skills)) == (true_desc, tuple(true_skills)) and tries < 5:
                cf_desc, cf_skills = rng.choice(profile_pool)
                tries += 1

            user_text = USER_TEMPLATE.format(
                query=query,
                self_description=cf_desc if cf_desc else "(no self-description provided)",
                skills=json.dumps(cf_skills, ensure_ascii=False),
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Best-effort JSON parse
            from plan_utils import parse_plan
            cf_plan = parse_plan(gen)
            if cf_plan is None:
                n_fail += 1
                continue

            fout.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "cf_profile_self_description": cf_desc,
                        "cf_profile_skills": cf_skills,
                        "cf_plan": cf_plan,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_ok += 1

            if i % 20 == 0:
                print(f"  {i}/{len(src)}  ok={n_ok} fail={n_fail}")

    print(f"\nDone. cf cache → {out_path}  (ok={n_ok}, fail={n_fail})")


if __name__ == "__main__":
    main()
