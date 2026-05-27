"""Write experiment-plan checkpoint metadata JSON.

This helper records the fields required by EXPERIMENT_PLAN_2026-05-15:

  * data_split_sha is computed from maple_split_v1.json
  * git_sha is captured when git is available, otherwise null
  * hparams are extracted from the AutoTrain YAML params block when possible

Metric values such as dev_nll/dev_sv/dev_ar can be supplied after you read them
from AutoTrain logs or evaluator outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None", "~"}:
        return None
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def parse_params_block(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    params: dict[str, Any] = {}
    in_params = False
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line == "params:":
            in_params = True
            continue
        if in_params and raw and not raw.startswith(" "):
            break
        if in_params and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            params[key] = parse_scalar(value)
    params["config_path"] = str(config_path)
    params["config_sha256"] = sha256_file(config_path)
    return params


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="stage1a_PAD, stage1b_SDP, or stage2_JA")
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--epoch", type=float, required=True)
    ap.add_argument("--wall-clock-h", type=float, required=True)
    ap.add_argument("--dev-nll", type=float, required=True)
    ap.add_argument("--dev-sv", type=float)
    ap.add_argument("--dev-ar", type=float)
    ap.add_argument("--dev-dc-mean", type=float)
    ap.add_argument("--dev-atr-mean", type=float)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--split-file", type=Path, default=Path("maple_split_v1.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    root = Path.cwd()
    ckpt_path = Path(args.ckpt_path)
    out_path = args.out or ckpt_path / "metadata.json"

    metadata = {
        "stage": args.stage,
        "step": args.step,
        "epoch": args.epoch,
        "wall_clock_h": args.wall_clock_h,
        "dev_nll": args.dev_nll,
        "dev_sv": args.dev_sv,
        "dev_ar": args.dev_ar,
        "dev_dc_mean": args.dev_dc_mean,
        "dev_atr_mean": args.dev_atr_mean,
        "hparams": parse_params_block(args.config),
        "git_sha": git_sha(root),
        "data_split_sha": sha256_file(args.split_file),
        "ckpt_path": str(ckpt_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
