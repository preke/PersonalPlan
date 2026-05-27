"""
Light loader helpers for the upstream AOP repo.

The AOP repo hardcodes Linux paths (e.g. `os.chdir('/mnt/liao/planner')`)
and points its Sentence Transformer at a local `/mnt/liao/...` path; this
module neutralizes both by:

  * Loading the AOP `MLP.py` module from disk via importlib without
    executing it through normal sys.path resolution.
  * Returning a public Sentence Transformer (`sentence-transformers/
    all-MiniLM-L6-v2`) loaded with transformers.AutoTokenizer/AutoModel,
    so we never touch the original `/mnt/liao/...` path.

If torch, transformers, or the MLP weights are missing, the loader
returns None and the caller is expected to fall back to a degraded path
(no MLP, no replanning) - documented in plan.py module docstring.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Project root = baselines/aop/aop_patch.py → aop → baselines → ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
AOP_REPO = Path(os.environ.get(
    "AOP_REPO", _PROJECT_ROOT / "external" / "Agent-Oriented-Planning"
))
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_aop_module(name: str):
    """Load `<AOP_REPO>/<name>.py` while neutralizing any module-level
    `os.chdir` it may try to perform. Returns the loaded module or None
    if torch is not available (the AOP MLP module needs torch)."""
    repo_str = str(AOP_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    path = AOP_REPO / f"{name}.py"
    if not path.exists():
        return None

    _orig_chdir = os.chdir
    os.chdir = lambda *_a, **_kw: None
    try:
        spec = importlib.util.spec_from_file_location(f"aop_{name}", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as err:
            print(f"[aop_patch] failed to load aop {name}: "
                  f"{type(err).__name__}: {err}")
            return None
        return mod
    finally:
        os.chdir = _orig_chdir


def get_sentence_transformer() -> Optional[Tuple[object, object]]:
    """Return (tokenizer, model) for the public Sentence Transformer."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as err:
        print(f"[aop_patch] transformers not available: "
              f"{type(err).__name__}: {err}")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(ST_MODEL)
        model = AutoModel.from_pretrained(ST_MODEL)
        model.eval()
        return tokenizer, model
    except Exception as err:
        print(f"[aop_patch] failed to load {ST_MODEL}: "
              f"{type(err).__name__}: {err}")
        return None


def load_mlp_reward():
    """Load MLP_high.pt with the SimilarityMLP architecture; return None
    if either torch, the AOP MLP module, or the weights file is missing."""
    try:
        import torch
    except Exception as err:
        print(f"[aop_patch] torch not available: "
              f"{type(err).__name__}: {err}")
        return None

    aop_mlp_mod = load_aop_module("MLP")
    if aop_mlp_mod is None or not hasattr(aop_mlp_mod, "SimilarityMLP"):
        print("[aop_patch] could not load SimilarityMLP class from aop MLP.py")
        return None

    weights_path = AOP_REPO / "reward_model" / "MLP_high.pt"
    if not weights_path.exists():
        print(f"[aop_patch] MLP weights missing at {weights_path}; "
              f"running degraded path (no replanning).")
        return None

    mlp = aop_mlp_mod.SimilarityMLP()
    try:
        state = torch.load(str(weights_path), map_location="cpu",
                           weights_only=True)
    except TypeError:
        state = torch.load(str(weights_path), map_location="cpu")
    except Exception as err:
        print(f"[aop_patch] torch.load failed: "
              f"{type(err).__name__}: {err}")
        return None
    mlp.load_state_dict(state)
    mlp.eval()
    return mlp
