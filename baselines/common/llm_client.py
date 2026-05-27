"""
Single entry point for LLM calls. All baselines route through this.
Five backends:
  - "qwen3-32b":       DashScope OpenAI-compatible endpoint (default).
  - "qwen3-max":       DashScope OpenAI-compatible endpoint (T4/T5 backbone).
  - "gpt-5":           bianxie.ai proxy (only used by autogen_gpt5).
  - "claude-opus-4-6": bianxie.ai proxy.
  - "gpt-4o":          bianxie.ai proxy (with-key alternative for T4/T5).
"""
import os, json, time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Project root = baselines/common/llm_client.py → common → baselines → ROOT
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


class LLMClient:
    def __init__(self, backend: str = "qwen3-32b"):
        if backend == "qwen3-32b":
            self.model = "qwen3-32b"
            self.client = OpenAI(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.extra = {"extra_body": {"enable_thinking": False}}
        elif backend == "qwen3-max":
            self.model = "qwen3-max-2025-09-23"   # dated snapshot per user 2026-05-16
            self.client = OpenAI(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self.extra = {"extra_body": {"enable_thinking": False}}
        elif backend == "gpt-5":
            self.model = "gpt-5"
            self.client = OpenAI(
                api_key=os.environ["OPENAI_PROXY_API_KEY"],
                base_url=os.environ["OPENAI_PROXY_BASE_URL"],
            )
            self.extra = {}
        elif backend == "claude-opus-4-6":
            self.model = "claude-opus-4-6"
            self.client = OpenAI(
                api_key=os.environ["OPENAI_PROXY_API_KEY"],
                base_url=os.environ["OPENAI_PROXY_BASE_URL"],
            )
            self.extra = {}
        elif backend == "gpt-4o":
            self.model = "gpt-4o"
            self.client = OpenAI(
                api_key=os.environ["OPENAI_PROXY_API_KEY"],
                base_url=os.environ["OPENAI_PROXY_BASE_URL"],
            )
            self.extra = {}
        else:
            raise ValueError(backend)
        self.backend = backend

    def chat(self, messages, temperature=0.3, max_tokens=16000,
             retries=2) -> str:
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **self.extra,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise last_err
