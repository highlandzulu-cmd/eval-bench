"""Config schema for an eval-bench run.

A run config has three independent axes, matching the pipeline's goal:
point any supported *harness* at any supported *benchmark*, running any
configured *model*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Which model to run, and how to authenticate it.

    `provider` picks the credential/env-var convention. `name` is passed
    through to the harness verbatim (e.g. "deepseek-v4-flash",
    "claude-sonnet-5", "anthropic/claude-sonnet-5" for litellm-style
    harnesses that want a provider prefix).
    """

    provider: Literal[
        "anthropic", "openai", "deepseek-official", "openrouter", "openai-compatible"
    ]
    name: str
    api_key_env: str = Field(
        description="Name of the environment variable holding the API key."
    )
    base_url_env: str | None = Field(
        default=None,
        description="Env var holding a custom base URL. Required for "
        "openai-compatible (any OpenAI-protocol server: local vLLM, Ollama, "
        "Together, ...); optional for openrouter (defaults to "
        "https://openrouter.ai/api/v1).",
    )
    max_tokens: int = 49_152
    temperature: float = 0.0

    _DEFAULT_BASE_URLS = {"openrouter": "https://openrouter.ai/api/v1"}

    def resolve_api_key(self) -> str:
        import os

        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Model provider '{self.provider}' needs env var "
                f"'{self.api_key_env}' set, but it is empty or unset."
            )
        return key

    def resolve_base_url(self) -> str | None:
        import os

        resolved = os.environ.get(self.base_url_env) if self.base_url_env else None
        return resolved or self._DEFAULT_BASE_URLS.get(self.provider)

    def as_litellm_model_string(self) -> str:
        """Best-effort `provider/model` string for harnesses (mini-swe-agent,
        Terminal-Bench's `tb run --model`) that route through litellm."""
        if "/" in self.name:
            return self.name
        litellm_provider = {
            "anthropic": "anthropic",
            "openai": "openai",
            "deepseek-official": "deepseek",
            "openrouter": "openrouter",
            "openai-compatible": "openai",
        }[self.provider]
        return f"{litellm_provider}/{self.name}"


class HarnessConfig(BaseModel):
    """Which agent harness runs the task, and its harness-specific knobs.

    `name` is either one of eval-bench's built-in harness names (see
    `evalbench.harnesses.BUILTIN_HARNESSES`) or a `module.path:ClassName`
    import path to your own `Harness` subclass — no core file needs editing
    to add a new one. See evalbench/harnesses/base.py.
    """

    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Which benchmark to run, and its benchmark-specific knobs."""

    name: Literal["swebench", "terminal-bench"]
    options: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    run_id: str
    output_dir: Path = Path("./runs")
    max_workers: int = 1


class EvalBenchConfig(BaseModel):
    model: ModelConfig
    harness: HarnessConfig
    benchmark: BenchmarkConfig
    run: RunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalBenchConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
