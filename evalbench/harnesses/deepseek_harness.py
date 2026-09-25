"""Adapter for DeepSeek Harness (`dsh`), https://github.com/deepseek-ai/deepseek-harness.

Uses the published `deepseek-harness-sdk` Python package exactly as documented
in the project's Python SDK guide: construct a `DeepSeekHarness(provider=,
model=, cwd=, dsh_home=, profile=)`, call `.run(instruction, session_id=)`,
and read `.final_response`.

Model routing: DSH's base bundle mounts one native adapter directly
(`deepseek-official`, via `@deepseek-ai/dsh-llm-deepseek`). Every other
provider goes through `@deepseek-ai/dsh-llm-pi-ai`
(packages/llm/llm-pi-ai/README.md), the harness's multi-provider adapter,
which can either use its installed catalog (openai, anthropic — just needs
`apiKeyEnv`) or a fully "hand-declared gateway" (`api` + `baseURL` +
`models`) for anything OpenAI-protocol-compatible, which is exactly what
OpenRouter, a local vLLM server, etc. are. We build that Cordis patch on the
fly and pass it via the SDK's documented `patches=(...)` constructor arg —
the same mechanism used below to opt into the `str_replace_editor` tool.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from evalbench.config import ModelConfig
from evalbench.harnesses.base import Harness, HarnessResult

_EDITOR_PATCH_YAML = textwrap.dedent(
    """\
    - insert:
        - id: fs-local
          name: '@deepseek-ai/dsh-fs-local'
          config:
            cwd: !!js process.cwd()
        - id: tool-str-replace-editor
          name: '@deepseek-ai/dsh-tool-str-replace-editor'
    """
)

# provider -> (pi-ai route name, needs hand-declared api/baseURL/models)
_PI_AI_ROUTES = {
    "openai": ("openai", False),
    "anthropic": ("anthropic", False),
    "openrouter": ("openrouter", True),
    "openai-compatible": ("custom", True),
}


def _pi_ai_patch_yaml(model: ModelConfig) -> str:
    route, hand_declared = _PI_AI_ROUTES[model.provider]
    provider_config: dict = {"apiKeyEnv": model.api_key_env}
    if hand_declared:
        base_url = model.resolve_base_url()
        if not base_url:
            raise ValueError(
                f"model.provider '{model.provider}' needs base_url_env set "
                "(dsh-llm-pi-ai needs an explicit baseURL for a hand-declared gateway)"
            )
        provider_config["api"] = "openai-completions"
        provider_config["baseURL"] = base_url
        provider_config["models"] = [{"id": model.name}]
    elif model.base_url_env:
        # optional override even for a catalog-known provider (e.g. an Azure/OpenAI proxy)
        if base_url := model.resolve_base_url():
            provider_config["baseURL"] = base_url

    config = {"providers": {route: provider_config}}
    # Cordis patch YAML: a plain "insert" row; JSON is valid YAML so this
    # avoids hand-rolling YAML indentation for a nested dict.
    return "- insert:\n    - id: llm-pi-ai-evalbench\n      name: '@deepseek-ai/dsh-llm-pi-ai'\n      config: " + json.dumps(
        config
    )


class DeepSeekHarnessAdapter(Harness):
    """options:
    - profile: str = "sdk-minimal"
    - enable_editor_tool: bool = False   (see warning below; default is the
      proven-working path)
    - dsh_home_root: str | None  (defaults to a run-scoped temp dir)

    enable_editor_tool warning: live-tested against OpenRouter (gpt-4o-mini
    and gemma-3-27b-it) and found NOT to actually expose str_replace_editor
    to the model in the sdk-minimal profile — both models correctly reported
    they had no way to write files, no hallucination involved. Mounting
    `dsh-fs-local` + `dsh-tool-str-replace-editor` via a patch isn't
    sufficient by itself; something else (likely wiring the tool into
    dsh-tools'/dsh-agent's active tool list) is still needed and hasn't been
    root-caused yet. Plain bash-only (the default) was verified working
    end-to-end: model runs a shell command, file gets written, `git diff`
    captures it. Leave this off until that's fixed, or verify it yourself
    with `dsh --profile sdk-minimal --patch <this patch> --dump-config`
    before trusting it on a real run.
    """

    def __init__(self, model, options):
        super().__init__(model, options)
        if model.provider != "deepseek-official" and model.provider not in _PI_AI_ROUTES:
            raise ValueError(f"deepseek-harness adapter doesn't know how to route provider '{model.provider}'")
        try:
            from deepseek_harness import DeepSeekHarness  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The deepseek-harness adapter needs the "
                "'deepseek-harness-sdk' package. Install with: "
                "pip install 'eval-bench[deepseek]'"
            ) from e

    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        from deepseek_harness import DeepSeekHarness

        os.environ[self.model.api_key_env] = self.model.resolve_api_key()

        profile = self.options.get("profile", "sdk-minimal")
        enable_editor = self.options.get("enable_editor_tool", False)

        patch_files: list = []
        if enable_editor and profile == "sdk-minimal":
            patch_files.append(_write_temp_patch(_EDITOR_PATCH_YAML))

        if self.model.provider == "deepseek-official":
            dsh_provider = "deepseek-official"
        else:
            dsh_provider, _ = _PI_AI_ROUTES[self.model.provider]
            patch_files.append(_write_temp_patch(_pi_ai_patch_yaml(self.model)))

        dsh_home_root = self.options.get("dsh_home_root")
        dsh_home = Path(dsh_home_root or tempfile.mkdtemp(prefix="dsh-home-")) / session_id
        dsh_home.mkdir(parents=True, exist_ok=True)

        try:
            with DeepSeekHarness(
                provider=dsh_provider,
                model=self.model.name,
                max_tokens=self.model.max_tokens,
                cwd=str(workspace),
                dsh_home=str(dsh_home),
                profile=profile,
                patches=tuple(str(p) for p in patch_files),
            ) as harness:
                result = harness.run(instruction, session_id=session_id)
        finally:
            for p in patch_files:
                os.unlink(p)

        patch = _git_diff(workspace)
        return HarnessResult(
            patch=patch,
            final_response=result.final_response,
            exit_ok=True,
            extra={"dsh_home": str(dsh_home), "dsh_provider": dsh_provider},
        )


def _write_temp_patch(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".patch.yml", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def _git_diff(workspace: Path) -> str:
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--cached"],
        check=True,
        capture_output=True,
        text=True,
    )
    return diff.stdout
