"""Adapter for mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent),
"the 100 line AI agent that solves GitHub issues."

Uses its real Python bindings directly:

    agent = DefaultAgent(LitellmModel(model_name=...), LocalEnvironment(cwd=...))
    agent.run(task)

rather than mini-swe-agent's own SWE-bench batch runner (`run/benchmarks/
swebench.py`), so that this harness plugs into eval-bench's *own*
benchmark adapters (which handle repo checkout, patch capture, and grading
uniformly across every harness) instead of running a second, separate
Docker orchestration. mini-swe-agent's system/instance prompt templates
come from its shipped `default.yaml`; only the model and environment are
harness-specific here.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from evalbench.harnesses.base import Harness, HarnessResult

_LITELLM_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek-official": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
}


class MiniSweAgentAdapter(Harness):
    """options:
    - step_limit: int = 0 (0 = unlimited)
    - cost_limit: float = 3.0
    - config_spec: str | None  (path to an additional mini-swe-agent yaml
      config to merge over the shipped default.yaml, e.g. its swebench.yaml)
    """

    def __init__(self, model, options):
        super().__init__(model, options)
        try:
            import minisweagent  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The mini-swe-agent adapter needs the 'mini-swe-agent' "
                "package. Install with: pip install 'eval-bench[mini-swe-agent]'"
            ) from e

    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import get_config_from_spec
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.utils.serialize import recursive_merge

        os.environ[_LITELLM_ENV_VAR[self.model.provider]] = self.model.resolve_api_key()

        model_kwargs = {}
        base_url = self.model.resolve_base_url()
        if base_url:
            model_kwargs["api_base"] = base_url

        config_specs = ["default.yaml", *([self.options["config_spec"]] if self.options.get("config_spec") else [])]
        config = recursive_merge(*(get_config_from_spec(spec) for spec in config_specs))
        agent_config = {
            **config.get("agent", {}),
            "step_limit": self.options.get("step_limit", 0),
            "cost_limit": self.options.get("cost_limit", 3.0),
        }

        model = LitellmModel(
            model_name=self.model.as_litellm_model_string(),
            model_kwargs=model_kwargs,
        )
        env = LocalEnvironment(cwd=str(workspace), **config.get("environment", {}))
        agent = DefaultAgent(model, env, **agent_config)

        info = agent.run(instruction)
        fd, trace_path_str = tempfile.mkstemp(prefix=f"mini-swe-{session_id}-", suffix=".traj.json")
        os.close(fd)
        trace_path = Path(trace_path_str)
        agent.save(trace_path)

        patch = _git_diff(workspace)
        return HarnessResult(
            patch=patch,
            final_response=info.get("submission", ""),
            exit_ok=info.get("exit_status") == "Submitted",
            raw_log_path=trace_path,
            extra={"exit_status": info.get("exit_status")},
        )


def _git_diff(workspace: Path) -> str:
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
    diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--cached"],
        check=True,
        capture_output=True,
        text=True,
    )
    return diff.stdout
