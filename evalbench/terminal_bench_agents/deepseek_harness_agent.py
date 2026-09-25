"""Terminal-Bench agent plugin for DeepSeek Harness (`dsh`).

Terminal-Bench runs each task inside its own Docker container and expects
agents that aren't wired in natively to subclass `AbstractInstalledAgent`:
it copies an install script into the container, sources it, then sends
shell commands into the task's tmux session (see
`terminal_bench.agents.installed_agents.abstract_installed_agent` and
`.../claude_code/claude_code_agent.py` for the pattern this mirrors).

Register with Terminal-Bench via an import path, no fork needed:

    tb run --agent-import-path evalbench.terminal_bench_agents.deepseek_harness_agent:DeepSeekHarnessAgent \\
           --model deepseek/deepseek-v4-flash --dataset-name terminal-bench-core --dataset-version 0.1.1

Model selection caveat: the `dsh` CLI's one-shot `headless` profile only
documents the task text as a positional argument (see `dsh` CLI reference,
"App arguments" table) — there is no documented `--model` flag. We set
`DSH_MODEL` as a best-effort default (the Python SDK's own example script,
`minimal.py`, reads that env var) and additionally support pointing at an
explicit Cordis patch file via `--agent-kwarg patch=/path/to/model.patch.yml`
for anyone who has inspected `dsh --profile headless --dump-default-config`
and knows their adapter row's exact id.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand


class DeepSeekHarnessAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "deepseek-harness"

    def __init__(self, model_name: str | None = None, patch: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._patch = patch
        self._version = kwargs.get("version", "")

    @property
    def _env(self) -> dict[str, str]:
        env = {"DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]}
        if base_url := os.environ.get("DEEPSEEK_BASE_URL"):
            env["DEEPSEEK_BASE_URL"] = base_url
        if self._model_name:
            env["DSH_MODEL"] = self._model_name.removeprefix("deepseek/")
        elif "DSH_MODEL" in os.environ:
            env["DSH_MODEL"] = os.environ["DSH_MODEL"]
        return env

    @property
    def _install_agent_script_path(self) -> Path:
        return self._get_templated_script_path("dsh-setup.sh.j2")

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        escaped_instruction = shlex.quote(instruction)
        patch_flag = f" --patch {shlex.quote(self._patch)}" if self._patch else ""
        return [
            TerminalCommand(
                command=f"dsh --profile headless{patch_flag} {escaped_instruction}",
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
