"""Generic Terminal-Bench agent: installs nothing beyond what `install_script`
does, then runs `command` (a template with an `{instruction}` placeholder)
inside the task's tmux session. This is the Terminal-Bench-side counterpart
to `evalbench.harnesses.generic_cli.GenericCliAdapter` — a stand-in for
wiring up a harness that doesn't have a first-class adapter yet.

Register via:

    tb run --agent-import-path evalbench.terminal_bench_agents.generic_agent:GenericInstalledAgent \\
           --agent-kwarg install_script=/abs/path/install.sh \\
           --agent-kwarg command='my-agent-cli --task {instruction}'
"""
from __future__ import annotations

import shlex
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand


class GenericInstalledAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "generic-cli"

    def __init__(
        self,
        install_script: str | None = None,
        command: str | None = None,
        env: dict[str, str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not install_script or not command:
            raise ValueError(
                "generic-cli terminal-bench agent needs --agent-kwarg "
                "install_script=<path> --agent-kwarg command='<template with {instruction}>'"
            )
        self._install_script = Path(install_script)
        self._command_template = command
        self._extra_env = env or {}

    @property
    def _env(self) -> dict[str, str]:
        return self._extra_env

    @property
    def _install_agent_script_path(self) -> Path:
        return self._install_script

    def _run_agent_commands(self, instruction: str) -> list[TerminalCommand]:
        command = self._command_template.format(instruction=shlex.quote(instruction))
        return [
            TerminalCommand(
                command=command,
                min_timeout_sec=0.0,
                max_timeout_sec=float("inf"),
                block=True,
                append_enter=True,
            ),
        ]
