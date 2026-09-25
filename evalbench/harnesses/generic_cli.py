"""Fallback adapter for any agent harness that isn't wired up as a first-class
adapter yet: it runs a user-supplied shell command template in the task
workspace and captures whatever the agent changed via `git diff`.

Use this to plug in a harness by hand before writing a real adapter for it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from evalbench.harnesses.base import Harness, HarnessResult


class GenericCliAdapter(Harness):
    """options:
    - command: str (required) — a template with {instruction}, {workspace},
      {session_id}, {model_name} placeholders, e.g.:
      "my-agent-cli --model {model_name} --repo {workspace} --task {instruction!r}"
    - env: dict[str, str] — extra environment variables to set
    - timeout_sec: int = 1800
    """

    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        template = self.options.get("command")
        if not template:
            raise ValueError("generic-cli harness requires options.command")

        command = template.format(
            instruction=instruction,
            workspace=str(workspace),
            session_id=session_id,
            model_name=self.model.name,
        )

        env = os.environ.copy()
        env[self.model.api_key_env] = self.model.resolve_api_key()
        if self.model.base_url_env:
            base_url = self.model.resolve_base_url()
            if base_url:
                env[self.model.base_url_env] = base_url
        env.update(self.options.get("env", {}))

        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.options.get("timeout_sec", 1800),
        )

        subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True)
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--cached"],
            check=True,
            capture_output=True,
            text=True,
        )
        return HarnessResult(
            patch=diff.stdout,
            final_response=proc.stdout,
            exit_ok=proc.returncode == 0,
            extra={"stderr": proc.stderr, "returncode": proc.returncode},
        )
