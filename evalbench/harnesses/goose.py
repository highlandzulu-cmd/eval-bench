"""Adapter for Goose (https://github.com/block/goose), Block's open-source
AI agent. Drives the real `goose run` CLI headlessly:

    goose run --no-session --output-format stream-json --provider <p> \\
        --model <m> --with-builtin developer -t "<instruction>"

`--with-builtin developer` is required, not cosmetic: without it Goose has
no file-editing/shell tools at all (see the CLI reference's "Extension
Options" — Goose ships with none of its tool extensions enabled by
default). `-t/--text` + default (non-interactive, no `-s`) makes this a
one-shot call that exits after responding; `--no-session` skips goose's own
session-file bookkeeping (we persist our own trace instead, see below).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from evalbench.harnesses.base import Harness, HarnessResult

# provider -> (goose --provider value, env var goose reads for its key)
# Goose's own provider ids/env vars per `goose configure`; anthropic/openai/
# openrouter are long-standing goose providers. deepseek-official is passed
# through as "deepseek" (goose added first-class DeepSeek support) but
# hasn't been verified against a live `goose configure` run — check
# `goose configure` on your install if it doesn't recognize the provider id.
_GOOSE_PROVIDER = {
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
    "deepseek-official": ("deepseek", "DEEPSEEK_API_KEY"),
    "openrouter": ("openrouter", "OPENROUTER_API_KEY"),
}


class GooseAdapter(Harness):
    """options:
    - builtins: str = "developer"   (comma-separated --with-builtin extensions)
    - max_turns: int | None
    - timeout_sec: int = 1800
    """

    def __init__(self, model, options):
        super().__init__(model, options)
        if model.provider not in _GOOSE_PROVIDER and model.provider != "openai-compatible":
            raise ValueError(f"goose adapter doesn't know how to route provider '{model.provider}'")
        if subprocess.run(["which", "goose"], capture_output=True).returncode != 0:
            raise RuntimeError(
                "The goose adapter needs the `goose` CLI on PATH. "
                "See https://github.com/block/goose for install instructions "
                "(e.g. `curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash`)."
            )

    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        env = os.environ.copy()
        if self.model.provider == "openai-compatible":
            goose_provider = "openai"
            key_env = self.model.api_key_env
        else:
            goose_provider, key_env = _GOOSE_PROVIDER[self.model.provider]
        env[key_env] = self.model.resolve_api_key()
        if base_url := self.model.resolve_base_url():
            # goose reads provider-specific *_HOST vars for custom endpoints;
            # OPENAI_HOST is the documented one for the openai provider.
            env["OPENAI_HOST"] = base_url

        cmd = [
            "goose",
            "run",
            "--no-session",
            "--output-format",
            "stream-json",  # every step, not just the final answer - this is our trace
            "--provider",
            goose_provider,
            "--model",
            self.model.name,
            "--with-builtin",
            self.options.get("builtins", "developer"),
            "-t",
            instruction,
        ]
        if max_turns := self.options.get("max_turns"):
            cmd += ["--max-turns", str(max_turns)]

        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.options.get("timeout_sec", 1800),
        )

        trace_fd, trace_path_str = tempfile.mkstemp(prefix=f"goose-{session_id}-", suffix=".stream.jsonl")
        with os.fdopen(trace_fd, "w") as f:
            f.write(proc.stdout)

        patch = _git_diff(workspace)
        return HarnessResult(
            patch=patch,
            # stdout is now a stream-json event log, not readable prose - the
            # trace file is the source of truth; we haven't verified goose's
            # stream-json schema well enough to safely extract just the final
            # text here, so this is intentionally left for the trace to cover.
            final_response="",
            exit_ok=proc.returncode == 0,
            raw_log_path=Path(trace_path_str),
            extra={"stderr": proc.stderr, "returncode": proc.returncode},
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
