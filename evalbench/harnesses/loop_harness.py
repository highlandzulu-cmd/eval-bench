"""Adapter for Loop (https://github.com/soketlabs/loop), Soket AI's Rust
coding-agent harness. Uses this branch specifically, per the user's request:
https://github.com/soketlabs/loop/tree/feat/no-default-model

Drives the real `loop --print` headless mode (crates/loop-cli/src/print_mode.rs):
it's explicitly built for benchmark runners — its own doc comment says so,
and it ships `--trace-name`/`--trace-tag`/`--trace-metadata`/`--trace-session`
flags plus a `loop-trace-id: <id>` line on stderr for exactly this use case,
exporting full step-by-step traces to Langfuse (or any OTLP collector) when
configured. `main.rs` sets `interactive: false` whenever `--print` is used,
which is what lets tool calls (bash/write/edit) proceed without the
interactive approval prompt that gates them in the normal TUI.

Provider support: Loop's built-in providers are `soket` (its own hosted
default), `openai`, and `openrouter` — there's no native Anthropic or
DeepSeek provider. Anything else needs a "custom" OpenAI-compatible
provider registered via `/login custom` or a hand-written
`~/.loop/agent/models.json` entry; we haven't verified that file's exact
schema closely enough to safely generate it blind, so `openai-compatible`
here requires you to register the provider yourself once (interactively,
pointed at the same `LOOP_CODING_AGENT_DIR` this adapter will reuse) and
pass its name via `options.provider_name`.

Note: Soket's own gateway (`https://api.tensorstudio.ai/v1`, provider id
`soket`, env `SOKET_API_KEY`/`TENSORSTUDIO_API_KEY`/`LOOP_API_KEY`) is
reached the same way - `model.provider: openai-compatible` +
`options.provider_name: soket` - and its catalog (seen in a local
`models-store.json`) includes several open-weight models directly:
gemma-4-31b, qwen3-8-27b, gpt-oss-120b among them. Worth checking against
before assuming OpenRouter is the only route for those.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from evalbench.harnesses.base import Harness, HarnessResult

_NATIVE_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class LoopHarnessAdapter(Harness):
    """options:
    - no_context_files: bool = True   (skip auto-loading AGENTS.md/CLAUDE.md from the checked-out repo)
    - provider_name: str              (required for provider: openai-compatible - see module docstring)
    - loop_home_root: str | None      (defaults to a run-scoped temp dir; reused as LOOP_CODING_AGENT_DIR)
    - langfuse_host / langfuse_public_key_env / langfuse_secret_key_env: str
      (optional - wires up real step-by-step tracing via Loop's own Langfuse export;
      without these, only stdout/stderr are captured as the trace)
    - timeout_sec: int = 1800
    """

    def __init__(self, model, options):
        super().__init__(model, options)
        if model.provider not in ("openai", "openrouter", "openai-compatible"):
            raise ValueError(
                "loop-harness only has native support for provider: openai or "
                f"openrouter (or openai-compatible with a pre-registered custom "
                f"provider) - Loop has no built-in Anthropic or DeepSeek provider; "
                f"got '{model.provider}'"
            )
        if model.provider == "openai-compatible" and not options.get("provider_name"):
            raise ValueError(
                "provider: openai-compatible needs options.provider_name - register "
                "it once yourself with `loop --cwd . ` then `/login custom`, pointed "
                "at the same LOOP_CODING_AGENT_DIR you pass via options.loop_home_root"
            )
        if subprocess.run(["which", "loop"], capture_output=True).returncode != 0:
            raise RuntimeError(
                "The loop-harness adapter needs the `loop` CLI on PATH. Build it from "
                "https://github.com/soketlabs/loop (feat/no-default-model branch): "
                "`cargo install --path crates/loop-cli`, or grab a prebuilt release binary."
            )

    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        env = os.environ.copy()

        loop_home_root = self.options.get("loop_home_root")
        loop_home = Path(loop_home_root or tempfile.mkdtemp(prefix="loop-home-")) / session_id
        loop_home.mkdir(parents=True, exist_ok=True)
        env["LOOP_CODING_AGENT_DIR"] = str(loop_home)

        if self.model.provider == "openai-compatible":
            # e.g. Soket's own gateway (provider id "soket", env SOKET_API_KEY /
            # TENSORSTUDIO_API_KEY / LOOP_API_KEY) or any other pre-registered
            # custom provider - Loop reads whichever env var that provider expects,
            # which is exactly model.api_key_env here.
            provider = self.options["provider_name"]
            env[self.model.api_key_env] = self.model.resolve_api_key()
        else:
            provider = self.model.provider
            env[_NATIVE_PROVIDER_ENV[self.model.provider]] = self.model.resolve_api_key()

        for opt_key, env_var in (
            ("langfuse_host", "LANGFUSE_HOST"),
            ("langfuse_public_key_env", "LANGFUSE_PUBLIC_KEY"),
            ("langfuse_secret_key_env", "LANGFUSE_SECRET_KEY"),
        ):
            if value := self.options.get(opt_key):
                env[env_var] = os.environ.get(value, value) if opt_key.endswith("_env") else value

        cmd = [
            "loop",
            "--cwd",
            str(workspace),
            "--provider",
            provider,
            "--model",
            self.model.name,
            "--trace-name",
            "eval-bench",
            "--trace-tag",
            "swebench",
            "--trace-session",
            session_id,
        ]
        if self.options.get("no_context_files", True):
            cmd.append("--no-context-files")
        cmd += ["--print", instruction]

        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.options.get("timeout_sec", 1800),
        )

        trace_id = next(
            (
                line.removeprefix("loop-trace-id: ").strip()
                for line in proc.stderr.splitlines()
                if line.startswith("loop-trace-id: ")
            ),
            None,
        )

        trace_fd, trace_path_str = tempfile.mkstemp(prefix=f"loop-{session_id}-", suffix=".log")
        with os.fdopen(trace_fd, "w") as f:
            f.write("=== stdout ===\n")
            f.write(proc.stdout)
            f.write("\n=== stderr ===\n")
            f.write(proc.stderr)

        patch = _git_diff(workspace)
        return HarnessResult(
            patch=patch,
            final_response=proc.stdout,
            exit_ok=proc.returncode == 0,
            raw_log_path=Path(trace_path_str),
            extra={
                "returncode": proc.returncode,
                "trace_id": trace_id,
                "scratch_dir": str(loop_home),
            },
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
