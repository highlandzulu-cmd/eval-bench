"""Harness interface for patch-producing benchmarks (SWE-bench-style).

Terminal-Bench does not use this interface: it drives agents itself inside
a tmux session in a container, so eval-bench instead maps a `HarnessConfig`
onto Terminal-Bench's own `--agent` / `--agent-import-path` flags. See
`evalbench/benchmarks/terminal_bench.py` and
`evalbench/terminal_bench_agents/`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from evalbench.config import ModelConfig


@dataclass
class HarnessResult:
    patch: str
    final_response: str = ""
    exit_ok: bool = True
    raw_log_path: Path | None = None
    extra: dict = field(default_factory=dict)


class Harness(ABC):
    """One task solve: given a task instruction and a repo checked out at
    `workspace`, run the agent and return the git diff it produced."""

    def __init__(self, model: ModelConfig, options: dict):
        self.model = model
        self.options = options

    @abstractmethod
    def solve(self, instruction: str, workspace: Path, session_id: str) -> HarnessResult:
        ...
