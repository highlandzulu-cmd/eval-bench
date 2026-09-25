from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from evalbench.config import BenchmarkConfig, HarnessConfig, ModelConfig, RunConfig


@dataclass
class BenchmarkReport:
    run_id: str
    benchmark: str
    total: int = 0
    resolved: int = 0
    details_path: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def resolve_rate(self) -> float:
        return self.resolved / self.total if self.total else 0.0


class Benchmark(ABC):
    def __init__(
        self,
        model: ModelConfig,
        harness_config: HarnessConfig,
        benchmark_config: BenchmarkConfig,
        run_config: RunConfig,
    ):
        self.model = model
        self.harness_config = harness_config
        self.options = benchmark_config.options
        self.run = run_config

    @abstractmethod
    def execute(self) -> BenchmarkReport:
        ...
