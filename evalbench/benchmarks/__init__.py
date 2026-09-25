from evalbench.benchmarks.base import Benchmark, BenchmarkReport
from evalbench.config import BenchmarkConfig, HarnessConfig, ModelConfig, RunConfig

_REGISTRY = {}


def _lazy_registry() -> dict:
    if not _REGISTRY:
        from evalbench.benchmarks.swebench import SWEBenchBenchmark
        from evalbench.benchmarks.terminal_bench import TerminalBenchBenchmark

        _REGISTRY.update(
            {
                "swebench": SWEBenchBenchmark,
                "terminal-bench": TerminalBenchBenchmark,
            }
        )
    return _REGISTRY


def build_benchmark(
    model: ModelConfig,
    harness_config: HarnessConfig,
    benchmark_config: BenchmarkConfig,
    run_config: RunConfig,
) -> Benchmark:
    registry = _lazy_registry()
    if benchmark_config.name not in registry:
        raise ValueError(f"Unknown benchmark '{benchmark_config.name}'. Available: {list(registry)}")
    return registry[benchmark_config.name](model, harness_config, benchmark_config, run_config)


__all__ = ["Benchmark", "BenchmarkReport", "build_benchmark"]
