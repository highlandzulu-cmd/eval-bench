from evalbench.config import HarnessConfig, ModelConfig
from evalbench.harnesses.base import Harness, HarnessResult

_REGISTRY = {}


def _lazy_registry() -> dict:
    if not _REGISTRY:
        from evalbench.harnesses.deepseek_harness import DeepSeekHarnessAdapter
        from evalbench.harnesses.generic_cli import GenericCliAdapter
        from evalbench.harnesses.mini_swe_agent import MiniSweAgentAdapter

        _REGISTRY.update(
            {
                "deepseek-harness": DeepSeekHarnessAdapter,
                "mini-swe-agent": MiniSweAgentAdapter,
                "generic-cli": GenericCliAdapter,
            }
        )
    return _REGISTRY


def build_harness(model: ModelConfig, config: HarnessConfig) -> Harness:
    registry = _lazy_registry()
    if config.name not in registry:
        raise ValueError(f"Unknown harness '{config.name}'. Available: {list(registry)}")
    return registry[config.name](model, config.options)


__all__ = ["Harness", "HarnessResult", "build_harness"]
