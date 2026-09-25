import importlib

from evalbench.config import HarnessConfig, ModelConfig
from evalbench.harnesses.base import Harness, HarnessResult

BUILTIN_HARNESSES = ("deepseek-harness", "mini-swe-agent", "goose", "loop-harness", "generic-cli")

_REGISTRY = {}


def _lazy_registry() -> dict:
    if not _REGISTRY:
        from evalbench.harnesses.deepseek_harness import DeepSeekHarnessAdapter
        from evalbench.harnesses.generic_cli import GenericCliAdapter
        from evalbench.harnesses.goose import GooseAdapter
        from evalbench.harnesses.loop_harness import LoopHarnessAdapter
        from evalbench.harnesses.mini_swe_agent import MiniSweAgentAdapter

        _REGISTRY.update(
            {
                "deepseek-harness": DeepSeekHarnessAdapter,
                "mini-swe-agent": MiniSweAgentAdapter,
                "goose": GooseAdapter,
                "loop-harness": LoopHarnessAdapter,
                "generic-cli": GenericCliAdapter,
            }
        )
    return _REGISTRY


def _import_harness_class(import_path: str) -> type[Harness]:
    """Load a user's own Harness subclass from `module.path:ClassName`,
    mirroring how Terminal-Bench itself resolves `--agent-import-path`. This
    is the escape hatch for any harness eval-bench doesn't know about by
    name — write a class implementing `Harness.solve()` anywhere on your
    PYTHONPATH and point a config at it, no eval-bench source edits needed.
    """
    if ":" not in import_path:
        raise ValueError(
            f"Unknown harness '{import_path}'. Use one of {BUILTIN_HARNESSES}, "
            "or a 'module.path:ClassName' import path to your own Harness subclass."
        )
    module_name, class_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    harness_class = getattr(module, class_name)
    if not issubclass(harness_class, Harness):
        raise ValueError(f"'{class_name}' in '{module_name}' is not a subclass of evalbench.harnesses.base.Harness")
    return harness_class


def build_harness(model: ModelConfig, config: HarnessConfig) -> Harness:
    registry = _lazy_registry()
    harness_class = registry.get(config.name) or _import_harness_class(config.name)
    return harness_class(model, config.options)


__all__ = ["Harness", "HarnessResult", "build_harness", "BUILTIN_HARNESSES"]
