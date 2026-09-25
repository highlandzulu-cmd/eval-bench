"""Terminal-Bench adapter (https://github.com/harbor-framework/terminal-bench-1,
pip package `terminal-bench`, CLI `tb`).

Terminal-Bench owns the whole loop itself — dataset, Docker sandbox, agent
plugin, grading — so eval-bench's job here is just translating a unified
(model, harness) config into the right `tb run` invocation, and reading back
`<output_path>/<run_id>/results.json` (see `terminal_bench.harness.harness.
Harness._results_output_path` / `models.BenchmarkResults`).

harness.name -> tb agent mapping:
    mini-swe-agent   -> --agent mini-swe-agent   (built into tb)
    deepseek-harness -> --agent-import-path evalbench.terminal_bench_agents.deepseek_harness_agent:DeepSeekHarnessAgent
    generic-cli      -> --agent-import-path evalbench.terminal_bench_agents.generic_agent:GenericInstalledAgent
                        (needs harness.options.install_script and harness.options.command)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from evalbench.benchmarks.base import Benchmark, BenchmarkReport

# tb's own built-in --agent names (terminal_bench.agents.agent_name.AgentName)
# we pass straight through instead of writing an adapter.
_BUILTIN_AGENTS = {"mini-swe-agent", "goose", "claude-code", "aider", "codex", "openhands"}
_IMPORT_PATH_AGENTS = {
    "deepseek-harness": "evalbench.terminal_bench_agents.deepseek_harness_agent:DeepSeekHarnessAgent",
    "generic-cli": "evalbench.terminal_bench_agents.generic_agent:GenericInstalledAgent",
}


class TerminalBenchBenchmark(Benchmark):
    """options:
    - dataset_name: str = "terminal-bench-core"
    - dataset_version: str = "0.1.1"
    - task_ids: list[str] | None
    - n_concurrent: int = 4
    - no_rebuild: bool = False
    """

    def execute(self) -> BenchmarkReport:
        output_dir = Path(self.run.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "tb",
            "run",
            "--dataset-name",
            self.options.get("dataset_name", "terminal-bench-core"),
            "--dataset-version",
            self.options.get("dataset_version", "0.1.1"),
            "--model",
            self.model.as_litellm_model_string(),
            "--output-path",
            str(output_dir),
            "--run-id",
            self.run.run_id,
            "--n-concurrent",
            str(self.options.get("n_concurrent", self.run.max_workers)),
        ]

        harness_name = self.harness_config.name
        if harness_name in _BUILTIN_AGENTS:
            cmd += ["--agent", harness_name]
        else:
            # a known eval-bench harness with a written tb agent plugin, or
            # (the escape hatch) a raw "module:Class" import path to your
            # own AbstractInstalledAgent subclass — see terminal_bench_agents/.
            import_path = _IMPORT_PATH_AGENTS.get(harness_name, harness_name)
            if ":" not in import_path:
                raise ValueError(
                    f"No Terminal-Bench agent mapping for harness '{harness_name}'. "
                    f"Use one of {_BUILTIN_AGENTS | set(_IMPORT_PATH_AGENTS)}, or a "
                    "'module.path:ClassName' import path to your own AbstractInstalledAgent."
                )
            cmd += ["--agent-import-path", import_path]
            for key, value in self.harness_config.options.items():
                cmd += ["--agent-kwarg", f"{key}={value}"]

        for task_id in self.options.get("task_ids", []) or []:
            cmd += ["--task-id", task_id]
        if self.options.get("no_rebuild"):
            cmd.append("--no-rebuild")

        env = os.environ.copy()
        env[self.model.api_key_env] = self.model.resolve_api_key()
        if self.model.base_url_env:
            base_url = self.model.resolve_base_url()
            if base_url:
                env[self.model.base_url_env] = base_url

        subprocess.run(cmd, check=True, env=env)

        results_path = output_dir / self.run.run_id / "results.json"
        report = BenchmarkReport(
            run_id=self.run.run_id, benchmark="terminal-bench", details_path=results_path
        )
        if results_path.exists():
            results = json.loads(results_path.read_text())
            report.total = len(results.get("results", []))
            report.resolved = results.get("n_resolved", 0)
            report.extra["raw_results"] = results
        else:
            print(f"[terminal-bench] expected results at {results_path}, not found; check `tb run` output above")
        return report
