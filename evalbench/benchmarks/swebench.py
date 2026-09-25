"""SWE-bench adapter (https://github.com/SWE-bench/SWE-bench).

Grading always goes through SWE-bench's own `swebench` CLI (Docker-based;
we never reimplement that), reading the run summary it writes to
`logs/evaluation/<run_id>/results.json`. `swebench eval --help` / `swebench
infer --help` are the source of truth if these flags drift — this CLI
(v5, `swebench eval|infer|images|report`) replaced the older
`python -m swebench.harness.run_evaluation` entry point in 2026, though the
README says the old form still works.

Two ways to get predictions, chosen automatically by harness:

- harness = mini-swe-agent: shell out to `swebench infer`, which runs
  mini-swe-agent itself end-to-end (repo checkout, agent loop, patch
  capture) against any litellm model string. No need to duplicate that.
- any other harness (deepseek-harness, generic-cli, ...): eval-bench does
  the instance loop itself — git-checkout each repo at `base_commit` into a
  scratch workspace, call the harness's `solve()`, capture the diff — and
  writes the same predictions file format SWE-bench's own tooling uses, so
  grading is identical either way.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from evalbench.benchmarks.base import Benchmark, BenchmarkReport
from evalbench.harnesses import build_harness

DATASET_ALIASES = {
    "full": "princeton-nlp/SWE-bench",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "multimodal": "princeton-nlp/SWE-bench_Multimodal",
    "multilingual": "swe-bench/SWE-bench_Multilingual",
}


class SWEBenchBenchmark(Benchmark):
    """options:
    - dataset: str = "lite"            (alias above, or any HF dataset id/path)
    - split: str = "test"
    - max_instances: int | None
    - instance_ids: list[str] | None
    - task_repo: str | None            (local swe-bench-tasks checkout, for `swebench eval`'s local image builds)
    - skip_eval: bool = False          (only generate predictions, skip Docker grading)
    """

    def execute(self) -> BenchmarkReport:
        dataset_alias = self.options.get("dataset", "lite")
        run_dir = Path(self.run.output_dir) / self.run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        preds_path = run_dir / "preds.json"

        if self.harness_config.name == "mini-swe-agent":
            self._infer_native(dataset_alias, run_dir, preds_path)
        else:
            self._infer_generic(dataset_alias, run_dir, preds_path)

        report = BenchmarkReport(
            run_id=self.run.run_id, benchmark="swebench", details_path=preds_path
        )
        if not self.options.get("skip_eval", False):
            self._grade(dataset_alias, preds_path, report)
        else:
            preds = json.loads(preds_path.read_text())
            report.total = len(preds)
        return report

    def _infer_native(self, dataset_alias: str, run_dir: Path, preds_path: Path) -> None:
        """Let SWE-bench's own `swebench infer` drive mini-swe-agent."""
        cmd = [
            "swebench",
            "infer",
            dataset_alias,
            "-m",
            self.model.as_litellm_model_string(),
            "-o",
            str(run_dir),
            "-w",
            str(self.run.max_workers),
        ]
        if instance_ids := self.options.get("instance_ids"):
            for iid in instance_ids:
                cmd += ["-i", iid]
        _run_env_checked(cmd, self.model)
        # `swebench infer` writes preds.json under -o; normalize the path in
        # case a future CLI version nests it (e.g. run_dir/preds/preds.json).
        if not preds_path.exists():
            candidates = list(run_dir.rglob("preds.json"))
            if not candidates:
                raise RuntimeError(f"swebench infer did not produce preds.json under {run_dir}")
            preds_path.write_text(candidates[0].read_text())

    def _infer_generic(self, dataset_alias: str, run_dir: Path, preds_path: Path) -> None:
        from datasets import load_dataset

        dataset_path = DATASET_ALIASES.get(dataset_alias, dataset_alias)
        split = self.options.get("split", "test")
        instances = list(load_dataset(dataset_path, split=split))

        if instance_ids := self.options.get("instance_ids"):
            wanted = set(instance_ids)
            instances = [i for i in instances if i["instance_id"] in wanted]
        if max_instances := self.options.get("max_instances"):
            instances = instances[:max_instances]

        harness = build_harness(self.model, self.harness_config)
        preds: dict[str, dict] = {}
        if preds_path.exists():
            preds = json.loads(preds_path.read_text())

        for instance in instances:
            instance_id = instance["instance_id"]
            if instance_id in preds and not self.options.get("redo_existing", False):
                continue
            workspace = run_dir / "workspaces" / instance_id
            _checkout_repo(instance["repo"], instance["base_commit"], workspace)
            try:
                result = harness.solve(
                    instruction=instance["problem_statement"],
                    workspace=workspace,
                    session_id=instance_id,
                )
                patch = result.patch
            except Exception as e:  # noqa: BLE001 - record and keep going
                patch = ""
                print(f"[swebench] {instance_id} failed: {e}")
            preds[instance_id] = {
                "model_name_or_path": self.model.name,
                "instance_id": instance_id,
                "model_patch": patch,
            }
            preds_path.write_text(json.dumps(preds, indent=2))

    def _grade(self, dataset_alias: str, preds_path: Path, report: BenchmarkReport) -> None:
        cmd = [
            "swebench",
            "eval",
            dataset_alias,
            "-p",
            str(preds_path),
            "--run-id",
            self.run.run_id,
            "-j",
            str(self.run.max_workers),
        ]
        if task_repo := self.options.get("task_repo"):
            cmd += ["--task-repo", task_repo]
        subprocess.run(cmd, check=True)

        results_path = Path("logs/evaluation") / self.run.run_id / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            resolved = results.get("resolved_ids") or results.get("resolved", [])
            report.total = len(json.loads(preds_path.read_text()))
            report.resolved = len(resolved) if isinstance(resolved, list) else resolved
            report.extra["results_path"] = str(results_path)
            report.extra["raw_results"] = results
        else:
            print(f"[swebench] expected results at {results_path}, not found; check `swebench eval` output above")


def _checkout_repo(repo: str, base_commit: str, workspace: Path) -> None:
    if workspace.exists():
        return
    workspace.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", f"https://github.com/{repo}.git", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", base_commit], check=True)


def _run_env_checked(cmd: list[str], model) -> None:
    import os

    env = os.environ.copy()
    env[model.api_key_env] = model.resolve_api_key()
    if model.base_url_env:
        base_url = model.resolve_base_url()
        if base_url:
            env[model.base_url_env] = base_url
    subprocess.run(cmd, check=True, env=env)
