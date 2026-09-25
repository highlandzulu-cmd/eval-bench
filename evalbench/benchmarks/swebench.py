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

_TASK_INSTRUCTION_TEMPLATE = """\
You are an autonomous coding agent working directly in a git repository \
checked out at {workspace}. You have a persistent shell tool: use it to \
explore the codebase, reproduce the problem, edit files (e.g. with sed, a \
heredoc, or python), and verify your fix. This is not a conversation — do \
not just explain or discuss the issue, actually make the code changes.

Issue to resolve:

{problem_statement}

When you are done, make sure your changes are saved to disk in this \
checked-out repository. Do not create a new branch or commit; leaving the \
edits as uncommitted working-tree changes is correct and expected."""

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
    - skip_eval: bool = False          (skip grading entirely, only generate predictions)
    - grading: "docker" | "deepeval" | "both" = "docker"
      - docker: the real FAIL_TO_PASS/PASS_TO_PASS test run via `swebench eval` (needs Docker; can be slow/
        resource-heavy, see README - this is the only grading mode that produces an official SWE-bench verdict)
      - deepeval: an LLM-judge (https://deepeval.com) GEval score of whether the patch plausibly resolves the
        issue, compared against the dataset's own reference patch - no Docker, much faster/cheaper, but it's
        an opinion, not a test run. Options: deepeval_judge (a model dict like the top-level `model:` block;
        defaults to the harness's own model) and deepeval_threshold (float, default 0.5).
      - both: run both and keep results from each separately under report.extra.
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
        grading = self.options.get("grading", "docker")
        if self.options.get("skip_eval", False):
            preds = json.loads(preds_path.read_text())
            report.total = len(preds)
        else:
            if grading in ("docker", "both"):
                self._grade(dataset_alias, preds_path, report)
            if grading in ("deepeval", "both"):
                self._grade_with_deepeval(dataset_alias, preds_path, report)
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
            instruction = _TASK_INSTRUCTION_TEMPLATE.format(
                workspace=workspace, problem_statement=instance["problem_statement"]
            )
            try:
                result = harness.solve(
                    instruction=instruction,
                    workspace=workspace,
                    session_id=instance_id,
                )
                patch = result.patch
                _persist_trace(result, run_dir, instance_id)
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

    def _grade_with_deepeval(self, dataset_alias: str, preds_path: Path, report: BenchmarkReport) -> None:
        from datasets import load_dataset
        from deepeval import evaluate
        from deepeval.evaluate.configs import DisplayConfig
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        from evalbench.config import ModelConfig
        from evalbench.evaluators.litellm_judge import LitellmJudgeModel

        judge_options = self.options.get("deepeval_judge")
        judge_model = ModelConfig(**judge_options) if judge_options else self.model
        judge = LitellmJudgeModel(judge_model)

        dataset_path = DATASET_ALIASES.get(dataset_alias, dataset_alias)
        split = self.options.get("split", "test")
        instances = {i["instance_id"]: i for i in load_dataset(dataset_path, split=split)}

        preds = json.loads(preds_path.read_text())
        metric = GEval(
            name="SWE-bench Patch Correctness",
            criteria=(
                "Determine whether the 'actual output' (a code diff/patch) correctly "
                "resolves the software issue described in 'input', achieving a similar "
                "functional effect to the reference fix in 'expected output'. Judge on "
                "functional correctness, not exact textual match - a differently-written "
                "patch that fixes the same underlying bug the same way should score well. "
                "A patch that is empty, unrelated, or only adds a test/reproduction script "
                "without touching the buggy code should score low."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            threshold=self.options.get("deepeval_threshold", 0.5),
        )

        test_cases = [
            LLMTestCase(
                name=instance_id,
                input=instances[instance_id]["problem_statement"],
                actual_output=pred["model_patch"] or "(no patch produced)",
                expected_output=instances[instance_id].get("patch", ""),
            )
            for instance_id, pred in preds.items()
            if instance_id in instances
        ]

        result = evaluate(
            test_cases,
            [metric],
            display_config=DisplayConfig(print_results=False, show_indicator=False),
        )

        per_instance = [
            {
                "instance_id": r.name,
                "success": r.success,
                "score": r.metrics_data[0].score if r.metrics_data else None,
                "reason": r.metrics_data[0].reason if r.metrics_data else None,
            }
            for r in result.test_results
        ]
        resolved = sum(1 for r in per_instance if r["success"])
        report.extra["deepeval"] = {
            "judge_model": judge.get_model_name(),
            "total": len(per_instance),
            "resolved": resolved,
            "per_instance": per_instance,
        }
        if self.options.get("grading") == "deepeval":
            report.total = len(per_instance)
            report.resolved = resolved


def _persist_trace(result, run_dir: Path, instance_id: str) -> None:
    """Copy the harness's raw session/trajectory log out of wherever the
    harness happened to write it (typically an OS temp dir - see each
    harness's `solve()`) into `run_dir/traces/`, then clean up the harness's
    own scratch directory so repeated runs don't silently leak disk space.
    """
    import shutil

    if result.raw_log_path and result.raw_log_path.exists():
        traces_dir = run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        suffix = "".join(result.raw_log_path.suffixes) or ".log"
        shutil.copy2(result.raw_log_path, traces_dir / f"{instance_id}{suffix}")
        # dsh's log lives inside dsh_home (removed below as a whole tree);
        # mini-swe-agent/goose write a standalone temp file - remove that too.
        if not result.extra.get("dsh_home"):
            result.raw_log_path.unlink(missing_ok=True)

    for key in ("dsh_home", "scratch_dir"):
        if scratch_dir := result.extra.get(key):
            shutil.rmtree(scratch_dir, ignore_errors=True)


def _checkout_repo(repo: str, base_commit: str, workspace: Path) -> None:
    """Get `workspace` to a clean checkout of `repo` at `base_commit`.

    If the directory already exists (a prior attempt on this instance, e.g.
    a `redo_existing` rerun), reset it instead of reusing whatever state a
    previous harness run left behind - a harness that edited files without
    ever producing a captured patch (a real failure mode we hit with Loop
    Harness: it can leave real file edits on disk while printing nothing)
    would otherwise silently contaminate every later attempt on that same
    instance with its leftover, unrelated changes.
    """
    if workspace.exists():
        subprocess.run(["git", "-C", str(workspace), "reset", "--hard", base_commit], check=True)
        subprocess.run(["git", "-C", str(workspace), "clean", "-fdx"], check=True)
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
