# eval-bench

Point any agent harness, at any model, at any eval benchmark — one YAML config per run.

```
model     -> which LLM, and how to auth it
harness   -> which agent drives the model — DeepSeek Harness, mini-swe-agent, Goose,
             Loop, or any harness of your own via a generic-cli command template or an import path
benchmark -> which suite grades the result (SWE-bench, Terminal-Bench)
```

`harness.name` is never a closed list: it's one of `evalbench.harnesses.BUILTIN_HARNESSES`
(`deepseek-harness`, `mini-swe-agent`, `goose`, `loop-harness`, `generic-cli`) **or** a
`module.path:ClassName` import path to a `Harness` subclass you wrote yourself,
resolved the same way Terminal-Bench resolves its own `--agent-import-path`. Nothing
in eval-bench's source needs editing to add a new one.

`evalbench run configs/deepseek-harness.swebench-lite.yaml` runs the whole thing end to end and prints a resolve rate.

Every run also persists each instance's full agent trace (every tool call, every model turn) to `runs/<run_id>/traces/<instance_id>.*` — not just the final patch. The harness's own scratch state (e.g. DeepSeek Harness's temp `dsh_home`, which otherwise leaks a full `node_modules` install per task) is cleaned up right after the trace is copied out, in `evalbench/benchmarks/swebench.py:_persist_trace`.

## Install

```bash
pip install -e ".[all]"   # or pick extras: [deepseek], [mini-swe-agent], [swebench], [terminalbench], [deepeval]
```

You'll also need, depending on what you run:
- **Docker** — both SWE-bench's grading harness and Terminal-Bench's task sandbox use it.
- **`DEEPSEEK_API_KEY`** (or provider-appropriate key) exported in your shell.
- The `swebench` CLI (`pip install swebench`) and/or `tb` CLI (`pip install terminal-bench`) on PATH — installed automatically by the `swebench`/`terminalbench` extras.

## Layout

```
evalbench/
  config.py                    # ModelConfig / HarnessConfig / BenchmarkConfig / RunConfig (pydantic)
  cli.py                       # `evalbench run <config.yaml>`
  harnesses/
    base.py                    # Harness ABC: solve(instruction, workspace, session_id) -> patch
    deepseek_harness.py        # real adapter for DeepSeek Harness's Python SDK
    mini_swe_agent.py          # real adapter for mini-swe-agent's Python bindings
    goose.py                   # real adapter for Goose's `goose run` headless CLI
    loop_harness.py            # real adapter for Soket AI's Loop (`loop --print`), feat/no-default-model branch
    generic_cli.py             # fallback: run any command template, capture `git diff`
  benchmarks/
    base.py                    # Benchmark ABC: execute() -> BenchmarkReport
    swebench.py                # drives the harness per-instance OR shells out to `swebench infer`,
                                # always grades via the real `swebench eval` CLI
    terminal_bench.py          # shells out to `tb run` with the right --agent / --agent-import-path
  terminal_bench_agents/       # Terminal-Bench "installed agent" plugins (its own extension point)
    deepseek_harness_agent.py  # installs+runs `dsh` inside the task container
    generic_agent.py           # generic version of the same, for a harness with no adapter yet
```

## Adding a harness

Two extension points, because SWE-bench and Terminal-Bench have fundamentally different execution models. Neither requires editing eval-bench's own source — that's what "harness-agnostic" means here.

1. **Patch-producing harnesses** (SWE-bench, and eval-bench's own generic loop): write a class anywhere on your PYTHONPATH that subclasses `evalbench.harnesses.base.Harness` and implements `solve(instruction, workspace, session_id) -> HarnessResult`, then point a config at it: `harness.name: mypackage.myharness:MyHarness`. It's resolved dynamically (`evalbench/harnesses/__init__.py:_import_harness_class`) exactly like Terminal-Bench resolves its own `--agent-import-path`. No PR to eval-bench needed. The four built-ins (`deepseek-harness`, `mini-swe-agent`, `goose`, `generic-cli`) are just names that skip that step.
2. **Terminal-Bench**: Terminal-Bench drives the agent itself inside a Docker container over tmux, so the class shape is different — subclass `terminal_bench.agents.installed_agents.abstract_installed_agent.AbstractInstalledAgent` instead (see `terminal_bench_agents/deepseek_harness_agent.py` for a worked example) and set `harness.name` directly to its import path; `evalbench/benchmarks/terminal_bench.py` passes any name it doesn't recognize as a built-in tb agent (`mini-swe-agent`, `goose`, `claude-code`, `aider`, `codex`, `openhands`) straight through to `tb run --agent-import-path`.

Don't have a real adapter yet for a harness? Use `harness.name: generic-cli` (SWE-bench side) and `harness.name: generic-cli` with `install_script`/`command` options (Terminal-Bench side) as a stand-in — see the docstrings in `generic_cli.py` / `generic_agent.py`. It just runs a command template and captures `git diff`, no Python class needed.

## Things worth double-checking before a big/expensive run

This was built by reading each project's actual source and docs rather than guessing, but a couple of spots are genuinely underdocumented upstream and worth a `--help` check the first time:

- **DeepSeek Harness model selection for `dsh --profile headless`**: the CLI reference only documents the task text as a positional argument for the `headless` profile — no `--model` flag. `deepseek_harness_agent.py` sets `DSH_MODEL` as a best-effort env var (confirmed to work for the SDK's own `minimal.py` example, not confirmed for the raw CLI) and also accepts an explicit `--patch` file if you've inspected `dsh --profile headless --dump-default-config` and know your adapter row's id.
- **`tb run` flag names** (`--agent-kwarg`, `--task-id`, `--n-concurrent`, etc.) were taken from the README's own example and the `Harness.__init__` constructor parameter names; run `tb run --help` once to confirm on your installed version.
- **`swebench infer` flags** beyond `-m`/`-o`/`-w` (shown in the SWE-bench README) — run `swebench infer --help` before relying on `--instance_ids`/filtering behavior we pass through.

## Example configs

- `configs/deepseek-harness.swebench-lite.yaml` — DSH's Python SDK generates patches for SWE-bench Lite; grading via `swebench eval`.
- `configs/mini-swe-agent.swebench-lite.yaml` — delegates entirely to `swebench infer` (which runs mini-swe-agent itself).
- `configs/deepseek-harness.terminal-bench.yaml` — DSH installed and run inside each Terminal-Bench task container.
- `configs/deepseek-harness.openrouter-gemma.yaml` — DSH driven by a Gemma model via OpenRouter instead of DeepSeek's own models.
- `configs/goose.swebench-lite.yaml` — Block's Goose agent, via its real `goose run` headless CLI.
- `configs/loop-harness.swebench-lite.yaml` — Soket AI's Loop, via its real `loop --print` headless mode (built by Loop itself for benchmark runners, with native Langfuse tracing).
- `configs/loop-harness.soket.swebench-deepeval.yaml` / `configs/deepseek-harness.swebench-deepeval.yaml` — grade with a DeepEval GEval LLM-judge instead of Docker (see below).

## Grading with an LLM judge instead of Docker

`swebench eval`'s real FAIL_TO_PASS/PASS_TO_PASS test run is the only *official* SWE-bench verdict, but it needs Docker and can be genuinely slow (a single instance's environment build took 30+ minutes under QEMU emulation on Apple Silicon in testing — see the ARM64 notes below). Set `benchmark.options.grading: deepeval` to instead score each patch with [DeepEval](https://deepeval.com)'s `GEval` metric — an LLM judge compares the generated patch against the dataset's own reference patch for functional correctness, no Docker involved. `grading: both` runs both and keeps them separate under `report.extra`.

```yaml
benchmark:
  options:
    grading: deepeval
    deepeval_judge:              # optional: defaults to the run's own model if omitted
      provider: openrouter
      name: anthropic/claude-sonnet-5
      api_key_env: OPENROUTER_API_KEY
    deepeval_threshold: 0.5
```

Use a judge model different from (and ideally stronger than) whichever model generated the patch — verified in testing that a weak judge (gpt-4o-mini judging its own kind of output) gives noisier, less decisive scores even on a clearly-correct patch. `evalbench/evaluators/litellm_judge.py` wraps any `ModelConfig` as a DeepEval-compatible judge via litellm.

Have a specific "loop harness" or other project in mind that isn't wired up here? Point me at its repo/docs and I'll build a real adapter the same way — or use the `module:Class` / `generic-cli` escape hatches above right now without waiting.
