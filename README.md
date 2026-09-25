# eval-bench

Point any agent harness, at any model, at any eval benchmark — one YAML config per run.

```
model     -> which LLM, and how to auth it
harness   -> which agent drives the model (DeepSeek Harness, mini-swe-agent, or a generic CLI you plug in)
benchmark -> which suite grades the result (SWE-bench, Terminal-Bench)
```

`evalbench run configs/deepseek-harness.swebench-lite.yaml` runs the whole thing end to end and prints a resolve rate.

## Install

```bash
pip install -e ".[all]"   # or pick extras: [deepseek], [mini-swe-agent], [swebench], [terminalbench]
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

Two extension points, because SWE-bench and Terminal-Bench have fundamentally different execution models:

1. **Patch-producing harnesses** (SWE-bench, and eval-bench's own generic loop): subclass `evalbench.harnesses.base.Harness` and implement `solve(instruction, workspace, session_id) -> HarnessResult`. Register it in `evalbench/harnesses/__init__.py`.
2. **Terminal-Bench**: Terminal-Bench drives the agent itself inside a Docker container over tmux. Subclass `terminal_bench.agents.installed_agents.abstract_installed_agent.AbstractInstalledAgent` (see `terminal_bench_agents/deepseek_harness_agent.py` for a worked example) and point `tb run` at it with `--agent-import-path module:Class`. Wire the harness name to that import path in `evalbench/benchmarks/terminal_bench.py`'s `_IMPORT_PATH_AGENTS`.

Don't have a real adapter yet for a harness? Use `harness.name: generic-cli` (SWE-bench side) and `harness.name: generic-cli` with `install_script`/`command` options (Terminal-Bench side) as a stand-in — see the docstrings in `generic_cli.py` / `generic_agent.py`.

## Things worth double-checking before a big/expensive run

This was built by reading each project's actual source and docs rather than guessing, but a couple of spots are genuinely underdocumented upstream and worth a `--help` check the first time:

- **DeepSeek Harness model selection for `dsh --profile headless`**: the CLI reference only documents the task text as a positional argument for the `headless` profile — no `--model` flag. `deepseek_harness_agent.py` sets `DSH_MODEL` as a best-effort env var (confirmed to work for the SDK's own `minimal.py` example, not confirmed for the raw CLI) and also accepts an explicit `--patch` file if you've inspected `dsh --profile headless --dump-default-config` and know your adapter row's id.
- **`tb run` flag names** (`--agent-kwarg`, `--task-id`, `--n-concurrent`, etc.) were taken from the README's own example and the `Harness.__init__` constructor parameter names; run `tb run --help` once to confirm on your installed version.
- **`swebench infer` flags** beyond `-m`/`-o`/`-w` (shown in the SWE-bench README) — run `swebench infer --help` before relying on `--instance_ids`/filtering behavior we pass through.

## Example configs

- `configs/deepseek-harness.swebench-lite.yaml` — DSH's Python SDK generates patches for SWE-bench Lite; grading via `swebench eval`.
- `configs/mini-swe-agent.swebench-lite.yaml` — delegates entirely to `swebench infer` (which runs mini-swe-agent itself).
- `configs/deepseek-harness.terminal-bench.yaml` — DSH installed and run inside each Terminal-Bench task container.
