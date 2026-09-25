from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from evalbench.benchmarks import build_benchmark
from evalbench.config import EvalBenchConfig

app = typer.Typer(add_completion=False, help="Point any harness at any benchmark, with any model.")
console = Console()


@app.command()
def run(config: str = typer.Argument(..., help="Path to a run config YAML file.")):
    """Run one harness x model x benchmark combination end to end."""
    cfg = EvalBenchConfig.from_yaml(config)

    console.print(
        f"[bold]run[/bold]        {cfg.run.run_id}\n"
        f"[bold]model[/bold]      {cfg.model.provider}/{cfg.model.name}\n"
        f"[bold]harness[/bold]    {cfg.harness.name}\n"
        f"[bold]benchmark[/bold]  {cfg.benchmark.name}"
    )

    benchmark = build_benchmark(cfg.model, cfg.harness, cfg.benchmark, cfg.run)
    report = benchmark.execute()

    table = Table(title=f"Results: {report.run_id}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("benchmark", report.benchmark)
    table.add_row("total", str(report.total))
    table.add_row("resolved", str(report.resolved))
    table.add_row("resolve rate", f"{report.resolve_rate:.1%}")
    if report.details_path:
        table.add_row("details", str(report.details_path))
    console.print(table)


@app.command()
def validate(config: str = typer.Argument(..., help="Path to a run config YAML file.")):
    """Parse and print a run config without executing anything."""
    cfg = EvalBenchConfig.from_yaml(config)
    console.print_json(json.dumps(cfg.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    app()
