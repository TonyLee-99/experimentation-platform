"""Command-line interface: ``expkit <command>``."""

from __future__ import annotations

import json

import typer

app = typer.Typer(add_completion=False, help="Experimentation analysis platform (stats engine + scorecards).")


@app.command()
def data() -> None:
    """Download the real datasets and print row counts (acceptance D1-D3)."""
    from expkit.analyses.common import write_json
    from expkit.data import disk_usage_bytes, prepare_all

    rows = [s.__dict__ for s in prepare_all()]
    write_json("datasets", {"datasets": rows, "data_dir_bytes": disk_usage_bytes()})
    typer.echo(json.dumps(rows, indent=2))


@app.command()
def cupac(force: bool = False) -> None:
    """Fit cross-fitted CUPAC covariates and the predicted-exposure trigger."""
    from expkit.analyses.cupac import fit_cupac

    typer.echo(json.dumps({k: v for k, v in fit_cupac(force).items() if k != "params"}, indent=2))


@app.command()
def analyze(config: str = typer.Option(..., "--config", "-c", help="experiment YAML")) -> None:
    """Run the full scorecard for one experiment and write the HTML readout."""
    from expkit.config import ExperimentConfig
    from expkit.report.scorecard import analyze as run_analysis
    from expkit.report.scorecard import render

    cfg = ExperimentConfig.from_yaml(config)
    readout = run_analysis(cfg)
    path = render(readout, cfg.output)
    if cfg.results_json:
        from expkit.analyses.common import write_json
        from expkit.report.scorecard import summary_json

        write_json(cfg.results_json, summary_json(readout))
    typer.echo(f"{readout.decision.verdict} [{readout.decision.rule}] {readout.decision.reason}")
    typer.echo(f"wrote {path}")


@app.command()
def validate(
    suite: str = typer.Argument("all", help="all | crosscheck | aa | asos | noncompliance | heavy | hte | balance"),
) -> None:
    """Run validation suites / analyses and write results/*.json."""
    from expkit.analyses import asos, balance, heavy_tails, hte, noncompliance
    from expkit.validation import aa, crosscheck

    suites = {
        "crosscheck": crosscheck.main,
        "aa": aa.main,
        "asos": asos.main,
        "noncompliance": noncompliance.main,
        "heavy": heavy_tails.main,
        "hte": hte.main,
        "balance": balance.main,
    }
    for name, fn in suites.items():
        if suite in ("all", name):
            typer.echo(f"== {name}")
            fn()


@app.command()
def docs() -> None:
    """Render README.md and docs/*.md from docs/templates and results/*.json."""
    from expkit.docs import render_all

    for path in render_all():
        typer.echo(f"wrote {path}")


@app.command()
def acceptance() -> None:
    """Evaluate every acceptance criterion from results/*.json (writes results/acceptance.json)."""
    from expkit.validation.acceptance import evaluate, update_doc

    rows = evaluate()
    update_doc(rows)
    for row in rows:
        typer.echo(f"{row['id']:4s} {row['status']:7s} {row['evidence']}")


if __name__ == "__main__":
    app()
