"""wobbl uncertainty, validate-against, credibility and test-plan: the evidence a reviewer asks for."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui


def _result_dir(result_dir: str) -> Path:
    from .main import _latest_dir

    d = Path(result_dir) if result_dir else _latest_dir()
    if d is None or not (d / "outcome.json").exists():
        ui.fail(f"no Wobbl results in '{result_dir or 'wobbl-out'}'", "Pass the folder a run printed.")
    return d


@click.command()
@click.argument("result_dir", default="")
@click.option("--vary", "vary", multiple=True,
              help="An uncertain input and its scatter, e.g. 'load 10%', 'yield lognormal 8%', "
                   "'E uniform 70000..75000', 'thickness tol 0.2', 'film uniform 3..8' (repeatable).")
@click.option("--samples", type=int, default=200000, show_default=True, help="Samples drawn from the inputs.")
@click.option("--method", type=click.Choice(["lhs", "mc"]), default="lhs", show_default=True,
              help="Latin hypercube or plain Monte Carlo sampling.")
@click.option("--solves", type=int, default=None,
              help="New solves for the surrogate when an input changes the solve (default: twice the terms of a "
                   "degree 2 polynomial chaos).")
@click.option("--seed", type=int, default=1, show_default=True)
@click.option("--no-mesh-error", is_flag=True, help="Leave out the discretization error the checks measured.")
@click.option("--json", "as_json", is_flag=True)
def uncertainty(result_dir, vary, samples, method, solves, seed, no_mesh_error, as_json):
    """Carry input scatter through a finished run: distributions, probability of failure and Sobol indices.

    \b
    Load, stiffness and strength scatter in a strength run scale the converged
    solution exactly, so they need no new solve. Geometry parameters of a .py
    part, heat inputs and circuit values need new solves, and the answer comes
    from a polynomial chaos fit whose leave one out error is reported.
    """
    from ..credibility.models import ModelError
    from ..credibility.uncertainty import run_uncertainty
    from ..credibility.uq import VaryError

    d = _result_dir(result_dir)
    try:
        with ui.progress(not as_json) as cb:
            r = run_uncertainty(d, list(vary), samples, method, seed, solves, not no_mesh_error, cb)
    except (VaryError, ModelError) as e:
        ui.fail(str(e))
    if as_json:
        click.echo(json.dumps({k: v for k, v in r.items() if k != "histogram"}, indent=1))
        return
    show_uncertainty(r, d)


def show_uncertainty(r: dict, d: Path):
    from rich.table import Table

    from ..credibility import uq
    from ..units import fmt

    ui.console.print()
    ui.console.print(f"  {r['headline']}")
    ui.section("answer")
    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("output", "mean", "sd", "5%", "95%"):
        tb.add_column(c)
    for name, s in r["outputs"].items():
        tb.add_row(name.replace("_", " "), fmt(s["mean"]), fmt(s["sd"]), fmt(s["p05"]), fmt(s["p95"]))
    ui.console.print(tb)
    if r["probability"]:
        ui.section("probability")
        for k, p in r["probability"].items():
            ui.console.print(f"  {k:<13} {uq.odds(p)}  [dim]against {p['against']}[/dim]")
        for a in r.get("alternatives", []):
            ui.console.print(f"  [dim]with {a['label']}: {uq.odds(a['probability'])}[/dim]")
    ui.section("what matters most")
    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("input", "first order", "total"):
        tb.add_column(c)
    for x in r["sensitivity"]["indices"]:
        tb.add_row(x["input"], f"{x['first_order']:.3f}", f"{x['total']:.3f}")
    ui.console.print(tb)
    ui.console.print(f"\n  [dim]{r['model']['method']}[/dim]")
    ui.section("files")
    ui.console.print(f"  uncertainty  {d / 'uncertainty.json'}")
    ui.console.print(f"  report       {d / 'report.html'}\n")
