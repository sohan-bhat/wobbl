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
    """Carry input scatter through a finished run of any domain: distributions, probability of failure and Sobol
    indices.

    \b
    Any number the run was made from can scatter, by the name the error
    message lists. The run is solved again at a design of values and fitted
    with polynomial chaos, whose leave one out error is reported. Load,
    stiffness and strength scatter in a strength run scale the converged
    solution exactly instead, and heat solves again on its converged mesh.
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


@click.command("validate-against")
@click.argument("result_dir")
@click.argument("tests")
@click.option("--vary", "vary", multiple=True,
              help="Scatter of an input the test article carries, e.g. 'E 5%' or 'film uniform 3..8', on top of "
                   "any in uncertainty.json (repeatable).")
@click.option("--load-uncertainty", default="1%", show_default=True,
              help="Standard uncertainty of the test load, as a percentage of it.")
@click.option("--solves", type=int, default=None, help="New solves for the surrogate of heat and circuit runs.")
@click.option("--no-calibrate", is_flag=True, help="Compare only, without calibrating the uncertain inputs.")
@click.option("--json", "as_json", is_flag=True)
def validate_against(result_dir, tests, vary, load_uncertainty, solves, no_calibrate, as_json):
    """Compare a finished run of any domain with physical test readings as ASME V&V 20 does, with the area metric,
    and calibrate the uncertain inputs from them.

    \b
    A reading's quantity is any value the run reports (such as apogee_m or
    t_max_C), a column of a time series it wrote read at the reading's time,
    or a gauge on a strength run (strain, displacement) or a heat run
    (temperature). CSV columns: name, quantity, value, unit, uncertainty,
    x, y, z, direction, load, time, gauge_length. A note line (# ...) that
    says the readings are synthetic marks every result as synthetic.
    """
    from ..credibility.against import ModelError, ReadingError, run_validation
    from ..credibility.uq import VaryError

    d = _result_dir(result_dir)
    if not Path(tests).exists():
        ui.fail(f"file not found: {tests}")
    try:
        lu = float(load_uncertainty.strip().rstrip("%")) / 100.0
    except ValueError:
        ui.fail(f"--load-uncertainty takes a percentage, like 1%, not '{load_uncertainty}'")
    try:
        with ui.progress(not as_json) as cb:
            r = run_validation(d, tests, list(vary), lu, solves, not no_calibrate, cb)
    except (VaryError, ModelError, ReadingError) as e:
        ui.fail(str(e))
    if as_json:
        out = {k: v for k, v in r.items() if k != "calibration"}
        if r.get("calibration"):
            out["calibration"] = {k: v for k, v in r["calibration"].items() if k != "draws"}
        click.echo(json.dumps(out, indent=1))
        return
    show_validation(r, d)


def show_validation(r: dict, d: Path):
    from rich.table import Table

    from ..credibility.report import _reading
    from ..units import fmt

    ui.console.print()
    ui.console.print(f"  {r['headline']}")
    ui.section("readings")
    tb = Table(box=None, header_style="dim", padding=(0, 2))
    for c in ("gauge", "reading", "simulation", "E", "u_val", ""):
        tb.add_column(c)
    for x in r["readings"]:
        q = x["quantity"]
        tb.add_row(x["name"], _reading(x["value"], q), _reading(x["simulation"], q), _reading(x["error"], q),
                   _reading(x["u_val"], q), "[green]agrees[/green]" if x["within"] else "[yellow]differs[/yellow]")
    ui.console.print(tb)
    cal = r.get("calibration")
    if cal:
        ui.section("calibration")
        for c in cal["inputs"]:
            ui.console.print(f"  {c['input']:<12} {fmt(c['prior_mean'])} -> {fmt(c['mean'])} {c['unit']}  "
                             f"[dim]95% {fmt(c['ci95'][0])} to {fmt(c['ci95'][1])}, {c['narrowed']:.2g}x "
                             "narrower[/dim]")
    for n in r.get("notes", []):
        ui.console.print(f"  [dim]{n}[/dim]")
    ui.section("files")
    ui.console.print(f"  validation   {d / 'validation.json'}")
    ui.console.print(f"  report       {d / 'report.html'}\n")


LEVEL = click.Choice(["low", "medium", "high"])


@click.command()
@click.argument("result_dir", default="")
@click.option("--influence", type=LEVEL, default="medium", show_default=True,
              help="How much the decision leans on the model rather than on other evidence.")
@click.option("--consequence", type=LEVEL, default="medium", show_default=True,
              help="How bad a wrong decision would be.")
@click.option("--question", default="", help="The question the model answers (default: the run's).")
@click.option("--suite", default=None, help="results.json from 'wobbl validate', for code verification.")
@click.option("--json", "as_json", is_flag=True)
def credibility(result_dir, influence, consequence, question, suite, as_json):
    """Assess a run's credibility in the manner of ASME V&V 40 and write credibility.html for a reviewer.

    \b
    The model risk comes from the model's influence on the decision and the
    decision's consequence. Each factor (code and solution verification, model
    form, inputs, the tests, the comparison, applicability and the answer's own
    uncertainty) gets a level from the evidence next to the run and a proposed
    goal. Wobbl supplies the evidence and does not certify anything.
    """
    from ..credibility.assess import run_credibility
    from ..credibility.models import ModelError

    d = _result_dir(result_dir)
    try:
        r = run_credibility(d, influence, consequence, question, suite)
    except ModelError as e:
        ui.fail(str(e))
    if as_json:
        click.echo(json.dumps(r, indent=1))
        return
    from rich.table import Table

    ui.console.print(f"\n  {r['headline']}")
    ui.section("factors")
    tb = Table(box=None, header_style="dim", padding=(0, 1))
    for c in ("factor", "level", "goal", ""):
        tb.add_column(c)
    for f in r["factors"]:
        tb.add_row(f["factor"], f["level"], f["goal"],
                   "[green]meets[/green]" if f["meets"] else "[yellow]short[/yellow]")
    ui.console.print(tb)
    ui.console.print(f"\n  [dim]{r['disclaimer']}[/dim]")
    _show_plan(d)
    ui.section("files")
    ui.console.print(f"  credibility  {d / 'credibility.html'}")
    ui.console.print(f"  report       {d / 'report.html'}\n")


@click.command("test-plan")
@click.argument("result_dir", default="")
@click.option("--json", "as_json", is_flag=True)
def test_plan(result_dir, as_json):
    """The smallest set of physical tests that would most reduce the risk, with gauge positions, loads and
    predicted readings, and which usual tests the simulation evidence can or cannot stand in for."""
    from ..credibility.report import refresh
    from ..credibility.testplan import plan
    from ..credibility.uncertainty import _clean

    d = _result_dir(result_dir)
    r = _clean(plan(d))
    (d / "test-plan.json").write_text(json.dumps(r, indent=1))
    refresh(d)
    if as_json:
        click.echo(json.dumps(r, indent=1))
        return
    _show_plan(d)
    ui.console.print(f"\n  plan  {d / 'test-plan.json'}\n")


def _show_plan(d: Path):
    r = json.loads((d / "test-plan.json").read_text())
    ui.section("test plan")
    ui.console.print(f"  {r['headline']}")
    for t in r["tests"]:
        ui.console.print(f"\n  [bold]{t['order']}. {t['kind']}[/bold]  {t['what'].rstrip('.')}.")
        ui.console.print(f"  [dim]{t['why']}[/dim]")
    ui.console.print()
    for x in r["replace"]:
        style = {"likely": "green", "not yet": "yellow", "cannot": "red"}[x["verdict"]]
        ui.console.print(f"  [{style}]{x['verdict']:<8}[/{style}] {x['test']}  [dim]{x['why']}[/dim]")
