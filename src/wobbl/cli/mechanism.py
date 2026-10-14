"""wobbl mechanism: robot arms, linkages, gear trains, impacts and tipping, with MuJoCo."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui


def show_mechanism(o):
    ui.console.print()
    ui.verdict_panel(o.verdict.upper(), o.verdict, o.headline, list(o.metrics), o.confidence, o.confidence_reasons)
    ui.section("checks")
    ui.checks_table(o.checks)
    if o.events:
        ui.section("events")
        for e in o.events:
            ui.console.print(f"  {e['t_s']:8.3f} s  {e['name']}" + (f"  [dim]{e['detail']}[/dim]" if e.get("detail")
                                                                    else ""))
    if o.fixes:
        ui.section("how to fix it")
        for f in o.fixes:
            ui.console.print(f"  [bold]{f['title']}[/bold]")
            ui.console.print(f"  [dim]{f['summary']}[/dim]")
    ui.section("assumptions")
    ui.bullets(o.assumptions)
    ui.section("files")
    for k, v in o.images.items():
        ui.console.print(f"  {k:<13} {v}")
    ui.console.print(f"  trajectory    {Path(o.workdir) / 'trajectory.csv'}")
    ui.console.print(f"  report        {o.report}\n")


def _run(target: str, params: dict, out, question: str, as_json: bool, guesses=()):
    from ..mechanism.study import run_mechanism
    from .main import _out_dir

    out_dir = _out_dir(target, out)
    try:
        with ui.progress(not as_json) as cb:
            for g in guesses:
                cb("assume", g)
            o = run_mechanism(target, params, out_dir, question, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if guesses:
        o.assumptions = [g[0].upper() + g[1:] + "." for g in guesses] + o.assumptions
        o.write(out_dir)
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_mechanism(o)


@click.command()
@click.argument("target")
@click.option("--set", "sets", multiple=True,
              help="Template value, e.g. --set payload=500g --set servo=mg996r. Bare numbers are mm, g, degrees, "
                   "N m, rpm and s.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def mechanism(target, sets, out, question, as_json):
    """Mechanisms with MuJoCo: arm, fourbar, gears, impact, tip, pendulum, a spec file or an MJCF or URDF model.

    \b
    wobbl mechanism arm --set servo=mg996r --set payload=500g
    wobbl mechanism fourbar --set crank=30 --set coupler=125 --set rocker=45 --set ground=120
    wobbl mechanism gears --set teeth=12:36,10:40 --set motor=fa-130 --set load=0.005
    wobbl mechanism impact --set mass=200g --set drop=500mm --set material=pla --set limit=300
    wobbl mechanism tip --set payload=500g --set reach=120 --set force=3
    wobbl mechanism robot.toml
    """
    params = {}
    for s in sets:
        if "=" not in s:
            ui.fail(f"--set needs key=value, got '{s}'")
        k, v = s.split("=", 1)
        params[k.strip()] = v.strip()
    _run(target, params, out, question, as_json)


def mechanism_ask(part, question, kw):
    from ..mechanism.tools import read_mechanism_question

    if part not in ("-", "") and Path(part).exists():
        return _run(part, {}, kw.get("out"), question, kw.get("as_json"))
    try:
        tpl, params, guesses = read_mechanism_question(question)
    except ValueError as e:
        ui.fail(str(e))
    _run(tpl, params, kw.get("out"), question, kw.get("as_json"), guesses)
