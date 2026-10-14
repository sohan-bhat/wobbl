"""wobbl plastic, wobbl contact and wobbl bolt: past first yield, parts that touch, and bolted joints."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui

STYLE = {"holds": "holds", "yields": "marginal", "marginal": "marginal", "collapses": "fails", "fails": "fails",
         "slips": "fails", "separates": "fails"}


def show(o):
    ui.console.print()
    ui.verdict_panel(o.verdict.upper(), STYLE.get(o.verdict, o.verdict), o.headline, list(o.metrics)[:6],
                     o.confidence, o.confidence_reasons)
    ui.section("checks")
    ui.checks_table(o.checks)
    if o.fixes:
        ui.section("how to fix it")
        for f in o.fixes:
            ui.console.print(f"  [bold]{f['title']}[/bold]")
            ui.console.print(f"  [dim]{f['summary']}[/dim]")
    ui.section("assumptions")
    ui.bullets(o.assumptions)
    ui.section("files")
    for k, v in o.images.items():
        ui.console.print(f"  {k:<12} {v}")
    ui.console.print(f"  report       {o.report}\n")


def _finish(o, as_json):
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show(o)


def _part_options(f):
    opts = [
        click.option("--material", "-m", required=True, help="A metal from 'wobbl materials', or custom:E=..,yield=.."),
        click.option("--load", "-l", "loads", multiple=True, help="Load, as in wobbl run, e.g. '3 kN -z @ x=max'."),
        click.option("--fix", "-f", "fixes", multiple=True, help="Support, as in wobbl run, e.g. 'bolt holes d=8.5'."),
        click.option("--quality", type=click.Choice(["fast", "normal", "fine"]), default="normal"),
        click.option("--scale", type=float, default=1.0, help="Scale the geometry (1000 for files in meters)."),
        click.option("--param", "params", multiple=True, help="Parameter of a .py part, e.g. thickness=6."),
        click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<part>)."),
        click.option("--question", default="", help="The question this run answers (shown in the report)."),
        click.option("--keep", is_flag=True, help="Keep the solver's work files next to the result."),
        click.option("--json", "as_json", is_flag=True, help="Print the result as JSON."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


@click.command()
@click.argument("part")
@_part_options
@click.option("--target", type=float, default=None,
              help="Margin wanted on the collapse load (default the material's safety factor, 1.5 for most metals).")
@click.option("--curve", default="auto",
              help="Past yield: auto (the material's true curve), perfect (flat at yield), 'true 250@0, 300@0.02' "
                   "(true stress MPa at plastic strain), 'engineering 250@0.00125, 400@0.2' or a CSV file.")
@click.option("--body", type=int, default=None, help="Pick one body from a multi body file.")
@click.option("--playback", is_flag=True, help="Also write playback.json with the load going on and off.")
def plastic(part, material, loads, fixes, quality, scale, params, out, question, keep, as_json, target, curve, body,
            playback):
    """Past first yield: permanent set after unloading and the load that collapses a metal part.

    \b
    wobbl plastic hook.py -m mild-steel --fix "bolt holes d=8.5" --load "3 kN -z @ hole d=20"
    wobbl plastic bracket.step -m 6061-t6 --fix "x=min" --load "1.5 kN down @ x=max" --target 2
    """
    from ..plastic.study import PlasticStudy, run_plastic
    from .main import _out_dir

    if not loads or not fixes:
        ui.fail("wobbl plastic needs at least one --load and one --fix",
                "Example: wobbl plastic hook.step -m mild-steel --fix 'holes d=8.5' --load '3 kN down @ top'")
    out_dir = _out_dir(part, out)
    try:
        st = PlasticStudy(str(Path(part).resolve()), material, list(loads), list(fixes), target, curve, quality,
                          question, scale, body, dict(p.split("=", 1) for p in params if "=" in p), keep,
                          playback)
        with ui.progress(not as_json) as cb:
            o = run_plastic(st, out_dir, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    _finish(o, as_json)
