"""wobbl thermal: steady and transient heat."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui


def show_generic(o):
    ui.console.print()
    ui.verdict_panel(o.verdict.upper(), o.verdict, o.headline, list(o.metrics), o.confidence, o.confidence_reasons)
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


def _run(path, material, power, heat_faces, films, ambient, fixed, duration, limit, radiation, out, question,
         as_json, quality, guesses=()):
    from ..thermal.study import run_thermal
    from ..thermal.tools import study_from
    from .main import _out_dir

    out_dir = _out_dir(path, out)
    try:
        st = study_from(path, material, power, heat_faces, list(films), ambient, list(fixed), duration, limit,
                        radiation, question, quality)
        with ui.progress(not as_json) as cb:
            for g in guesses:
                cb("assume", g)
            o = run_thermal(st, out_dir, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if guesses:
        o.assumptions = [f"Wobbl guessed: {g}." for g in guesses] + o.assumptions
        o.write(out_dir)
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


@click.command()
@click.argument("part")
@click.option("--material", "-m", required=True)
@click.option("--power", "-p", default="", help="Heat made inside the part, e.g. '12 W'.")
@click.option("--heat-faces", default="", help="Put the power on these faces instead of through the volume.")
@click.option("--film", "films", multiple=True, help="Convection, e.g. '5 W/m2K @ all' or '25 W/m2K at 30 C @ top'.")
@click.option("--ambient", default="25 C", show_default=True)
@click.option("--fixed-temp", "fixed", multiple=True, help="Held temperature, e.g. '40 C @ bottom'.")
@click.option("--duration", default="", help="Transient run length, e.g. '30 min'. Omit for steady state.")
@click.option("--limit", default="", help="Temperature limit, e.g. '60 C' (default: the material's).")
@click.option("--radiation", type=float, default=0.0, help="Surface emissivity (0 turns radiation off).")
@click.option("--quality", type=click.Choice(["fast", "normal"]), default="normal")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def thermal(part, material, power, heat_faces, films, ambient, fixed, duration, limit, radiation, quality, out,
            question, as_json):
    """Steady or transient heat: will it overheat?"""
    _run(part, material, power, heat_faces, films, ambient, fixed, duration, limit, radiation, out, question,
         as_json, quality)


def thermal_ask(part, question, kw):
    from ..ask import find_material
    from ..thermal.tools import read_thermal_question

    r = read_thermal_question(question)
    guesses = list(r.pop("guesses"))
    mat = kw.get("material") or find_material(question)
    if mat is None:
        mat = "6061-t6"
        guesses.append("material not mentioned: assumed aluminum 6061-T6 (override with --material)")
    if "power" not in r:
        ui.fail("the question has no heat load I can read; say it like '12 W' or pass --power")
    _run(part, mat, r.get("power", ""), "", [r["film"]], r.get("ambient", "25 C"), [], r.get("duration", ""),
         r.get("limit", ""), 0.0, kw.get("out"), question, kw.get("as_json"), "normal", guesses)
