"""wobbl circuit: any circuit (a SPICE netlist, a KiCad schematic or a description in words), with the LED, RC,
buck and MOSFET templates as shortcuts."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui
from .thermal import show_generic


def _run(target: str, params: dict, probes, out, question, as_json, guesses=(), physics: str = "ratings"):
    from ..circuits.study import simulate_circuit
    from .main import _out_dir

    name = target if len(target) < 60 and "\n" not in target else "circuit"
    out_dir = _out_dir(name if Path(target).exists() or len(target.split()) == 1 else "circuit", out)
    said = list(guesses)
    try:
        with ui.progress(not as_json) as cb:
            for g in guesses:
                cb("assume", g)

            def assume(text, override=""):
                said.append(text + (f" (override with {override})" if override else ""))
                cb("assume", said[-1])

            cb("solve", "running ngspice")
            o = simulate_circuit(target, question, params, out_dir, list(probes), physics,
                                 log=lambda stage, msg: cb(stage, msg), assume=assume)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    if said:
        o.assumptions = [f"Wobbl guessed: {g}." for g in said] + [a for a in o.assumptions if "guessed" not in a]
        o.write(out_dir)
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


@click.command()
@click.argument("target")
@click.option("--set", "sets", multiple=True,
              help="A value: a template's (supply=5V), a part's (R1=470, R1.power=1W, C3.voltage=25V, J2.load=2A) or "
                   "the ambient (ambient=40C).")
@click.option("--probe", "probes", multiple=True, help="A signal to report, e.g. 'v(out)'.")
@click.option("--physics", type=click.Choice(["ratings", "operating-point", "transient", "ac", "tolerance"]),
              default="ratings", show_default=True, help="Which analysis leads; every part is judged in each.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def circuit(target, sets, probes, physics, out, question, as_json):
    """Any circuit: a SPICE netlist, a KiCad schematic, words, or the led, rc, buck and mosfet templates.

    \b
    wobbl circuit board.kicad_sch --set J2.load=2A
    wobbl circuit amp.cir --probe 'v(out)'
    wobbl circuit "three red LEDs in parallel from a 9 V battery, each with 330 ohm"
    wobbl circuit led --set supply=5V --set resistor=100 --set led=blue
    wobbl circuit buck --set vin=12 --set vout=3.3 --set current=2 --set capacitor=10u
    """
    params = {}
    for s in sets:
        if "=" not in s:
            ui.fail(f"--set needs key=value, got '{s}'")
        k, v = s.split("=", 1)
        params[k.strip()] = v.strip()
    _run(target, params, probes, out, question, as_json, physics=physics)


def circuit_ask(part, question, kw):
    """A plain question: about a file, or a circuit described in the question itself."""
    from ..circuits.domain import template_fit
    from ..circuits.tools import read_circuit_question

    if Path(part).exists():
        return _run(part, {}, (), kw.get("out"), question, kw.get("as_json"))
    try:
        tpl, params, guesses = read_circuit_question(question)
    except ValueError:
        tpl, params, guesses = "", {}, []
    if tpl and template_fit(tpl, question):
        return _run(tpl, params, (), kw.get("out"), question, kw.get("as_json"), guesses)
    _run(question, {}, (), kw.get("out"), question, kw.get("as_json"))
