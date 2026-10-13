"""wobbl circuit: LED, RC, buck ripple and MOSFET heating checks, or any netlist."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui
from .thermal import show_generic


def _run(target: str, params: dict, probes, out, question, as_json, guesses=()):
    from ..circuits.run import run_circuit, run_netlist
    from .main import _out_dir

    out_dir = _out_dir(target, out)
    try:
        with ui.progress(not as_json) as cb:
            for g in guesses:
                cb("assume", g)
            cb("solve", "running ngspice")
            # a template name stays a template even when a folder of that name (its own output) sits here
            if Path(target).is_file():
                o = run_netlist(target, list(probes), out_dir, question)
            else:
                o = run_circuit(target, params, out_dir, question)
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
@click.argument("target")
@click.option("--set", "sets", multiple=True, help="Template parameter, e.g. --set supply=5V --set resistor=100.")
@click.option("--probe", "probes", multiple=True, help="Signal to read from a netlist, e.g. 'v(out)'.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def circuit(target, sets, probes, out, question, as_json):
    """Circuit checks: led, rc, buck, mosfet, or a SPICE netlist file.

    \b
    wobbl circuit led --set supply=5V --set resistor=100 --set led=blue
    wobbl circuit buck --set vin=12 --set vout=3.3 --set current=2 --set capacitor=10u
    wobbl circuit mosfet --set current=15 --set rds_on=8m --set theta_ja=40
    wobbl circuit board.cir --probe 'v(out)'
    """
    params = {}
    for s in sets:
        if "=" not in s:
            ui.fail(f"--set needs key=value, got '{s}'")
        k, v = s.split("=", 1)
        params[k.strip()] = v.strip()
    _run(target, params, probes, out, question, as_json)


def circuit_ask(part, question, kw):
    from ..circuits.tools import read_circuit_question

    if Path(part).exists() and part.lower().endswith((".cir", ".sp", ".spice", ".net")):
        ui.fail("for a netlist, say which signals to read: wobbl circuit board.cir --probe 'v(out)'")
    try:
        tpl, params, guesses = read_circuit_question(question)
    except ValueError as e:
        ui.fail(str(e))
    _run(tpl, params, (), kw.get("out"), question, kw.get("as_json"), guesses)
