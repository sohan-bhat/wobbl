"""wobbl motor, wobbl flame and wobbl propellant: solid rocket motors, flame temperatures and propellant performance."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui
from .thermal import show_generic


def _finish(o, as_json):
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


@click.command()
@click.argument("spec", required=False, default="")
@click.option("--propellant", "-p", default="", help="knsb, kndx, knsu, ocean-water or cherry-limeade.")
@click.option("--grain", "-g", "grains", multiple=True,
              help="A grain, forward end first, e.g. 'bates d=48 core=18 L=80 x4' (mm). Repeatable.")
@click.option("--throat", default="", help="Nozzle throat diameter, e.g. '13 mm' (a bare number is mm).")
@click.option("--exit", "exit_", default="", help="Nozzle exit diameter.")
@click.option("--efficiency", type=float, default=None, help="Nozzle efficiency, 0.85 to 0.95 (default 0.90).")
@click.option("--casing", default="", help="Casing, e.g. '6061-t6 od=54 wall=1.6 L=400 sf=2' (mm).")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def motor(spec, propellant, grains, throat, exit_, efficiency, casing, out, question, as_json):
    """Solid rocket motor: thrust curve, chamber pressure against the casing, Kn and erosive burning.

    \b
    wobbl motor motor.toml
    wobbl motor --propellant knsb --grain "bates d=48 core=18 L=80 x4" --throat 13 --exit 33 \\
        --casing "6061-t6 od=54 wall=1.6 L=400"

    Writes thrust.csv and a RASP motor.eng next to the report; a rocket spec for
    'wobbl motion' can name the design with design = "motor.toml" under [motor].
    """
    from ..chem.motor import run_motor
    from ..chem.spec import read_spec, spec_from_flags
    from .main import _out_dir

    try:
        source = spec or spec_from_flags(propellant, list(grains), throat, exit_, casing, efficiency)
        ms = read_spec(source, question)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    out_dir = _out_dir(spec or ms.name, out)
    try:
        with ui.progress(not as_json) as cb:
            o = run_motor(ms, out_dir, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    _finish(o, as_json)
    if not as_json:
        d = Path(o.workdir)
        ui.console.print(f"  thrust curve {d / o.values['files']['thrust_csv']}\n"
                         f"  motor file   {d / o.values['files']['eng']}\n")


@click.command()
@click.argument("fuel")
@click.option("--oxidizer", default="air", show_default=True, help="air or oxygen.")
@click.option("--phi", type=float, default=1.0, show_default=True, help="Equivalence ratio, fuel over stoichiometric.")
@click.option("--pressure", default="1 atm", show_default=True)
@click.option("--temperature", default="25 C", show_default=True, help="Temperature of the fuel and oxidizer.")
@click.option("--limit", default="", help="Temperature the flame must stay under, e.g. '1800 C'.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def flame(fuel, oxidizer, phi, pressure, temperature, limit, out, question, as_json):
    """Adiabatic flame temperature and equilibrium products of a fuel in air or oxygen.

    \b
    wobbl flame methane --phi 1
    wobbl flame propane --phi 0.9 --limit "1800 C"
    """
    from ..chem.combustion import run_flame
    from ..units import parse_quantity
    from .main import _out_dir

    try:
        P = parse_quantity(pressure, "stress").value * 1e6
        T = parse_quantity(temperature, "temperature").value + 273.15
        lim = parse_quantity(limit, "temperature").value + 273.15 if limit else None
        if phi <= 0:
            raise ValueError("--phi has to be above zero")
        with ui.progress(not as_json) as cb:
            cb("equilibrium", f"{fuel} in {oxidizer} at phi {phi:g}")
            o = run_flame(fuel, oxidizer, phi, T, P, _out_dir(f"{fuel}-flame", out), question, lim)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    _finish(o, as_json)


@click.command()
@click.argument("propellant")
@click.option("--pressure", "chamber", default="6.895 MPa", show_default=True, help="Chamber pressure.")
@click.option("--area-ratio", type=float, default=None, help="Nozzle exit to throat area ratio.")
@click.option("--exit-pressure", default="", help="Expand to this pressure instead (default the ambient).")
@click.option("--ambient", default="1 atm", show_default=True)
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def propellant(propellant, chamber, area_ratio, exit_pressure, ambient, out, question, as_json):
    """Rocket propellant performance: c* and Isp, shifting and frozen, CEA style.

    \b
    wobbl propellant knsb --pressure "5 MPa" --area-ratio 8
    wobbl propellant "ap:70,al:16,htpb:14" --pressure "6.9 MPa" --area-ratio 10
    wobbl propellant "lox:6,lh2:1" --pressure "1000 psi" --area-ratio 40 --ambient "0 Pa"
    """
    from ..chem.combustion import run_propellant
    from ..units import parse_quantity
    from .main import _out_dir

    try:
        pc = parse_quantity(chamber, "stress").value * 1e6
        pa = parse_quantity(ambient, "stress").value * 1e6
        pe = parse_quantity(exit_pressure, "stress").value * 1e6 if exit_pressure else None
        if area_ratio is not None and area_ratio < 1:
            raise ValueError("--area-ratio is the exit area over the throat area, at least 1")
        name = propellant.replace(",", "-").replace(":", "") if ":" in propellant else propellant
        with ui.progress(not as_json) as cb:
            cb("equilibrium", f"{propellant} at {chamber}")
            o = run_propellant(propellant, pc, area_ratio, pe, _out_dir(name, out), question, pa)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    _finish(o, as_json)


@click.command()
@click.option("--cell", default="18650", show_default=True,
              help="18650, 21700, 26650 or 14500, or a size like 18x65 mm.")
@click.option("--ambient", default="25 C", show_default=True, help="Temperature around the cell.")
@click.option("--film", default="7.17 W/m2K", show_default=True,
              help="Cooling on the cell, e.g. '25 W/m2K', or 'sealed box', 'open air', 'a small fan'.")
@click.option("--emissivity", type=float, default=0.8, show_default=True)
@click.option("--from", "from_", default="", help="A heat result folder whose hottest temperature surrounds the cell.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def runaway(cell, ambient, film, emissivity, from_, out, question, as_json):
    """Lithium ion thermal runaway onset: when a cell's self heating outruns its cooling.

    \b
    wobbl runaway --cell 18650 --ambient "60 C"
    wobbl runaway --cell 21700 --film "sealed box" --from wobbl-out/pack
    """
    from ..chem.runaway import cell_from, run_runaway, surroundings_from
    from ..units import parse_quantity
    from .main import _out_dir

    try:
        c = cell_from(cell, film, emissivity)
        source = ""
        if from_:
            Ta, source = surroundings_from(from_)
        else:
            Ta = parse_quantity(ambient, "temperature").value + 273.15
        with ui.progress(not as_json) as cb:
            cb("kinetics", f"{c.label} cell around {Ta - 273.15:.4g} C")
            o = run_runaway(c, Ta, _out_dir(f"{cell}-runaway", out), question, source)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e), getattr(e, "hint", ""))
    _finish(o, as_json)
