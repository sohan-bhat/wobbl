"""wobbl flow: air and water flow, from pipes and ducts to drag, fan cooling, propellers and hulls."""
from __future__ import annotations

import json
import re

import click

from . import ui
from .thermal import show_generic


def _finish(o, as_json: bool):
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


def _common(f):
    opts = [click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<name>)."),
            click.option("--question", default="", help="The question this run answers (shown in the report)."),
            click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")]
    for o in reversed(opts):
        f = o(f)
    return f


def _q(text, kind, what):
    from ..flow.fluids import qty

    try:
        return qty(text, kind)
    except ValueError as e:
        ui.fail(f"could not read {what} '{text}': {e}")


@click.group()
def flow():
    """Air and water flow through pipes and around bodies, fan cooling, propellers and hulls.

    \b
    wobbl flow prop apcsf_10x4.7 --mass 1.2kg --rotors 4
    wobbl flow hull --length 12m --beam 3.6m --draft 0.9m --displacement 16t --speed 8kn
    """


@flow.command()
@click.argument("prop", default="apcsf_10x4.7")
@click.option("--rpm", default="", help="Speed to run at, e.g. 6000.")
@click.option("--thrust", default="", help="Thrust wanted from each rotor, e.g. '4 N'.")
@click.option("--mass", default="", help="Mass of the whole craft to hover, e.g. '1.2 kg'.")
@click.option("--rotors", type=int, default=1, show_default=True, help="Rotors sharing the weight.")
@click.option("--speed", default="0 m/s", show_default=True, help="Forward or climb speed.")
@click.option("--air-temp", default="20 C", show_default=True)
@click.option("--altitude", default="0 m", show_default=True)
@click.option("--max-rpm", default="", help="Top speed of the motor or prop (default: the maker's limit if known).")
@_common
def prop(prop, rpm, thrust, mass, rotors, speed, air_temp, altitude, max_rpm, out, question, as_json):
    """Propeller thrust, power and hover margin by blade element momentum theory.

    PROP is a bundled UIUC prop (apcsf_10x4.7, apcsf_9x4.7, apce_10x5, ...) or diameter x pitch in inches,
    like 10x4.7 or 12x6x3 for three blades.
    """
    from ..flow.propeller import PropStudy, run_prop
    from .main import _out_dir

    st = PropStudy(prop, rpm=float(rpm) if rpm else 0.0, thrust=_q(thrust, "force", "the thrust") if thrust else 0.0,
                   mass=_q(mass, "mass", "the mass") if mass else 0.0, rotors=rotors,
                   speed=_q(speed, "velocity", "the speed"), air_temp=_q(air_temp, "temperature", "the air temperature"),
                   altitude=_q(altitude, "length", "the altitude"), max_rpm=float(max_rpm) if max_rpm else 0.0,
                   question=question)
    try:
        with ui.progress(not as_json) as cb:
            cb("solve", f"blade element momentum theory on {prop}")
            o = run_prop(st, _out_dir(re.sub(r"[^\w-]", "_", prop), out))
    except ValueError as e:
        ui.fail(str(e))
    _finish(o, as_json)


def _displacement(text: str) -> tuple[float, str]:
    """A volume in m3, or a mass in t, kg or lb, which becomes a volume once the water is known."""
    t = text.strip().lower().replace(" ", "")
    m = re.match(r"^([\d.eE+-]+)(m3|m\^3)$", t)
    if m:
        return float(m.group(1)), "volume"
    return _q(text, "mass", "the displacement"), "mass"


@flow.command()
@click.option("--length", required=True, help="Waterline length, e.g. '12 m'.")
@click.option("--beam", required=True, help="Waterline beam.")
@click.option("--draft", required=True, help="Mean draft.")
@click.option("--displacement", required=True, help="Volume ('16 m3') or mass ('16 t', '35000 lb').")
@click.option("--speed", required=True, help="Speed, e.g. '8 kn'.")
@click.option("--power", default="", help="Installed engine power, e.g. '60 kW', to judge against.")
@click.option("--efficiency", type=float, default=0.55, show_default=True,
              help="Propulsive efficiency from engine to tow rope.")
@click.option("--cm", type=float, default=0.0, help="Midship coefficient (estimated when left out).")
@click.option("--cwp", type=float, default=0.0, help="Waterplane coefficient (estimated when left out).")
@click.option("--lcb", type=float, default=0.0, help="Centre of buoyancy forward of mid length, percent of length.")
@click.option("--transom", default="0 m", help="Immersed transom area as a side, e.g. '0.8 m' for 0.64 m2.")
@click.option("--fresh", is_flag=True, help="Fresh water instead of seawater.")
@_common
def hull(length, beam, draft, displacement, speed, power, efficiency, cm, cwp, lcb, transom, fresh, out, question,
         as_json):
    """Calm water resistance and power of a displacement hull (Holtrop and Mennen)."""
    from ..flow.hull import RHO_SEA, Hull, HullStudy, run_hull
    from .main import _out_dir

    disp, kind = _displacement(displacement)
    if kind == "mass":
        disp /= 999.1 if fresh else RHO_SEA
    side = _q(transom, "length", "the transom")
    h = Hull(_q(length, "length", "the length"), _q(beam, "length", "the beam"), _q(draft, "length", "the draft"),
             disp, cm=cm, cwp=cwp, lcb=lcb, transom_area=side * side)
    st = HullStudy(h, _q(speed, "velocity", "the speed"), _q(power, "power", "the power") if power else 0.0,
                   efficiency, fresh, question)
    try:
        o = run_hull(st, _out_dir("hull", out))
    except (ValueError, ZeroDivisionError) as e:
        ui.fail(str(e))
    _finish(o, as_json)
