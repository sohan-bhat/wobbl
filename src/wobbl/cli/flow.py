"""wobbl flow: air and water flow, from pipes and ducts to drag, fan cooling, propellers and hulls."""
from __future__ import annotations

import inspect
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


def _run(kind: str, params: dict, name: str, out, as_json: bool, note: str):
    """Build the study the same way the MCP tool does, run it and show the result."""
    from ..flow.tools import build
    from .main import _out_dir

    try:
        study, run = build(kind, params)
        with ui.progress(not as_json) as cb:
            cb("solve", note)
            kw = {"progress": cb} if "progress" in inspect.signature(run).parameters else {}
            o = run(study, _out_dir(re.sub(r"[^\w-]", "_", name), out), **kw)
    except ValueError as e:
        ui.fail(str(e), getattr(e, "hint", ""))
    _finish(o, as_json)


@click.group()
def flow():
    """Air and water flow through pipes and around bodies, fan cooling, propellers and hulls.

    \b
    wobbl flow pipe --length 40m --diameter 25mm --flow "30 L/min" --fittings "3 elbow90"
    wobbl flow drag sphere --size 22cm --speed 20m/s
    wobbl flow heatsink heatsink.py --fan "10 CFM, 20 Pa" --power 30W --ambient 40C
    wobbl flow prop apcsf_10x4.7 --mass 1.2kg --rotors 4
    wobbl flow hull --length 12m --beam 3.6m --draft 0.9m --displacement 16t --speed 8kn
    """


@flow.command()
@click.option("--length", required=True, help="Pipe length, e.g. '40 m'.")
@click.option("--flow", "flow_rate", required=True, help="Flow rate, e.g. '30 L/min', '2 m3/h' or '40 CFM'.")
@click.option("--diameter", default="", help="Bore of a round pipe, e.g. '25 mm'.")
@click.option("--width", default="", help="Width of a rectangular duct.")
@click.option("--height", default="", help="Height of a rectangular duct.")
@click.option("--fluid", default="water", show_default=True, help="air, water, seawater or custom:rho=..,mu=..")
@click.option("--temp", default="20 C", show_default=True, help="Fluid temperature.")
@click.option("--material", default="pvc", show_default=True,
              help="Wall material, one of smooth, pvc, copper, pex, stainless, steel, galvanized, cast-iron "
                   "or concrete.")
@click.option("--roughness", default="", help="Wall roughness, e.g. '0.05 mm' (overrides the material).")
@click.option("--fittings", default="", help="e.g. '3 elbow90, gate-valve, exit' or 'K=4.5'.")
@click.option("--rise", default="0 m", show_default=True, help="Height of the outlet above the inlet.")
@click.option("--max-dp", default="", help="Pressure the pump or supply can give, e.g. '50 kPa'.")
@_common
def pipe(length, flow_rate, diameter, width, height, fluid, temp, material, roughness, fittings, rise, max_dp, out,
         question, as_json):
    """Pressure drop through a pipe or duct with its fittings, on the Moody chart."""
    _run("pipe", dict(length=length, flow=flow_rate, diameter=diameter, width=width, height=height, fluid=fluid,
                      temp=temp, material=material, roughness=roughness, fittings=fittings, rise=rise, max_dp=max_dp,
                      question=question), "pipe", out, as_json, "Darcy Weisbach and the Moody chart")


@flow.command()
@click.argument("shape", type=click.Choice(["sphere", "cylinder", "disk", "square-plate", "cube"]))
@click.option("--size", required=True, help="Diameter or side, e.g. '22 cm'.")
@click.option("--speed", required=True, help="Speed of the stream, e.g. '20 m/s' or '30 km/h'.")
@click.option("--length", default="", help="Span of a cylinder (left out for a very long one).")
@click.option("--fluid", default="air", show_default=True, help="air, water, seawater or custom:rho=..,mu=..")
@click.option("--temp", default="20 C", show_default=True)
@click.option("--cd", type=float, default=0.0, help="A known drag coefficient instead of the published curve.")
@click.option("--max-drag", default="", help="Drag the design can take, e.g. '15 N'.")
@_common
def drag(shape, size, speed, length, fluid, temp, cd, max_drag, out, question, as_json):
    """Drag force on a sphere, cylinder, plate or cube from published drag curves."""
    _run("drag", dict(shape=shape, size=size, speed=speed, length=length, fluid=fluid, temp=temp, cd=cd,
                      max_drag=max_drag, question=question), shape, out, as_json, "published drag curves")


@flow.command()
@click.argument("part")
@click.option("--fan", default="", help="The fan's free flow and shut off pressure, e.g. '10 CFM, 20 Pa'.")
@click.option("--velocity", default="", help="Air speed between the fins instead of a fan, e.g. '2 m/s'.")
@click.option("--power", "-p", required=True, help="Heat the chip makes, e.g. '30 W'.")
@click.option("--heat-faces", default="", help="Faces the heat enters through, else it is made through the volume.")
@click.option("--material", "-m", default="6063-t5", show_default=True)
@click.option("--ambient", default="25 C", show_default=True, help="Air entering the fins.")
@click.option("--limit", default="", help="Temperature limit, e.g. '85 C'.")
@click.option("--radiation", type=float, default=0.0, help="Surface emissivity (0 turns radiation off).")
@click.option("--keep", is_flag=True, help="Keep the CFD and FEA work folders.")
@_common
def heatsink(part, fan, velocity, power, heat_faces, material, ambient, limit, radiation, keep, out, question,
             as_json):
    """Fan cooled heat sink, with CFD between the fins setting the cooling and CalculiX the temperature.

    PART is a parametric .py heat sink whose PARAMS give base, size, fins, fin and fin_height in mm, with the
    fins standing on a base at z=0 and running along y.
    """
    from pathlib import Path

    _run("heatsink", dict(part=part, fan=fan, velocity=velocity, power=power, heat_faces=heat_faces,
                          material=material, ambient=ambient, limit=limit, radiation=radiation, keep=keep,
                          question=question), Path(part).stem, out, as_json, "OpenFOAM between the fins")


@flow.command()
@click.argument("prop", default="apcsf_10x4.7")
@click.option("--rpm", default="", help="Speed to run at, e.g. 6000.")
@click.option("--thrust", default="", help="Thrust wanted from each rotor, e.g. '4 N'.")
@click.option("--mass", default="", help="Mass of the whole craft to hover, e.g. '1.2 kg'.")
@click.option("--rotors", type=int, default=1, show_default=True, help="Rotors sharing the weight.")
@click.option("--speed", default="0 m/s", show_default=True, help="Forward or climb speed.")
@click.option("--air-temp", default="20 C", show_default=True)
@click.option("--altitude", default="0 m", show_default=True)
@click.option("--max-rpm", default="", help="Top speed of the motor or prop, else the maker's limit when known.")
@_common
def prop(prop, rpm, thrust, mass, rotors, speed, air_temp, altitude, max_rpm, out, question, as_json):
    """Propeller thrust, power and hover margin by blade element momentum theory.

    PROP is a bundled UIUC prop (apcsf_10x4.7, apcsf_9x4.7, apce_10x5, ...) or diameter x pitch in inches,
    like 10x4.7 or 12x6x3 for three blades.
    """
    _run("prop", dict(prop=prop, rpm=rpm, thrust=thrust, mass=mass, rotors=rotors, speed=speed, air_temp=air_temp,
                      altitude=altitude, max_rpm=max_rpm, question=question), prop, out, as_json,
         f"blade element momentum theory on {prop}")


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
@click.option("--transom-area", type=float, default=0.0, help="Immersed transom area at rest, m2.")
@click.option("--fresh", is_flag=True, help="Fresh water instead of seawater.")
@_common
def hull(length, beam, draft, displacement, speed, power, efficiency, cm, cwp, lcb, transom_area, fresh, out,
         question, as_json):
    """Calm water resistance and power of a displacement hull (Holtrop and Mennen)."""
    _run("hull", dict(length=length, beam=beam, draft=draft, displacement=displacement, speed=speed, power=power,
                      efficiency=efficiency, cm=cm, cwp=cwp, lcb=lcb, transom_area=transom_area, fresh=fresh,
                      question=question), "hull", out, as_json, "Holtrop and Mennen")
