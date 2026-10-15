"""wobbl magnetic, electric, board, antenna and sound: field studies solved with GetDP."""
from __future__ import annotations

import json

import click

from . import ui
from .thermal import show_generic


def _common(f):
    opts = [click.option("--set", "sets", multiple=True,
                         help="Template value with its unit, e.g. --set current=0.5A --set stroke=6mm (repeatable)."),
            click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<template>)."),
            click.option("--question", default="", help="The question this run answers (shown in the report)."),
            click.option("--keep", is_flag=True, help="Keep the meshes and solver files in <out>/work."),
            click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")]
    for o in reversed(opts):
        f = o(f)
    return f


def _run(domain: str, template: str, sets, out, question, keep, as_json):
    from ..fields.tools import check_params, runner
    from .main import _out_dir

    params = {}
    for s in sets:
        if "=" not in s:
            ui.fail(f"--set needs key=value, got '{s}'")
        k, v = s.split("=", 1)
        params[k.strip()] = v.strip()
    try:
        check_params(domain, template, params)
        with ui.progress(not as_json) as cb:
            o = runner(domain)(template, params, _out_dir(template, out), question, keep, log=cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


@click.command()
@click.argument("template", type=click.Choice(["solenoid", "electromagnet", "magnet"]))
@_common
def magnetic(template, sets, out, question, keep, as_json):
    """Magnetic force, inductance and saturation of actuators and magnets.

    \b
    wobbl magnetic solenoid --set turns=800 --set current=0.6A --set stroke=6mm --set need=8N
    wobbl magnetic electromagnet --set diameter=50mm --set turns=500 --set current=1A --set need=100N
    wobbl magnetic magnet --set diameter=20mm --set height=10mm --set grade=n42 --set plate=5mm
    """
    _run("magnetic", template, sets, out, question, keep, as_json)


@click.command()
@click.argument("template", type=click.Choice(["plates", "coax", "sphere"]))
@_common
def electric(template, sets, out, question, keep, as_json):
    """Capacitance, peak field and breakdown margin.

    \b
    wobbl electric plates --set diameter=100mm --set gap=2mm --set voltage=3kV
    wobbl electric coax --set inner_d=1mm --set outer_d=4.6mm --set voltage=5kV --set dielectric=pe
    wobbl electric sphere --set diameter=20mm --set gap=5mm --set voltage=20kV
    """
    _run("electric", template, sets, out, question, keep, as_json)


@click.command()
@click.argument("template", type=click.Choice(["microstrip", "stripline", "coplanar", "coupled", "coax"]))
@_common
def board(template, sets, out, question, keep, as_json):
    """Trace impedance, inductance and capacitance per length, and crosstalk.

    \b
    wobbl board microstrip --set width=0.3mm --set height=0.17mm --set dielectric=fr4 --set length=20mm
    wobbl board coupled --set width=0.2mm --set spacing=0.15mm --set height=0.1mm --set length=50mm --set rise=0.5ns
    wobbl board coax --set inner_d=0.91mm --set outer_d=2.95mm --set dielectric=pe
    """
    _run("board", template, sets, out, question, keep, as_json)


@click.command()
@click.argument("template", type=click.Choice(["dipole", "monopole", "patch", "loop", "invf", "link"]))
@_common
def antenna(template, sets, out, question, keep, as_json):
    """Antenna resonance, impedance, S11 and gain, and wireless range.

    \b
    wobbl antenna dipole --set frequency=2.4GHz --set length=58mm --set diameter=1mm
    wobbl antenna monopole --set frequency=915MHz --set tx_power=14dBm --set sensitivity=-120dBm --set environment=suburban
    wobbl antenna patch --set frequency=2.4GHz --set height=1.6mm --set dielectric=fr4
    wobbl antenna patch --set frequency=2.4GHz --set length=29mm --set width=38mm --question "what is its S11?"
    wobbl antenna loop --set frequency=868MHz --set circumference=345mm
    wobbl antenna invf --set frequency=2.4GHz --set dielectric=fr4 --set height=1.6mm
    wobbl antenna link --set frequency=2.4GHz --set tx_power=10dBm --set sensitivity=-90dBm --set distance=30m
    """
    _run("antenna", template, sets, out, question, keep, as_json)


@click.command()
@click.argument("template", type=click.Choice(["room", "enclosure", "muffler", "panel"]))
@_common
def sound(template, sets, out, question, keep, as_json):
    """Room and enclosure modes, muffler transmission loss and the sound a vibrating panel makes.

    \b
    wobbl sound room --set size=5x4x2.7 --set fmax=120Hz
    wobbl sound muffler --set pipe_d=50mm --set chamber_d=150mm --set chamber_length=300mm
    wobbl sound panel --set diameter=200mm --set velocity=0.01 --set distance=1m --set limit=70
    """
    _run("sound", template, sets, out, question, keep, as_json)
