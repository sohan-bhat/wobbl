"""wobbl buckling, wobbl vibration and wobbl fatigue: how a part lasts, not only whether it holds once."""
from __future__ import annotations

import json
from pathlib import Path

import click

from . import ui
from .thermal import show_generic


def _part_options(f):
    opts = [
        click.option("--material", "-m", required=True, help="Material name or custom:E=..,nu=..,yield=..,density=.."),
        click.option("--fix", "-f", "fixes", multiple=True, help="Support, e.g. 'x=min' or 'bolt holes d=3.2'."),
        click.option("--target-sf", type=float, default=None, help="Required safety factor (default: the material's)."),
        click.option("--quality", type=click.Choice(["fast", "normal", "fine"]), default="normal"),
        click.option("--mesh-size", type=float, default=None, help="Global element size in mm."),
        click.option("--scale", type=float, default=1.0, help="Scale the geometry (1000 for files in meters)."),
        click.option("--body", type=int, default=None, help="Pick one body from a multi body file."),
        click.option("--param", "params", multiple=True, help="Parameter of a .py part, e.g. thickness=6."),
        click.option("--keep", is_flag=True, help="Keep the solver's work files for a closer look."),
        click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<part>)."),
        click.option("--question", default="", help="The question this run answers (shown in the report)."),
        click.option("--json", "as_json", is_flag=True, help="Print the result as JSON."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _part_kw(kw) -> dict:
    """The part options every life command shares, as the entry functions take them."""
    return {"quality": kw["quality"], "scale": kw["scale"], "body": kw["body"], "mesh_size": kw["mesh_size"],
            "question": kw["question"], "params": dict(p.split("=", 1) for p in kw.get("params", ()) if "=" in p)}


def _finish(run, out_dir, as_json):
    try:
        with ui.progress(not as_json) as cb:
            o = run(out_dir, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


@click.command()
@click.argument("part")
@_part_options
@click.option("--load", "-l", "loads", multiple=True, help="The load that compresses it, e.g. '1000 N -x @ x=max'.")
@click.option("--shape", type=click.Choice(["auto", "column", "plate", "cylinder", "dome", "general"]), default="auto",
              help="Shape class for the imperfection knockdown (default: found from the part).")
@click.option("--modes", type=int, default=4, show_default=True, help="Buckling modes to find.")
def buckling(part, loads, shape, modes, **kw):
    """Buckling load factor with an imperfection knockdown for the shape.

    \b
    wobbl buckling strut.step -m 6061-t6 --fix "z=min" --load "2 kN -z @ z=max" --target-sf 3
    """
    from ..life import api
    from .main import _out_dir

    if not loads or not kw["fixes"]:
        ui.fail("wobbl buckling needs at least one --load and one --fix",
                "Example: wobbl buckling strut.step -m 6061-t6 --fix 'z=min' --load '2 kN -z @ z=max'")
    _finish(lambda out, cb: api.buckling(part, kw["material"], kw["fixes"], loads, target_sf=kw["target_sf"],
                                         shape=shape, modes=modes, out=out, progress=cb, keep=kw["keep"],
                                         **_part_kw(kw)), _out_dir(part, kw["out"]), kw["as_json"])


@click.command()
@click.argument("part")
@_part_options
@click.option("--mass", "masses", multiple=True,
              help="A mass the part carries, e.g. '45 g at 150,0,25 @ top and within 15 of 150,0,15' (repeatable).")
@click.option("--modes", type=int, default=15, show_default=True, help="Natural frequencies to find (up to 40).")
@click.option("--rpm", multiple=True, help="Rotor speed or range, e.g. '9000 rpm' or '4000-9000 rpm' (repeatable).")
@click.option("--blades", type=int, default=2, show_default=True, help="Blades on the rotor, for blade pass.")
@click.option("--frequency", "frequencies", multiple=True, help="Other tone or band to avoid, e.g. '120 Hz'.")
@click.option("--margin", type=float, default=0.2, show_default=True,
              help="Keep natural frequencies this share away from every excitation.")
@click.option("--sine", default="",
              help="Sine sweep: of the base, e.g. '2 g z 10-500 Hz', or a band alone, '10-500 Hz', to sweep the loads.")
@click.option("--load", "-l", "loads", multiple=True,
              help="A load that shakes the part as a harmonic force across the --sine band, e.g. '5 N z @ top'.")
@click.option("--psd", default="", help="Random base input: gevs-qual, gevs-accept or '20:0.01, 80:0.04, 2000:0.01'.")
@click.option("--axis", default="z", show_default=True, help="Axis of the random input.")
@click.option("--duration", default="", help="Time at the random or sine input, e.g. '2 min', for fatigue.")
@click.option("--damping", type=float, default=None, help="Damping ratio, e.g. 0.02 (default by material).")
def vibration(part, masses, modes, rpm, blades, frequencies, margin, sine, loads, psd, axis, duration, damping,
              **kw):
    """Natural frequencies against rotors and tones, and the stress of a sine sweep or random input.

    \b
    wobbl vibration arm.py -m pa12 --fix "bolt holes d=3.2" --mass "60 g at 150,0,25 @ top and within 15 of 150,0,15" --rpm "4000-9000 rpm"
    wobbl vibration bracket.step -m 6061-t6 --fix "bolt holes" --psd gevs-qual --axis z
    """
    from ..life import api
    from ..life.vibspec import VibrationStudy
    from ..structural.spec import StructuralStudy
    from .main import _out_dir

    if not kw["fixes"]:
        ui.fail("wobbl vibration needs at least one --fix, the way the part is mounted",
                "Example: wobbl vibration arm.py -m pa12 --fix 'bolt holes d=3.2' --rpm '6000 rpm'")
    try:   # read the speeds and tones before any solving, so a typo fails at once
        VibrationStudy(StructuralStudy(part=part, material=kw["material"]), rpm=list(rpm), blades=blades,
                       frequencies=list(frequencies)).excitations()
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e))
    _finish(lambda out, cb: api.vibration(part, kw["material"], kw["fixes"], loads=loads, masses=masses, rpm=rpm,
                                          blades=blades, frequencies=frequencies, margin=margin, sine=sine, psd=psd,
                                          axis=axis, duration=duration, damping=damping, modes=modes,
                                          target_sf=kw["target_sf"], out=out, progress=cb, keep=kw["keep"],
                                          **_part_kw(kw)), _out_dir(part, kw["out"]), kw["as_json"])


@click.command()
@click.argument("source", required=False, default="")
@click.option("--material", "-m", default="", help="Material, for stresses given directly.")
@click.option("--amplitude", default="", help="Nominal stress amplitude, e.g. '80 MPa' (with --material).")
@click.option("--mean", default="", help="Nominal mean stress, e.g. '40 MPa'.")
@click.option("--kt", type=float, default=None, help="Stress concentration at the notch.")
@click.option("--radius", default="", help="Notch radius, e.g. '1 mm', for the notch sensitivity.")
@click.option("--ratio", "-R", type=float, default=0.0, show_default=True,
              help="Load ratio min/max for a strength result: 0 on and off, -1 fully reversed.")
@click.option("--cycles", default="1e7", show_default=True, help="Life the part needs, in cycles.")
@click.option("--block", "blocks", multiple=True, help="Duty cycle block, e.g. '100% x 2e4' or '0.6 x 1e6 R=-1'.")
@click.option("--history", default="",
              help="A load history to rainflow count: a file or numbers, multiples of the studied load (MPa for --material).")
@click.option("--repeats", type=float, default=1.0, show_default=True, help="Times the duty cycle or history runs.")
@click.option("--fix", "-f", "fixes", multiple=True, help="Support, when SOURCE is a part to solve first.")
@click.option("--load", "-l", "loads", multiple=True, help="Load, when SOURCE is a part to solve first.")
@click.option("--quality", type=click.Choice(["fast", "normal", "fine"]), default="normal",
              help="Mesh quality of the strength study, when SOURCE is a part.")
@click.option("--duration", default="", help="Time at the input, for a vibration result.")
@click.option("--surface", type=click.Choice(["auto", "polished", "ground", "machined", "cold-drawn", "hot-rolled",
                                              "as-cast", "as-forged"]), default="auto")
@click.option("--size", default="", help="Equivalent diameter of the section, e.g. '12 mm'.")
@click.option("--loading", type=click.Choice(["auto", "bending", "axial", "torsion"]), default="auto")
@click.option("--reliability", type=float, default=0.9, show_default=True, help="Share of parts that must survive.")
@click.option("--mean-stress", "mean_model", type=click.Choice(["goodman", "gerber", "soderberg"]), default="goodman",
              show_default=True)
@click.option("--target-sf", type=float, default=None, help="Required safety factor on stress.")
@click.option("--out", "-o", default=None)
@click.option("--question", default="")
@click.option("--json", "as_json", is_flag=True)
def fatigue(source, material, amplitude, mean, kt, radius, ratio, cycles, blocks, history, repeats, fixes, loads,
            quality, duration, surface, size, loading, reliability, mean_model, target_sf, out, question, as_json):
    """Fatigue life from a strength or vibration result, a part and its loads, or stresses you give.

    \b
    wobbl fatigue wobbl-out/bracket --ratio 0 --cycles 1e6
    wobbl fatigue wobbl-out/bracket --block "100% x 2e4" --block "60% x 1e6" --repeats 10
    wobbl fatigue wobbl-out/bracket --history loads.csv --repeats 1000
    wobbl fatigue bracket.step -m 6061-t6 --fix "bolt holes" --load "300 N down @ top" --ratio -1
    wobbl fatigue --material 4140 --amplitude "200 MPa" --mean "100 MPa" --kt 2 --radius "1 mm"
    """
    from ..life import api
    from .main import _out_dir

    is_part = bool(source) and Path(source).is_file()
    if source and not Path(source).exists():
        ui.fail(f"no such result folder or part: {source}")
    if is_part and (not material or not fixes or not loads):
        ui.fail("fatigue on a part solves it first, so it needs --material, --fix and --load",
                "Example: wobbl fatigue bracket.step -m 6061-t6 --fix 'bolt holes' --load '300 N down @ top'")
    kw = dict(material=material, amplitude=amplitude, mean=mean, kt=kt, radius=radius, ratio=ratio, cycles=cycles,
              blocks=blocks, history=history, repeats=repeats, duration=duration, surface=surface, size=size,
              loading=loading, reliability=reliability, mean_stress=mean_model, target_sf=target_sf,
              question=question)
    if is_part:
        kw.update(part=source, fixtures=fixes, loads=loads, quality=quality)
    else:
        kw["source"] = source
    name = (Path(source).stem if is_part else Path(source).name) + "-fatigue" if source else "fatigue"
    _finish(lambda o, cb: api.fatigue(out=o, progress=cb, **kw), _out_dir(name, out), as_json)
