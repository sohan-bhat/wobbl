"""wobbl crack: where a fatigue crack goes, when it fails the part, and how lives spread over many parts."""
from __future__ import annotations

import json

import click

from . import ui
from .thermal import show_generic


@click.command()
@click.argument("part")
@click.option("--material", "-m", required=True, help="Material with crack growth data, e.g. 7075-t6 (see --growth).")
@click.option("--load", "-l", "loads", multiple=True, required=True,
              help="Peak load of each cycle, as in wobbl run, e.g. '4 kN -x bearing @ hole d=8' (repeatable).")
@click.option("--fix", "-f", "fixes", multiple=True, required=True,
              help="Support, as in wobbl run, e.g. 'bolt holes d=6.6' (repeatable).")
@click.option("--crack", "-c", "crack", required=True,
              help="Starting crack, e.g. 'through 1.27 mm at hole d=8', 'corner 1 mm at hole d=8', "
                   "'surface a=1 c=2 at hottest'.")
@click.option("--r-ratio", type=float, default=0.1, show_default=True,
              help="Valley over peak load for constant amplitude cycles.")
@click.option("--spectrum", default="", help="File of cycles as load factors, one 'max,min' or 'max,min,count' "
                                             "per line, repeated as a block.")
@click.option("--law", type=click.Choice(["nasgro", "paris"]), default="nasgro", show_default=True)
@click.option("--growth", default="", help="Your own constants, 'C=2.7e-8,n=3.7,dkth=1.8,kic=29' (mm/cycle and "
                                           "MPa sqrt(m)).")
@click.option("--kc", type=float, default=None, help="Fracture toughness in MPa sqrt(m), in place of the material's.")
@click.option("--wheeler", type=float, default=0.0, help="Wheeler retardation exponent from tests (0 turns it off).")
@click.option("--target-life", type=float, default=None, help="Cycles every part should last.")
@click.option("--max-pof", type=float, default=0.01, show_default=True,
              help="Share of parts allowed to fail before the target life.")
@click.option("--detectable", default="1.27 mm", show_default=True,
              help="Crack size an inspection finds reliably, for the inspection interval.")
@click.option("--flaw-scatter", type=float, default=0.3, show_default=True,
              help="Scatter of the starting crack size (standard deviation of its logarithm).")
@click.option("--growth-scatter", type=float, default=None, help="Scatter of the growth coefficient C.")
@click.option("--load-scatter", type=float, default=0.05, show_default=True, help="Scatter of the load.")
@click.option("--samples", type=int, default=20000, show_default=True, help="Parts in the Monte Carlo.")
@click.option("--steps", type=int, default=12, show_default=True, help="Most growth steps to solve.")
@click.option("--budget", type=float, default=1200.0, show_default=True, help="Seconds for the crack path.")
@click.option("--quality", type=click.Choice(["fast", "normal", "fine"]), default="normal")
@click.option("--scale", type=float, default=1.0, help="Scale the geometry (1000 for files in meters).")
@click.option("--param", "params", multiple=True, help="Parameter of a .py part, e.g. thickness=6.")
@click.option("--keep", is_flag=True, help="Keep the work files (meshes and solver files).")
@click.option("--out", "-o", default=None, help="Output folder (default wobbl-out/<part>).")
@click.option("--question", default="", help="The question this run answers (shown in the report).")
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@click.option("--quiet", "-q", is_flag=True, help="No progress lines.")
def crack(part, material, loads, fixes, crack, r_ratio, spectrum, law, growth, kc, wheeler, target_life, max_pof,
          detectable, flaw_scatter, growth_scatter, load_scatter, samples, steps, budget, quality, scale, params,
          keep, out, question, as_json, quiet):
    """Fatigue crack growth: the crack's path through the part, cycles to failure and their spread.

    \b
    wobbl crack lug.step -m 7075-t6 --fix "bolt holes d=6.6" --load "4 kN -x bearing @ hole d=8" \\
      --crack "through 1.27 mm at hole d=8" --r-ratio 0.1 --target-life 100000
    """
    from pathlib import Path

    from ..fracture.study import run_fracture
    from .main import _out_dir

    if not Path(part).exists():
        ui.fail(f"file not found: {part}")
    out_dir = _out_dir(part, out)
    try:
        st = study_from_options(part, material, loads, fixes, crack, r_ratio=r_ratio, spectrum=spectrum, law=law,
                                growth=growth, kc=kc, wheeler=wheeler, target_life=target_life, max_pof=max_pof,
                                detectable=detectable, flaw_scatter=flaw_scatter, growth_scatter=growth_scatter,
                                load_scatter=load_scatter, samples=samples, steps=steps, budget=budget,
                                quality=quality, question=question, scale=scale, params=params, keep=keep)
        with ui.progress(not quiet and not as_json) as cb:
            o = run_fracture(st, out_dir, cb)
    except Exception as e:  # noqa: BLE001
        ui.fail(str(e).strip("'\""), getattr(e, "hint", ""))
    if as_json:
        click.echo(json.dumps(o.to_dict(), indent=1))
    else:
        show_generic(o)


def study_from_options(part, material, loads, fixes, crack, detectable="1.27 mm", params=(), **kw):
    """The study a crack command line describes, checked the way the run would read it."""
    from pathlib import Path

    from ..fracture.growth import law_for, parse_spectrum
    from ..fracture.spec import study_from
    from ..materials import resolve_material
    from ..units import parse_quantity

    st = study_from(str(Path(part).resolve()), material, list(loads), list(fixes), crack,
                    detectable=parse_quantity(detectable, "length", "mm").value, **kw)
    st.params = dict(p.split("=", 1) for p in params if "=" in p)
    law_for(resolve_material(material).key, st.law, st.growth, st.kc)
    if st.spectrum:
        parse_spectrum(st.spectrum, st.r_ratio)
    return st
